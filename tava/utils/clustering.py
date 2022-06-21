# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn.functional as F
from sklearn_extra.cluster import KMedoids
from tava.utils.transforms import matrix_to_quaternion


@torch.no_grad()
def so3_geodesic_distance(rot_mat1, rot_mat2):
    """Geodesic distance between two rotation matrix in SO(3)
    
    :params rot_mat1: [M, ..., 3, 3]
    :params rot_mat2: [N, ..., 3, 3]
    :return [M, N, ...]
    """
    assert rot_mat1.shape[-2:] == rot_mat2.shape[-2:] == (3, 3)
    r1 = matrix_to_quaternion(rot_mat1)
    r2 = matrix_to_quaternion(rot_mat2)
    inner = torch.einsum("m...i,n...i->mn...", r1, r2)
    dists = (1 - inner ** 2).clamp(0, 1)
    return dists


@torch.no_grad()
def verts_euclidean_distance(verts1, verts2):
    """Euclidean distance between two set of vertices
    
    :params verts1: [M, ..., 3]
    :params verts2: [N, ..., 3]
    :return [M, N, ...]
    """
    diffs = verts1[:, None, ...] - verts2[None, :, ...]
    dists = torch.linalg.norm(diffs, dim=-1)
    return dists


@torch.no_grad()
def pose_distance(metric, **kwargs):
    assert metric in ["rotation", "verts"]
    if metric == "verts":
        subsample = kwargs.get("subsample", 1)
        transform_global1 = kwargs["transform_global1"]
        transform_global2 = kwargs["transform_global2"]
        verts1 = kwargs["verts1"][..., ::subsample, :]
        verts2 = kwargs["verts2"][..., ::subsample, :]
        
        verts1_proj = torch.einsum(
            "...ij,...nj->...ni", 
            transform_global1.inverse(), 
            F.pad(verts1, (0, 1), value=1)
        )[..., 0:3]
        verts2_proj = torch.einsum(
            "...ij,...nj->...ni", 
            transform_global2.inverse(), 
            F.pad(verts2, (0, 1), value=1)
        )[..., 0:3]
        return verts_euclidean_distance(verts1_proj, verts2_proj).mean(dim=-1)
    
    elif metric == "rotation":
        transform_global1 = kwargs["transform_global1"]
        transform_global2 = kwargs["transform_global2"]
        bones1 = kwargs["bones1"]
        bones2 = kwargs["bones2"]
        matrix1 = bones1.matrix_relative("world", transform_global1)[..., 0:3, 0:3]
        matrix2 = bones2.matrix_relative("world", transform_global2)[..., 0:3, 0:3]
        return so3_geodesic_distance(matrix1, matrix2).mean(dim=-1)

@torch.no_grad()
def clustering(dists, n = 10, **kwargs):
    assert dists.dim() == 2
    kmedoids = KMedoids(
        n_clusters=n, metric="precomputed", method="pam", **kwargs
    ).fit(dists.cpu().numpy())
    return kmedoids.labels_, kmedoids.medoid_indices_


@torch.no_grad()
def train_val_test_split(transform_global, verts, ncluster=10, seed=None):
    assert verts.dim() == 3  # [B, N, 3]

    if seed is None:
        seed = verts.numel()
    # for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    # first exclude a random continuous clip with 1/10 data as test sequences
    total_size = verts.shape[0]
    n_test = total_size // 10
    test_start = torch.randint(
        0, total_size - n_test, size=(1,), generator=g
    ).item()
    test_end = test_start + n_test
    ids_test = torch.arange(start=test_start, end=test_end).to(verts.device)
    ids_trainval = torch.cat([
        torch.arange(start=0, end=test_start),
        torch.arange(start=test_end, end=total_size)
    ])

    # select the trainval set for split.
    verts = verts[ids_trainval]
    transform_global = transform_global[ids_trainval]    

    # calculate dists on all poses
    dists = pose_distance(
        metric="verts", subsample=verts.shape[1] // 100,
        verts1=verts, transform_global1=transform_global,
        verts2=verts, transform_global2=transform_global,
    )
    
    # cluster the pose into nclusters
    cluster_labels, cluster_centers = clustering(dists, n=ncluster)
    cluster_labels = torch.from_numpy(cluster_labels).long().to(verts.device)
    cluster_centers = torch.from_numpy(cluster_centers).long().to(verts.device)
    
    # withhold the cluster with largest distance to others as Out-Of-Dist val set
    icluster_val_ood = dists[
        torch.meshgrid(cluster_centers, cluster_centers, indexing="ij")
    ].mean(dim=-1).argmax()
    ids_val_ood = torch.where(icluster_val_ood == cluster_labels)[0]
    
    # withhold random 1/3 of the data in each other cluster as In-Dist val set
    # the rest as train set.
    ids_val_ind, ids_train = [], []
    for icluster in torch.unique(cluster_labels):
        if icluster == icluster_val_ood:
            continue
        else:
            icluster_ids = torch.where(icluster == cluster_labels)[0]
            icluster_ids = icluster_ids[torch.randperm(len(icluster_ids), generator=g)]
            n_total = icluster_ids.shape[0]
            n_val, n_train = n_total // 3, n_total - n_total // 3
            icluster_ids_val, icluster_ids_train = torch.split(icluster_ids, [n_val, n_train])
            ids_val_ind.append(icluster_ids_val)
            ids_train.append(icluster_ids_train)
    ids_val_ind = torch.cat(ids_val_ind)
    ids_train = torch.cat(ids_train)

    # resume the ids because of test set chunk split
    ids_train[ids_train >= test_start] += n_test
    ids_val_ind[ids_val_ind >= test_start] += n_test
    ids_val_ood[ids_val_ood >= test_start] += n_test

    assert torch.cat(
        [ids_train, ids_val_ind, ids_val_ood, ids_test]
    ).unique().shape[0] == total_size

    return {
        "all": torch.arange(total_size),
        "train": ids_train,  "test": ids_test,
        "val_ind": ids_val_ind, "val_ood": ids_val_ood,
    }
