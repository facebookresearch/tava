# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F
from tava.utils.bone import project_to_bone_space
from tava.utils.structures import Bones
from torch import Tensor


class _PartWiseDPEncoder(nn.Module):
    """Part Wise Rigid Deform Positional Encoder

    Rigidly deform the point x back to the **bone** space {x'_i}, where

        x'_i = T_i ^ {-1} @ x

    The representation of x can then be {x'_i} with positional encoding:

        x_enc = {PE(x'_1), ... PE(x'_J)}

    The viewdirs v can also be represented similarly:

        v_enc = {PE(v'_1), ... PE(v'_J)}
        where v'_i = R_i ^ {-1} @ v

    In this part-wise encoding, each transform is processed **independently**,
    So this one requires multiple **independent** NeRF, one for each part.

    Reference:
        Neural Articulated Radiance Field. (The `NeRF_P` version)

    Note in the paper NARF,
    1. the 6D SE(3) representation with positional encoding [6 x PE x J],
    is injected to the color branch of NeRF.
    2. an extra PE(bone_length) \in [3 x PE x J] is used as the input
    to the entire NeRF mlp, to distinguish different subjects.
    """

    def __init__(
        self,
        posi_enc: nn.Module,
        n_transforms: int,
        with_bkgd: bool = True,
    ):
        super().__init__()
        self.posi_enc = posi_enc
        self.with_bkgd = with_bkgd
        self.n_transforms = n_transforms

    def forward(
        self,
        x: Tensor,
        x_cov: Tensor,
        bones: Bones,
        rigid_clusters: Tensor = None,
    ):
        """
        :params x: [..., 3],
        :params x_cov: For mipnerf posi enc. [..., 3(, 3)]
        :params bones: bones in the world space [n_bones,]
        :params rigid_clusters: Optionally. The cluster id for each bone in
            torch.int32. The bones with the same cluster id are supposed to
            be moved rigidly together. `None` means each bones moves
            individually. [n_bones,]
        :return
            x_enc: [..., n_bones (+ 1), self.out_dim]
            x_proj: [..., n_bones (+ 1), 3]
        """
        # project x from world space to bone space
        x_proj = project_to_bone_space(bones, x)  # [..., n_bones, 3]
        if rigid_clusters is not None:
            # keep those that are unique.
            x_proj = torch.stack(
                [
                    x_proj[..., rigid_clusters == cluster_id, :].mean(dim=-2)
                    for cluster_id in torch.unique(rigid_clusters)
                ],
                dim=-2,
            )
        # attach identity transform to it for background / empty space.
        if self.with_bkgd:
            # [..., n_bones + 1, 3]
            x_proj = torch.cat([x_proj, x[..., None, :]], dim=-2)
        # positional encoding
        if x_cov is None:  # PE
            x_enc = self.posi_enc(x_proj)
        else:  # IPE
            x_cov = (
                x_cov.unsqueeze(-2).expand(list(x_proj.shape[:-1]) + [3])
                if self.posi_enc.diag
                else x_cov.unsqueeze(-3).expand(
                    list(x_proj.shape[:-1]) + [3, 3]
                )
            )
            x_enc = self.posi_enc((x_proj, x_cov))
        return x_enc, x_proj


class DisentangledDPEncoder(_PartWiseDPEncoder):
    """Disentangled Rigid Deform Positional Encoder

    Reference:
        Neural Articulated Radiance Field. (The `NeRF_D` version)

    Similar to `HolisticDPEncoder` but instead of a naive concatenation,
    some light networks are learned to produce "probability" of each part
    independently (one for each). The probability is first multiplied to each
    positional encoding, then perform concatenation.
    """

    def __init__(
        self,
        posi_enc: nn.Module,
        n_transforms: int,
        with_bkgd: bool = True,
    ):
        super().__init__(posi_enc, n_transforms, with_bkgd=with_bkgd)
        # networks for part probability
        self.nets = nn.ModuleList(
            [
                (
                    nn.Sequential(
                        nn.Linear(posi_enc.out_dim, 10),
                        nn.ReLU(inplace=True),
                        nn.Linear(10, 1),
                    )
                )
                for _ in range(self.n_transforms + int(self.with_bkgd))
            ]
        )

    @property
    def warp_dim(self):
        return (self.n_transforms + int(self.with_bkgd)) * 3

    @property
    def out_dim(self):
        return (self.n_transforms + int(self.with_bkgd)) * self.posi_enc.out_dim

    @property
    def diag(self):
        return self.posi_enc.diag

    def forward(
        self,
        x: Tensor,
        x_cov: Tensor,
        bones: Bones,
        rigid_clusters: Tensor = None,
    ):
        """
        :params x: [..., 3],
        :params x_cov: For mipnerf posi enc. [..., 3(, 3)]
        :params bones: bones in the world space [n_bones,]
        :params rigid_clusters: Optionally. The cluster id for each bone in
            torch.int32. The bones with the same cluster id are supposed to
            be moved rigidly together. `None` means each bones moves
            individually. [n_bones,]
        :return
            x_enc: [..., self.out_dim]
            x_warp: [..., self.warp_dim]
        """
        x_enc, x_proj = super().forward(x, x_cov, bones, rigid_clusters)
        x_prob = torch.stack(
            [self.nets[i](x_enc[..., i, :]) for i in range(x_enc.shape[-2])],
            dim=-2,
        )  # [..., n_bones (+ 1), 1]
        x_enc = x_enc * F.softmax(x_prob, dim=-2)
        x_enc = x_enc.reshape(list(x.shape[:-1]) + [self.out_dim])
        x_warp = x_proj * F.softmax(x_prob, dim=-2)
        x_warp = x_warp.reshape(list(x.shape[:-1]) + [self.warp_dim])
        return x_enc, x_warp
