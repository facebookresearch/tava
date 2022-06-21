# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn.functional as F
from tava.utils.transforms import axis_angle_to_matrix


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, new_params=False):
    """ SMPL Linear Blend Skinning.

    Args:
        betas: [batch_size, n_betas]
        pose: [batch_size, J + 1, 3]
        v_template: [batch_size, n_verts, 3]
        shapedirs: [n_verts, 3, n_betas]
        posedirs: [J * 9, n_verts * 3]
        J_regressor: [J, n_verts]
        parents: [J,]
        lbs_weights: [n_verts, J + 1]      
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device
    dtype = betas.dtype

    # Add shape blend shapes
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    J = vertices2joints(J_regressor, v_shaped)

    if new_params:
        # Add pose blend shapes
        ident = torch.eye(3, dtype=dtype, device=device)
        rot_mats = batch_rodrigues(pose.view([batch_size, -1, 3]))
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
        v_posed = pose_offsets + v_shaped
    else:
        v_posed = v_shaped

    # Get the global joint location
    J_transformed, A, A_bone = batch_rigid_transform(rot_mats, J, parents)

    # Do skinning:
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(
        W, A.view(batch_size, num_joints, 16)
    ).view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, A, A_bone


def blend_shapes(betas, shape_disps):
    """ Calculate blended shapes for vertex displacements.
    
    Args:
        betas: [batch_size, n_betas]
        shape_disps: [n_verts, 3, n_betas]
    Returns:
        Vertex displacements due to shape variabltion.
        [batch_size, n_verts, 3]
    """
    blend_shape = torch.einsum('bk,vjk->bvj', betas, shape_disps)
    return blend_shape


def vertices2joints(J_regressor, vertices):
    """ Regress joint locations from vertices.

    Args:
        J_regressor: [n_joints, n_verts]
        vertices: [batch_size, n_verts, 3]
    Returns:
        Joint locations. [batch_size, n_joints, 3]
    """
    return torch.einsum('jv,bvk->bjk', J_regressor, vertices)


def to_se4(R, t):
    """ Convert R and t to 4 x 4 transformation matrices
    
    Args:
        R: rotation matrix. [batch_size, 3, 3]
        t: translation vector. [batch_size, 3, 1]
    Returns:
        SE(4) transformation matrix. [batch_size, 4, 4]
    """
    return torch.cat([
        F.pad(R, [0, 0, 0, 1], value=0),
        F.pad(t, [0, 0, 0, 1], value=1)
    ], dim=-1)


def batch_rodrigues(rot_vecs):
    return axis_angle_to_matrix(rot_vecs)


def batch_rigid_transform(rot_mats, joints, parents):
    """ Go through forward kinematic chain.
    
    Args:
        rot_mats: local rotation matrix. [batch_size, n_joints, 3, 3]
        joints: rest joint locations. [batch_size, n_joints, 3]
        parents: the kinematic tree. [batch_size, n_joints]
    Returns:
        posed_joints: posed joint locations. [batch_size, n_joints, 3]
        rel_transforms: relative (w.r.t the root joint) transformation
            for all the joints. [batch_size, n_joints, 4, 4]
        transforms_bone: bone transformation matrix. [batch_size, 
            n_joints, 4, 4]
    """
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    
    # Note(ruilongli): `transforms_mat` seems to be problematic.
    # Shouldn't it be `[R, 0] @ [I, T] = [R, RT]`, instead of `[I, T] @ [R, 0] = [R, T]`?
    # In the case of `[R, T]`, posed_joints[:, 0:4] always equal to joints[:, 0:4].
    transforms_mat = to_se4(
        rot_mats.view(-1, 3, 3),
        rel_joints.contiguous().view(-1, 3, 1)
    ).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    # bone tranformations: 
    transform_chain_bone = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = transform_chain[parents[i]]
        transform_chain_bone.append(curr_res)
    transforms_bone = torch.stack(transform_chain_bone, dim=1)

    # santity check:
    diff = posed_joints[:, 0:4] - joints[:, 0:4, :, 0]
    assert torch.allclose(posed_joints[:, 0:4], joints[:, 0:4, :, 0]), diff.abs().mean()

    return posed_joints, rel_transforms, transforms_bone
