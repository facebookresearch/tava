# Copyright (c) Meta Platforms, Inc. and affiliates.
""" Modified from

https://github.com/zju3dv/neuralbody/blob/master/zju_smpl/smplmodel/body_model.py
"""
import os.path as osp
import pickle

import numpy as np
import torch
import torch.nn as nn

from lbs import batch_rodrigues, lbs


def to_tensor(array, dtype=torch.float32, device=torch.device('cpu')):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class SMPLlayer(nn.Module):

    def __init__(self,
                 model_path,
                 gender='neutral',
                 device=None) -> None:
        super(SMPLlayer, self).__init__()
        dtype = torch.float32
        self.dtype = dtype
        self.device = device
        # create the SMPL model
        if osp.isdir(model_path):
            model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
            smpl_path = osp.join(model_path, model_fn)
        else:
            smpl_path = model_path
        assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
            smpl_path)

        with open(smpl_path, 'rb') as smpl_file:
            data = pickle.load(smpl_file, encoding='latin1')
        self.faces = data['f']
        self.register_buffer(
            'faces_tensor',
            to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long))
        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = data['posedirs']
        data['posedirs'] = np.reshape(data['posedirs'], [-1, num_pose_basis]).T

        for key in [
            'J_regressor', 'v_template', 'weights', 'posedirs', 'shapedirs'
        ]:
            val = to_tensor(to_np(data[key]), dtype=dtype)
            self.register_buffer(key, val)
        # indices of parents for each joints
        parents = to_tensor(to_np(data['kintree_table'][0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents)

    def forward(self,
                poses,
                shapes,
                Rh=None,
                Th=None,
                scale=1,
                new_params=False,
                **kwargs):
        """ Forward pass for SMPL model
        Args:
            poses (n, 72)
            shapes (n, 10)
            Rh (n, 3): global orientation
            Th (n, 3): global translation
            return_verts (bool, optional): if True return (6890, 3). Defaults to False.
        """
        if 'torch' not in str(type(poses)):
            dtype, device = self.dtype, self.device
            poses = to_tensor(poses, dtype, device)
            shapes = to_tensor(shapes, dtype, device)
            Rh = to_tensor(Rh, dtype, device)
            Th = to_tensor(Th, dtype, device)
        bn = poses.shape[0]
        if Rh is None:
            Rh = torch.zeros(bn, 3, device=poses.device)
        rot = batch_rodrigues(Rh)
        transl = Th.unsqueeze(dim=1)
        if shapes.shape[0] < bn:
            shapes = shapes.expand(bn, -1)
        vertices, joints, joints_transform, bones_transform = lbs(
            shapes,
            poses,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.weights,
            new_params=new_params,
        )

        global_transform = torch.eye(4, dtype=rot.dtype, device=rot.device)
        global_transform[:3, :3] = rot * scale
        global_transform[:3, 3] = transl

        vertices = torch.matmul(vertices, rot.transpose(1, 2)) * scale + transl
        joints = torch.matmul(joints, rot.transpose(1, 2)) * scale + transl
        joints_transform = torch.einsum("ij,...jk->...ik", global_transform, joints_transform)
        bones_transform = torch.einsum("ij,...jk->...ik", global_transform, bones_transform)
        return vertices, joints, joints_transform, bones_transform
