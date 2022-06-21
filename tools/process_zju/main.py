# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

import numpy as np
import torch
import tqdm

from body_model import SMPLlayer

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")


def load_rest_pose_info(subject_id: int, body_model):
    data_dir = os.path.join(
        PROJECT_DIR, "data", "zju", "CoreView_%d" % subject_id, "new_params",
    )
    fp = os.path.join(data_dir, "1.npy")
    smpl_data = np.load(fp, allow_pickle=True).item()
    vertices, joints, joints_transform, bones_transform = body_model(
        poses=np.zeros((1, 72), dtype=np.float32),
        shapes=smpl_data["shapes"],
        Rh=np.zeros((1, 3), dtype=np.float32),
        Th=np.zeros((1, 3), dtype=np.float32),
        scale=1,
        new_params=True,
    )
    return (
        vertices.squeeze(0), 
        joints.squeeze(0), 
        joints_transform.squeeze(0),
        bones_transform.squeeze(0),
    )


def load_pose_info(subject_id: int, frame_id: int, body_model):
    data_dir = os.path.join(
        PROJECT_DIR, "data", "zju", "CoreView_%d" % subject_id, "new_params",
    )
    if subject_id in [313, 315]:
        fp = os.path.join(data_dir, "%d.npy" % (frame_id + 1))
    else:
        fp = os.path.join(data_dir, "%d.npy" % frame_id)
    # smpl_data['shapes'] is actually the same across frames (checked)
    smpl_data = np.load(fp, allow_pickle=True).item()
    vertices, joints, joints_transform, bones_tranform = body_model(
        poses=np.array(smpl_data['poses']),
        shapes=np.array(smpl_data['shapes']),
        Rh=np.array(smpl_data['Rh']),
        Th=np.array(smpl_data['Th']),
        scale=1,
        new_params=True,
    )
    pose_params = torch.cat(
        [
            torch.tensor(smpl_data['poses']),
            torch.tensor(smpl_data['Rh']),
            torch.tensor(smpl_data['Th']),
        ], dim=-1
    ).float()
    return (
        vertices.squeeze(0),
        joints.squeeze(0),
        joints_transform.squeeze(0),
        pose_params.squeeze(0),
        bones_tranform.squeeze(0),
    )


def cli(subject_id: int):
    print ("processing subject %d" % subject_id)
    # smpl body model
    body_model = SMPLlayer(
        model_path=os.path.join(PROJECT_DIR, "data"), gender="neutral", 
    )

    # parsing frame ids
    meta_fp = os.path.join(
        PROJECT_DIR, "data", "zju", "CoreView_%d" % subject_id, "annots.npy"
    )
    meta_data = np.load(meta_fp, allow_pickle=True).item()
    frame_ids = list(range(len(meta_data['ims'])))

    # rest state info
    rest_verts, rest_joints, rest_tfs, rest_tfs_bone = (
        load_rest_pose_info(subject_id, body_model)
    )
    lbs_weights = body_model.weights.float()

    # pose state info
    verts, joints, tfs, params, tf_bones = [], [], [], [], []
    for frame_id in tqdm.tqdm(frame_ids):
        _verts, _joints, _tfs, _params, _tfs_bone = (
            load_pose_info(subject_id, frame_id, body_model)
        )
        verts.append(_verts)
        joints.append(_joints)
        tfs.append(_tfs)
        params.append(_params)
        tf_bones.append(_tfs_bone)
    verts = torch.stack(verts)
    joints = torch.stack(joints)
    tfs = torch.stack(tfs)
    params = torch.stack(params)
    tf_bones = torch.stack(tf_bones)

    data = {
        "lbs_weights": lbs_weights,  # [6890, 24]
        "rest_verts": rest_verts,  # [6890, 3]
        "rest_joints": rest_joints,  # [24, 3]
        "rest_tfs": rest_tfs,  # [24, 4, 4]
        "rest_tfs_bone": rest_tfs_bone, # [24, 4, 4]
        "verts": verts,  # [1470, 6890, 3]
        "joints": joints,  # [1470, 24, 3]
        "tfs": tfs,  # [1470, 24, 4, 4]
        "tf_bones": tf_bones,  # [1470, 24, 4, 4]
        "params": params  # [1470, 72 + 3 + 3]
    }
    save_path = os.path.join(
        PROJECT_DIR, "data", "zju", "CoreView_%d" % subject_id, "pose_data.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)


if __name__ == "__main__":
    for subject_id in [313, 315, 377, 386, 387, 390, 392, 393, 394]:
        cli(subject_id)
