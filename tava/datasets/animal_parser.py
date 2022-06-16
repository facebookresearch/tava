import json
import os

import imageio
import numpy as np
import torch


class SubjectParser:
    """Single subject data parser."""

    WIDTH = 800
    HEIGHT = 800

    def __init__(self, subject_id: str, root_fp: str, legacy: bool = False):
        self.subject_id = subject_id
        self.root_fp = root_fp
        self.root_dir = os.path.join(root_fp, subject_id)

        actions = sorted(
            [
                fp
                for fp in os.listdir(self.root_dir)
                if os.path.exists(
                    os.path.join(self.root_dir, fp, "camera.json")
                )
            ]
        )
        if legacy:
            # for legacy reasons, we skip the first action to match with the paper.
            # we also shuffle the actions in a fixed way
            actions.pop(0)
            g = torch.Generator()
            g = g.manual_seed(56789)
            idxs = torch.randperm(len(actions), generator=g)
            self.actions = [actions[i] for i in idxs]
        else:
            self.actions = actions
        self.indexing = self._create_indexing()  # {action: {frame_id: [cid]}

    @property
    def camera_ids(self):
        return self.indexing[self.actions[0]][0]

    @property
    def frame_ids(self):
        return sorted(list(self.indexing[self.actions[0]].keys()))

    def _create_indexing(self):
        indexing = {}
        for action in self.actions:
            indexing[action] = {}
            for frame_id, camera_data in self.load_camera(action).items():
                indexing[action][int(frame_id)] = sorted(
                    list(camera_data.keys())
                )
        return indexing

    def load_camera(self, action, frame_id=None, camera_id=None):
        path = os.path.join(self.root_dir, action, "camera.json")
        with open(path, mode="r") as fp:
            data = json.load(fp)
        if (frame_id is not None) and (camera_id is not None):
            intrin = data[str(frame_id)][camera_id]["intrin"]
            extrin = data[str(frame_id)][camera_id]["extrin"]
            K = np.array(intrin, dtype=np.float32)
            c2w = np.linalg.inv(np.array(extrin, dtype=np.float32))
            return K, c2w  # shape [3, 3], [4, 4]
        else:
            return data

    def load_image(self, action, frame_id, camera_id):
        path = os.path.join(
            self.root_dir,
            action,
            "image",
            camera_id,
            "%08d.png" % int(frame_id),
        )
        image = imageio.imread(path)
        return image  # shape [HEIGHT, WIDTH, 4], value 0 ~ 255

    def load_depth(self, action, frame_id, camera_id):
        path = os.path.join(
            self.root_dir,  # "/tmp", self.subject_id, 
            action,
            "depth",
            camera_id,
            "%08d.exr" % int(frame_id),
        )
        depth = imageio.imread(path)[..., 0]
        return depth  # shape [HEIGHT, WIDTH], z-depth

    def load_pcd(self, action, frame_id, camera_id):
        K, c2w = self.load_camera(action, frame_id, camera_id)
        image = self.load_image(action, frame_id, camera_id)
        depth = self.load_depth(action, frame_id, camera_id)
        mask = image[..., 3] == 255
        height, width, _4 = image.shape

        x, y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            indexing="xy",
        )  # [height, width]
        camera_dirs = np.stack(
            [
                (x - K[0, 2] + 0.5) / K[0, 0],
                (y - K[1, 2] + 0.5) / K[1, 1],
                np.ones_like(x),
            ],
            axis=-1,
        )  # [height, width, 3]

        directions = (camera_dirs[..., None, :] * c2w[..., :3, :3]).sum(axis=-1)
        origins = np.broadcast_to(c2w[..., :3, -1], directions.shape)
        world_points = origins + directions * depth[..., None]
        return world_points[mask]  # shape [N, 3]

    def load_meta_data(self, action, frame_ids=None):
        fp = os.path.join(self.root_dir, action, "meta_data.npz")
        data = np.load(fp, allow_pickle=True)
        keys = [
            "rest_matrixs",
            "rest_tails",
            "lbs_weights",
            "rest_verts",
            "faces",
            "pose_matrixs",
            "pose_verts",
            "pose_tails",
        ]
        return {
            key: (
                data[key][frame_ids]
                if (frame_ids is not None and "pose_" in key)
                else data[key]
            )
            for key in keys
        }
