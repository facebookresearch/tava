import os

import cv2
import imageio
import numpy as np
import torch


class SubjectParser:
    """Single subject data parser."""

    WIDTH = 1024
    HEIGHT = 1024

    JOINT_NAMES = [
        "root",
        "lhip",
        "rhip",
        "belly",
        "lknee",
        "rknee",
        "spine",
        "lankle",
        "rankle",
        "chest",
        "ltoes",
        "rtoes",
        "neck",
        "linshoulder",
        "rinshoulder",
        "head",
        "lshoulder",
        "rshoulder",
        "lelbow",
        "relbow",
        "lwrist",
        "rwrist",
        "lhand",
        "rhand",
    ]

    BONE_NAMES = [
        ("root", "lhip"),
        ("root", "rhip"),
        ("root", "belly"),
        ("lhip", "lknee"),
        ("rhip", "rknee"),
        ("belly", "spine"),
        ("lknee", "lankle"),
        ("rknee", "rankle"),
        ("spine", "chest"),
        ("lankle", "ltoes"),
        ("rankle", "rtoes"),
        ("chest", "neck"),
        ("chest", "linshoulder"),
        ("chest", "rinshoulder"),
        ("neck", "head"),
        ("linshoulder", "lshoulder"),
        ("rinshoulder", "rshoulder"),
        ("lshoulder", "lelbow"),
        ("rshoulder", "relbow"),
        ("lelbow", "lwrist"),
        ("relbow", "rwrist"),
        ("lwrist", "lhand"),
        ("rwrist", "rhand"),
    ]

    RIGID_BONE_IDS = [
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ]

    def __init__(self, subject_id: int, root_fp: str):
        self.subject_id = subject_id
        self.root_fp = root_fp
        self.root_dir = os.path.join(root_fp, "CoreView_%d" % subject_id)

        self.mask_dir = os.path.join(self.root_dir, "mask_cihp")
        self.smpl_dir = os.path.join(self.root_dir, "new_params")
        self.eval_mask_dir = os.path.join(self.root_dir, "comparison")

        annots_fp = os.path.join(self.root_dir, "annots.npy")
        annots_data = np.load(annots_fp, allow_pickle=True).item()
        self.cameras = self._parse_camera(annots_data)

        # [1470 x 21]
        self.image_files = np.array(
            [[fp for fp in fps["ims"]] for fps in annots_data["ims"]], dtype=str
        )
        self._frame_ids = list(range(self.image_files.shape[0]))
        self._camera_ids = list(range(self.image_files.shape[1]))

    @property
    def camera_ids(self):
        return self._camera_ids

    @property
    def frame_ids(self):
        return self._frame_ids

    def _parse_camera(self, annots_data):
        K = np.array(annots_data["cams"]["K"]).astype(np.float32)
        R = np.array(annots_data["cams"]["R"]).astype(np.float32)
        T = np.array(annots_data["cams"]["T"]).astype(np.float32) / 1000.0
        D = np.array(annots_data["cams"]["D"]).astype(np.float32)
        cameras = {
            cid: {
                "K": K[cid],
                "D": D[cid],
                "w2c": np.concatenate(
                    [
                        np.concatenate([R[cid], T[cid]], axis=-1),
                        np.array([[0.0, 0.0, 0.0, 1.0]], dtype=R.dtype),
                    ],
                    axis=0,
                ),
            }
            for cid in range(K.shape[0])
        }
        return cameras

    def load_image(self, frame_id, camera_id):
        path = os.path.join(
            self.root_dir, self.image_files[frame_id, camera_id]
        )
        image = imageio.imread(path)
        return image  # shape [HEIGHT, WIDTH, 3], value 0 ~ 255

    def load_mask(self, frame_id, camera_id, trimap=True):
        path = os.path.join(
            self.mask_dir,
            self.image_files[frame_id, camera_id].replace(".jpg", ".png"),
        )
        mask = (imageio.imread(path) != 0).astype(np.uint8) * 255
        if trimap:
            mask = self._process_mask(mask)
        return mask  # shape [HEIGHT, WIDTH], value [0, (128,) 255]

    def load_meta_data(self, frame_ids=None):
        data = torch.load(os.path.join(self.root_dir, "pose_data.pt"))
        keys = [
            "lbs_weights",
            "rest_verts",
            "rest_joints",
            "rest_tfs",
            "rest_tfs_bone",
            "verts",
            "joints",
            "tfs",
            "tf_bones",
            "params",
        ]
        return {
            key: (
                data[key][frame_ids].numpy()
                if (
                    frame_ids is not None
                    and key in ["verts", "joints", "tfs", "tf_bones", "params"]
                )
                else data[key].numpy()
            )
            for key in keys
        }

    def _process_mask(self, mask, border: int = 5, ignore_value: int = 128):
        kernel = np.ones((border, border), np.uint8)
        mask_erode = cv2.erode(mask.copy(), kernel)
        mask_dilate = cv2.dilate(mask.copy(), kernel)
        mask[mask_dilate != mask_erode] = ignore_value
        return mask
