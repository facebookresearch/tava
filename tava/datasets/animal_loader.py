# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

import cv2
import numpy as np
import torch
from tava.datasets.abstract import CachedIterDataset
from tava.datasets.animal_parser import SubjectParser
from tava.utils.camera import generate_rays, transform_cameras
from tava.utils.structures import Bones, Cameras, namedtuple_map


def _dataset_view_split(parser, split):
    if split == "all":
        camera_ids = parser.camera_ids
    elif split == "train":
        camera_ids = parser.camera_ids[::2]
    elif split in ["val_ind", "val_ood", "val_view"]:
        camera_ids = parser.camera_ids[1::2]
    elif split == "test":
        camera_ids = parser.camera_ids[1:2]
    return camera_ids


def _dataset_frame_split(parser, split):
    if split in ["train", "val_view"]:
        splits_fp = os.path.join(parser.root_dir, "splits/train.txt")
    else:
        splits_fp = os.path.join(parser.root_dir, f"splits/{split}.txt")
    with open(splits_fp, mode="r") as fp:
        frame_list = np.loadtxt(fp, dtype=str).tolist()
    frame_list = [(action, int(frame_id)) for (action, frame_id) in frame_list]
    return frame_list


def _dataset_index_list(parser, split):
    camera_ids = _dataset_view_split(parser, split)
    frame_list = _dataset_frame_split(parser, split)
    index_list = []
    for action, frame_id in frame_list:
        index_list.extend(
            [(action, frame_id, camera_id) for camera_id in camera_ids]
        )
    return index_list


class SubjectLoader(CachedIterDataset):
    """Single subject data loader for training and evaluation."""

    SPLIT = ["all", "train", "val_ind", "val_ood", "val_view", "test"]

    @classmethod
    def encode_meta_id(cls, action, frame_id):
        return "%s___%05d" % (action, int(frame_id))

    @classmethod
    def decode_meta_id(cls, meta_id: str):
        action, frame_id = meta_id.split("___")
        return action, int(frame_id)

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        resize_factor: float = 1.0,
        color_bkgd_aug: str = None,
        num_rays: int = None,
        cache_n_repeat: int = 0,
        near: float = None,
        far: float = None,
        legacy: bool = False,
        **kwargs,
    ):
        assert split in self.SPLIT, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        self.resize_factor = resize_factor
        self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.training = (num_rays is not None) and (split in ["train", "all"])
        self.color_bkgd_aug = color_bkgd_aug if self.training else "white"
        self.parser = SubjectParser(
            subject_id=subject_id, root_fp=root_fp, legacy=legacy
        )
        self.index_list = _dataset_index_list(self.parser, split)
        self.dtype = torch.get_default_dtype()
        super().__init__(self.training, cache_n_repeat)

    def __len__(self):
        return len(self.index_list)

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        image, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, dtype=rgba.dtype)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, dtype=rgba.dtype)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, dtype=rgba.dtype)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, dtype=rgba.dtype)

        image = image * alpha + color_bkgd * (1.0 - alpha)

        if self.num_rays is not None:
            resolution = image.shape[0] * image.shape[1]
            ray_indices = torch.randperm(resolution)[: self.num_rays]
            pixels = image.reshape(resolution, 3)[ray_indices]
            rays = namedtuple_map(
                lambda r: r.reshape([resolution] + list(r.shape[2:])), rays
            )
            rays = namedtuple_map(lambda x: x[ray_indices], rays)
        else:
            pixels = image

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        # load data
        action, frame_id, camera_id = self.index_list[index]
        K, c2w = self.parser.load_camera(action, frame_id, camera_id)
        rgba = self.parser.load_image(action, frame_id, camera_id)

        # create pixels
        rgba = (
            torch.from_numpy(
                cv2.resize(
                    rgba,
                    (0, 0),
                    fx=self.resize_factor,
                    fy=self.resize_factor,
                    interpolation=cv2.INTER_AREA,
                )
            ).to(self.dtype)
            / 255.0
        )

        # create rays from camera
        cameras = Cameras(
            intrins=torch.from_numpy(K).to(self.dtype),
            extrins=torch.from_numpy(c2w).to(self.dtype).inverse(),
            distorts=None,
            width=self.parser.WIDTH,
            height=self.parser.HEIGHT,
        )
        cameras = transform_cameras(cameras, self.resize_factor)
        rays = generate_rays(
            cameras, opencv_format=True, near=self.near, far=self.far
        )

        return {
            "subject_id": self.parser.subject_id,
            "camera_id": camera_id,
            # `meta_id` is used to query pose info from `pose_meta_info`
            "meta_id": self.encode_meta_id(action, frame_id),
            "rgba": rgba,  # [h, w, 4]
            "rays": rays,  # [h, w]
            "rigid_clusters": None,
        }

    def build_pose_meta_info(self):
        # create indexing for this split
        indexing = {}
        for action, frame_id, _ in self.index_list:
            if action not in indexing:
                indexing[action] = []
            indexing[action].append(frame_id)

        # load canonical meta info using any action because they are the same.
        _meta_data = self.parser.load_meta_data(action=list(indexing.keys())[0])
        # filter the active bones (exclude helper bones and root bones).
        # here we use the `lbs_weight`s to automatically filter it but
        # an alternative way is just to manually set it up.
        bone_ids = np.where(_meta_data["lbs_weights"].max(axis=0) > 0)[
            0
        ].tolist()
        bones_rest = Bones(
            heads=None,
            tails=torch.from_numpy(_meta_data["rest_tails"][bone_ids]).to(self.dtype),
            transforms=torch.from_numpy(
                _meta_data["rest_matrixs"][bone_ids]
            ).to(self.dtype),
        )

        # load meta info for all poses.
        bones_posed, meta_ids = [], []
        for action, frame_ids in indexing.items():
            frame_ids = sorted(list(set(frame_ids)))
            meta_data = self.parser.load_meta_data(action)
            tails = torch.from_numpy(meta_data["pose_tails"]).to(self.dtype)
            transforms = torch.from_numpy(meta_data["pose_matrixs"]).to(self.dtype)
            for frame_id in frame_ids:
                meta_ids.append(self.encode_meta_id(action, frame_id))
                bones_posed.append(
                    Bones(
                        heads=None,
                        tails=tails[frame_id, bone_ids],
                        transforms=transforms[frame_id, bone_ids],
                    )
                )
        return {
            "meta_ids": meta_ids,
            "bones_rest": bones_rest,
            "bones_posed": bones_posed,
        }
