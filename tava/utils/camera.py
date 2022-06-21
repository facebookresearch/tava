# Copyright (c) Meta Platforms, Inc. and affiliates.
import math
from typing import Tuple

import torch
from tava.utils.point import homo
from tava.utils.structures import Cameras, Rays


def project_points_to_image_plane(
    cameras: Cameras, points: torch.Tensor
) -> Tuple[torch.Tensor]:
    """Project 3D points to 2D on image plane.

    :params cameras: a single or multiple cameras [(n_cams,)]
    :params points: [..., 3]
    :returns
        coordinates on image plane. [(n_cams,) ..., 2]
        depth [(n_cams,) ...]
    """
    assert points.shape[-1] == 3
    if cameras.intrins.dim() == 3:
        # multiple cameras
        equation = "nij,njk,...k->n...i"
    else:
        # a single camera
        equation = "ij,jk,...k->...i"
    points_c = torch.einsum(
        equation,
        cameras.intrins[..., :3, :3],
        cameras.extrins[..., :3, :4],
        homo(points),
    )
    points2d = points_c[..., 0:2] / points_c[..., 2:3]
    depth = points_c[..., 2]
    return points2d, depth


def transform_cameras(cameras: Cameras, resize_factor: float) -> torch.Tensor:
    intrins = cameras.intrins
    intrins[..., :2, :] = intrins[..., :2, :] * resize_factor
    width = int(cameras.width * resize_factor + 0.5)
    height = int(cameras.height * resize_factor + 0.5)
    return Cameras(
        intrins=intrins,
        extrins=cameras.extrins,
        distorts=cameras.distorts,
        width=width,
        height=height,
    )


def generate_rays(
    cameras: Cameras,
    opencv_format: bool = True,
    near: float = None,
    far: float = None,
) -> Rays:
    """Generating rays for a single or multiple cameras.

    :params cameras [(n_cams,)]
    :returns: Rays, [(n_cams,) height, width]
    """
    K = cameras.intrins[..., None, None, :, :]
    c2w = cameras.extrins[..., None, None, :, :].inverse()

    x, y = torch.meshgrid(
        torch.arange(cameras.width, dtype=K.dtype),
        torch.arange(cameras.height, dtype=K.dtype),
        indexing="xy",
    )  # [height, width]

    camera_dirs = homo(
        torch.stack(
            [
                (x - K[..., 0, 2] + 0.5) / K[..., 0, 0],
                (y - K[..., 1, 2] + 0.5) / K[..., 1, 1],
            ],
            dim=-1,
        )
    )  # [n_cams, height, width, 3]
    if not opencv_format:
        camera_dirs[..., [1, 2]] *= -1

    # [n_cams, height, width, 3]
    directions = (camera_dirs[..., None, :] * c2w[..., :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[..., :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = torch.sqrt(
        torch.sum(
            (directions[..., :-1, :, :] - directions[..., 1:, :, :]) ** 2,
            dim=-1,
        )
    )
    dx = torch.cat([dx, dx[..., -2:-1, :]], dim=-2)
    radii = dx * 2 / math.sqrt(12)  # [n_cams, height, width]

    if near is not None:
        near = near * torch.ones_like(origins[..., 0:1])
    if far is not None:
        far = far * torch.ones_like(origins[..., 0:1])
    rays = Rays(
        origins=origins,  # [n_cams, height, width, 3]
        directions=directions,  # [n_cams, height, width, 3]
        viewdirs=viewdirs,  # [n_cams, height, width, 3]
        radii=radii[..., None],  # [n_cams, height, width, 1]
        # near far is not needed when they are estimated by skeleton.
        near=near,
        far=far,
    )
    return rays
