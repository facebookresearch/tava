# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Tuple

import torch
from tava.utils.point import homo, transform_points
from tava.utils.structures import Bones, Rays
from torch import Tensor


def _dot(a, b):
    return (a * b).sum(dim=-1, keepdim=True)


def get_end_points(bones: Bones) -> Tuple[Tensor]:
    """Get the bone heads and bone tails."""
    if bones.heads is not None:
        heads = bones.heads
    else:
        heads = bones.transforms[..., 0:3, 3]
    tails = bones.tails
    return heads, tails


def transform_bones(bones: Bones, transforms: Tensor) -> Bones:
    """Apply transformations to the bones."""
    return Bones(heads=bones.heads, tails=bones.tails, transforms=transforms)


def lbs(bones: Bones, points: Tensor, weights: Tensor) -> Tensor:
    """Apply Linear Blend Skinning (LBS) to the points.

    :params bones: [n_bones,]
    :params points: [..., 3]
    :params weights: [..., n_bones]
    :returns deformed points [..., 3]
    """
    transforms = torch.einsum("...b,bij->...ij", bones.transforms, weights)
    points = transform_points(points, transforms)
    return points


def project_to_bone_space(
    bones: Bones, points: Tensor, individual: bool = False
) -> Tensor:
    """Project points to the bone space.

    :params bones: [n_bones,]
    :params points: If individual is true, the shape should be [..., n_bones, 3].
        else, the shape is [..., 3]
    :returns projecte points in bone space (relative to bone heads) [..., n_bones]
    """
    if individual:
        # the points and bones have one-to-one correspondence.
        # [..., n_bones, 3] <-> [n_bones,]
        assert points.shape[-2:] == (bones.tails.shape[0], 3)
    else:
        points = points[..., None, :]  # [..., 1, 3]
    return torch.einsum(
        "bij,...bj->...bi",
        bones.transforms.inverse()[..., :3, :4],
        homo(points),
    )


def sample_on_bones(
    bones: Bones, n_per_bone=5, range=(0.0, 1.0), uniform=False
) -> Tensor:
    """Sample points on the bones.

    :params bones: [n_bones,]
    :returns samples [n_per_bone, n_bones, 3]
    """
    heads, tails = get_end_points(bones)
    device = heads.device
    dtype = heads.dtype
    n_bones = heads.shape[0]
    if uniform:
        t_vals = torch.linspace(
            range[0], range[1], n_per_bone, dtype=dtype, device=device
        )
        t_vals = t_vals[:, None].expand((n_per_bone, n_bones))
    else:
        t_vals = (
            torch.rand((n_per_bone, n_bones), dtype=dtype, device=device)
            * (range[1] - range[0])
            + range[0]
        )
    samples = (
        heads[None, :, :] + (tails - heads)[None, :, :] * t_vals[:, :, None]
    )
    return samples


def closest_distance_to_points(
    bones: Bones, points: Tensor, individual: bool = False
) -> Tensor:
    """Cartesian distance from points to bones (line segments).

    https://zalo.github.io/blog/closest-point-between-segments/

    :params bones: [n_bones,]
    :params points: If individual is true, the shape should be [..., n_bones, 3].
        else, the shape is [..., 3]
    :returns distances [..., n_bones]
    """
    if individual:
        # the points and bones have one-to-one correspondence.
        # [..., n_bones, 3] <-> [n_bones,]
        assert points.shape[-2:] == (bones.tails.shape[0], 3)
    else:
        points = points[..., None, :]  # [..., 1, 3]
    heads, tails = get_end_points(bones)
    t = _dot(points - heads, tails - heads) / _dot(tails - heads, tails - heads)
    p = heads + (tails - heads) * torch.clamp(t, 0, 1)
    dists = torch.linalg.norm(p - points, dim=-1)
    return dists


def closest_distance_to_rays(bones: Bones, rays: Rays) -> Tuple[Tensor]:
    """Cartesian distance in from bones to rays.

    Equivalent to distance between two line segments.
    https://zalo.github.io/blog/closest-point-between-segments/

    :params bones: [n_bones,]
    :params rays: [n_rays,]
    :returns:
        - dists: shape of [n_rays, n_bones]
        - t_vals: closest points on the rays. [n_rays, n_bones]
    """
    RAY_LENGTH = 100

    heads, tails = get_end_points(bones)
    a = heads
    u = tails - heads
    b = rays.origins[:, None, :]
    v = rays.directions[:, None, :] * RAY_LENGTH

    r = b - a
    ru = _dot(r, u)
    rv = _dot(r, v)
    uu = _dot(u, u)
    uv = _dot(u, v)
    vv = _dot(v, v)
    det = uu * vv - uv * uv

    # if det is too close to 0, then they're parallel
    if False:
        pass
    else:
        # compute optimal values for s and t
        s = (ru * vv - rv * uv) / torch.clamp(det, min=1e-6)
        t = (ru * uv - rv * uu) / torch.clamp(det, min=1e-6)

        # constrain values s and t so that they describe points on the segments
        s = torch.clamp(s, 0.0, 1.0)
        t = torch.clamp(t, 0.0, 1.0)

    # convert value s for segA into the corresponding closest value t
    # for segB and vice versa
    S = torch.clamp((t * uv + ru) / uu, 0.0, 1.0)
    T = torch.clamp((s * uv - rv) / vv, 0.0, 1.0)

    A = a + S * u
    B = b + T * v

    dists = torch.linalg.norm(A - B, dim=-1)
    z_vals = T * RAY_LENGTH
    return dists, z_vals.squeeze(-1)
