import torch
import torch.nn.functional as F
from torch import Tensor


def homo(points: Tensor) -> Tensor:
    """Get the homogeneous coordinates."""
    return F.pad(points, (0, 1), value=1)


def transform_points(points: Tensor, transforms: Tensor) -> Tensor:
    """Apply transformations to the points."""
    return torch.einsum(
        "...ij,...j->...i", transforms[..., :3, :4], homo(points)
    )
