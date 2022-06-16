import torch
import torch.nn as nn
from torch import Tensor


class PoseConditionDPEncoder(nn.Module):
    """Naivest Deform Positional Encoder

    Concat the 1-dim pose representation to every posi_enc(x)

    Reference:
        Neural Articulated Radiance Field. (The `P-NeRF` baseline)

    Note in the paper NARF,
    1. they are using 6D SE(3) to represent each transformation,
    with positional encoding, resulting a `pose_latent` with shape
    [6 x PE x J],
    2. the `pose_latent` is not only the input of the entire NeRF
    mlp, but also injected again to the color branch.
    3. an extra PE(bone_length) \in [3 x PE x J] is used as the input
    to the entire NeRF mlp, to distinguish different subjects.
    """

    def __init__(self, posi_enc: nn.Module, pose_dim: int):
        super().__init__()
        self.pose_dim = pose_dim
        self.posi_enc = posi_enc

    @property
    def warp_dim(self):
        return 3 + self.pose_dim

    @property
    def out_dim(self):
        return self.posi_enc.out_dim + self.pose_dim

    @property
    def diag(self):
        return self.posi_enc.diag

    def forward(self, x: Tensor, x_cov: Tensor, pose_latent: Tensor):
        """
        :params x: [..., 3],
        :params x_cov: For mipnerf posi enc. [..., 3(, 3)]
        :params pose_latent: [self.pose_dim,]
        :return
            x_enc: [..., self.out_dim]
            x_warp: [..., self.warp_dim]
        """
        assert pose_latent.dim() == 1, "%s" % str(pose_latent.shape)
        pose_latent = pose_latent.expand(list(x.shape[:-1]) + [self.pose_dim])
        if x_cov is None:
            x_enc = torch.cat([self.posi_enc(x), pose_latent], dim=-1)
        else:
            x_enc = torch.cat([self.posi_enc((x, x_cov)), pose_latent], dim=-1)
        x_warp = torch.cat([x, pose_latent], dim=-1)
        return x_enc, x_warp
