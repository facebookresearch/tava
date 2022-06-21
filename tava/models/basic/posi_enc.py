# Copyright (c) Meta Platforms, Inc. and affiliates.
""" Positional Encoding. """
import math

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        append_identity: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.append_identity = append_identity

    @property
    def out_dim(self):
        return (
            self.in_dim if self.append_identity else 0
        ) + self.in_dim * 2 * (self.max_deg - self.min_deg)

    def forward(self, x: torch.Tensor):
        """
        :params x: [..., 3]
        :return x_enc: [..., self.out_dim]
        """
        scales = torch.tensor(
            [2**i for i in range(self.min_deg, self.max_deg)],
            dtype=x.dtype,
            device=x.device,
        )
        xb = torch.reshape(
            (x[Ellipsis, None, :] * scales[:, None]),
            list(x.shape[:-1]) + [scales.shape[0] * x.shape[-1]],
        )
        four_feat = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.append_identity:
            return torch.cat([x] + [four_feat], dim=-1)
        else:
            return four_feat


class IntegratedPositionalEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        diag: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.diag = diag

    @property
    def out_dim(self):
        return self.in_dim * 2 * (self.max_deg - self.min_deg)

    def forward(self, x_coord: torch.Tensor):
        """
        :params x_coord: ([..., 3], [..., 3] or [..., 3, 3])
        :return x_enc: [..., self.out_dim]
        """
        if self.diag:
            x, x_cov_diag = x_coord
            scales = torch.tensor(
                [2**i for i in range(self.min_deg, self.max_deg)],
                device=x.device,
            )
            shape = list(x.shape[:-1]) + [x.shape[-1] * scales.shape[0]]
            y = torch.reshape(x[..., None, :] * scales[:, None], shape)
            y_var = torch.reshape(
                x_cov_diag[..., None, :] * scales[:, None] ** 2, shape
            )
        else:
            x, x_cov = x_coord
            num_dims = x.shape[-1]
            basis = torch.cat(
                [
                    2**i * torch.eye(num_dims, device=x.device)
                    for i in range(self.min_deg, self.max_deg)
                ],
                1,
            )
            y = torch.matmul(x, basis)
            # Get the diagonal of a covariance matrix (ie, variance).
            # This is equivalent to jax.vmap(torch.diag)((basis.T @ covs) @ basis).
            y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)
        return self._expected_sin(
            torch.cat([y, y + 0.5 * math.pi], dim=-1),
            torch.cat([y_var] * 2, dim=-1),
        )[0]

    def _expected_sin(self, x, x_var):
        """Estimates mean and variance of sin(z), z ~ N(x, var)."""
        # When the variance is wide, shrink sin towards zero.
        y = torch.exp(-0.5 * x_var) * torch.sin(x)
        y_var = torch.clip(
            0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2, min=0
        )
        return y, y_var


class WindowedPositionalEncoder(PositionalEncoder):
    """ `AnnealedSinusoidalEncoder` in Nefies:
    https://github.com/google/nerfies/blob/main/nerfies/modules.py#L231
    """
    
    def forward(self, x: torch.Tensor, alpha: float):
        """
        :params x: [..., 3]
        :params alpha: float
        :return x_enc: [..., self.out_dim]
        """
        features = super().forward(x)
        if self.append_identity:
            identity, features = torch.split(
                features, 
                (self.in_dim, self.in_dim * 2 * (self.max_deg - self.min_deg)), 
                dim=-1
            )
        features = features.reshape(
            list(x.shape[:-1]) + [self.max_deg - self.min_deg, self.in_dim, 2]
        )
        window = self.cosine_easing_window(alpha).reshape(
            (self.max_deg - self.min_deg, 1, 1)
        ).to(features)
        features = window * features
        if self.append_identity:
            return torch.cat([
                identity, features.reshape(list(x.shape[:-1]) + [-1])
            ], dim=-1)
        else:
            return features
        
    def cosine_easing_window(self, alpha):
        bands = torch.linspace(0, self.max_deg - 1, self.max_deg)
        x = torch.clamp(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(math.pi * x + math.pi))
