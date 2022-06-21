# Copyright 2021 Google LLC
# Modified by Ruilong Li from https://github.com/google-research/jaxnerf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn


class NerfModel(nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(
        self,
        mlp_coarse: nn.Module,
        mlp_fine: nn.Module = None,
        # positional encoding for coordinates
        pos_enc: nn.Module = None,
        # positional encoding for view directions
        view_enc: nn.Module = None,
        num_samples_coarse: int = 64,  # The number of samples.
        num_samples_fine: int = 128,  # The number of samples.
        stop_level_grad: bool = True,  # If True, don't backprop across levels')
        # If True, use view directions as a condition.
        use_viewdirs: bool = True,
        # If True, sample linearly in disparity, not in depth.
        lindisp: bool = False,
        density_activation = torch.nn.ReLU(),  # Density activation.
        # Standard deviation of noise added to raw density.
        density_noise: float = 0.0,
        # The shift added to raw densities pre-activation.
        density_bias: float = 0.0,
        rgb_activation = torch.nn.Sigmoid(),  # The RGB activation.
        rgb_padding: float = 0.000,  # Padding added to the RGB outputs.
        num_levels: int = 2,
        coarse_sample_with_fine: bool = True,
    ):
        super().__init__()
        assert num_levels in [1, 2]
        self.mlp_coarse = mlp_coarse
        self.mlp_fine = mlp_fine if num_levels == 2 else None
        self.pos_enc = pos_enc
        self.view_enc = view_enc
        self.num_levels = num_levels
        self.num_samples_coarse = num_samples_coarse
        self.num_samples_fine = num_samples_fine
        self.stop_level_grad = stop_level_grad
        self.use_viewdirs = use_viewdirs
        self.lindisp = lindisp
        self.density_activation = density_activation
        self.density_noise = density_noise
        self.density_bias = density_bias
        self.rgb_activation = rgb_activation
        self.rgb_padding = rgb_padding
        self.coarse_sample_with_fine = coarse_sample_with_fine

    def _query_mlp(self, rays, samples, i_level, randomized=True, **kwargs):
        if self.pos_enc:
            samples_enc = self.pos_enc(samples)
        else:
            samples_enc = samples

        mlp = self.mlp_coarse if i_level == 0 else self.mlp_fine

        # Point attribute predictions
        if self.use_viewdirs:
            viewdirs_enc = self.view_enc(rays.viewdirs)
            raw_rgb, raw_density = mlp(samples_enc, cond_view=viewdirs_enc)
        else:
            raw_rgb, raw_density = mlp(samples_enc)

        # Add noise to regularize the density predictions if needed.
        if randomized and (self.density_noise > 0):
            raw_density += self.density_noise * torch.randn(
                raw_density.shape,
                dtype=raw_density.dtype,
                device=raw_density.device,
            )

        # Volumetric rendering.
        rgb = self.rgb_activation(raw_rgb)
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        density = self.density_activation(raw_density + self.density_bias)
        return rgb, density

    def forward(self, rays, color_bkgd, randomized: bool = True, **kwargs):
        ret = []
        extra_info = None
        weights = None
        
        for i_level in range(self.num_levels):
            if i_level == 0:
                # coarse sampling along rays
                t_vals, samples = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    self.num_samples_coarse,
                    rays.near,
                    rays.far,
                    randomized,
                    self.lindisp,
                )
            else:
                # fine sampling
                t_vals_mid = 0.5 * (t_vals[Ellipsis, 1:] + t_vals[Ellipsis, :-1])
                t_vals, samples = sample_pdf(
                    t_vals_mid,
                    weights[Ellipsis, 1:-1],
                    rays.origins,
                    rays.directions,
                    t_vals,
                    self.num_samples_fine,
                    randomized,
                    self.coarse_sample_with_fine,
                )

            rgb, density = self._query_mlp(
                rays, samples, i_level=i_level, randomized=randomized, **kwargs
            )

            comp_rgb, disp, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_vals,
                rays.directions,
                color_bkgd=color_bkgd,
            )
            points = (weights[..., None] * samples[0]).sum(dim=-2)
            ret.append((comp_rgb, disp, acc, points))
        return ret, extra_info


def cast_rays(t_vals, origins, directions):
    return (
        origins[Ellipsis, None, :]
        + t_vals[Ellipsis, None] * directions[Ellipsis, None, :]
    )


def volumetric_rendering(rgb, density, t_vals, dirs, color_bkgd):
    """Volumetric Rendering Function.
    Args:
        rgb: torch.ndarray(float32), color, [batch_size, num_samples, 3]
        density: torch.ndarray(float32), density, [batch_size, num_samples, 1].
        t_vals: torch.ndarray(float32), [batch_size, num_samples].
        dirs: torch.ndarray(float32), [batch_size, 3].
        color_bkgd: torch.ndarray(float32), [3].
    Returns:
        comp_rgb: torch.ndarray(float32), [batch_size, 3].
        disp: torch.ndarray(float32), [batch_size].
        acc: torch.ndarray(float32), [batch_size].
        weights: torch.ndarray(float32), [batch_size, num_samples]
    """
    t_dists = torch.cat(
        [
            t_vals[Ellipsis, 1:] - t_vals[Ellipsis, :-1],
            torch.tensor(
                [1e10], dtype=t_vals.dtype, device=t_vals.device
            ).expand(t_vals[Ellipsis, :1].shape),
        ],
        -1,
    )
    delta = t_dists * torch.linalg.norm(dirs[Ellipsis, None, :], dim=-1)

    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(
        -torch.cat(
            [
                torch.zeros_like(density_delta[..., :1]),
                torch.cumsum(density_delta[..., :-1], dim=-1),
            ],
            dim=-1,
        )
    )
    weights = alpha * trans

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    acc = weights.sum(dim=-1)
    # distance = (weights * t_mids).sum(dim=-1) / acc
    # distance = torch.clip(
    #     torch.nan_to_num(distance, torch.finfo().max), t_vals[:, 0], t_vals[:, -1]
    # )
    depth = (weights * t_vals).sum(dim=-1)
    eps = 1e-10
    inv_eps = 1 / eps
    # torch.where accepts <scaler, double tensor>
    disp = (acc / depth).double()
    disp = torch.where(
        (disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps
    )
    disp = disp.to(acc.dtype)

    comp_rgb = comp_rgb + color_bkgd * (1.0 - acc[..., None])
    return comp_rgb, depth, acc, weights


def sample_along_rays(
    origins,
    directions,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
):
    """Stratified sampling along the rays.
    Args:
        origins: torch.ndarray(float32), [batch_size, 3], ray origins.
        directions: torch.ndarray(float32), [batch_size, 3], ray directions.
        num_samples: int.
        near: torch.ndarray, [batch_size, 1], near clip.
        far: torch.ndarray, [batch_size, 1], far clip.
        randomized: bool, use randomized stratified sampling.
        lindisp: bool, sampling linearly in disparity rather than depth.
    Returns:
        t_vals: torch.ndarray, [batch_size, num_samples], sampled z values.
        means: torch.ndarray, [batch_size, num_samples, 3], sampled means.
    """
    batch_size = origins.shape[0]
    device = origins.device

    t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand([batch_size, num_samples], device=device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, num_samples])
    coords = cast_rays(t_vals, origins, directions)
    return t_vals, coords


def piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """Piecewise-Constant PDF sampling.
    Args:
      bins: torch.tensor(float32), [batch_size, num_bins + 1].
      weights: torch.tensor(float32), [batch_size, num_bins].
      num_samples: int, the number of samples.
      randomized: bool, use randomized samples.
    Returns:
      z_samples: torch.tensor(float32), [batch_size, num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdims=True)
    padding = torch.clamp(eps - weight_sum, min=0)
    weights = weights + padding / weights.shape[-1]  # avoid +=
    weight_sum = weight_sum + padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.clamp(torch.cumsum(pdf[Ellipsis, :-1], dim=-1), max=1)
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], dtype=cdf.dtype, device=cdf.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], dtype=cdf.dtype, device=cdf.device),
        ],
        dim=-1,
    )

    # Draw uniform samples.
    if randomized:
        # Note that `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], 
                       dtype=cdf.dtype, device=cdf.device)
    else:
        # Match the behavior of torch.rand() by spanning [0, 1-eps].
        u = torch.linspace(0.0, 1.0 - torch.finfo().eps, num_samples, 
                           dtype=cdf.dtype, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[Ellipsis, None, :] >= cdf[Ellipsis, :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = torch.max(torch.where(mask, x[Ellipsis, None], x[Ellipsis, :1, None]), dim=-2)[0]
        x1 = torch.min(torch.where(~mask, x[Ellipsis, None], x[Ellipsis, -1:, None]), dim=-2)[0]
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    # `nan_to_num` exists in pytorch>=1.8.0
    t = torch.clamp(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)

    # Prevent gradient from backprop-ing through `samples`.
    return samples.detach()


def sample_pdf(
    bins, weights, origins, directions, z_vals, num_samples, randomized,
    coarse_sample_with_fine,
):
    """Hierarchical sampling.
    Args:
      bins: torch.tensor(float32), [batch_size, num_bins + 1].
      weights: torch.tensor(float32), [batch_size, num_bins].
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      z_vals: torch.tensor(float32), [batch_size, num_coarse_samples].
      num_samples: int, the number of samples.
      randomized: bool, use randomized samples.
    Returns:
      z_vals: torch.tensor(float32),
        [batch_size, num_coarse_samples + num_fine_samples].
      points: torch.tensor(float32),
        [batch_size, num_coarse_samples + num_fine_samples, 3].
    """
    z_samples = piecewise_constant_pdf(bins, weights, num_samples, randomized)
    if coarse_sample_with_fine:
        # Compute united z_vals and sample points
        z_vals = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)[0]
    else:
        z_vals = z_samples
    coords = cast_rays(z_vals, origins, directions)
    return z_vals, coords
