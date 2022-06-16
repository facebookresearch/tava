# Copyright 2021 Google LLC
# Modified by Ruilong Li from https://github.com/google/mipnerf
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


class MipNerfModel(nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(
        self,
        mlp: nn.Module,
        # integrated positional encoding for coordinates
        pos_enc: nn.Module,
        # positional encoding for view directions
        view_enc: nn.Module = None,
        num_samples: int = 128,  # The number of samples per level.
        num_levels: int = 2,  # The number of sampling levels.
        # Dirichlet/alpha "padding" on the histogram.
        resample_padding: float = 0.01,
        stop_level_grad: bool = True,  # If True, don't backprop across levels')
        # If True, use view directions as a condition.
        use_viewdirs: bool = True,
        # If True, sample linearly in disparity, not in depth.
        lindisp: bool = False,
        # The shape of cast rays ('cone' or 'cylinder').
        ray_shape: str = "cone",
        density_activation=torch.nn.Softplus(),  # Density activation.
        # Standard deviation of noise added to raw density.
        density_noise: float = 0.0,
        # The shift added to raw densities pre-activation.
        density_bias: float = -1.0,
        rgb_activation=torch.nn.Sigmoid(),  # The RGB activation.
        rgb_padding: float = 0.001,  # Padding added to the RGB outputs.
        disable_integration: bool = False,  # If True, use PE instead of IPE.
    ):
        super().__init__()
        self.mlp = mlp
        self.pos_enc = pos_enc
        self.view_enc = view_enc
        self.num_samples = num_samples
        self.num_levels = num_levels
        self.resample_padding = resample_padding
        self.stop_level_grad = stop_level_grad
        self.use_viewdirs = use_viewdirs
        self.lindisp = lindisp
        self.ray_shape = ray_shape
        self.density_activation = density_activation
        self.density_noise = density_noise
        self.density_bias = density_bias
        self.rgb_activation = rgb_activation
        self.rgb_padding = rgb_padding
        self.disable_integration = disable_integration
        self.diag = self.pos_enc.diag

    def _query_mlp(self, rays, samples, randomized=True, **kwargs):
        if self.disable_integration:
            samples = (samples[0], torch.zeros_like(samples[1]))
        samples_enc = self.pos_enc(samples)

        # Point attribute predictions
        if self.use_viewdirs:
            viewdirs_enc = self.view_enc(rays.viewdirs)
            raw_rgb, raw_density = self.mlp(samples_enc, cond_view=viewdirs_enc)
        else:
            raw_rgb, raw_density = self.mlp(samples_enc)

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
                # Stratified sampling along rays
                t_vals, samples = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.lindisp,
                    self.ray_shape,
                    diag=self.diag,
                )
            else:
                t_vals, samples = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_vals,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_level_grad,
                    resample_padding=self.resample_padding,
                    diag=self.diag,
                )

            rgb, density = self._query_mlp(
                rays, samples, randomized=randomized, **kwargs
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


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]

    d_mag_sq = torch.clip(torch.sum(d**2, dim=-1, keepdims=True), min=1e-10)

    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1], device=d.device)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `d` is normalized.
    Args:
        d: torch.float32 3-vector, the axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable: boolean, whether or not to use the stable computation described in
            the paper (setting this to False will cause catastrophic failure).
    Returns:
        a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * (
            (hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2
        )
        r_var = base_radius**2 * (
            (mu**2) / 4
            + (5 / 12) * hw**2
            - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2)
        )
    else:
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = base_radius**2 * (
            3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3)
        )
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and radius is the
    radius. Does not renormalize `d`.
    Args:
        d: torch.float32 3-vector, the axis of the cylinder
        t0: float, the starting distance of the cylinder.
        t1: float, the ending distance of the cylinder.
        radius: float, the radius of the cylinder
        diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    Returns:
        a Gaussian (mean and covariance).
    """
    t_mean = (t0 + t1) / 2
    r_var = radius**2 / 4
    t_var = (t1 - t0) ** 2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
    Args:
        t_vals: float array, the "fencepost" distances along the ray.
        origins: float array, the ray origin coordinates.
        directions: float array, the ray direction vectors.
        radii: float array, the radii (base radii for cones) of the rays.
        ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
        diag: boolean, whether or not the covariance matrices should be diagonal.
    Returns:
        a tuple of arrays of means and covariances.
    """
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    if ray_shape == "cone":
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == "cylinder":
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False  # noqa
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs


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
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
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
    depth = (weights * t_mids).sum(dim=-1)
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
    radii,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
    ray_shape,
    diag,
):
    """Stratified sampling along the rays.
    Args:
        origins: torch.ndarray(float32), [batch_size, 3], ray origins.
        directions: torch.ndarray(float32), [batch_size, 3], ray directions.
        radii: torch.ndarray(float32), [batch_size, 3], ray radii.
        num_samples: int.
        near: torch.ndarray, [batch_size, 1], near clip.
        far: torch.ndarray, [batch_size, 1], far clip.
        randomized: bool, use randomized stratified sampling.
        lindisp: bool, sampling linearly in disparity rather than depth.
        ray_shape: string, which shape ray to assume.
    Returns:
        t_vals: torch.ndarray, [batch_size, num_samples], sampled z values.
        means: torch.ndarray, [batch_size, num_samples, 3], sampled means.
        covs: torch.ndarray, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    batch_size = origins.shape[0]
    device = origins.device

    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand([batch_size, num_samples + 1], device=device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, num_samples + 1])
    means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape, diag)
    return t_vals, (means, covs)


def resample_along_rays(
    origins,
    directions,
    radii,
    t_vals,
    weights,
    randomized,
    ray_shape,
    stop_grad,
    resample_padding,
    diag,
):
    """Resampling.
    Args:
        origins: torch.ndarray(float32), [batch_size, 3], ray origins.
        directions: torch.ndarray(float32), [batch_size, 3], ray directions.
        radii: torch.ndarray(float32), [batch_size, 3], ray radii.
        t_vals: torch.ndarray(float32), [batch_size, num_samples+1].
        weights: torch.array(float32), weights for t_vals
        randomized: bool, use randomized samples.
        ray_shape: string, which kind of shape to assume for the ray.
        stop_grad: bool, whether or not to backprop through sampling.
        resample_padding: float, added to the weights before normalizing.
    Returns:
        t_vals: torch.ndarray(float32), [batch_size, num_samples+1].
        points: torch.ndarray(float32), [batch_size, num_samples, 3].
    """
    # Do a blurpool.
    weights_pad = torch.cat(
        [weights[..., :1], weights, weights[..., -1:]], dim=-1
    )
    weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    # Add in a constant (the sampling function will renormalize the PDF).
    weights = weights_blur + resample_padding

    new_t_vals = sorted_piecewise_constant_pdf(
        t_vals,
        weights,
        t_vals.shape[-1],
        randomized,
    )
    if stop_grad:
        new_t_vals = new_t_vals.detach()
    means, covs = cast_rays(
        new_t_vals, origins, directions, radii, ray_shape, diag
    )
    return new_t_vals, (means, covs)


def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """Piecewise-Constant PDF sampling from sorted bins.
    Args:
        bins: torch.ndarray(float32), [batch_size, num_bins + 1].
        weights: torch.ndarray(float32), [batch_size, num_bins].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.
    Returns:
        t_samples: torch.ndarray(float32), [batch_size, num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdims=True)
    padding = torch.clip(eps - weight_sum, min=0)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.clip(torch.cumsum(pdf[..., :-1], dim=-1), max=1)
    cdf = torch.cat(
        [
            torch.zeros(
                list(cdf.shape[:-1]) + [1], dtype=cdf.dtype, device=cdf.device
            ),
            cdf,
            torch.ones(
                list(cdf.shape[:-1]) + [1], dtype=cdf.dtype, device=cdf.device
            ),
        ],
        dim=-1,
    )

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = torch.arange(num_samples, device=pdf.device) * s
        u = u + s * torch.rand(
            list(cdf.shape[:-1]) + [num_samples], device=cdf.device
        )
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.clip(u, max=1.0 - torch.finfo().eps)
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(
            0.0, 1.0 - torch.finfo().eps, num_samples, device=cdf.device
        )
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = torch.max(
            torch.where(mask, x[..., None], x[..., :1, None]), -2
        ).values
        x1 = torch.min(
            torch.where(~mask, x[..., None], x[..., -1:, None]), -2
        ).values
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples
