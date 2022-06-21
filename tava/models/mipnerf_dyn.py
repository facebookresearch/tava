# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
from tava.models.basic.mipnerf import (
    MipNerfModel,
    resample_along_rays,
    sample_along_rays,
    volumetric_rendering,
)
from tava.models.basic.mlp import MLP
from tava.models.deform_posi_enc.rigid import DisentangledDPEncoder
from tava.models.deform_posi_enc.snarf import SNARFDPEncoder
from tava.utils.bone import (
    closest_distance_to_points,
    closest_distance_to_rays,
    get_end_points,
)
from tava.utils.structures import Bones, Rays, namedtuple_map


def _restore_and_fill(values, masks, fill_in=0.0):
    assert masks.dim() == 1
    restored_values = torch.zeros(
        [masks.shape[0]] + list(values.shape[1:]),
        dtype=values.dtype,
        device=values.device,
    )
    restored_values[masks] = values
    restored_values[~masks] += fill_in
    return restored_values


def _select_rays_near_to_bones(rays: Rays, bones: Bones, threshold: float):
    """Select rays near to the bones and calculate per-ray near far plane."""
    dists, t_vals = closest_distance_to_rays(bones, rays)  # [n_rays, n_bones]

    heads, tails = get_end_points(bones)
    margin = torch.linalg.norm(heads - tails, dim=-1) / 2.0 + threshold

    # get near far but relax with margin
    t_margin = margin / torch.linalg.norm(rays.directions, dim=-1, keepdim=True)
    t_vals[dists >= threshold] = -1e10
    far = (t_vals + t_margin).max(dim=-1).values
    t_vals[dists >= threshold] = 1e10
    near = (t_vals - t_margin).min(dim=-1).values.clamp(min=0)
    selector = near < far

    rays = namedtuple_map(lambda x: x[selector], rays)
    near = near[selector]
    far = far[selector]
    return rays, near, far, selector


def _interp_along_rays(masks, z_vals, values_list, dim=-1):
    assert masks.dim() == 2 and dim == -1
    t = torch.arange(masks.shape[dim], device=masks.device) + 1
    indices_next = (masks.shape[dim] - 1) - torch.cummax(
        masks.flip([dim]) * t, dim
    ).indices.flip([dim])
    indices_before = torch.cummax(masks * t, dim).indices

    z_vals_next = torch.gather(z_vals, dim, indices_next)
    z_vals_before = torch.gather(z_vals, dim, indices_before)
    z_weight = (z_vals - z_vals_before) / (z_vals_next - z_vals_before + 1e-10)

    masks_next = torch.gather(masks, 1, indices_next)
    masks_before = torch.gather(masks, 1, indices_before)
    masks_new = masks_next & masks_before

    outs = [masks_new]
    for values in values_list:
        values_next = torch.gather(
            values, 1, indices_next.unsqueeze(-1).expand_as(values)
        )
        values_before = torch.gather(
            values, 1, indices_before.unsqueeze(-1).expand_as(values)
        )
        values_interp = (values_next - values_before) * z_weight.unsqueeze(
            -1
        ) + values_before
        outs.append(
            values_interp * masks_new.to(values_interp.dtype).unsqueeze(-1)
        )
    return outs


class DynMipNerfModel(MipNerfModel):
    def __init__(
        self,
        pos_enc: nn.Module,
        shading_mode: int = None,  # shading mode
        # The dim of the pose-dependent condition for shading
        # None or zero means no shading condition.
        shading_pose_dim: int = None,
        # Sample-bone distance threshold in the world space.
        world_dist: float = None,
    ):
        # `pos_enc` is a deformable positional encoding, that maps a world
        # coordinate `x_w` to its representation `x_c` conditioned on the
        # pose latent `p`. pos_enc(x, p) -> x_c

        assert shading_mode in [None, "implicit", "implicit_AO"]
        self.shading_mode = shading_mode
        self.shading_pose_dim = shading_pose_dim
        self.world_dist = world_dist

        # Define the MLP that query the color & density etc from the
        # representation `x_c`. We treat object as lambertian so we don't
        # model view-dependent color in TAVA.
        if shading_mode is None:
            # the color is not shaded (not pose-dependent).
            mlp = MLP(
                input_dim=pos_enc.out_dim,
                # disable pose-conditioned color.
                condition_dim=0,
                # diable AO output
                num_ao_channels=0,
                condition_ao_dim=0,
            )
            ao_activation = None
        elif shading_mode == "implicit":
            # the color is implicitly conditioned on the shading condition
            assert shading_pose_dim is not None
            mlp = MLP(
                input_dim=pos_enc.out_dim,
                # implicitly shading-conditioned color
                condition_dim=shading_pose_dim,
                # diable AO output
                num_ao_channels=0,
                condition_ao_dim=0,
            )
            ao_activation = None
        elif shading_mode == "implicit_AO":
            assert shading_pose_dim is not None
            # the color is scaled by ambiant occlution
            # which is learnt implicitly
            mlp = MLP(
                input_dim=pos_enc.out_dim,
                # disable implicitly conditioned color
                condition_dim=0,
                # enable AO output
                num_ao_channels=1,
                condition_ao_dim=shading_pose_dim,
            )
            ao_activation = torch.nn.Sigmoid()
        else:
            raise ValueError(shading_mode)

        super().__init__(mlp, pos_enc, num_samples=64, use_viewdirs=False)
        self.ao_activation = ao_activation

    def _query_mlp(
        self,
        _rays,
        samples,
        bones_posed,
        bones_rest,
        randomized=True,
        **kwargs,
    ):
        x, x_cov = samples
        if self.disable_integration:
            x_cov = torch.zeros_like(x_cov)

        # Deform and encode the world coordinates conditioned on Bone movements.
        # `x_enc` is ready to enter the MLP for querying color and density etc,
        # `x_warp` is basically `x_enc` but strips out the [sin, cos] frequency encoding.
        mask, valid = None, None
        if isinstance(self.pos_enc, DisentangledDPEncoder):
            x_enc, x_warp = self.pos_enc(
                x,
                x_cov,
                bones_posed,
                rigid_clusters=kwargs.get("rigid_clusters", None),
            )
        elif isinstance(self.pos_enc, SNARFDPEncoder):
            x_enc, x_warp, mask, valid = self.pos_enc(
                x,
                x_cov,
                bones_posed,
                bones_rest,
                rigid_clusters=kwargs.get("rigid_clusters", None),
                pose_latent=kwargs.get("pose_latent", None),
            )
        else:
            raise ValueError(type(self.pos_enc))

        # Point attribute predictions
        # TODO(ruilong): The naming of `cond_view` and `cond_extra` are confusing.
        if self.shading_mode is None:
            raw_rgb, raw_density = self.mlp(x_enc, masks=mask)
        elif self.shading_mode == "implicit":
            # the `cond_view` field affects color implicitly.
            raw_rgb, raw_density = self.mlp(
                x_enc, cond_view=kwargs["pose_latent"], masks=mask
            )
        elif self.shading_mode == "implicit_AO":
            raw_rgb, raw_density, raw_ao = self.mlp(
                x_enc, cond_extra=kwargs["pose_latent"], masks=mask
            )

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
        if self.shading_mode == "implicit_AO":
            ao = self.ao_activation(raw_ao)
            rgb = rgb * ao

        # For SNARF encoder, there are multiple candidates, so we need to
        # merge them in the end.
        if isinstance(self.pos_enc, SNARFDPEncoder):
            # The `mask` stores which candidates are inside a certain range
            # of the bones in canonical space, and is converged during root
            # finding. So for the other points we set the density to zero.
            density = density * mask[..., None]
            # Then we select the "best" candidate by just finding the one
            # with the maximum density. Note we need to be careful with the
            # case where all candidates may be diverged during root finding.
            # We here quitely update the `valid` tensor that records convergence
            # from multiple candidates to the best candidate.
            density, rgb, x_warp, valid = self.pos_enc.aggregate(
                density, rgb, x_warp, valid
            )
        return rgb, density, x_warp, valid

    def forward(
        self,
        rays: Rays,
        color_bkgd: torch.Tensor,
        bones_posed: Bones,
        bones_rest: Bones,
        randomized: bool = True,
        **kwargs,
    ):
        ret = []
        extra_info = {}
        if isinstance(self.pos_enc, SNARFDPEncoder):
            # {"loss_bone_w", "loss_bone_offset"}
            extra_info.update(
                self.pos_enc.get_extra_losses(
                    bones_rest,
                    rigid_clusters=kwargs.get("rigid_clusters", None),
                    pose_latent=kwargs.get("pose_latent", None),
                )
            )

        # Calculate per-ray near far distance based on the bones
        # and select rays near to the bones to process
        rays, near, far, selector = _select_rays_near_to_bones(
            rays, bones_posed, threshold=self.world_dist
        )

        # skip all rays in this batch if no ray intersection.
        if ~selector.any():
            for _ in range(self.num_levels):
                ret.append(
                    (
                        torch.ones(
                            (selector.shape[0], 3), 
                            dtype=near.dtype,
                            device=selector.device
                        )
                        * color_bkgd,  # rgb
                        torch.zeros(
                            (selector.shape[0]), 
                            dtype=near.dtype,
                            device=selector.device
                        ),  # depth
                        torch.zeros(
                            (selector.shape[0]), 
                            dtype=near.dtype,
                            device=selector.device
                        ),  # acc
                        torch.zeros(
                            (selector.shape[0], self.pos_enc.warp_dim),
                            dtype=near.dtype,
                            device=selector.device
                        ),  # warp
                    )
                )
            return ret, extra_info

        # start rendering
        weights = None
        for i_level in range(self.num_levels):
            if i_level == 0:
                # Stratified sampling along rays
                t_vals, samples = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    near[:, None],
                    far[:, None],
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
            rgb, density, warp, valid = self._query_mlp(
                rays,
                samples,
                bones_posed,
                bones_rest,
                randomized=randomized,
                **kwargs,
            )
            if valid is not None:
                # If there are missing values (not valid), linearly interpolate
                # along the rays. Be careful: In MipNeRF, the samples are
                # (mean, cov) and t_vals are borders with shape
                # (num_rays, num_samples + 1)
                t_mids = (t_vals[..., :-1] + t_vals[..., 1:]) / 2.0
                _, rgb, density, warp = _interp_along_rays(
                    valid, t_mids, [rgb, density, warp]
                )

            # Samples out of range should have zero density no matter what.
            samples_selector = (
                closest_distance_to_points(bones_posed, samples[0])
                .min(dim=-1)
                .values
                < self.world_dist
            )
            density = density * samples_selector[..., None]

            # basic volumetric rendering same as nerf
            comp_rgb, depth, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_vals,
                rays.directions,
                color_bkgd=color_bkgd,
            )
            comp_warp = (weights[..., None] * warp).sum(dim=-2)

            # fill in values for those rays we skipped.
            ret.append(
                (
                    _restore_and_fill(comp_rgb, selector, fill_in=color_bkgd),
                    _restore_and_fill(depth, selector, fill_in=0.0),
                    _restore_and_fill(acc, selector, fill_in=0.0),
                    _restore_and_fill(comp_warp, selector, fill_in=0),
                )
            )

        # if valid is not None:
        #     extra_info["root_success"] = valid[samples_selector].float().mean()
        return ret, extra_info
