from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tava.models.basic.mlp import StraightMLP
from tava.models.basic.posi_enc import PositionalEncoder
from tava.models.deform_posi_enc.naive import PoseConditionDPEncoder
from tava.utils.bone import closest_distance_to_points, sample_on_bones
from tava.utils.knn import knn_gather
from tava.utils.point import homo
from tava.utils.structures import Bones
from torch import Tensor


class _SkinWeightsNet(nn.Module):
    def __init__(self, n_transforms: int):
        super().__init__()
        self.posi_enc = PositionalEncoder(
            in_dim=3, min_deg=0, max_deg=4, append_identity=True
        )
        self.net = StraightMLP(
            net_depth=4,
            net_width=128,
            input_dim=self.posi_enc.out_dim,
            output_dim=n_transforms,
        )

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Query skinning weights in canonical space.

        :params x: Canonical points. [..., 3]
        :params mask: Optionally mask out some points. [...]
        :return logits of the skinning weights. [..., n_transforms]
        """
        return self.net(self.posi_enc(x), mask=mask)


class _OffsetsNet(nn.Module):
    def __init__(self, pose_dim: int):
        super().__init__()
        self.posi_enc = PoseConditionDPEncoder(
            posi_enc=PositionalEncoder(
                in_dim=3, min_deg=0, max_deg=4, append_identity=True
            ),
            pose_dim=pose_dim,
        )
        self.net = StraightMLP(
            net_depth=4,
            net_width=128,
            input_dim=self.posi_enc.out_dim,
            output_dim=3,
        )

    def forward(
        self, x: Tensor, pose_latent: Tensor, mask: Tensor = None
    ) -> Tensor:
        """Query skinning offsets.

        :params x: Canonical points. [..., 3]
        :params mask: Optionally mask out some points. [...]
        :return offsets in the world space. [..., output_dim]
        """
        x_enc, _ = self.posi_enc(x, None, pose_latent)
        return self.net(x_enc, mask=mask)


class SNARFDPEncoder(nn.Module):
    """Differential Forward Skinning Module based on SNARF."""

    def __init__(
        self,
        posi_enc: nn.Module,  # IPE for the *final* coordinate encoding.
        n_transforms: int,  # n transforms for LBS.
        with_bkgd: bool = True,  # with the background virtual bone.
        search_n_init: int = 5,  # K initialization for root finding.
        soft_blend: int = 5,  # soften the skinning weights softmax.
        # disable the non-linear offset during inference
        offset_net_enabled: bool = True,  # enable non-linear offset network.
        offset_pose_dim: int = None,  # pose-dependent latent dim.
        # zero the offset during inference.
        offset_constant_zero: bool = False,
        # sample-bone distance threshold in the canonical space.
        cano_dist: float = None,
    ):
        super().__init__()
        self.posi_enc = posi_enc
        self.n_transforms = n_transforms
        self.with_bkgd = with_bkgd
        self.search_n_init = search_n_init
        self.soft_blend = soft_blend
        self.offset_net_enabled = offset_net_enabled
        self.offset_constant_zero = offset_constant_zero
        self.cano_dist = cano_dist

        self.skin_net = _SkinWeightsNet(n_transforms + int(with_bkgd))
        if offset_net_enabled:
            assert offset_pose_dim is not None
            self.offset_net = _OffsetsNet(offset_pose_dim)

    @property
    def warp_dim(self):
        return 3

    @property
    def out_dim(self):
        return self.posi_enc.out_dim

    @property
    def diag(self):
        return self.posi_enc.diag

    def aggregate(self, density, rgb, x_cano, mask):
        """The aggregation function for multiple candidates.

        :params density: [..., I, 1]
        :params color: [..., I, 3]
        :params x_cano: [..., I, 3]
        :params mask: [..., I]
        :return
            aggregated density: [..., 1]
            aggregated color: [..., 3]
            aggregated x_cano: [..., 3]
            aggregated valid: which values are all valid.
        """
        density, indices = torch.max(density, dim=-2)
        rgb = torch.gather(
            rgb,
            -2,
            indices.unsqueeze(-2).expand(list(rgb.shape[:-2]) + [1, 3]),
        ).squeeze(-2)
        x_cano = torch.gather(
            x_cano,
            -2,
            indices.unsqueeze(-2).expand(list(x_cano.shape[:-2]) + [1, 3]),
        ).squeeze(-2)
        mask = mask.any(dim=-1)
        return density, rgb, x_cano, mask

    def forward(
        self,
        x: Tensor,
        x_cov: Tensor,
        bones_world: Bones,
        bones_cano: Bones,
        rigid_clusters: Tensor = None,
        pose_latent: Tensor = None,
    ):
        """
        :params x: Points in the world space. [B, N, 3]
        :params x_cov: For mipnerf posi enc. [B, N, 3(, 3)]
        :params bones_world: World space bones. [n_bones,]
        :params bones_cano: Canonical space bones. [n_bones,]
        :params rigid_clusters: Optionally. The cluster id for each bone in
            torch.int32. The bones with the same cluster id are supposed to
            be moved rigidly together. `None` means each bones moves
            individually. [n_bones,]
        :params pose_latent: Optionally passed in for the offsets network.
        :return
            x_enc: The candidated encoding for x. [B, N, I, self.out_dim]
            x_warp: The candidated canonical points for x. [B, N, I, self.warp_dim]
            x_cano: [B, N, I] with boolen value. Whether the candidated canonical
                points are near enough to the bones to be considered. (Out range
                points will be set to zero density later)
            valid: [B, N, I] with boolen value. Whether the candidated canonical
                points are valid from root finding. (Failures from root finding
                will be interpolated later)
        """
        with torch.no_grad():
            # Initialize I = n_init (+ 1) candidates in canonical.
            x_cano_init = _initialize_canonical(
                x,
                bones_world,
                bones_cano,
                rigid_clusters=rigid_clusters,
                n_init=self.search_n_init,
                with_bkgd=self.with_bkgd,
            )  # [B, N, I, 3]

            # Root finding to get the corresponded canonical points.
            x_cano, valid, _ = self._search_canonical(
                x,
                x_cano_init,
                bones_world,
                bones_cano,
                rigid_clusters=rigid_clusters,
                pose_latent=pose_latent,
            )  # [B, N, I, 3], [B, N, I]

            # Identify points that are out of range in canonical space.
            if self.cano_dist > 0:
                mask = (
                    closest_distance_to_points(bones_cano, x_cano)
                    .min(dim=-1)
                    .values
                    < self.cano_dist
                )  # [B, N, I]
                mask = mask * valid
            else:
                mask = valid

        if self.training:
            # Inject gradients to `x_cano`, does not change its values.
            x_cano, _, _ = self.forward_skinning_inject_grads(
                x_cano,
                bones_world,
                bones_cano,
                rigid_clusters=rigid_clusters,
                pose_latent=pose_latent,
            )
            
        # positional encoding
        if x_cov is not None:
            x_cov = (
                x_cov.unsqueeze(-2).expand(list(x_cano.shape[:-1]) + [3])
                if self.posi_enc.diag
                else x_cov.unsqueeze(-3).expand(list(x_cano.shape[:-1]) + [3, 3])
            )
            x_enc = self.posi_enc((x_cano, x_cov))
        else:
            x_enc = self.posi_enc(x_cano)
        return x_enc, x_cano, mask, valid

    def query_weights(self, x_cano: Tensor, mask: Tensor = None):
        weights_logit = self.skin_net(x_cano, mask=mask)
        weights = F.softmax(weights_logit * self.soft_blend, dim=-1)
        return weights

    def forward_skinning(
        self,
        x_cano: Tensor,
        bones_world: Bones,
        bones_cano: Bones,
        rigid_clusters: Tensor = None,
        pose_latent: Tensor = None,
        mask: Tensor = None,
    ) -> Tensor:
        """Skinning the canonical points to world space using LBS.

        :params x_cano: Canonical points [..., 3]
        :params bones_world: World space bones. [n_bones,]
        :params bones_cano: Canonical space bones. [n_bones,]
        :params rigid_clusters: Optionally. The cluster id for each bone in
            torch.int32. The bones with the same cluster id are supposed to
            be moved rigidly together. `None` means each bones moves
            individually. [n_bones,]
        :params pose_latent: Optionally passed in for the offsets network.
        :params mask: Optionally to skip some points. [...]
        :returns:
            World space points. [..., 3]
        """
        # Query skinning weights. [..., n_transforms (+ 1)]
        weights = self.query_weights(x_cano, mask=mask)

        # Get transforms that match with the skinning weights. [n_transforms, 4, 4]
        transforms = _try_collapse_rigid_bones(
            # (cano -> bone -> world)
            bones_world.transforms @ bones_cano.transforms.inverse(),
            rigid_clusters=rigid_clusters,
            collapse_func=lambda x: x[0],
        )

        # Append the global transforms for background samples
        if self.with_bkgd:
            assert weights.shape[-1] == transforms.shape[0] + 1
            # [n_transforms + 1, 4, 4]
            transforms = torch.cat(
                [transforms, torch.eye(4)[None, :, :].to(transforms)], dim=-3
            )

        if self.offset_net_enabled and not self.offset_constant_zero:
            assert (
                pose_latent is not None
            ), "The offset needs to be pose-dependent."
            offsets = self.offset_net(x_cano, pose_latent, mask=mask)
        else:
            offsets = None

        x_world = _linear_skinning(
            x_cano, weights, transforms, offsets_world=offsets
        )
        return x_world

    def forward_skinning_inject_grads(
        self,
        x_cano: Tensor,
        bones_world: Bones,
        bones_cano: Bones,
        rigid_clusters: Tensor = None,
        pose_latent: Tensor = None,
    ) -> Tensor:
        """Skinning the canonical points to world space using LBS.

        :params x_cano: Canonical points [..., 3]
        :params bones_world: World space bones. [n_bones,]
        :params bones_cano: Canonical space bones. [n_bones,]
        :params rigid_clusters: Optionally. The cluster id for each bone in
            torch.int32. The bones with the same cluster id are supposed to
            be moved rigidly together. `None` means each bones moves
            individually. [n_bones,]
        :params pose_latent: Optionally passed in for the offsets network.
        :returns:
            World space points. [..., 3]
        """
        assert (
            self.training
        ), "Please use `forward_skinning` instead in test mode."
        x_world, grads = self._calc_skining_gradients(
            x_cano,
            bones_world,
            bones_cano,
            rigid_clusters=rigid_clusters,
            pose_latent=pose_latent,
        )
        # trick for implicit diff with autodiff:
        # x_cano = x_cano + 0 and x_cano' = correction'
        x_cano = x_cano.detach()
        grads_inv = grads.inverse().detach()
        correction = x_world - x_world.detach()
        correction = torch.einsum(
            "...ij,...j->...i", -grads_inv.detach(), correction
        )
        x_cano = x_cano + correction
        return x_cano, x_world, grads

    @torch.enable_grad()
    def _calc_skining_gradients(
        self,
        x_cano: Tensor,
        bones_world: Bones,
        bones_cano: Bones,
        rigid_clusters: Tensor = None,
        pose_latent: Tensor = None,
    ):
        """Calculate forward skinning gradients (jacobian).

        This function will also return the world space points.

        :params x_cano: Canonical points [..., 3]
        :params bones_world: World space bones. [n_bones,]
        :params bones_cano: Canonical space bones. [n_bones,]
        :params rigid_clusters: Optionally. The cluster id for each bone in
            torch.int32. The bones with the same cluster id are supposed to
            be moved rigidly together. `None` means each bones moves
            individually. [n_bones,]
        :params pose_latent: Optionally passed in for the offsets network.
        :return
            World space points. [..., 3]
            Jacobian of d(x_world) / d(x_cano). [..., 3, 3]
        """
        x_cano.requires_grad_(True)
        x_world = self.forward_skinning(
            x_cano,
            bones_world,
            bones_cano,
            rigid_clusters=rigid_clusters,
            pose_latent=pose_latent,
        )
        grads = []
        for i in range(x_world.shape[-1]):
            d_out = torch.zeros_like(
                x_world, requires_grad=False, device=x_world.device
            )
            d_out[..., i] = 1
            grad = torch.autograd.grad(
                outputs=x_world,
                inputs=x_cano,
                grad_outputs=d_out,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        return x_world, grads

    @torch.no_grad()
    def _search_canonical(
        self,
        x: Tensor,
        x_cano_init: Tensor,
        bones_world: Bones,
        bones_cano: Bones,
        rigid_clusters: Tensor = None,
        pose_latent: Tensor = None,
    ):
        """Search canonical correspondences of the world points.

        This function uses Newton's method to solve the root finding
        problem f(x_cano) - x_world = 0, given x_cano_init. Thus the
        speed-wise bottleneck lies in this function.

        :params x: World space points. [..., 3]
        :params x_cano_init: Canonical canonidated points. [..., I, 3]
        :params bones_world: World space bones. [n_bones,]
        :params bones_cano: Canonical space bones. [n_bones,]
        :params rigid_clusters: Optionally. The cluster id for each bone in
            torch.int32. The bones with the same cluster id are supposed to
            be moved rigidly together. `None` means each bones moves
            individually. [n_bones,]
        :params pose_latent: Optionally passed in for the offsets network.
        :return
            Canonical space points. [..., I, 3]
            Whether the root finding converged. boolen values. [..., I]
            Estimated inverse Jacobian. [..., I, 3, 3]
        """
        x = x.unsqueeze(-2).expand_as(x_cano_init).reshape(-1, 3)

        # compute init jacobians
        _, J_init = self._calc_skining_gradients(
            x_cano_init,
            bones_world,
            bones_cano,
            rigid_clusters=rigid_clusters,
            pose_latent=pose_latent,
        )
        J_inv_init = J_init.inverse()

        # construct function for root finding
        def _func(x_cano, mask):
            # [*, 3, 1] -> [**, 3, 1]
            x_cano = x_cano.view(-1, 3)
            mask = mask.view(-1)
            x_opt = self.forward_skinning(
                x_cano,
                bones_world,
                bones_cano,
                rigid_clusters=rigid_clusters,
                pose_latent=pose_latent,
                mask=mask,
            )
            error = x_opt - x
            error = error[mask].reshape(-1, 3, 1)
            return error

        x_cano, _, mask, J_inv = _broyden(
            _func,  # [*, 3, 1] -> [**, 3, 1]
            x_cano_init.reshape(-1, 3, 1),
            J_inv_init.reshape(-1, 3, 3),
        )

        x_cano = x_cano.reshape(x_cano_init.shape)
        mask = mask.reshape(x_cano_init.shape[:-1])
        J_inv = J_inv.reshape(list(x_cano_init.shape) + [3])
        return x_cano, mask, J_inv

    def get_extra_losses(
        self,
        bones_rest: Bones,
        rigid_clusters: Tensor = None,
        pose_latent: Tensor = None,
    ):
        losses = {}

        # Skinning weights on the bones should be one-hot.
        # The joints have some ambiguities because they can listen to whichever
        # bone connected to this joint. So we don't sample near to the joints.
        bone_samples = sample_on_bones(
            bones_rest, n_per_bone=5, range=(0.1, 0.9)
        )  # [n_per_bone, n_bones, 3]
        weights = self.query_weights(bone_samples)
        if rigid_clusters is None:
            rigid_clusters = torch.arange(self.n_transforms)
        weights_gt = (
            F.one_hot(rigid_clusters, num_classes=weights.shape[-1])
            .expand_as(weights)
            .to(weights)
        )
        losses["loss_bone_w"] = F.mse_loss(weights, weights_gt)

        # Offsets on the bones should be all zero, including the joints.
        if self.offset_net_enabled:
            assert pose_latent is not None
            # FIXME(ruilongli): The paper's code is using world space sample
            # which is not intended but kind of equal to MC sampling.
            bone_samples = sample_on_bones(
                bones_rest, n_per_bone=5, range=(0.0, 1.0)
            )  # [n_per_bone, n_bones, 3]
            offsets = self.offset_net(bone_samples, pose_latent)
            losses["loss_bone_offset"] = (offsets**2).mean()

        return losses


def _linear_skinning(
    x: torch.Tensor,
    weights: torch.Tensor,
    transforms: torch.Tensor,
    offsets_cano: torch.Tensor = None,
    offsets_world: torch.Tensor = None,
    inverse: bool = False,
) -> torch.Tensor:
    """Linear blend skinning

    x' = sum_{i} {w_i * T_i} dot (x + offset_c) + offset_w

    :params x: Canonical / world points. [..., N, 3]
    :params weights: Skinning weights [..., N, J]
    :params transforms: Joint transformations. [..., J, 4, 4]
    :params offsets_cano: Optional, x offsets in canonical space. [..., N, 3]
    :params offsets_world: Optional, x offsets in world space. [..., N, 3]
    :return World / canonical points. [..., N, 3]
    """
    offsets_cano = 0.0 if offsets_cano is None else offsets_cano
    offsets_world = 0.0 if offsets_world is None else offsets_world

    tf = torch.einsum("...pn,...nij->...pij", weights, transforms)
    if inverse:
        x_in = F.pad(x - offsets_world, (0, 1), value=1.0)
        x_out = torch.einsum("...pij,...pj->...pi", tf.inverse(), x_in)
        return x_out[..., 0:3] - offsets_cano
    else:
        x_in = F.pad(x + offsets_cano, (0, 1), value=1.0)
        x_out = torch.einsum("...pij,...pj->...pi", tf, x_in)
        return x_out[..., 0:3] + offsets_world


@torch.no_grad()
def _initialize_canonical(
    x: Tensor,
    bones_world: Bones,
    bones_cano: Bones,
    rigid_clusters: Tensor = None,
    n_init: int = 5,
    with_bkgd: bool = True,
) -> Tensor:
    """Initialize the canonical correspondances of x in the world space.

    :params x: Points in the world space. [B, N, 3]
    :params bones_world: World space bones. [n_bones,]
    :params bones_cano: Canonical space bones. [n_bones,]
    :params rigid_clusters: The cluster id for each bone in torch.int32.
        [B,] The bones with the same cluster id are supposed to be moved
        rigidly together.
    :params n_init: Number of initializations.
    :params with_bkgd: If true, add itself as an additional initialization
        to keep its possiblilty to be a background point.
    :return x_cano: Initial canonical points. [B, N, n_init(+ 1), 3]
    """
    B, N, _3 = x.shape

    # distance between bones and samples in the world space.
    dists = closest_distance_to_points(bones_world, x)  # [B, N, n_bones]

    # get data for transforms
    dists = _try_collapse_rigid_bones(
        dists.permute(2, 0, 1),
        rigid_clusters=rigid_clusters,
        collapse_func=lambda x: x.min(dim=0).values,
    ).permute(
        1, 2, 0
    )  # [B, N, n_transforms]
    transforms = _try_collapse_rigid_bones(
        # (world -> bone -> cano)
        bones_cano.transforms @ bones_world.transforms.inverse(),
        rigid_clusters=rigid_clusters,
        collapse_func=lambda x: x[0],
    )  # [n_transforms, 4, 4]

    # select the k nearest
    knn = torch.topk(dists, n_init, dim=-1, largest=False)
    knn_idxs = knn.indices  # [B, N, n_init]

    # Gather the knn transforms
    transforms = knn_gather(
        transforms.reshape(1, -1, 4 * 4), knn_idxs.reshape(1, -1, n_init)
    ).view(
        [B, N, n_init, 4, 4]
    )  # [B, N, n_init, 4, 4]

    if with_bkgd:
        # Append the identity transforms for background samples
        transforms = torch.cat(
            [transforms, torch.eye(4).to(transforms).expand([B, N, 1, 4, 4])],
            dim=-3,
        )  # [B, N, n_init + 1, 4, 4]

    # Inverse transformation to canonical space.
    x_cano = torch.einsum("bnipq,bnq->bnip", transforms[..., :3, :4], homo(x))
    return x_cano


def _try_collapse_rigid_bones(
    data: Tensor,
    rigid_clusters: Tensor = None,
    collapse_func: Callable = None,
) -> Tensor:
    """Try collapse somes bone attributes.

    The reason we are doing this is because some bones may be rigidly
    attached to each other so there any some redundancies in the data and
    sometimes it can cause ambiguity. For example, in the task of
    skinning, a surface point can listen to either bone if there are
    multiple bones moving with the same transformation.

    Warning: This function always assume you are trying to collapse the
    **first** dimension of the data.

    :params data: Bone attribute to be collapsed. [B, ...]
    :params rigid_clusters: The cluster id for each bone in torch.int32.
        [B,] The bones with the same cluster id are supposed to be moved
        rigidly together.
    :params collapse_func: Callable function to decide how to collapse.
        For example, `lambda x: x[0]`; `lambda x: x.min(dim=0)` etc.
    :returns
        the collapsed data. The shape should be [n_clusters, ...], where
        n_clusters = len(unique(rigid_clusters)). It may also return
        the original data if you pass None to rigid_clusters.
    """
    if rigid_clusters is None:
        # don't do collapse
        return data
    assert collapse_func is not None
    assert len(rigid_clusters) == len(data)
    data_collapsed = []
    for cluster_id in torch.unique(rigid_clusters):
        selector = rigid_clusters == cluster_id
        # the bones map to the same transform_id should have the
        # same transformation because they are supposed to be
        # rigidly attached to each other.
        # TODO(ruilong) We skip the check here for speed but maybe
        # better to have a check?
        data_collapsed.append(collapse_func(data[selector]))
    data_collapsed = torch.stack(data_collapsed, dim=0)
    return data_collapsed


def _broyden(
    g, x_init, J_inv_init, max_steps=50, cvg_thresh=1e-5, dvg_thresh=1, eps=1e-6
):
    """Find roots of the given function g(x) = 0.
    This function is impleneted based on https://github.com/locuslab/deq.
    Tensor shape abbreviation:
        N: number of points
        D: space dimension
    Args:
        g (function): the function of which the roots are to be determined. shape: [N, D, 1]->[N, D, 1]
        x_init (tensor): initial value of the parameters. shape: [N, D, 1]
        J_inv_init (tensor): initial value of the inverse Jacobians. shape: [N, D, D]
        max_steps (int, optional): max number of iterations. Defaults to 50.
        cvg_thresh (float, optional): covergence threshold. Defaults to 1e-5.
        dvg_thresh (float, optional): divergence threshold. Defaults to 1.
        eps (float, optional): a small number added to the denominator to prevent numerical error. Defaults to 1e-6.
    Returns:
        result (tensor): root of the given function. shape: [N, D, 1]
        diff (tensor): corresponding loss. [N]
        valid_ids (tensor): identifiers of converged points. [N]
    """

    # initialization
    x = x_init.clone().detach()
    J_inv = J_inv_init.clone().detach()

    ids_val = torch.ones(x.shape[0], device=x.device).bool()

    gx = g(x, mask=ids_val)
    update = -J_inv.bmm(gx)

    x_opt = x.clone()
    gx_norm_opt = torch.linalg.norm(gx.squeeze(-1), dim=-1)

    delta_gx = torch.zeros_like(gx)
    delta_x = torch.zeros_like(x)

    ids_val = torch.ones_like(gx_norm_opt).bool()

    for _ in range(max_steps):

        # update paramter values
        delta_x[ids_val] = update
        x[ids_val] += delta_x[ids_val]
        delta_gx[ids_val] = g(x, mask=ids_val) - gx[ids_val]
        gx[ids_val] += delta_gx[ids_val]

        # store values with minial loss
        gx_norm = torch.linalg.norm(gx.squeeze(-1), dim=-1)
        ids_opt = gx_norm < gx_norm_opt
        gx_norm_opt[ids_opt] = gx_norm.clone().detach()[ids_opt]
        x_opt[ids_opt] = x.clone().detach()[ids_opt]

        # exclude converged and diverged points from furture iterations
        ids_val = (gx_norm_opt > cvg_thresh) & (gx_norm < dvg_thresh)
        if ids_val.sum() <= 0:
            break

        # compute paramter update for next iter
        vT = (delta_x[ids_val]).transpose(-1, -2).bmm(J_inv[ids_val])
        a = delta_x[ids_val] - J_inv[ids_val].bmm(delta_gx[ids_val])
        b = vT.bmm(delta_gx[ids_val])
        b[b >= 0] += eps
        b[b < 0] -= eps
        u = a / b
        J_inv[ids_val] += u.bmm(vT)
        update = -J_inv[ids_val].bmm(gx[ids_val])

    diff = gx_norm_opt
    valid_ids = gx_norm_opt < cvg_thresh
    return x_opt, diff, valid_ids, J_inv
