""" https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/knn.py """
from typing import Union

import torch


def knn_gather(
    x: torch.Tensor,
    idx: torch.Tensor,
    lengths: Union[torch.Tensor, None] = None,
):
    """
    A helper function for knn that allows indexing a tensor x with the indices `idx`
    returned by `knn_points`.
    For example, if `dists, idx = knn_points(p, x, lengths_p, lengths, K)`
    where p is a tensor of shape (N, L, D) and x a tensor of shape (N, M, D),
    then one can compute the K nearest neighbors of p with `p_nn = knn_gather(x, idx, lengths)`.
    It can also be applied for any tensor x of shape (N, M, U) where U != D.
    Args:
        x: Tensor of shape (N, M, U) containing U-dimensional features to
            be gathered.
        idx: LongTensor of shape (N, L, K) giving the indices returned by `knn_points`.
        lengths: LongTensor of shape (N,) of values in the range [0, M], giving the
            length of each example in the batch in x. Or None to indicate that every
            example has length M.
    Returns:
        x_out: Tensor of shape (N, L, K, U) resulting from gathering the elements of x
            with idx, s.t. `x_out[n, l, k] = x[n, idx[n, l, k]]`.
            If `k > lengths[n]` then `x_out[n, l, k]` is filled with 0.0.
    """
    N, M, U = x.shape
    _N, L, K = idx.shape

    if N != _N:
        raise ValueError("x and idx must have same batch dimension.")

    if lengths is None:
        lengths = torch.full(
            (x.shape[0],), M, dtype=torch.int64, device=x.device
        )

    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, U)
    # idx_expanded has shape [N, L, K, U]

    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)
    # p2_nn has shape [N, L, K, U]

    needs_mask = lengths.min() < K
    if needs_mask:
        # mask has shape [N, K], true where idx is irrelevant because
        # there is less number of points in p2 than K
        mask = lengths[:, None] <= torch.arange(K, device=x.device)[None]

        # expand mask to shape [N, L, K, U]
        mask = mask[:, None].expand(-1, L, -1)
        mask = mask[:, :, :, None].expand(-1, -1, -1, U)
        x_out[mask] = 0.0

    return x_out
