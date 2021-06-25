# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import utils.utils as utils


def _has_nan_or_inf(x):
    return torch.isnan(x).any() or torch.isinf(x).any()


@torch.no_grad()
def sinkhorn(cost_mat, eps, niter, r_prob=None, c_prob=None):
    """
    cost_mat: s1, s2, ..., sn, M, N
    r_prob: s1, s2, ..., sn, M
    c_prob: s1, s2, ..., sn, N
    """
    Q = torch.exp(-cost_mat / eps)
    Q = Q / Q.sum(dim=[-2, -1], keepdim=True)
    M, N = Q.shape[-2], Q.shape[-1]

    if r_prob is not None:
        # s1, ..., sn, M -> s1, ..., sn, M, 1
        r_prob = (r_prob / r_prob.sum(dim=-1, keepdim=True)).unsqueeze(-1)
        assert not _has_nan_or_inf(r_prob)
    else:
        r_prob = 1 / M

    if c_prob is not None:
        # s1, ..., sn, N -> s1, ..., sn, 1, N
        c_prob = (c_prob / c_prob.sum(dim=-1, keepdim=True)).unsqueeze(-2)
        assert not _has_nan_or_inf(c_prob)
    else:
        c_prob = 1 / N

    for _ in range(niter):
        # normalize each row: total weight per row must be r_prob
        Q /= Q.sum(dim=-1, keepdim=True)
        Q *= r_prob
        # normalize each column: total weight per column must be c_prob
        Q /= Q.sum(dim=-2, keepdim=True)
        Q *= c_prob
    return Q


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not utils.is_dist_avail_and_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
