# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import get_rank


class ContrastiveLoss(nn.Module):
    """
    InfoNCE Loss that supports label_smoothing and learnable temperature.
    """
    def __init__(self, T=3.9, label_smoothing=0.0, temp_grad=True):
        super().__init__()
        alpha = 1.0 - label_smoothing
        if alpha < 1.0:
            self.loss = SmoothContrastiveLoss(T, temp_grad, alpha)
        else:
            self.loss = HardContrastiveLoss(T, temp_grad)

    def forward(self, logits):
        return self.loss(logits)

    @property
    def T(self):
        return self.loss.T


class HardContrastiveLoss(nn.Module):
    def __init__(self, T=3.9, temp_grad=True):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(T, requires_grad=temp_grad))

    def forward(self, logits):
        logits = logits * torch.clamp(torch.exp(self.T), min=1.0, max=100.0)
        batch = logits.shape[0]
        offset = get_rank() * batch
        labels = torch.arange(
            offset, offset + batch, device=logits.device, dtype=torch.long,
        )
        loss = F.cross_entropy(logits, labels, reduction="mean")
        return loss


class SmoothContrastiveLoss(nn.Module):
    def __init__(self, T=3.9, temp_grad=True, alpha=1.0):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(T, requires_grad=temp_grad))
        assert alpha > 0 and alpha < 1
        self.alpha = alpha
        self.kld = nn.KLDivLoss(reduction="batchmean")
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, logits):
        T_s = torch.clamp(torch.exp(self.T), min=1.0, max=100.0)
        pred_logprob = self.logsoftmax(logits * T_s)

        with torch.no_grad():
            eps = (1 - self.alpha) / (logits.shape[1] - 1)
            batch = logits.shape[0]
            offset = get_rank() * batch

            t_prob = torch.ones(*logits.shape) * eps
            t_prob[:, offset:offset + batch] += torch.eye(batch) * (self.alpha - eps)
            t_prob = t_prob.to(logits)
            t_ent = (-t_prob * torch.log(t_prob)).sum(dim=1).mean()

        return self.kld(input=pred_logprob, target=t_prob) + t_ent
