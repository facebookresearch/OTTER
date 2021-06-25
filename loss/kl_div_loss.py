# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class KLDivLoss(nn.Module):
    """
    Wrapped KLDivergence Loss with a learnable temperature.
    """
    def __init__(self):
        super().__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.T_s = nn.Parameter(torch.tensor(3.9, requires_grad=True))

    def forward(self, pred, target_prob):
        """
        Pred is logits and target is probabilities.
        """
        T_s = torch.clamp(torch.exp(self.T_s), min=1.0, max=100.0)
        pred_logprob = self.logsoftmax(pred * T_s)

        return self.loss(input=pred_logprob, target=target_prob)
