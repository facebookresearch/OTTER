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
        temp = torch.clamp(torch.exp(self.T_s), min=0, max=100.0)
        pred_logprob = self.logsoftmax(pred * temp)

        return self.loss(input=pred_logprob, target=target_prob)
