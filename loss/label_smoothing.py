import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        # ignore_index = -100, reduction = 'mean'
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, smoothed_target,_mask):
        logprobs = F.log_softmax(x, dim=-1)
        loss = - logprobs * smoothed_target * _mask
        loss = loss[torch.nonzero(loss,as_tuple=True)].mean()
        return loss