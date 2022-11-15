"""
Loss functions

Created on Sun nov 23 11:17:09 2022

@author: mukul
"""

import torch
from torch import nn
import torch.nn.functional as F


class IMSE(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IMSE, self).__init__()

    def forward(self, inputs, targets, beta=0.7):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        rmse = (1 - beta) * (torch.sqrt(torch.mean((targets - inputs) ** 2))) + beta * (
            torch.sqrt(torch.sum((targets - inputs) ** 2) / union))

        return rmse

class IoULoss(nn.Module):
    def  __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, pred, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # pred = F.sigmoid(pred)
        SMOOTH = 1e-6

        # pred = pred.clone()
        # targets = targets.clone()
        # targets[targets > 0] = 1
        # targets = targets.to(torch.int)
        # pred[pred > 0] = 1
        # pred = pred.to(torch.int)

        # union is the mutually inclusive area of all labels & predictions
        # intersection = (pred * targets).sum((1, 2, 3))
        # total = (pred + targets).sum((1, 2, 3))
        intersection = (pred * targets)
        intersection[intersection > 0] = 1
        intersection = intersection.sum((1, 2, 3))
        total = (pred + targets)
        total[total > 0] = 1
        total = total.sum((1, 2, 3))
        union = total - intersection
        IoU = torch.mean((intersection + SMOOTH) / (union + SMOOTH))
        if IoU > 1:
            print('woh')
        if(any(union <0)):
            print('ahh')

        return abs(1 - IoU)
