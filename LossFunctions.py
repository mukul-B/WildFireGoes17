"""
Loss functions

Created on Sun nov 23 11:17:09 2022

@author: mukul
"""

import torch
from torch import nn

SMOOTH = 1e-6

# Global MSE
class BCE(nn.Module):
    def __init__(self, beta):
        self.last_activation = "sigmoid"
        super(BCE, self).__init__()

    def forward(self, pred, targets):
        targets[targets > 0] = 1
        # critr = nn.BCELoss()
        # loss = critr(pred, targets)
        pred = pred.view(-1)
        targets = targets.view(-1)
        loss = torch.sqrt(torch.mean((targets - pred) ** 2))
        return loss


class jaccard_loss(nn.Module):
    def __init__(self, beta):
        super(jaccard_loss, self).__init__()
        self.last_activation = "sigmoid"
        print('jacard_loss intialized')


    def forward(self, pred, targets):
        # pred = pred.view(-1)
        # targets = targets.view(-1)
        targets[targets>0]=1
        # pred[pred >= 0.5] = 1
        # pred[pred < 0.5] = 0
        intersection = torch.sum(pred * targets,(1,2,3))
        sum_pred= torch.sum(pred ,(1,2,3))
        sum_targets = torch.sum(targets, (1, 2, 3))
        loss =  - torch.mean((intersection + SMOOTH) / (torch.sum(sum_targets) + torch.sum(sum_pred) + intersection + SMOOTH))
        return loss


class two_branch_loss_rmses(nn.Module):
    def __init__(self, beta):
        super(two_branch_loss_rmses, self).__init__()
        self.last_activation = "both"
        self.beta = beta
        print('two_branch_loss intialized')

    # def forward(self,  pred_sup,pred_seg,target,binary_target):
    def forward(self, pred, target):
        pred_sup, pred_seg= pred[0],pred[1]

        # rmse
        pred_sup = pred_sup.view(-1)
        target_sup = target.view(-1)
        rmse_loss = torch.sqrt(torch.mean((target_sup - pred_sup) ** 2))
        # rmse_loss = nn.functional.mse_loss(pred_sup, target)
        # rmse_loss = torch.sqrt(rmse_loss)

        # jaccard
        binary_target = target.cuda()
        binary_target[binary_target > 0] = 1
        pred_seg = pred_seg.view(-1)
        binary_target = binary_target.view(-1)
        binary_loss = torch.sqrt(torch.mean((binary_target - pred_seg) ** 2))

        total_loss = self.beta * rmse_loss + (1 - self.beta) * binary_loss
        return rmse_loss,binary_loss,total_loss
class two_branch_loss(nn.Module):
    def __init__(self, beta):
        super(two_branch_loss, self).__init__()
        self.last_activation = "both"
        self.beta = beta
        print('two_branch_loss intialized')

    # def forward(self,  pred_sup,pred_seg,target,binary_target):
    def forward(self, pred, target):
        pred_sup, pred_seg= pred[0],pred[1]

        # rmse
        pred_sup = pred_sup.view(-1)
        target_sup = target.view(-1)
        rmse_loss = torch.sqrt(torch.mean((target_sup - pred_sup) ** 2))
        # rmse_loss = nn.functional.mse_loss(pred_sup, target)
        # rmse_loss = torch.sqrt(rmse_loss)

        # jaccard
        binary_target = target.cuda()
        binary_target[binary_target > 0] = 1
        intersection = torch.sum(pred_seg * binary_target, (1, 2, 3))
        sum_pred = torch.sum(pred_seg, (1, 2, 3))
        sum_targets = torch.sum(binary_target, (1, 2, 3))
        jaccard_loss = - torch.mean(
            (intersection + SMOOTH) / (torch.sum(sum_targets) + torch.sum(sum_pred) + intersection + SMOOTH))

        total_loss = self.beta * rmse_loss + (1 - self.beta) * jaccard_loss
        return rmse_loss,jaccard_loss,total_loss

# Global MSE
class GMSE(nn.Module):
    def __init__(self, beta):
        self.last_activation = "relu"
        super(GMSE, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        rmse = torch.sqrt(torch.mean((targets - pred) ** 2))
        return rmse

# Global MSE
class GLMSE(nn.Module):
    def __init__(self, beta):
        self.last_activation = "relu"
        self.beta = beta
        super(GLMSE, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        global_rmse = torch.sqrt(torch.mean((targets - pred) ** 2))

        a = (targets - pred) ** 2
        a = a * targets
        local_rmse = torch.sqrt(torch.sum(a) / torch.sum(targets))
        total_loss =  self.beta * local_rmse +  (1 - self.beta) * global_rmse
        return local_rmse,global_rmse,total_loss

class IMSE2(nn.Module):
    def __init__(self, beta):
        super(IMSE2, self).__init__()
        self.last_activation = "sigmoid"



    def forward(self, pred, targets):
        intersection = (pred * targets).sum((1, 2, 3))
        total = (pred + targets).sum((1, 2, 3))
        union = total - intersection
        unionS = union.sum()
        rmse = torch.sqrt(torch.sum((targets - pred) ** 2) / unionS)
        return rmse

class MSE(nn.Module):
    def __init__(self, beta):
        super(MSE, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        rmse = torch.sqrt(torch.mean((targets - pred) ** 2))
        return rmse


# local MSE
class LMSE(nn.Module):
    def __init__(self, beta):
        super(LMSE, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        a = (targets - pred) ** 2
        # a[targets == 0] = 0
        a = a * targets
        rmse = torch.sqrt(torch.sum(a) / torch.sum(targets))
        # rmse = torch.sqrt(torch.sum(a) / torch.count_nonzero(targets))
        return rmse





# MSE based on A union B
class MSEunion(nn.Module):
    def __init__(self, beta):
        super(MSEunion, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        total = pred + targets
        diff = (targets - pred) ** 2
        diff[total == 0] = 0
        rmse = torch.sqrt(torch.sum(diff) / torch.count_nonzero(total))
        # rmse = torch.sqrt(torch.mean((targets - pred) ** 2))
        return rmse


# MSE based on A intersection B
class MSEintersection(nn.Module):
    def __init__(self, beta):
        super(MSEintersection, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        intersection = pred * targets
        diff = (targets - pred) ** 2
        diff[intersection == 0] = 0
        rmse = torch.sqrt(torch.sum(diff) / torch.count_nonzero(intersection))
        # rmse = torch.sqrt(torch.mean((targets - pred) ** 2))
        return rmse


# MSE based on IOU
class MSEiou(nn.Module):
    def __init__(self, beta):
        super(MSEiou, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        intersection = pred * targets
        total = pred + targets
        diff = (targets - pred) ** 2
        # try later diff * pred
        diff = diff * targets
        # diff[intersection == 0] = 0
        # rmse = torch.sqrt(torch.mean(diff)/torch.sum(total))
        rmse = torch.sqrt(torch.sum(diff) / torch.count_nonzero(total))
        # rmse = torch.sqrt(torch.mean((targets - pred) ** 2))
        return rmse


# MSE based on IOU
class MSEiou2(nn.Module):
    def __init__(self, beta):
        super(MSEiou2, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        intersection = pred * targets
        total = pred + targets
        diff = (targets - pred) ** 2
        diff = diff * targets
        rmse = torch.sqrt(torch.sum(diff) / torch.sum(intersection))
        return rmse


# IOU number
class IOU_number(nn.Module):
    def __init__(self, beta):
        super(IOU_number, self).__init__()

    def forward(self, pred, targets):
        # pred = pred.view(-1)
        # targets = targets.view(-1)
        inter = pred * targets
        tot = pred + targets
        # inter[inter > 0] = 1
        # tot[tot > 0] = 1
        intersection = torch.sum(inter, (1, 2, 3))
        total = torch.sum(tot, (1, 2, 3))
        # intersection = torch.sum(inter)
        # total = torch.sum(tot)
        union = total - intersection
        IOU = torch.mean((intersection + SMOOTH) / (union + SMOOTH))
        # IOU = (intersection + SMOOTH) / (union + SMOOTH)

        return 1 - IOU


# IOU number
class IOU_nonBinary(nn.Module):
    def __init__(self, beta):
        super(IOU_nonBinary, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        intersection = pred * targets
        total = pred + targets
        diffI = (targets - pred) ** 2
        diffU = (targets - pred) ** 2
        diffI[intersection == 0] = 0
        diffU[total == 0] = 0
        IOU = torch.sum(diffI) / torch.sum(diffU)

        return IOU


class IMSE(nn.Module):
    def __init__(self, beta):
        super(IMSE, self).__init__()

    def forward(self, pred, targets):
        pred = pred.view(-1)
        targets = targets.view(-1)
        intersection = (pred * targets).sum()
        total = (pred + targets).sum()
        union = total - intersection
        rmse = torch.sqrt(torch.sum((targets - pred) ** 2) / intersection)
        return rmse





class IoULoss(nn.Module):
    def __init__(self, beta=0.5):
        self.beta = beta
        super(IoULoss, self).__init__()

    def forward(self, pred, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # pred = F.sigmoid(pred)
        SMOOTH = 1e-6
        # union is the mutually inclusive area of all labels & predictions

        intersection = (pred * targets)
        intersection = intersection.sum((1, 2, 3))
        total = (pred + targets)
        total = total.sum((1, 2, 3))
        union = total - intersection
        unionS = union.sum()
        IoU = torch.mean((intersection + SMOOTH) / (union + SMOOTH))

        beta = self.beta
        return (1 - beta) * (1 - IoU) + beta * (torch.sqrt(torch.sum((targets - pred) ** 2) / unionS))
        # return (1 - beta) * (1 - IoU) + beta * (torch.sqrt(torch.mean((targets - pred) ** 2)))


class IoULoss2(nn.Module):
    def __init__(self, beta=0.5):
        self.beta = beta
        super(IoULoss2, self).__init__()

    def forward(self, pred, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # pred = F.sigmoid(pred)
        SMOOTH = 1e-6
        # union is the mutually inclusive area of all labels & predictions

        intersection = (pred * targets)
        intersection = intersection.sum((1, 2, 3))
        total = (pred + targets)
        total = total.sum((1, 2, 3))
        union = total - intersection
        unionS = union.sum()
        IoU = torch.mean((intersection + SMOOTH) / (union + SMOOTH))

        beta = self.beta
        return (1 - beta) * (1 - IoU) + beta * (torch.sqrt(torch.sum((targets - pred) ** 2) / unionS))


# MSE based on A intersection B
class MSENew(nn.Module):
    def __init__(self, beta):
        super(MSENew, self).__init__()

    def forward(self, pred, targets):
        SMOOTH = 0.006
        pred = pred.view(-1)
        targets = targets.view(-1)
        intersection = pred * targets
        diff1 = (targets - pred) ** 2
        diff1[intersection == 0] = 0
        rmseI = torch.sqrt(torch.sum(diff1) / torch.count_nonzero(intersection))

        total = pred + targets
        diff2 = (targets - pred) ** 2
        diff2[total == 0] = 0
        rmseU = torch.sqrt(torch.sum(diff2) / torch.count_nonzero(total))

        # loss = ((SMOOTH + rmseI)(SMOOTH + rmseU)(SMOOTH + rmseU - rmseI))
        # # /(intersection + SMOOTH)

        loss = ((SMOOTH + rmseI)(SMOOTH + rmseU)(SMOOTH + rmseU - rmseI))
        # /(intersection + SMOOTH)

        # rmse = torch.sqrt(torch.mean((targets - pred) ** 2))
        return loss
