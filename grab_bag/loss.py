import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def bce_dice_loss(pred, target, metrics, bce_weight=50.0):
    # print(f"{pred.shape=}")
    # print(f"{target.shape=}")
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = torch.sigmoid(pred)

    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (100.0 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

# Not clear this stuff is used:

# class BCELoss(nn.Module):
#     def __init__(self):
#         super(BCELoss, self).__init__()
#         self.bceloss = nn.BCELoss()

#     def forward(self, pred, target):
#         size = pred.size(0)
#         pred_ = pred.view(size, -1)
#         target_ = target.view(size, -1)

#         return self.bceloss(pred_, target_)


# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()

#     def forward(self, pred, target):
#         smooth = 1
#         size = pred.size(0)

#         pred_ = pred.view(size, -1)
#         target_ = target.view(size, -1)
#         intersection = pred_ * target_
#         dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
#         dice_loss = 1 - dice_score.sum()/size

#         return dice_loss


# class BceDiceLoss(nn.Module):
#     def __init__(self, wb=1, wd=1):
#         super(BceDiceLoss, self).__init__()
#         self.bce = BCELoss()
#         self.dice = DiceLoss()
#         self.wb = wb
#         self.wd = wd

#     def forward(self, pred, target):
#         bceloss = self.bce(pred, target)
#         diceloss = self.dice(pred, target)

#         loss = self.wd * diceloss + self.wb * bceloss
#         return loss


# class GT_BceDiceLoss(nn.Module):
#     def __init__(self, wb=1, wd=1):
#         super(GT_BceDiceLoss, self).__init__()
#         self.bcedice = BceDiceLoss(wb, wd)

#     def forward(self, gt_pre, out, target):
#         bcediceloss = self.bcedice(out, target)
#         gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
#         print('gt_pre', gt_pre5.shape, gt_pre4.shape, gt_pre3.shape, gt_pre2.shape, gt_pre1.shape, 'target', target.shape)
#         gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
#         return bcediceloss + gt_loss
