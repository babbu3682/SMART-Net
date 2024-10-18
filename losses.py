import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import functools

def binary_dice_loss(y_true, y_pred, smooth=0.0, eps=1e-7, return_score=False):
    bs = y_true.size(0)
    y_true = y_true.view(bs, 1, -1)
    y_pred = y_pred.view(bs, 1, -1)

    intersection = torch.sum(y_true * y_pred, dim=(0, 2))
    cardinality  = torch.sum(y_true + y_pred, dim=(0, 2))

    dice_score = (2. * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    dice_loss  = 1 - dice_score
    if return_score:
        return dice_score.mean()
    else:
        return dice_loss.mean()

def binary_focal_loss(y_pred, y_true, gamma=2.0, alpha=0.25):
    # alpha: Weight constant that penalize model for FNs (False Negatives)
    logpt  = F.binary_cross_entropy(y_pred, y_true, reduction="none")
    focal_term = (1.0 - torch.exp(-logpt)).pow(gamma)
    loss = focal_term * logpt
    loss *= alpha*y_true + (1-alpha)*(1-y_true)
    return loss.mean()

def binary_tversky_loss(y_pred, y_true, alpha=0.5, beta=0.5, smooth=0.0, eps=1e-7):
    # With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    # alpha: Weight constant that penalize model for FPs (False Positives)
    # beta: Weight constant that penalize model for FNs (False Negatives)    
    bs = y_true.size(0)
    y_true = y_true.view(bs, 1, -1)
    y_pred = y_pred.view(bs, 1, -1)

    intersection = torch.sum(y_true * y_pred, dim=(0, 2))
    fp = torch.sum(y_pred * (1.0 - y_true), dim=(0, 2))
    fn = torch.sum((1 - y_pred) * y_true, dim=(0, 2))

    tversky_score = (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)
    tversky_loss  = 1 - tversky_score
    return tversky_loss.mean()



class MTL_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_cls     = F.binary_cross_entropy
        self.loss_seg     = binary_dice_loss
        self.loss_rec     = F.l1_loss
        self.loss_consist = F.mse_loss

    def forward(self, pred_cls, pred_seg, pred_rec, label, mask, image, pooled_seg=None):
        assert pred_cls.size() == label.size(), f"{pred_cls.size()} != {label.size()}"
        assert pred_seg.size() == mask.size(),  f"{pred_seg.size()} != {mask.size()}"
        assert pred_rec.size() == image.size(), f"{pred_rec.size()} != {image.size()}"

        cls_loss  = self.loss_cls(input=pred_cls, target=label)
        seg_loss  = self.loss_seg(y_pred=pred_seg, y_true=mask)
        rec_loss  = self.loss_rec(input=pred_rec, target=image)

        total_loss = cls_loss + seg_loss + rec_loss
        loss_dict  = {"total_loss": total_loss.item(), "cls_loss": cls_loss.item(), "seg_loss": seg_loss.item(), "rec_loss": rec_loss.item()}

        if pooled_seg is not None:
            total_loss += self.loss_consist(pred_cls, pooled_seg)
            loss_dict["consist_loss"] = self.loss_consist(pred_cls, pooled_seg).item()
            return total_loss, loss_dict
        else:
            return total_loss, loss_dict


# 3D - 2D transfer
class CLS_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_cls = F.binary_cross_entropy
        # self.loss_cls_binary_focal = functools.partial(binary_focal_loss, gamma=0.0, alpha=0.7)

    def forward(self, pred_cls, label):
        # print(logit_cls.shape, logit_seg.shape, logit_det.shape, logit_rec.shape, logit_idx.shape, label.shape, mask.shape, bbox.shape, image.shape, idx.shape)
        assert pred_cls.size() == label.size(), f"{pred_cls.size()} != {label.size()}"
        
        cls_loss = self.loss_cls(input=pred_cls, target=label)
        # cls_binary_focal_loss  = self.loss_cls_binary_focal(y_pred=pred_cls, y_true=label)

        # return cls_bce_loss + cls_binary_focal_loss, {'cls_bce_loss': cls_bce_loss, 'cls_binary_focal_loss': cls_binary_focal_loss}
        total_loss = cls_loss
        loss_dict  = {"total_loss": total_loss.item(), "cls_bce_loss": cls_loss.item()}
        return total_loss, loss_dict
    
class SEG_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_seg_dice = binary_dice_loss
        self.loss_seg_bce  = F.binary_cross_entropy

    def forward(self, pred_seg, mask):
        assert pred_seg.size() == mask.size(), f"{pred_seg.size()} != {mask.size()}"
        seg_dice_loss = self.loss_seg_dice(y_pred=pred_seg, y_true=mask)
        seg_bce_loss  = self.loss_seg_bce(input=pred_seg, target=mask)

        return seg_dice_loss + seg_bce_loss, {"seg_dice_loss": seg_dice_loss, "seg_bce_loss": seg_bce_loss}




def get_loss(name):
    # 2D
    if name == 'MTL_Loss':
        return MTL_Loss()                   
    
    # elif name == 'MTL_CLS_SEG_REC_Loss_2':
    #     return MTL_CLS_SEG_REC_Loss_2()                       

    # 3D - 2D transfer
    elif name == 'CLS_Loss':
        return CLS_Loss()    

    elif name == 'SEG_Loss':
        return SEG_Loss()    

    else:
        raise NotImplementedError