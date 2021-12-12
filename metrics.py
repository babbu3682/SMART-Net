import torch
from torch import nn

import monai
from monai.metrics.utils import MetricReduction, do_metric_reduction
from monai.metrics import compute_meandice, DiceMetric, ConfusionMatrixMetric 

# from .fpn import FPN

# metric 
dice_metric    = DiceMetric(include_background=True, reduction="mean_batch")
confuse_metric = ConfusionMatrixMetric(include_background=True, compute_sample=False, reduction='mean')


######################################################################################################################################################################
######################################################                   Cls / Seg Metrics                    ########################################################
######################################################################################################################################################################

class Uptask_Loss(torch.nn.Module):
    def __init__(self, mode='mtl'):
        super().__init__()
        self.loss_cls     = torch.nn.BCEWithLogitsLoss()
        self.loss_seg     = Dice_BCE_Loss()
        self.loss_consist = Consistency_16x16_Loss()
        self.loss_recon   = torch.nn.MSELoss()
        
        self.cls_weight       = 1.0
        self.seg_weight       = 1.0
        self.consist_weight   = 1.0
        self.recon_weight     = 1.0

        self.mode = mode

    def forward(self, cls_pred=None, seg_pred=None, recon_pred=None, cls_gt=None, seg_gt=None, recon_gt=None):
        if self.mode == 'only_cls':
            loss_cls     = self.loss_cls(cls_pred, cls_gt)
            return self.cls_weight*loss_cls, None
        
        elif self.mode == 'only_seg':
            loss_seg     = self.loss_seg(seg_pred, seg_gt)
            return self.seg_weight*loss_seg, None
        
        elif self.mode == 'mtl':
            loss_cls     = self.loss_cls(cls_pred, cls_gt)
            loss_seg     = self.loss_seg(seg_pred, seg_gt)
            total = self.cls_weight*loss_cls + self.seg_weight*loss_seg
            return total, {'Cls_Loss':(self.cls_weight*loss_cls).item(), 'Seg_Loss':(self.seg_weight*loss_seg).item()}
        
        elif self.mode == 'mtl+consist':  # Ours
            loss_cls     = self.loss_cls(cls_pred, cls_gt)
            loss_seg     = self.loss_seg(seg_pred, seg_gt)
            loss_consist = self.loss_consist(cls_pred, seg_pred)
            total = self.cls_weight*loss_cls + self.seg_weight*loss_seg + self.consist_weight*loss_consist
            return total, {'Cls_Loss':(self.cls_weight*loss_cls).item(), 'Seg_Loss':(self.seg_weight*loss_seg).item(), 'Consist_Loss':(self.consist_weight*loss_consist).item()}

        elif self.mode == 'mtl+consist+recon':
            loss_cls     = self.loss_cls(cls_pred, cls_gt)
            loss_seg     = self.loss_seg(seg_pred, seg_gt)
            loss_consist = self.loss_consist(cls_pred, seg_pred)
            loss_recon   = self.loss_recon(cls_pred, seg_pred)
            total = self.cls_weight*loss_cls + self.seg_weight*loss_seg + self.consist_weight*loss_consist + self.recon_weight*loss_recon
            return total, {'Cls_Loss':(self.cls_weight*loss_cls).item(), 'Seg_Loss':(self.seg_weight*loss_seg).item(), 'Consist_Loss':(self.consist_weight*loss_consist).item(), 'Recon_Loss':(self.recon_weight*loss_recon).item()}

        else :
            print("Please Select Loss mode...! [only_cls, only_seg, mtl, mtl+consist, mtl+consist+recon]")

   