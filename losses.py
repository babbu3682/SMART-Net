import torch
from torch import nn
from utils.CLS_SEG_REC_structure.losses import DiceLoss


class Dice_BCE_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function_1 = DiceLoss(mode='binary', from_logits=True)
        self.loss_function_2 = torch.nn.BCEWithLogitsLoss()
        self.dice_weight     = 1.0   
        self.bce_weight      = 1.0   

    def forward(self, y_pred, y_true):
        dice_loss  = self.loss_function_1(y_pred, y_true)
        bce_loss   = self.loss_function_2(y_pred, y_true)

        return self.dice_weight*dice_loss + self.bce_weight*bce_loss

class Consistency_16x16_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.L2loss  = torch.nn.MSELoss()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=16, stride=16, padding=0)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=16, stride=16, padding=0)

    def forward(self, y_cls, y_seg):
        # if valid, y_cls.shape --> ([36, 1]) // y_seg.shape --> ([1, 1, 256, 256, 36])
        y_cls = torch.sigmoid(y_cls)
        y_seg = torch.sigmoid(y_seg)

        if y_seg.dim() == 5:
            y_seg = y_seg.transpose(0, 4).squeeze(4)
            y_seg = self.avgpool(self.maxpool(y_seg)).flatten(start_dim=1, end_dim=-1)
            loss  = self.L2loss(y_seg, y_cls)
        
        else : 
            y_seg = self.avgpool(self.maxpool(y_seg)).flatten(start_dim=1, end_dim=-1)
            loss  = self.L2loss(y_seg, y_cls)

        return loss


######################################################################################################################################################################
######################################################                    Uptask Loss                         ########################################################
######################################################################################################################################################################

class Uptask_Loss(torch.nn.Module):
    def __init__(self, mode='mtl', uncert=False):
        super().__init__()
        self.loss_cls     = torch.nn.BCEWithLogitsLoss()
        self.loss_seg     = Dice_BCE_Loss()
        self.loss_recon   = torch.nn.L1Loss()
        self.loss_consist  = Consistency_16x16_Loss()
        
        self.mode   = mode
        self.uncert = uncert

        if uncert:
            self.cls_weight       = nn.Parameter(torch.tensor([1.0], requires_grad=True, device='cuda'))
            self.seg_weight       = nn.Parameter(torch.tensor([1.0], requires_grad=True, device='cuda'))
            self.consist_weight   = nn.Parameter(torch.tensor([1.0], requires_grad=True, device='cuda'))
            # self.cls_weight.to('cuda')
            # self.seg_weight.to('cuda')
            # self.consist_weight.to('cuda')

        else:
            self.cls_weight       = 1.0
            self.seg_weight       = 1.0
            self.consist_weight   = 1.0
            self.recon_weight     = 1.0

        
    def cal_weighted_loss(self, loss, uncert_w):
        return torch.exp(-uncert_w)*loss + 0.5*uncert_w

    def forward(self, cls_pred=None, seg_pred=None, rec_pred=None, cls_gt=None, seg_gt=None, rec_gt=None):
        if self.mode == 'cls+seg+rec+consist':
            loss_cls     = self.loss_cls(cls_pred, cls_gt)
            loss_seg     = self.loss_seg(seg_pred, seg_gt)
            loss_recon   = self.loss_recon(rec_pred, rec_gt)
            total = self.cls_weight*loss_cls + self.seg_weight*loss_seg + self.recon_weight*loss_recon
            return total, {'Cls_Loss':(self.cls_weight*loss_cls).item(), 'Seg_Loss':(self.seg_weight*loss_seg).item(), 'Recon_Loss':(self.recon_weight*loss_recon).item()}
        
        else: 
            print("Please Select Loss mode...! [only_cls, only_seg, mtl, mtl+consist, mtl+consist+recon]")

######################################################################################################################################################################
######################################################                    Downtask Loss                       ########################################################
######################################################################################################################################################################

class Downtask_Loss(torch.nn.Module):
    def __init__(self, mode='3d_cls'):
        super().__init__()
        self.loss_cls     = torch.nn.BCEWithLogitsLoss()
        self.loss_seg     = Dice_BCE_Loss()
        self.cls_weight   = 1.0
        self.seg_weight   = 1.0

        self.mode = mode

    def forward(self, cls_pred=None, seg_pred=None, cls_gt=None, seg_gt=None):
        if self.mode == '3d_cls' or self.mode == 'other_cls':
            loss_cls = self.loss_cls(cls_pred, cls_gt)
            return self.cls_weight*loss_cls, None
        
        elif self.mode == '3d_seg' or self.mode == 'other_seg':
            loss_seg = self.loss_seg(seg_pred, seg_gt)
            return self.seg_weight*loss_seg, None

        else :
            print("Please Select Loss mode...! [3d_cls, 3d_seg]")            