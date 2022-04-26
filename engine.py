import math
import utils
import numpy as np
import torch
import cv2

from metrics import *





def freeze_params(model: torch.nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def unfreeze_params(model: torch.nn.Module):
    """Set requires_grad=True for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = True

def predict(self, x):
    """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

    Args:
        x: 4D torch tensor with shape (batch_size, channels, height, width)

    Return:
        prediction: 4D torch tensor with shape (batch_size, classes, height, width)

    """
    if self.training:
        self.eval()

    with torch.no_grad():
        x = self.forward(x)

    return x


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

features   = {}
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


def Activation_Map(x):
    mean = torch.mean(x, dim=1)
    mean = torch.sigmoid(mean).squeeze().cpu().detach().numpy()
    mean = np.stack([ cv2.resize(i, (256, 256), interpolation=cv2.INTER_CUBIC) for i in mean ], axis=0)
    return mean


########################################################
# Uptask Task
    # SMART_Net
def train_Up_SMART_Net(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("image == ", batch_data['image_meta_dict']['filename_or_obj'])
            print("label == ", batch_data['label_meta_dict']['filename_or_obj'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


# For Highly Imbalanced Dataset, We have to get a batch that has a balanced class distribution. However, It is slow training...
def train_Up_Imbalance_SMART_Net(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for _ in metric_logger.log_every(range(len(data_loader[0])), print_freq, header):
        
        batch_data_pos = next(iter(data_loader[0]))
        batch_data_neg = next(iter(data_loader[1]))
        
        inputs  = torch.cat([batch_data_pos["image"], batch_data_neg["image"]], dim=0).squeeze(4).to(device)  # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = torch.cat([batch_data_pos["label"], batch_data_neg["label"]], dim=0).squeeze(4).to(device)  # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def valid_Up_SMART_Net(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        # Metric CLS
        auc            = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!

        # Metrics SEG
        result_dice    = dice_metric(y_pred=seg_pred.round(), y=seg_gt)              # pred_seg must be round() !! 

        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    dice               = dice_metric.aggregate().item()    
    
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    metric_logger.update(dice=dice)
    
    auc_metric.reset()
    confuse_metric.reset()
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
    
@torch.no_grad()
def test_Up_SMART_Net(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        # Metric CLS
        auc              = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix   = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!

        # Metrics SEG
        result_dice      = dice_metric(y_pred=seg_pred.round(), y=seg_gt)              # pred_seg must be round() !! 
  
        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    dice               = dice_metric.aggregate().item()    
    
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    metric_logger.update(dice=dice)

    auc_metric.reset()
    confuse_metric.reset()
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
     
@torch.no_grad()
def test_Up_SMART_Net_Patient_Level(model, criterion, data_loader, device, print_freq, batch_size):
    # 3d patient-level evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'
    fpv = 0

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)      # (B, C, H, W, D) 
        seg_gt  = batch_data["label"]                 # (B, C, H, W, D)
        cls_gt  = torch.stack([ seg_gt[..., i].flatten(1).bool().any(dim=1, keepdim=True).float() for i in range(seg_gt.shape[-1]) ], dim=0).squeeze(2) #  ---> (B, 1)

        cls_pred  = torch.stack([ model(inputs[..., i])[0].detach().cpu() for i in range(inputs.shape[-1]) ], dim = 0).flatten(1)
        seg_pred  = torch.stack([ model(inputs[..., i])[1].detach().cpu() for i in range(inputs.shape[-1]) ], dim = -1)
        # rec_pred  = torch.stack([ model(inputs[..., i])[2].detach().cpu() for i in range(inputs.shape[-1]) ], dim = -1)    
  
        # loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, cls_gt=cls_gt, seg_gt=seg_gt)
        # loss_value = loss.item()

        # if not math.isfinite(loss_value):
            # print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        # metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        # if loss_detail is not None:
            # metric_logger.update(**loss_detail)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        # Metric CLS
        auc             = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix  = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!

        # Metrics SEG
        result_dice     = dice_metric(y_pred=seg_pred.round(), y=seg_gt)      # pred_seg must be round() !! 
        if seg_gt.max()!=1:
            fpv += seg_pred.round().sum()
  
        # # Metrics REC
        # mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        # metric_logger.update(mae=mae)

    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    dice               = dice_metric.aggregate().item()    
    
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    metric_logger.update(dice=dice, fpv=fpv)

    auc_metric.reset()
    confuse_metric.reset()
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


    # Dual
        # CLS+SEG
def train_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred = model(inputs)
        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, cls_gt=cls_gt, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, cls_gt=cls_gt, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!


        # Metrics SEG
        result_dice = dice_metric(y_pred=seg_pred.round(), y=seg_gt)              # pred_seg must be round() !! 

    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    dice               = dice_metric.aggregate().item()    
    
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    metric_logger.update(dice=dice)  

    auc_metric.reset()
    confuse_metric.reset()
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
    
@torch.no_grad()
def test_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, cls_gt=cls_gt, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!


        # Metrics SEG
        result_dice = dice_metric(y_pred=seg_pred.round(), y=seg_gt)              # pred_seg must be round() !! 

    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    dice               = dice_metric.aggregate().item()    
    
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    metric_logger.update(dice=dice)
    
    auc_metric.reset()
    confuse_metric.reset()
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
    

        # CLS+REC
def train_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, rec_pred=rec_pred, cls_gt=cls_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, rec_pred=rec_pred, cls_gt=cls_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)

        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!
  
        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
    
@torch.no_grad()
def test_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred, seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, rec_pred=rec_pred, cls_gt=cls_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)

        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!

        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


        # SEG+REC
def train_Up_SMART_Net_Dual_SEG_REC(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

        seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, rec_pred=rec_pred, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Up_SMART_Net_Dual_SEG_REC(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

        seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, rec_pred=rec_pred, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)

        # Metrics SEG
        result_dice = dice_metric(y_pred=seg_pred.round(), y=seg_gt)              # pred_seg must be round() !! 
  
        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    # Aggregatation
    dice               = dice_metric.aggregate().item()    
    metric_logger.update(dice=dice)
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
    
@torch.no_grad()
def test_Up_SMART_Net_Dual_SEG_REC(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

        seg_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, rec_pred=rec_pred, seg_gt=seg_gt, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)

        # Metrics SEG
        dice = dice_metric(y_pred=seg_pred.round(), y=seg_gt)              # pred_seg must be round() !!
        metric_logger.update(dice=dice)
  
        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    # Aggregatation
    dice               = dice_metric.aggregate().item()    
    metric_logger.update(dice=dice)
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


    # Single
        # CLS
def train_Up_SMART_Net_Single_CLS(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Up_SMART_Net_Single_CLS(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)

        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!

    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
    
@torch.no_grad()
def test_Up_SMART_Net_Single_CLS(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1, keepdim=True).float() #  ---> (B, 1)

        cls_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)

        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!

    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
     
        # SEG
def train_Up_SMART_Net_Single_SEG(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

        seg_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Up_SMART_Net_Single_SEG(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

        seg_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)

        # Metrics SEG
        result_dice = dice_metric(y_pred=seg_pred.round(), y=seg_gt)              # pred_seg must be round() !! 

    # Aggregatation
    dice               = dice_metric.aggregate().item()    
    metric_logger.update(dice=dice)
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
    
@torch.no_grad()
def test_Up_SMART_Net_Single_SEG(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

        seg_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)

        # Metrics SEG
        result_dice = dice_metric(y_pred=seg_pred.round(), y=seg_gt)              # pred_seg must be round() !! 

    # Aggregatation
    dice               = dice_metric.aggregate().item()    
    metric_logger.update(dice=dice)
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

        # REC        
def train_Up_SMART_Net_Single_REC(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

        rec_pred = model(inputs)

        loss, loss_detail = criterion(rec_pred=rec_pred, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Up_SMART_Net_Single_REC(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

        rec_pred = model(inputs)

        loss, loss_detail = criterion(rec_pred=rec_pred, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}
    
@torch.no_grad()
def test_Up_SMART_Net_Single_REC(model, criterion, data_loader, device, print_freq, batch_size):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

        rec_pred = model(inputs)

        loss, loss_detail = criterion(rec_pred=rec_pred, rec_gt=inputs)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
  
        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=rec_pred, target=inputs).item()
        metric_logger.update(mae=mae)

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}


########################################################
# Down Task
    # CLS
def train_Down_SMART_Net_CLS(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size, gradual_unfreeze):
    # 3d patient-level based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    if gradual_unfreeze:
        # Gradual Unfreezing
        # 10 epoch 씩 one stage block 풀기, 100 epoch까지는 아예 고정
        if epoch >= 0 and epoch <= 100:
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            print("Freeze encoder ...!")
        elif epoch >= 101 and epoch < 111:
            print("Unfreeze encoder.layer4 ...!")
            unfreeze_params(model.module.encoder.layer4) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer4)
        elif epoch >= 111 and epoch < 121:
            print("Unfreeze encoder.layer3 ...!")
            unfreeze_params(model.module.encoder.layer3) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer3)
        elif epoch >= 121 and epoch < 131:
            print("Unfreeze encoder.layer2 ...!")
            unfreeze_params(model.module.encoder.layer2) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer2)
        elif epoch >= 131 and epoch < 141:
            print("Unfreeze encoder.layer1 ...!")
            unfreeze_params(model.module.encoder.layer1) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer1)
        else :
            print("Unfreeze encoder.stem ...!")
            unfreeze_params(model.module.encoder) if hasattr(model, 'module') else unfreeze_params(model.encoder)
    else :
        print("Freeze encoder ...!")
        freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                                        # (B, C, H, W, D)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1, keepdim=True).float().to(device)     #  ---> (B, 1)
        depths  = batch_data["depths"]                                                                  #  ---> (B, 1) Fix bug, change cpu()

        cls_pred = model(inputs, depths)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Down_SMART_Net_CLS(model, criterion, data_loader, device, print_freq, batch_size):
    # 3d patient-level based evaluate ...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                                        # (B, C, H, W, D)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1, keepdim=True).float().to(device)     #  ---> (B, 1)
        depths  = batch_data["depths"]                                                                  #  ---> (B, 1) Fix bug, change cpu()

        cls_pred = model(inputs, depths)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Post-processing
        cls_pred = torch.sigmoid(cls_pred)

        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!


    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate()
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_Down_SMART_Net_CLS(model, criterion, data_loader, device, print_freq, batch_size):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'TEST:'
    cls_list = []
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                                        # (B, C, H, W, D)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1, keepdim=True).float().to(device)     #  ---> (B, 1)
        depths  = batch_data["depths"]                                                                  #  ---> (B, 1) Fix bug, change cpu()
        

        cls_pred = model(inputs, depths)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Post-processing
        cls_pred = torch.sigmoid(cls_pred)
        cls_list.append(cls_pred)
        # Metric CLS
        auc                 = auc_metric(y_pred=cls_pred, y=cls_gt)
        confuse_matrix      = confuse_metric(y_pred=cls_pred.round(), y=cls_gt)   # pred_cls must be round() !!


    # Aggregatation
    auc                = auc_metric.aggregate()
    f1, acc, sen, spe  = confuse_metric.aggregate() 
    metric_logger.update(auc=auc, f1=f1, acc=acc, sen=sen, spe=spe)          
    
    auc_metric.reset()
    confuse_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

    # SEG
def train_Down_SMART_Net_SEG(model, criterion, data_loader, optimizer, device, epoch, print_freq, batch_size, gradual_unfreeze):
    # 3d patient-level based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if gradual_unfreeze:
        # Gradual Unfreezing
        # 10 epoch 씩 one stage block 풀기, 100 epoch까지는 아예 고정
        if epoch >= 0 and epoch <= 100:
            freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)
            print("Freeze encoder ...!")
        elif epoch >= 101 and epoch < 111:
            print("Unfreeze encoder.layer4 ...!")
            unfreeze_params(model.module.encoder.layer4) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer4)
        elif epoch >= 111 and epoch < 121:
            print("Unfreeze encoder.layer3 ...!")
            unfreeze_params(model.module.encoder.layer3) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer3)
        elif epoch >= 121 and epoch < 131:
            print("Unfreeze encoder.layer2 ...!")
            unfreeze_params(model.module.encoder.layer2) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer2)
        elif epoch >= 131 and epoch < 141:
            print("Unfreeze encoder.layer1 ...!")
            unfreeze_params(model.module.encoder.layer1) if hasattr(model, 'module') else unfreeze_params(model.encoder.layer1)
        else :
            print("Unfreeze encoder.stem ...!")
            unfreeze_params(model.module.encoder) if hasattr(model, 'module') else unfreeze_params(model.encoder)
    else :
        print("Freeze encoder ...!")
        freeze_params(model.module.encoder) if hasattr(model, 'module') else freeze_params(model.encoder)

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                 # (B, C, H, W, D)
        seg_gt  = batch_data["label"].to(device)                 # (B, C, H, W, D)

        seg_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
 

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Down_SMART_Net_SEG(model, criterion, data_loader, device, print_freq, batch_size):
    # 3d patient-level based evaluate ...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                   # (B, C, H, W, D)
        seg_gt  = batch_data["label"].to(device)                                   # (B, C, H, W, D)

        seg_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)

        # Metrics SEG
        result_dice = dice_metric(y_pred=seg_pred.round(), y=seg_gt)              # pred_seg must be round() !! 
        dice        = dice_metric.aggregate().item()              
        metric_logger.update(dice=dice)

    # Aggregatation
    dice               = dice_metric.aggregate().item()    
    metric_logger.update(dice=dice)
    dice_metric.reset()

    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_Down_SMART_Net_SEG(model, criterion, data_loader, device, print_freq, batch_size):
    # 3d patient-level based evaluate ...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=batch_size)
    header = 'Valid:'

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)                                   # (B, C, H, W, D)
        seg_gt  = batch_data["label"].to(device)                                   # (B, C, H, W, D)

        seg_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)

        # Metrics SEG
        result_dice = dice_metric(y_pred=seg_pred.round(), y=seg_gt)              # pred_seg must be round() !! 

    # Aggregatation
    dice               = dice_metric.aggregate().item()           
    metric_logger.update(dice=dice)
    dice_metric.reset()
    
    return {k: round(meter.global_avg, 7) for k, meter in metric_logger.meters.items()}



########################################################
## Inference code 
from monai.transforms import SaveImage
from monai.transforms import Resize

    # SMAET-Net
@torch.no_grad()
def infer_Up_SMART_Net(model, data_loader, device, print_freq, save_dir):
    # 2d slice-wise based evaluate...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'TEST:'
    
    save_dict = dict()
    img_path_list = []
    img_list = []
    cls_list = []
    seg_list = []
    rec_list = []
    act_list = []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

        model.encoder.layer4.register_forward_hook(get_activation('Activation Map')) # for Activation Map

        cls_pred, seg_pred, rec_pred = model(inputs)

        # post-processing
        cls_pred = torch.sigmoid(cls_pred)
        seg_pred = torch.sigmoid(seg_pred)

        img_path_list.append(batch_data["image_path"][0])
        img_list.append(inputs.detach().cpu().squeeze())
        cls_list.append(cls_pred.detach().cpu().squeeze())
        seg_list.append(seg_pred.detach().cpu().squeeze())
        rec_list.append(rec_pred.detach().cpu().squeeze())
        act_list.append(Activation_Map(activation['Activation Map']))


    save_dict['img_path_list']  = img_path_list
    save_dict['img_list']       = img_list
    save_dict['cls_pred']       = cls_list
    save_dict['seg_pred']       = seg_list
    save_dict['rec_pred']       = rec_list
    save_dict['activation_map'] = act_list
    np.savez(save_dir + '/result.npz', result=save_dict) 

    # CLS
@torch.no_grad()
def infer_Down_SMART_Net_CLS(model, data_loader, device, print_freq, save_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'TEST:'

    save_dict = dict()
    img_path_list = []
    img_list  = []
    cls_list  = []
    feat_list = []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        
        inputs  = batch_data["image"].to(device)                                                        # (B, C, H, W, D)
        depths  = batch_data["depths"]                                                                  #  ---> (B, 1) Fix bug, change cpu()

        model.fc.register_forward_hook(get_features('feat')) # for Representation
        
        cls_pred = model(inputs, depths)

        # Post-processing
        cls_pred = torch.sigmoid(cls_pred)

        img_path_list.append(batch_data["image_path"][0])
        img_list.append(inputs.detach().cpu().squeeze())
        cls_list.append(cls_pred.detach().cpu().squeeze())
        feat_list.append(features['feat'].detach().cpu().numpy())


    save_dict['img_path_list']  = img_path_list
    save_dict['img_list']       = img_list
    save_dict['cls_pred']       = cls_list
    save_dict['feat']           = feat_list
    np.savez(save_dir + '/result.npz', result=save_dict) 


    # SEG
@torch.no_grad()
def infer_Down_SMART_Net_SEG(model, data_loader, device, print_freq, save_dir):
    # 3d patient-level based evaluate ...! 
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", n=1)
    header = 'TEST:'

    image_saver = SaveImage(output_dir=save_dir, 
                    output_postfix='Image', 
                    output_ext='.nii.gz', 
                    resample=True, 
                    mode='bilinear', 
                    squeeze_end_dims=True, 
                    data_root_dir='', 
                    separate_folder=False, 
                    print_log=True)

    label_saver = SaveImage(output_dir=save_dir, 
                    output_postfix='Pred', 
                    output_ext='.nii.gz', 
                    resample=True, 
                    mode='nearest', 
                    squeeze_end_dims=True, 
                    data_root_dir='', 
                    separate_folder=False, 
                    print_log=True)



    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)  # (B, C, H, W, D)

        seg_pred = model(inputs)

        # post-processing
        seg_pred = torch.sigmoid(seg_pred)

        # resize 512 x 512
        inputs   = Resize(spatial_size=(512, 512, inputs.shape[-1]), mode='trilinear', align_corners=True)(inputs.detach().cpu().numpy().squeeze(0)) # Input = (C, H, W, D)
        seg_pred = Resize(spatial_size=(512, 512, seg_pred.shape[-1]), mode='nearest', align_corners=None)(seg_pred.detach().cpu().numpy().squeeze(0).round()) # Input = (C, H, W, D)

        # Save nii        
        image_save_dict = batch_data['image_meta_dict'][0]
        label_save_dict = batch_data['label_meta_dict'][0]
        
        image_saver(inputs, image_save_dict)    # Note: image should be channel-first shape: [C,H,W,[D]].
        label_saver(seg_pred, label_save_dict)  # Note: image should be channel-first shape: [C,H,W,[D]].

