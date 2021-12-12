import math
from pathlib import Path
from typing import Iterable, Optional

import utils
import torch

import numpy as np
import cv2

import monai
from monai.metrics.utils import MetricReduction, do_metric_reduction
from monai.metrics import compute_meandice, compute_roc_auc, DiceMetric, ConfusionMatrixMetric 
from sklearn.metrics import roc_auc_score, f1_score    

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


dice_metric    = DiceMetric(include_background=True, reduction="mean_batch")
confuse_metric = ConfusionMatrixMetric(include_background=True, compute_sample=False, reduction='mean')


######################################################                    Uptask Task                         ########################################################
def train_Uptask_CLS_SEG_REC(model, criterion, data_loader, optimizer, device, epoch):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].squeeze(4).to(device)     # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].squeeze(4).to(device)     # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = seg_gt.flatten(1).bool().any(dim=1).float().unsqueeze(1) #    ---> (B, 1)

        seg_pred, cls_pred, rec_pred = model(inputs)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt,  rec_gt=inputs)
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
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Evaluation code 
@torch.no_grad()
def valid_Uptask_CLS_SEG_REC(model, criterion, data_loader, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'

    # switch to evaluation mode
    model.eval()
    print_freq = 10
    metric_count = metric_sum = 0

    total_cls_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_cls_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   
        seg_gt  = batch_data["label"]              
        cls_gt  = torch.stack([ seg_gt[..., i].flatten(1).bool().any(dim=1).float() for i in range(seg_gt.shape[-1]) ], dim=0) #    ---> (B, 1)

        with torch.no_grad():
            seg_pred  = torch.stack([ model(inputs[..., i])[0].detach().cpu() for i in range(inputs.shape[-1]) ], dim = -1)
            cls_pred  = torch.stack([ model(inputs[..., i])[1].detach().cpu() for i in range(inputs.shape[-1]) ], dim = 0).flatten(1)
            rec_pred  = torch.stack([ model(inputs[..., i])[2].detach().cpu() for i in range(inputs.shape[-1]) ], dim = -1)


        total_cls_pred  = torch.cat([total_cls_pred,  torch.sigmoid(cls_pred)],   dim=0)
        total_cls_gt    = torch.cat([total_cls_gt,    cls_gt],     dim=0)

        loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=rec_pred, cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=inputs.detach().cpu())
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Metrics
        value, not_nans = dice_metric(y_pred=torch.sigmoid(seg_pred), y=seg_gt)
        metric_count   += not_nans
        metric_sum     += value * not_nans
  
    # Metric SEG
    Dice = (metric_sum / metric_count).item()
    
    # Metric CLS
    AUC = roc_auc_score(total_cls_gt, total_cls_pred)
    f1  = f1_score(y_true=total_cls_gt, y_pred=total_cls_pred.round())

    confuse_result      = confuse_metric(y_pred=total_cls_pred.round(), y=total_cls_gt)
    confusion_matrix, _ = do_metric_reduction(confuse_result, MetricReduction.SUM)

    TP = confusion_matrix[0].item()
    FP = confusion_matrix[1].item()
    TN = confusion_matrix[2].item()
    FN = confusion_matrix[3].item()
    
    Acc = (TP+TN) / (TP+FP+TN+FN)
    Sen = TP / (TP+FN)
    Spe = TN / (TN+FP)
    
    print('* Loss:{losses.global_avg:.3f} | F1:{F1:.3f} AUC:{AUC:.3f} Acc:{acc:.3f} Sen:{sen:.3f} Spe:{spe:.3f} | Dice:{dice:.3f} '.format(losses=metric_logger.loss, F1=f1, AUC=AUC, acc=Acc, sen=Sen, spe=Spe, dice=Dice))
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {'loss': metric_logger.loss.global_avg, 'F1':f1, 'AUC': AUC, 'Acc': Acc, 'Sen': Sen, 'Spe': Spe, 'Dice': Dice}

# test code 
@torch.no_grad()
def test_Uptask_CLS_SEG_REC(model, criterion, data_loader, device, test_name, save_path):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TEST:'

    # switch to evaluation mode
    model.eval()
    print_freq = 10
    metric_count = metric_sum = 0
    FalsePositive = 0

    total_cls_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_cls_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    total_seg_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_seg_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    # img_list      = []
    # mask_list     = []
    # label_list    = []
    # seg_prob_list = []
    # cls_prob_list = []
    # save_dict = dict()
    # activation_list = []
    # D_activation_list = []
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)     # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"]                # (B, C, H, W, 1) ---> (B, C, H, W)        
        cls_gt  = torch.stack([ seg_gt[..., i].flatten(1).bool().any(dim=1).float() for i in range(seg_gt.shape[-1]) ], dim=0) #    ---> (B, 1)
        
        # model.encoder.layer4.register_forward_hook(get_activation('Activation Map')) # for Activation Map
        # model.decoder.blocks[3].register_forward_hook(get_activation('Decoder')) # for Activation Map

        with torch.no_grad():
            seg_pred  = torch.stack([ model(inputs[..., i])[0].detach().cpu() for i in range(inputs.shape[-1]) ], dim = -1)
            cls_pred  = torch.stack([ model(inputs[..., i])[1].detach().cpu() for i in range(inputs.shape[-1]) ], dim = 0).flatten(1)

            seg_pred = torch.sigmoid(seg_pred)
            cls_pred = torch.sigmoid(cls_pred)

            if seg_gt.sum() == 0:
                FalsePositive += seg_pred.round().sum()

            # seg_pred = []
            # cls_pred = []
            # cam_pred = []
            # D_cam_pred = []
            # for i in range(inputs.shape[-1]):
            #     result = model(inputs[..., i])
            #     seg_pred.append(result[0].detach().cpu())
            #     cls_pred.append(result[1].detach().cpu())
            #     cam_pred.append(Activation_Map(activation['Activation Map']))
            #     D_cam_pred.append(Activation_Map(activation['Decoder']))

            # seg_pred = torch.stack(seg_pred, dim=-1)
            # cls_pred = torch.stack(cls_pred, dim=0).flatten(1)
            # cam_pred = np.stack(cam_pred, axis=-1)
            # D_cam_pred = np.stack(D_cam_pred, axis=-1)


        # total_cls_pred  = torch.cat([total_cls_pred,  torch.sigmoid(cls_pred)],   dim=0)
        # total_cls_gt    = torch.cat([total_cls_gt,    cls_gt],     dim=0)

        total_cls_pred  = torch.cat([total_cls_pred,  cls_pred],   dim=0)
        total_cls_gt    = torch.cat([total_cls_gt,    cls_gt],     dim=0)

        total_seg_pred  = torch.cat([total_seg_pred,  seg_pred.transpose(0, 4)],   dim=0)
        total_seg_gt    = torch.cat([total_seg_gt,    seg_gt.transpose(0, 4)],     dim=0)

        # loss, loss_detail = criterion(cls_pred=cls_pred, seg_pred=seg_pred, rec_pred=torch.ones(512,512), cls_gt=cls_gt, seg_gt=seg_gt, rec_gt=torch.ones(512,512))
        # loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))

        # # LOSS
        # metric_logger.update(loss=loss_value)  # 1 epoch의 배치들의 loss를 적립한뒤 epoch 끝나면 갯수 만큼 평균
        # if loss_detail is not None:
        #     metric_logger.update(**loss_detail)

        # Metrics
        value, not_nans = dice_metric(y_pred=seg_pred.round(), y=seg_gt)
        metric_count   += not_nans
        metric_sum     += value * not_nans

        # # save
        # img_list.append(inputs.cpu().detach().numpy())
        # mask_list.append(seg_gt)
        # label_list.append(cls_gt)
        # seg_prob_list.append(seg_pred)
        # cls_prob_list.append(cls_pred)
        # activation_list.append(cam_pred)
        # D_activation_list.append(D_cam_pred)

    # Metric SEG
    Dice = (metric_sum / metric_count).item()
    print("환자단위 dice = ", Dice)

    # Metrics
    value, not_nans = dice_metric(y_pred=total_seg_pred.squeeze(4).round(), y=total_seg_gt.squeeze(4))
    metric_count   += not_nans
    metric_sum     += value * not_nans
    print("슬라이스단위 dice = ", Dice)
    print("FP_seg = ", FalsePositive)

    # Metric CLS
    AUC = roc_auc_score(total_cls_gt, total_cls_pred)
    f1  = f1_score(y_true=total_cls_gt, y_pred=total_cls_pred.round())
    print("F1 score == ",  f1)   

    confuse_result      = confuse_metric(y_pred=total_cls_pred.round(), y=total_cls_gt)
    confusion_matrix, _ = do_metric_reduction(confuse_result, MetricReduction.SUM)

    TP = confusion_matrix[0].item()
    FP = confusion_matrix[1].item()
    TN = confusion_matrix[2].item()
    FN = confusion_matrix[3].item()
    
    Acc = (TP+TN) / (TP+FP+TN+FN)
    Sen = TP / (TP+FN)
    Spe = TN / (TN+FP)
    
    # # Save Prediction by using npz
    # save_dict['gt_img']       = img_list
    # save_dict['gt_mask']      = mask_list
    # save_dict['gt_label']     = label_list
    # save_dict['pred_mask']    = seg_prob_list
    # save_dict['pred_label']   = cls_prob_list
    # save_dict['activation']   = activation_list
    # save_dict['D_activation']   = D_activation_list

    # print("Saved npz...! => ", save_path + test_name + '.npz')

    # print("Saved npz...! => ", save_path + test_name + '_cls_3d[AUC_' + str(round(AUC, 3)) + '].npz')
    # np.savez(save_path + test_name + '.npz', x=save_dict) 


    print('* Loss:{losses.global_avg:.3f} | AUC:{AUC:.3f} Acc:{acc:.3f} Sen:{sen:.3f} Spe:{spe:.3f} | Dice:{dice:.3f} '.format(losses=metric_logger.loss, AUC=AUC, acc=Acc, sen=Sen, spe=Spe, dice=Dice))
    return {'loss': metric_logger.loss.global_avg, 'AUC': AUC, 'Acc': Acc, 'Sen': Sen, 'Spe': Spe, 'Dice': Dice}



######################################################                    Down Task                         ##########################################################
def train_Downtask_3dCls(model, criterion, data_loader, optimizer, device, epoch):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1).float().unsqueeze(1).to(device) #    ---> (B, 1)
        x_lens  = batch_data["z_shape"]            #    ---> (B, 1) 최근에 Bug 생김. cpu로 넣어줘야 함.
        # x_lens  = batch_data["z_shape"].to(device) #    ---> (B, 1)

        cls_pred = model(inputs, x_lens)

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
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def valid_Downtask_3dCls(model, criterion, data_loader, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'
    
    # switch to evaluation mode
    model.eval()
    print_freq = 10

    total_cls_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_cls_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt  = batch_data["label"].flatten(1).bool().any(dim=1).float().unsqueeze(1).to(device) #    ---> (B, 1)
        x_lens  = batch_data["z_shape"]            #    ---> (B, 1) 최근에 Bug 생김. cpu로 넣어줘야 함.
        # x_lens  = batch_data["z_shape"].to(device) #    ---> (B, 1)

        with torch.no_grad():
            cls_pred = model(inputs, x_lens)

        total_cls_pred  = torch.cat([total_cls_pred,  torch.sigmoid(cls_pred).detach().cpu()],   dim=0)
        total_cls_gt    = torch.cat([total_cls_gt,    cls_gt.detach().cpu()],     dim=0)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

            
    # Metric CLS
    AUC = roc_auc_score(total_cls_gt, total_cls_pred)
    f1  = f1_score(y_true=total_cls_gt, y_pred=total_cls_pred.round())
    print("F1 score == ",  f1)    
    confuse_result      = confuse_metric(y_pred=total_cls_pred.round(), y=total_cls_gt) 
    confusion_matrix, _ = do_metric_reduction(confuse_result, MetricReduction.SUM)

    TP = confusion_matrix[0].item()
    FP = confusion_matrix[1].item()
    TN = confusion_matrix[2].item()
    FN = confusion_matrix[3].item()
    
    Acc = (TP+TN) / (TP+FP+TN+FN)
    Sen = TP / (TP+FN)
    Spe = TN / (TN+FP)


    print('* Loss:{losses.global_avg:.3f} | AUC:{AUC:.3f} Acc:{acc:.3f} Sen:{sen:.3f} Spe:{spe:.3f} '.format(losses=metric_logger.loss, AUC=AUC, acc=Acc, sen=Sen, spe=Spe))
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {'loss': metric_logger.loss.global_avg, 'AUC':AUC, 'Acc': Acc, 'Sen': Sen, 'Spe': Spe}

@torch.no_grad()
def test_Downtask_3dCls(model, criterion, data_loader, device, test_name, save_path):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TEST:'
    
    # switch to evaluation mode
    model.eval()
    print_freq = 10

    # Save npz path 
    save_dict = dict()
    
    total_cls_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_cls_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    img_path_list  = []
    mask_path_list = []
    img_list       = []
    mask_list      = []
    label_list     = []
    cls_prob_list  = []
    feature_list   = []
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        img_path   = batch_data["img_path"][0]        # batch 1. so indexing [0]
        mask_path  = batch_data["mask_path"][0]       # batch 1. so indexing [0]
        inputs     = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt     = batch_data["label"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt     = batch_data["label"].flatten(1).bool().any(dim=1).float().unsqueeze(1).to(device) #    ---> (B, 1)
        x_lens     = batch_data["z_shape"]            #    ---> (B, 1) 최근에 Bug 생김. cpu로 넣어줘야 함.
        # x_lens     = batch_data["z_shape"].to(device) #    ---> (B, 1)

        with torch.no_grad():
            model.linear1.register_forward_hook(get_features('feat')) # for Representation
            cls_pred = model(inputs, x_lens)
            # print("체크 확인용", features['feat'].shape)  #torch.Size([1, 512]) 

            # Save
            img_path_list.append(img_path)
            mask_list.append(seg_gt.detach().cpu().numpy())
            mask_path_list.append(mask_path)
            img_list.append(inputs.detach().cpu().numpy())
            label_list.append(cls_gt.detach().cpu().numpy())            
            cls_prob_list.append(torch.sigmoid(cls_pred).detach().cpu().numpy())
            feature_list.append(features['feat'].detach().cpu().numpy())
            
        total_cls_pred  = torch.cat([total_cls_pred,  torch.sigmoid(cls_pred).detach().cpu()],   dim=0)
        total_cls_gt    = torch.cat([total_cls_gt,    cls_gt.detach().cpu()],     dim=0)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
    
    # Metric CLS
    AUC = roc_auc_score(total_cls_gt, total_cls_pred)
    confuse_result      = confuse_metric(y_pred=total_cls_pred.round(), y=total_cls_gt) 
    confusion_matrix, _ = do_metric_reduction(confuse_result, MetricReduction.SUM)
    f1  = f1_score(y_true=total_cls_gt, y_pred=total_cls_pred.round())
    print("F1 score == ",  f1)
    TP = confusion_matrix[0].item()
    FP = confusion_matrix[1].item()
    TN = confusion_matrix[2].item()
    FN = confusion_matrix[3].item()
    
    Acc = (TP+TN) / (TP+FP+TN+FN)
    Sen = TP / (TP+FN)
    Spe = TN / (TN+FP)

    # Save Prediction by using npz
    save_dict['gt_img_path']  = img_path_list
    save_dict['gt_mask_path'] = mask_path_list
    save_dict['gt_img']       = img_list
    save_dict['gt_mask']      = mask_list
    save_dict['gt_label']     = label_list
    save_dict['pred_label']   = cls_prob_list
    save_dict['feature']      = feature_list

    print("Saved npz...! => ", save_path + test_name + '_cls_3d[AUC_' + str(round(AUC, 3)) + '].npz')
    np.savez(save_path + test_name + '_cls_3d[AUC_' + str(round(AUC, 3)) + '].npz', cls_3d=save_dict) 

    print('* Loss:{losses.global_avg:.3f} | AUC:{AUC:.3f} Acc:{acc:.3f} Sen:{sen:.3f} Spe:{spe:.3f} '.format(losses=metric_logger.loss, AUC=AUC, acc=Acc, sen=Sen, spe=Spe))
    return {'loss': metric_logger.loss.global_avg, 'AUC':AUC, 'Acc': Acc, 'Sen': Sen, 'Spe': Spe}

# Inference code 
@torch.no_grad()
def inference_Downtask_3dCls(model, criterion, data_loader, device, test_name, save_path):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TEST:'
    
    # switch to evaluation mode
    model.eval()
    print_freq = 10

    # Save npz path 
    save_dict = dict()
    
    total_cls_pred  = torch.tensor([], dtype=torch.float32, device='cpu')
    total_cls_gt    = torch.tensor([], dtype=torch.float32, device='cpu')

    img_path_list  = []
    mask_path_list = []
    img_list       = []
    mask_list      = []
    label_list     = []
    cls_prob_list  = []
    feature_list   = []
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        img_path   = batch_data["img_path"][0]        # batch 1. so indexing [0]
        inputs     = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        cls_gt     = batch_data["label"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        x_lens     = batch_data["z_shape"]            #    ---> (B, 1) 최근에 Bug 생김. cpu로 넣어줘야 함.
        # x_lens     = batch_data["z_shape"].to(device) #    ---> (B, 1)

        with torch.no_grad():
            model.linear1.register_forward_hook(get_features('feat')) # for Representation
            cls_pred = model(inputs, x_lens)
            # print("체크 확인용", features['feat'].shape)  #torch.Size([1, 512]) 

            # Save
            img_path_list.append(img_path)
            img_list.append(inputs.detach().cpu().numpy())
            label_list.append(cls_gt.detach().cpu().numpy())            
            cls_prob_list.append(torch.sigmoid(cls_pred).detach().cpu().numpy())
            feature_list.append(features['feat'].detach().cpu().numpy())
            
        total_cls_pred  = torch.cat([total_cls_pred,  torch.sigmoid(cls_pred).detach().cpu()],   dim=0)
        total_cls_gt    = torch.cat([total_cls_gt,    cls_gt.detach().cpu()],     dim=0)

        loss, loss_detail = criterion(cls_pred=cls_pred, cls_gt=cls_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)
    
    # Metric CLS
    AUC = roc_auc_score(total_cls_gt, total_cls_pred)
    confuse_result      = confuse_metric(y_pred=total_cls_pred.round(), y=total_cls_gt) 
    confusion_matrix, _ = do_metric_reduction(confuse_result, MetricReduction.SUM)
    f1  = f1_score(y_true=total_cls_gt, y_pred=total_cls_pred.round())
    print("F1 score == ",  f1)
    TP = confusion_matrix[0].item()
    FP = confusion_matrix[1].item()
    TN = confusion_matrix[2].item()
    FN = confusion_matrix[3].item()
    
    Acc = (TP+TN) / (TP+FP+TN+FN)
    Sen = TP / (TP+FN)
    Spe = TN / (TN+FP)

    # Save Prediction by using npz
    save_dict['gt_img_path']  = img_path_list
    save_dict['gt_img']       = img_list
    save_dict['gt_label']     = label_list
    save_dict['pred_label']   = cls_prob_list
    save_dict['feature']      = feature_list

    print("Saved npz...! => ", save_path + test_name + '_cls_3d[AUC_' + str(round(AUC, 3)) + '].npz')
    np.savez(save_path + test_name + '_cls_3d[AUC_' + str(round(AUC, 3)) + '].npz', cls_3d=save_dict) 

    print('* Loss:{losses.global_avg:.3f} | AUC:{AUC:.3f} Acc:{acc:.3f} Sen:{sen:.3f} Spe:{spe:.3f} '.format(losses=metric_logger.loss, AUC=AUC, acc=Acc, sen=Sen, spe=Spe))
    return {'loss': metric_logger.loss.global_avg, 'AUC':AUC, 'Acc': Acc, 'Sen': Sen, 'Spe': Spe}




def train_Downtask_3dSeg(model, criterion, data_loader, optimizer, device, epoch, progressive_transfer_learning):
    # 2d slice-wise based Learning...! 
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)

        if progressive_transfer_learning:
            if epoch == 4:
                print("Unfreeze Layer 4...!")
                for name, param in model.named_parameters(): 
                    if 'encoder.layer4.' in name:
                        param.requires_grad = True
            elif epoch == 9:
                print("Unfreeze Layer 3,4...!")
                for name, param in model.named_parameters(): 
                    if 'encoder.layer3.' in name:
                        param.requires_grad = True

            elif epoch == 14:
                print("Unfreeze Layer 2,3,4...!")
                for name, param in model.named_parameters(): 
                    if 'encoder.layer2.' in name:
                        param.requires_grad = True

            elif epoch == 19:
                print("Unfreeze Layer 1,2,3,4...!")
                for name, param in model.named_parameters(): 
                    if 'encoder.layer1.' in name:
                        param.requires_grad = True                                   

            seg_pred = model(inputs)
        
        else:
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
            # metric_logger.update(RotationLoss=loss1.data.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # Gather the stats from all processes
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()

def valid_Downtask_3dSeg(model, criterion, data_loader, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'
    
    # switch to evaluation mode
    model.eval()
    print_freq = 10
    metric_count = metric_sum = 0
    
    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        inputs  = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt  = batch_data["label"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)

        with torch.no_grad():
            seg_pred = model(inputs)

        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Metrics
        value, not_nans = dice_metric(y_pred=torch.sigmoid(seg_pred).round(), y=seg_gt)
        metric_count   += not_nans
        metric_sum     += value * not_nans

        
    # Metric SEG
    Dice = (metric_sum / metric_count).item()

    print('* Loss:{losses.global_avg:.3f} | Dice:{dice:.3f} '.format(losses=metric_logger.loss, dice=Dice))
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {'loss': metric_logger.loss.global_avg, 'Dice': Dice}

@torch.no_grad()
def test_Downtask_3dSeg(model, criterion, data_loader, device, test_name, save_path):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TEST:'

    # switch to evaluation mode
    model.eval()
    print_freq = 10
    metric_count = metric_sum = 0    

    # Save npz path 
    save_dict = dict()

    img_path_list   = []
    mask_path_list  = []
    img_list        = []
    mask_list       = []
    seg_prob_list   = []
    active_map_list = []
    dice_list       = []

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        img_path   = batch_data["img_path"][0]        # batch 1. so indexing [0]
        mask_path  = batch_data["mask_path"][0]       # batch 1. so indexing [0]
        inputs     = batch_data["image"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)
        seg_gt     = batch_data["label"].to(device)   # (B, C, H, W, 1) ---> (B, C, H, W)

        with torch.no_grad():
            model.encoder.layer4[2].relu.register_forward_hook(get_activation('Activation Map')) # for Activation Map
            seg_pred = model(inputs)

            # print("체크 확인용", activation['Activation Map'].shape)  # torch.Size([32, 2048, 16, 16])
            # print("체크 확인용2", inputs.shape)  # [1, 1, 256, 256, 32]

            # plt.imshow(only_cls, 'jet', alpha=0.5)
            # Save
            img_path_list.append(img_path)
            mask_path_list.append(mask_path)
            img_list.append(inputs.detach().cpu().numpy())
            mask_list.append(seg_gt.detach().cpu().numpy())
            seg_prob_list.append(torch.sigmoid(seg_pred).detach().cpu().numpy())
            active_map_list.append(Activation_Map(activation['Activation Map']))


        loss, loss_detail = criterion(seg_pred=seg_pred, seg_gt=seg_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        # LOSS
        metric_logger.update(loss=loss_value)
        if loss_detail is not None:
            metric_logger.update(**loss_detail)

        # Metrics
        value, not_nans = dice_metric(y_pred=torch.sigmoid(seg_pred).round(), y=seg_gt)
        metric_count   += not_nans
        metric_sum     += value * not_nans
        dice_list.append(value * not_nans)

    Dice = (metric_sum / metric_count).item()
    # Save Prediction by using npz
    save_dict['gt_img_path']   = img_path_list
    save_dict['gt_mask_path']  = mask_path_list
    save_dict['gt_img']        = img_list
    save_dict['gt_mask']       = mask_list
    save_dict['pred_mask']     = seg_prob_list
    save_dict['active_map']    = active_map_list
    save_dict['dice']          = dice_list

    print("Saved npz...! => ", save_path + test_name + '_seg_3d[Dice_' + str(round(Dice, 3)) + '].npz')
    np.savez(save_path + test_name + '_seg_3d[Dice_' + str(round(Dice, 3)) + '].npz', seg_3d=save_dict) 

    print('* Loss:{losses.global_avg:.3f} | Dice:{dice:.3f} '.format(losses=metric_logger.loss, dice=Dice))
    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return {'loss': metric_logger.loss.global_avg, 'Dice': Dice}
