import os
import math
import utils
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from metrics import f1_metric, accuracy, sensitivity, specificity, binary_dice_score, fp_rate_score, binary_dice_score_np, fp_rate_score_np
from sklearn.metrics import roc_auc_score
# from monai.utils import ImageMetaKey as Key
# from monai.inferers import sliding_window_inference
from accelerate.utils import gather_object
from utils import check_2d_data_device, check_3d_cls_data_device, check_3d_seg_data_device, collect_dict_all_processor, save_image_and_prediction
from tqdm import tqdm


# Setting...!
fn_denorm  = lambda x: (x * 0.5) + 0.5
fn_tonumpy = lambda x: x.detach().cpu().numpy()

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

def freeze_params(model: torch.nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def unfreeze_params(model: torch.nn.Module):
    """Set requires_grad=True for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = True

def freeze_or_unfreeze_block(model, layer_name, stage_idxs=None, freeze=True):
    """
    Helper function to freeze or unfreeze blocks based on the model type and the specific stage indices.
    """
    if hasattr(model, 'module'):
        encoder = model.module.encoder
    else:
        encoder = model.encoder
    
    if stage_idxs:
        blocks = encoder._blocks[stage_idxs[0]:stage_idxs[1]]
    else:
        blocks = getattr(encoder, layer_name)

    if freeze:
        freeze_params(blocks)
        print(f"Freeze {layer_name} ...!")
    else:
        unfreeze_params(blocks)
        print(f"Unfreeze {layer_name} ...!")

def handle_efficientnet_freezing(model, epoch):
    """
    EfficientNet freezing/unfreezing logic based on the current epoch.
    """
    if epoch <= 100:
        freeze_or_unfreeze_block(model, 'encoder', freeze=True)
    elif 101 <= epoch < 111:
        freeze_or_unfreeze_block(model, 'encoder', freeze=True)
        freeze_or_unfreeze_block(model, 'blocks_stage_3', stage_idxs=[model.encoder._stage_idxs[2], None], freeze=False)
    elif 111 <= epoch < 121:
        freeze_or_unfreeze_block(model, 'encoder', freeze=True)
        freeze_or_unfreeze_block(model, 'blocks_stage_2', stage_idxs=[model.encoder._stage_idxs[1], model.encoder._stage_idxs[2]], freeze=False)
    elif 121 <= epoch < 131:
        freeze_or_unfreeze_block(model, 'encoder', freeze=True)
        freeze_or_unfreeze_block(model, 'blocks_stage_1', stage_idxs=[model.encoder._stage_idxs[0], model.encoder._stage_idxs[1]], freeze=False)
    elif 131 <= epoch < 141:
        freeze_or_unfreeze_block(model, 'encoder', freeze=True)
        freeze_or_unfreeze_block(model, 'blocks_stage_0', stage_idxs=[None, model.encoder._stage_idxs[0]], freeze=False)
    else:
        freeze_or_unfreeze_block(model, 'encoder', freeze=False)

def handle_resnet_freezing(model, epoch):
    """
    ResNet freezing/unfreezing logic based on the current epoch.
    """
    if epoch <= 100:
        freeze_or_unfreeze_block(model, 'encoder', freeze=True)
    elif 101 <= epoch < 111:
        freeze_or_unfreeze_block(model, 'layer4', freeze=False)
    elif 111 <= epoch < 121:
        freeze_or_unfreeze_block(model, 'layer3', freeze=False)
    elif 121 <= epoch < 131:
        freeze_or_unfreeze_block(model, 'layer2', freeze=False)
    elif 131 <= epoch < 141:
        freeze_or_unfreeze_block(model, 'layer1', freeze=False)
    else:
        freeze_or_unfreeze_block(model, 'encoder', freeze=False)

def apply_gradual_unfreezing(model, epoch, gradual_unfreeze=True, model_type='efficientnet'):
    """
    Function to apply gradual unfreezing based on the model type and epoch.
    """
    if gradual_unfreeze:
        if model_type == 'efficientnet':
            handle_efficientnet_freezing(model, epoch)
        elif model_type == 'resnet':
            handle_resnet_freezing(model, epoch)
    else:
        print("Freeze encoder ...!")
        freeze_or_unfreeze_block(model, 'encoder', freeze=True)

def activation_map(x):
    # print("mean 0 == ", x.shape)                                # x = (B, 2048, 16, 16, D)
    mean = torch.mean(x, dim=1, keepdim=True)                     # x = (B, 1, H, W ,D)
    mean = torch.sigmoid(mean).squeeze().detach().cpu().numpy()   # x = (H, W, D)
    mean = np.stack([ cv2.resize(mean[..., i], (256, 256), interpolation=cv2.INTER_CUBIC) for i in range(mean.shape[-1]) ], axis=-1)
    mean -= mean.min()
    mean /= mean.max()
    return torch.tensor(mean).unsqueeze(0)

########################################################
# Uptask Task
def train_smartnet_2d(train_loader, model, criterion, optimizer, device, epoch, use_consist, multi_gpu, accelerator):
    model.train()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(train_loader), disable=(not accelerator.is_local_main_process))
    
    for step, batch_data in enumerate(epoch_iterator):
        
        path, image, label, mask = batch_data
        image, label, mask = check_2d_data_device(multi_gpu, device, image, label, mask)

        with accelerator.autocast():
            if use_consist:
                pred_cls, pred_seg, pred_rec, pooled_seg = model(image) # pred.shape [20, 1]
                pred_cls = pred_cls.sigmoid()
                pred_seg = pred_seg.sigmoid()
                loss, loss_dict = criterion(pred_cls=pred_cls, pred_seg=pred_seg, pred_rec=pred_rec, label=label, mask=mask, image=image, pooled_seg=pooled_seg)
            else:
                pred_cls, pred_seg, pred_rec = model(image)
                pred_cls = pred_cls.sigmoid()
                pred_seg = pred_seg.sigmoid()
                loss, loss_dict = criterion(pred_cls=pred_cls, pred_seg=pred_seg, pred_rec=pred_rec, label=label, mask=mask, image=image)

        loss_value = loss.item()
        optimizer.zero_grad()
        if multi_gpu != 'DDP':
            loss.backward()
        else:
            accelerator.backward(loss)
        optimizer.step()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
        
        all_loss_dict  = gather_object([loss_dict])
        all_loss_dict  = collect_dict_all_processor(all_loss_dict, average=True)

        if accelerator.is_main_process:
            metric_logger.update(key='train_loss', value=loss_dict['total_loss'], n=image.shape[0]*accelerator.num_processes)
            for key in loss_dict:
                if key.startswith('cls_'):
                    metric_logger.update(key='train_'+key, value=loss_dict[key], n=image.shape[0]*accelerator.num_processes)
                elif key.startswith('seg_'):
                    metric_logger.update(key='train_'+key, value=loss_dict[key], n=image.shape[0]*accelerator.num_processes)
                elif key.startswith('rec_'):
                    metric_logger.update(key='train_'+key, value=loss_dict[key], n=image.shape[0]*accelerator.num_processes)

        epoch_iterator.set_description("Training: Epochs %d (%d / %d Steps), (train_loss=%2.5f)" % (epoch, step, len(train_loader), loss_value))

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

@torch.no_grad()
def valid_smartnet_2d(valid_loader, model, device, epoch, save_dir, use_consist, multi_gpu, accelerator):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(valid_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(valid_loader), disable=(not accelerator.is_local_main_process))
    path_list      = []
    pred_prob_list = []
    gt_binary_list = []
    dice_list      = []
    fp_list        = []
    mae_list       = []

    for step, batch_data in enumerate(epoch_iterator):
        
        path, image, label, mask = batch_data
        image, label, mask = check_2d_data_device(multi_gpu, device, image, label, mask)

        # model.encoder.layer4.register_forward_hook(get_activation('Activation Map')) # for Activation Map
        # act_list.append(Activation_Map(activation['Activation Map']))
        
        with accelerator.autocast():
            if use_consist:
                pred_cls, pred_seg, pred_rec, pooled_seg = model(image)
            else:
                pred_cls, pred_seg, pred_rec = model(image)

        pred_cls = pred_cls.sigmoid()
        pred_seg = pred_seg.sigmoid()

        epoch_iterator.set_description("Validating: Epochs %d (%d / %d Steps)" % (epoch, step, len(valid_loader)))
        
        path_list.extend(path)
        pred_prob_list.append(fn_tonumpy(pred_cls))
        gt_binary_list.append(fn_tonumpy(label))

        # Metrics SEG
        non_zero_mask    = mask.view(mask.shape[0], -1).any(dim=1)
        indices_non_zero = torch.where(non_zero_mask)[0]
        indices_zero     = torch.where(~non_zero_mask)[0]
        mask_non_zero, mask_zero         = mask[indices_non_zero], mask[indices_zero]
        pred_seg_non_zero, pred_seg_zero = pred_seg[indices_non_zero], pred_seg[indices_zero]
        dice_score = binary_dice_score(pred_seg_non_zero.round(), mask_non_zero).item() if indices_non_zero.numel() > 0 else -1.0
        fp_score   = fp_rate_score(pred_seg_zero.round(), mask_zero).item() if indices_zero.numel() > 0 else -1.0
        dice_list.append(dice_score)
        fp_list.append(fp_score)

        # Metrics REC
        mae = torch.nn.functional.l1_loss(input=pred_rec, target=image).item()
        mae_list.append(mae)

    # Summary
    pred_prob_list = gather_object(pred_prob_list)
    gt_binary_list = gather_object(gt_binary_list)
    all_dice_score = gather_object(dice_list)
    all_fp_score   = gather_object(fp_list)
    all_mae        = gather_object(mae_list)

    if accelerator.is_main_process:
        # Metric CLS
        pred_prob_list  = np.concatenate(pred_prob_list, axis=0).squeeze() # (B,)
        gt_binary_list  = np.concatenate(gt_binary_list, axis=0).squeeze() # (B,)        
        auc  = roc_auc_score(y_true=gt_binary_list, y_score=pred_prob_list)
        f1   = f1_metric(y_true=gt_binary_list, y_pred=pred_prob_list.round())
        acc  = accuracy(y_true=gt_binary_list, y_pred=pred_prob_list.round())
        sen  = sensitivity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
        spe  = specificity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
        
        metric_logger.update(key='valid_cls_auc', value=auc, n=1)
        metric_logger.update(key='valid_cls_f1',  value=f1,  n=1)
        metric_logger.update(key='valid_cls_acc', value=acc, n=1)
        metric_logger.update(key='valid_cls_sen', value=sen, n=1)
        metric_logger.update(key='valid_cls_spe', value=spe, n=1)

        # Metrics SEG & REC
        all_dice_score   = np.array(all_dice_score)
        all_fp_score     = np.array(all_fp_score)        
        valid_dice_score = all_dice_score[all_dice_score != -1]
        valid_fp_score   = all_fp_score[all_fp_score != -1]

        metric_logger.update(key='valid_seg_dice', value=np.mean(valid_dice_score), n=1)
        metric_logger.update(key='valid_seg_fp',   value=np.mean(valid_fp_score),   n=1)
        metric_logger.update(key='valid_rec_mae',  value=np.mean(all_mae),          n=1)
        
        # SAVE PNG
        if indices_non_zero.numel() > 0:
            save_image_and_prediction(image, mask, pred_seg, pred_rec, save_dir, epoch, index=indices_non_zero[0])
        else:
            save_image_and_prediction(image, mask, pred_seg, pred_rec, save_dir, epoch)

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

@torch.no_grad()
def test_smartnet_2d(test_loader, model, device, epoch, save_dir, use_consist):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(test_loader, desc="Testing (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(test_loader))
    dice_list      = []
    fp_list        = []
    pred_prob_list = []
    gt_binary_list = []

    for step, batch_data in enumerate(epoch_iterator):
        
        path, image, label, mask, depth = batch_data
        image = image.to(device).float()
        label = label.to(device).float()
        mask  = mask.to(device).float()        

        # model.encoder.layer4.register_forward_hook(get_activation('Activation Map')) # for Activation Map
        # act_list.append(Activation_Map(activation['Activation Map']))        
        
        B, C, D, H, W = image.shape
        # print("label == ", label.shape) # label ==  torch.Size([10, 64])

        image = image.permute(2, 0, 1, 3, 4)      # (B, C, D, H, W) --> (D, B, C, H, W)
        mask  = mask.permute(2, 0, 1, 3, 4)       # (B, C, D, H, W) --> (D, B, C, H, W)

        image = torch.reshape(image, [D*B, C, H, W])      # (D, B, C, H, W) --> (D*B, C, H, W)
        mask  = torch.reshape(mask, [D*B, C, H, W])       # (D, B, C, H, W) --> (D*B, C, H, W)
        label = torch.reshape(label, [D*B, 1])            # (D, B, 1) --> (D*B, 1)

        for i in range(image.shape[0]):
            target_image = image[i].unsqueeze(0)
            target_mask  = mask[i].unsqueeze(0)
            target_label = label[i].unsqueeze(0)

            if use_consist:
                pred_cls, pred_seg, pred_rec, pooled_seg = model(target_image)
            else:
                pred_cls, pred_seg, pred_rec = model(target_image)

            pred_seg = pred_seg.sigmoid()
            pred_cls = pred_cls.sigmoid()

            epoch_iterator.set_description("Testing: Epochs %d (%d / %d Steps)" % (epoch, step, len(test_loader)))

            pred_prob_list.append(fn_tonumpy(pred_cls))
            gt_binary_list.append(fn_tonumpy(target_label))

            # Metrics SEG
            non_zero_mask    = target_mask.view(target_mask.shape[0], -1).any(dim=1)
            indices_non_zero = torch.where(non_zero_mask)[0]
            indices_zero     = torch.where(~non_zero_mask)[0]
            mask_non_zero, mask_zero         = target_mask[indices_non_zero], target_mask[indices_zero]
            pred_seg_non_zero, pred_seg_zero = pred_seg[indices_non_zero], pred_seg[indices_zero]
            dice_score = binary_dice_score(pred_seg_non_zero.round(), mask_non_zero).item() if indices_non_zero.numel() > 0 else -1.0
            fp_score   = fp_rate_score(pred_seg_zero.round(), mask_zero).item() if indices_zero.numel() > 0 else -1.0
            dice_list.append(dice_score)
            fp_list.append(fp_score)

    # Metric CLS
    pred_prob_list  = np.concatenate(pred_prob_list, axis=0).squeeze() # (B,)
    gt_binary_list  = np.concatenate(gt_binary_list, axis=0).squeeze() # (B,)
    auc  = roc_auc_score(y_true=gt_binary_list, y_score=pred_prob_list)
    f1   = f1_metric(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    acc  = accuracy(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    sen  = sensitivity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    spe  = specificity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    
    metric_logger.update(key='test_cls_auc', value=auc, n=1)
    metric_logger.update(key='test_cls_f1',  value=f1,  n=1)
    metric_logger.update(key='test_cls_acc', value=acc, n=1)
    metric_logger.update(key='test_cls_sen', value=sen, n=1)
    metric_logger.update(key='test_cls_spe', value=spe, n=1)

    # Metrics SEG
    dice_list        = np.array(dice_list)
    fp_list          = np.array(fp_list)
    valid_dice_score = dice_list[dice_list != -1]
    valid_fp_score   = fp_list[fp_list != -1]

    metric_logger.update(key='test_seg_dice', value=np.mean(valid_dice_score), n=1)
    metric_logger.update(key='test_seg_fp',   value=np.mean(valid_fp_score),   n=1)

    return {k: round(v, 7) for k, v in metric_logger.average().items()}


########################################################
# Down Task
def train_smartnet_3d_2dtransfer_CLS(train_loader, model, criterion, optimizer, device, epoch, multi_gpu, accelerator):
    model.train()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(train_loader), disable=(not accelerator.is_local_main_process))
    
    # apply_gradual_unfreezing(model, epoch, gradual_unfreeze=True, model_type='efficientnet')

    for step, batch_data in enumerate(epoch_iterator):
        
        path, image, label, depth = batch_data
        image, label = check_3d_cls_data_device(multi_gpu, device, image, label)
        
        with accelerator.autocast():
            pred_cls = model(image, depth)
            
            pred_cls = pred_cls.sigmoid()
            loss, loss_dict = criterion(pred_cls, label)

        loss_value = loss.item()
        optimizer.zero_grad()
        if multi_gpu != 'DDP':
            loss.backward()
        else:
            accelerator.backward(loss)
        optimizer.step()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
        
        all_loss_dict  = gather_object([loss_dict])
        all_loss_dict  = collect_dict_all_processor(all_loss_dict, average=True)

        if accelerator.is_main_process:
            metric_logger.update(key='train_loss', value=loss_dict['total_loss'], n=image.shape[0]*accelerator.num_processes)
            for key in loss_dict:
                if key.startswith('cls_'):
                    metric_logger.update(key='train_'+key, value=loss_dict[key], n=image.shape[0]*accelerator.num_processes)

        epoch_iterator.set_description("Training: Epochs %d (%d / %d Steps), (train_loss=%2.5f)" % (epoch, step, len(train_loader), loss_value))

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

@torch.no_grad()
def valid_smartnet_3d_2dtransfer_CLS(valid_loader, model, device, epoch, save_dir, multi_gpu, accelerator):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(valid_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(valid_loader), disable=(not accelerator.is_local_main_process))
    path_list      = []
    pred_prob_list = []
    gt_binary_list = []
    
    for step, batch_data in enumerate(epoch_iterator):
        
        path, image, label, depth = batch_data
        image, label = check_3d_cls_data_device(multi_gpu, device, image, label)

        # model.fc.register_forward_hook(get_features('feat')) # for Representation
        # feat_list.append(features['feat'].detach().cpu().numpy())
        
        with accelerator.autocast():
            pred_cls = model(image, depth)
            pred_cls = pred_cls.sigmoid()

        epoch_iterator.set_description("Validating: Epochs %d (%d / %d Steps)" % (epoch, step, len(valid_loader)))
        
        path_list.extend(path)
        pred_prob_list.append(fn_tonumpy(pred_cls))
        gt_binary_list.append(fn_tonumpy(label))

    # Summary
    pred_prob_list = gather_object(pred_prob_list)
    gt_binary_list = gather_object(gt_binary_list)
    
    if accelerator.is_main_process:
        pred_prob_list  = np.concatenate(pred_prob_list, axis=0).squeeze() # (B,)
        gt_binary_list  = np.concatenate(gt_binary_list, axis=0).squeeze() # (B,)

        # Metric CLS
        auc  = roc_auc_score(y_true=gt_binary_list, y_score=pred_prob_list)
        f1   = f1_metric(y_true=gt_binary_list, y_pred=pred_prob_list.round())
        acc  = accuracy(y_true=gt_binary_list, y_pred=pred_prob_list.round())
        sen  = sensitivity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
        spe  = specificity(y_true=gt_binary_list, y_pred=pred_prob_list.round())

        metric_logger.update(key='valid_cls_auc', value=auc, n=1)
        metric_logger.update(key='valid_cls_f1',  value=f1,  n=1)
        metric_logger.update(key='valid_cls_acc', value=acc, n=1)
        metric_logger.update(key='valid_cls_sen', value=sen, n=1)
        metric_logger.update(key='valid_cls_spe', value=spe, n=1)

        # CAM ?

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

@torch.no_grad()
def test_smartnet_3d_2dtransfer_CLS(test_loader, model, device, epoch, save_dir):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(test_loader, desc="Testing (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(test_loader))
    path_list      = []
    pred_prob_list = []
    gt_binary_list = []

    for step, batch_data in enumerate(epoch_iterator):
        
        path, image, label, depth = batch_data
        image = image.to(device).float()
        label = label.to(device).float()

        pred_cls = model(image, depth)
        pred_cls = pred_cls.sigmoid()

        epoch_iterator.set_description("Testing: Epochs %d (%d / %d Steps)" % (epoch, step, len(test_loader)))
        
        path_list.extend(path)
        pred_prob_list.append(fn_tonumpy(pred_cls))
        gt_binary_list.append(fn_tonumpy(label))

    # Metric CLS
    pred_prob_list  = np.concatenate(pred_prob_list, axis=0).squeeze() # (B,)
    gt_binary_list  = np.concatenate(gt_binary_list, axis=0).squeeze() # (B,)

    auc  = roc_auc_score(y_true=gt_binary_list, y_score=pred_prob_list)
    f1   = f1_metric(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    acc  = accuracy(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    sen  = sensitivity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    spe  = specificity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    
    metric_logger.update(key='test_cls_auc', value=auc, n=1)
    metric_logger.update(key='test_cls_f1',  value=f1,  n=1)
    metric_logger.update(key='test_cls_acc', value=acc, n=1)
    metric_logger.update(key='test_cls_sen', value=sen, n=1)
    metric_logger.update(key='test_cls_spe', value=spe, n=1)

    # DataFrame
    df = pd.DataFrame()
    df['Patient_id'] = path_list
    df['Prob']       = pred_prob_list
    df['Label']      = gt_binary_list
    df['Decision']   = pred_prob_list.round()
    df.to_csv(save_dir+'/test_cls_result_epoch_'+str(epoch)+'.csv')

    return {k: round(v, 7) for k, v in metric_logger.average().items()}


# Down Task
def train_smartnet_3d_2dtransfer_SEG(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(train_loader))
    
    for step, batch_data in enumerate(epoch_iterator):
        
        path, image, mask, depth = batch_data

        image = image.to(device)
        mask  = mask.to(device)

        pred_seg = model(image)

        # act
        pred_seg = pred_seg.sigmoid()

        loss, loss_dict = criterion(pred_seg, mask)
        loss_value = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
        
        metric_logger.update(key='train_loss', value=loss_value, n=image.shape[0])
        # metric_logger.update(key='train_cls_bce_loss', value=loss_dict['cls_bce_loss'].item(), n=image.shape[0])

        epoch_iterator.set_description("Training: Epochs %d (%d / %d Steps), (train_loss=%2.5f)" % (epoch, step, len(train_loader), loss_value))

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

@torch.no_grad()
def valid_smartnet_3d_2dtransfer_SEG(valid_loader, model, device, epoch, save_dir):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(valid_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(valid_loader))
    
    for step, batch_data in enumerate(epoch_iterator):
        
        path, image, mask, depth = batch_data

        image = image.to(device)
        mask  = mask.to(device)

        pred_seg = model(image)

        # act
        pred_seg = pred_seg.sigmoid()

        epoch_iterator.set_description("Validating: Epochs %d (%d / %d Steps)" % (epoch, step, len(valid_loader)))
        
        # # Metrics SEG
        # if mask.any():
        #     dice = binary_dice_loss(y_pred=pred_seg.round(), y_true=mask, smooth=0.0, return_score=True)    # pred_seg must be round() !!  
        #     metric_logger.update(key='valid_dice', value=dice.item(), n=mask.shape[0])

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

@torch.no_grad()
def test_smartnet_3d_2dtransfer_SEG(test_loader, model, device, save_dir):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(test_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(test_loader))

    pred_prob_list = []
    gt_binary_list = []
    patient_id_list = []
    type_id_list = []

    for step, batch_data in enumerate(epoch_iterator):
        
        image, mask, _, patient_id, type_id = batch_data

        image = image.to(device)
        label = mask.to(device)

        pred_cls = model(image, depth)

        # act
        pred_cls = pred_cls.sigmoid()

        epoch_iterator.set_description("TEST: (%d / %d Steps)" % (step, len(test_loader)))
        
        pred_prob_list.append(fn_tonumpy(pred_cls))
        gt_binary_list.append(fn_tonumpy(label))
        patient_id_list.append(patient_id)
        type_id_list.append(type_id)

    pred_prob_list  = np.concatenate(pred_prob_list, axis=0).squeeze() # (B,)
    gt_binary_list  = np.concatenate(gt_binary_list, axis=0).squeeze() # (B,)
    patient_id_list = np.concatenate(patient_id_list, axis=0).squeeze() # (B,)
    type_id_list    = np.concatenate(type_id_list, axis=0).squeeze() # (B,)

    # Metric CLS
    auc            = roc_auc_score(y_true=gt_binary_list, y_score=pred_prob_list)
    f1             = f1_metric(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    acc            = accuracy(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    sen            = sensitivity(y_true=gt_binary_list, y_pred=pred_prob_list.round())
    spe            = specificity(y_true=gt_binary_list, y_pred=pred_prob_list.round())

    metric_logger.update(key='valid_auc', value=auc, n=1)
    metric_logger.update(key='valid_f1',  value=f1,  n=1)
    metric_logger.update(key='valid_acc', value=acc, n=1)
    metric_logger.update(key='valid_sen', value=sen, n=1)
    metric_logger.update(key='valid_spe', value=spe, n=1)

    # DataFrame
    df = pd.DataFrame()
    df['Patient_id'] = patient_id_list
    df['Type_id']    = type_id_list
    df['Prob']       = pred_prob_list
    df['Label']      = gt_binary_list
    df['Decision']   = pred_prob_list.round()
    df.to_csv(save_dir+'/pred_results.csv')

    return {k: round(v, 7) for k, v in metric_logger.average().items()}









# ########################################################
# ## Inference code 
# from monai.transforms import SaveImage
# from monai.transforms import Resize, Flip, Rotate90
# from monai.utils import ensure_tuple
# from typing import Dict, Optional, Union
# import logging
# import traceback


#     # SMAET-Net
# @torch.no_grad()
# def infer_Up_SMART_Net(model, data_loader, device, print_freq, save_dir):
#     # 2d slice-wise based evaluate...! 
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ", n=1)
#     header = 'TEST:'
    
#     save_dict = dict()
#     img_path_list = []
#     img_list = []
#     cls_list = []
#     seg_list = []
#     rec_list = []
#     act_list = []

#     for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
#         inputs  = batch_data["image"].squeeze(4).to(device)      # (B, C, H, W, 1) ---> (B, C, H, W)

#         model.encoder.layer4.register_forward_hook(get_activation('Activation Map')) # for Activation Map

#         cls_pred, seg_pred, rec_pred = model(inputs)

#         # post-processing
#         cls_pred = torch.sigmoid(cls_pred)
#         seg_pred = torch.sigmoid(seg_pred)

#         img_path_list.append(batch_data["image_path"][0])
#         img_list.append(inputs.detach().cpu().squeeze())
#         cls_list.append(cls_pred.detach().cpu().squeeze())
#         seg_list.append(seg_pred.detach().cpu().squeeze())
#         rec_list.append(rec_pred.detach().cpu().squeeze())
#         act_list.append(Activation_Map(activation['Activation Map']))


#     save_dict['img_path_list']  = img_path_list
#     save_dict['img_list']       = img_list
#     save_dict['cls_pred']       = cls_list
#     save_dict['seg_pred']       = seg_list
#     save_dict['rec_pred']       = rec_list
#     save_dict['activation_map'] = act_list
#     np.savez(save_dir + '/result.npz', result=save_dict) 

#     # CLS
# @torch.no_grad()
# def infer_Down_SMART_Net_CLS(model, data_loader, device, print_freq, save_dir):
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ", n=1)
#     header = 'TEST:'

#     save_dict = dict()
#     img_path_list = []
#     img_list  = []
#     cls_list  = []
#     feat_list = []


#     for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        
        
#         inputs  = batch_data["image"].to(device)                                                        # (B, C, H, W, D)
#         depths  = batch_data["depths"]                                                                  #  ---> (B, 1) Fix bug, change cpu()
#         paths   = batch_data["image_path"][0]


#         model.fc.register_forward_hook(get_features('feat')) # for Representation
        
#         cls_pred = model(inputs, depths)

#         # Post-processing
#         cls_pred = torch.sigmoid(cls_pred)

#         img_path_list.append(paths)
#         img_list.append(inputs.detach().cpu().numpy().squeeze())
#         cls_list.append(cls_pred.detach().cpu().numpy().squeeze())
#         feat_list.append(features['feat'].detach().cpu().numpy().squeeze())


#     save_dict['img_path_list']  = img_path_list
#     save_dict['img_list']       = img_list
#     save_dict['cls_pred']       = cls_list
#     save_dict['feat']           = feat_list
#     np.savez(save_dir + '/result.npz', result=save_dict) 


#     # SEG
# @torch.no_grad()
# def infer_Down_SMART_Net_SEG(model, data_loader, device, print_freq, save_dir):