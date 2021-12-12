import os
import sys

device = 'cuda'
sys.path.append(os.path.abspath('/workspace/sunggu'))
sys.path.append(os.path.abspath('/workspace/sunggu/1.Hemorrhage/utils/FINAL_utils'))
sys.path.append(os.path.abspath('/workspace/sunggu/MONAI'))

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
import volumentations

import skimage
import albumentations as albu
from sklearn.model_selection import train_test_split
from skimage.morphology import disk, binary_dilation

import re
import glob

from monai.transforms import *
from monai.data import Dataset, DataLoader

import segmentation_models_pytorch_final as smp

from torch.autograd.functional import jacobian

def list_sort_nicely(l):   
    def tryint(s):        
        try:            
            return int(s)        
        except:            
            return s
        
    def alphanum_key(s):
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]
    l.sort(key=alphanum_key)    
    return l

def Sampling_Z_axis_volumentation(x, patch_size=(256, 256, 1)):

    # Monai is (C, H, W, D) shape
    image = x['image'].squeeze(0)
    mask  = x['label'].squeeze(0)
    
    result = volumentations.Compose([volumentations.CropNonEmptyMaskIfExists(patch_size, always_apply=True)])(image=image, mask=mask)
    # (256, 256, 1)
    image = result['image']
    mask  = result['mask']
    
    x['image'] = np.expand_dims(image, axis=0)
    x['label'] = np.expand_dims(mask, axis=0)
    
    return x

def clahe_slice_wise(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    img = skimage.util.img_as_ubyte(img)
    
    assert img.dtype == np.uint8
    assert len(img.shape) == 3  # 2d --> (H, W, 1) / 3d --> (H, W, D)

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img = np.stack([clahe_mat.apply(img[..., i]) for i in range(img.shape[-1])], axis=-1)
    img = skimage.util.img_as_float32(img)        
    
    return img

def delete_small_pixel(mask, cut_pixel=20):  
    labels = skimage.measure.label(mask)

    # CUT pixel counts
    _, counts = np.unique(labels, return_counts=True)
    for idx, i in enumerate(counts):
        if(i <= cut_pixel):
            exec("labels[labels == "+ str(idx) + "] = 0")
    
    return labels.astype('bool').astype('float32')

def Preprocessing(x):
    image = x['image'].squeeze(0)
    mask  = x['label'].squeeze(0)
    
    image = clahe_slice_wise(image)
    mask  = delete_small_pixel(mask)
        
    x['image'] = np.expand_dims(image, axis=0)
    x['label'] = np.expand_dims(mask, axis=0)
    
    return x

def Resized_individual(x):
    z = x['image'].shape[-1]    
    x = Resized(keys=["image"], spatial_size=(256, 256, z), mode=['trilinear'], align_corners=True)(x)
    x = Resized(keys=["label"], spatial_size=(256, 256, z), mode=['nearest'],   align_corners=None)(x)
    
    return x



def pad_collate_fn(batches):

    if isinstance(batches[0], (list, tuple)):
        X          = [ batch[0]['image'] for batch in batches ]
        Y          = [ batch[0]['label'] for batch in batches ]
        img_list   = [ batch[0]['image_meta_dict']['filename_or_obj'] for batch in batches ]
        mask_list  = [ batch[0]['label_meta_dict']['filename_or_obj'] for batch in batches ]
        
    else : 
        X          = [ batch['image'] for batch in batches ]
        Y          = [ batch['label'] for batch in batches ]
        img_list   = [ batch['image_meta_dict']['filename_or_obj'] for batch in batches ]
        mask_list  = [ batch['label_meta_dict']['filename_or_obj'] for batch in batches ]    
        
    z_shapes = torch.IntTensor([x.shape[-1] for x in X])
    
    pad_image = []
    pad_label = []
    
    for img, label in zip(X, Y):
        assert img.shape == label.shape
        
        if (z_shapes.max() - img.shape[3] != 0):
            pad = torch.zeros( (img.shape[0], img.shape[1], img.shape[2], z_shapes.max()-img.shape[3]) )
            pad_image.append(torch.cat([img, pad], dim=-1))
            pad_label.append(torch.cat([label, pad], dim=-1))
            
        else :
            pad_image.append(img)
            pad_label.append(label)
            
    batch = dict()
    batch['img_path']    = img_list
    batch['mask_path']   = mask_list
    batch['image']       = torch.stack(pad_image, dim=0)
    batch['label']       = torch.stack(pad_label, dim=0)
    batch['z_shape']     = z_shapes

    
    return batch


val_transforms = Compose(
    [
        LoadNiftid(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        # Orientationd(keys=["image", "label"], axcodes="PLS"),
        Flipd(keys=["image", "label"], spatial_axis=1),
        Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),                

        ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
        Resized_individual,

        ##### Preprocessing
        # Sampling_Z_axis_volumentation,  # for Uptask 2d slice
        Preprocessing,
        # SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),   # for only previous 3d seg unet careful...!!!
        
        ToTensord(keys=["image", "label"]),
    ]
)




asan_img_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Asan_*/*_img.nii.gz"))
asan_label_list = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Asan_*/*_mask.nii.gz"))
asan_data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(asan_img_list, asan_label_list)]
asan_ds = Dataset(data=asan_data_dicts, transform=val_transforms)
asan_loader = DataLoader(asan_ds, batch_size=1, shuffle=False, num_workers=3, collate_fn=pad_collate_fn)

ulji_img_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Ulji_*/*_img.nii.gz"))
ulji_label_list = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Ulji_*/*_mask.nii.gz"))
ulji_data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(ulji_img_list, ulji_label_list)]
ulji_ds = Dataset(data=ulji_data_dicts, transform=val_transforms)
ulji_loader = DataLoader(ulji_ds, batch_size=1, shuffle=False, num_workers=3, collate_fn=pad_collate_fn)

pohang_img_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Pohang_*/*_img.nii.gz"))
pohang_label_list = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Pohang_*/*_mask.nii.gz"))
pohang_data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(pohang_img_list, pohang_label_list)]
pohang_ds = Dataset(data=pohang_data_dicts, transform=val_transforms)
pohang_loader = DataLoader(pohang_ds, batch_size=1, shuffle=False, num_workers=3, collate_fn=pad_collate_fn)

physionet_img_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Physionet_*/Images/*.nii"))
physionet_label_list = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Physionet_*/Labels/*.nii"))
physionet_data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(physionet_img_list, physionet_label_list)]
physionet_ds = Dataset(data=physionet_data_dicts, transform=val_transforms)
physionet_loader = DataLoader(physionet_ds, batch_size=1, shuffle=False, num_workers=3, collate_fn=pad_collate_fn)




only_cls         = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
only_seg         = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
only_rec         = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder

cls_seg          = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
cls_rec          = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
seg_rec          = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder

cls_seg_consist  = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
cls_seg_rec      = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
mr_net           = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
scratch          = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
imagenet         = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
model_genesis    = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder

# ablation
max_avg  = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
gmp      = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder
gap      = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=1, classes=1, activation=None).encoder

# Only Cls
check_point = torch.load('/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/[DownTASK]Hemo_resnet50_[Segmentation]_freeze_encoder_Conv3d_OnlyCls/epoch_370_best_metric_model.pth')
model_dict = only_cls.state_dict() 
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
only_cls.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])

# Only Seg
check_point = torch.load('/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/[DownTASK]Hemo_resnet50_[Segmentation]_freeze_encoder_Conv3d_OnlySeg/epoch_150_best_metric_model.pth')
model_dict = only_seg.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
only_seg.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])

# Only Recon
check_point = torch.load('/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/Downstream_Only_Recon_AutoEncoder_SEG/epoch_360_checkpoint.pth')
model_dict = only_rec.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
only_rec.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])


# CLS + SEG
check_point = torch.load("/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/[DownTASK]Hemo_resnet50_[Segmentation]_freeze_encoder_Conv3d_Aux/epoch_350_best_metric_model.pth")
model_dict = cls_seg.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
cls_seg.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])

# CLS + REC
check_point = torch.load('/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/Downstream_CLS_REC_AE_SEG/epoch_170_checkpoint.pth')
model_dict = cls_rec.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
cls_rec.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])

# SEG + REC
check_point = torch.load('/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/Downstream_SEG_REC_AE_SEG/epoch_135_checkpoint.pth')
model_dict = seg_rec.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
seg_rec.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])


# CLS + SEG + Consist
check_point = torch.load('/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/[DownTASK]Hemo_resnet50_[Segmentation]_freeze_encoder_Conv3d_Consistency_16_16_GAP/epoch_440_best_metric_model.pth')
model_dict = cls_seg_consist.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
cls_seg_consist.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])


# CLS + SEG + REC
check_point = torch.load("/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/Downstream_Consistency_Recon_AutoEncoder_Notconsist_SEG/epoch_220_checkpoint.pth")
model_dict = cls_seg_rec.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
cls_seg_rec.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])

# CLS + SEG + REC + Consist
check_point = torch.load("/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/Downstream_Consistency_16x16_GAP_Recon_AE_SEG/epoch_335_checkpoint.pth")
model_dict = mr_net.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
mr_net.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])


# Scratch
# check_point = torch.load("/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/[DownTASK]Hemo_resnet50_[Segmentation]_End-to-End_Conv3d_From_Scratch/epoch_80_best_metric_model.pth")
check_point = torch.load("/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/CLS/[DownTASK]Hemo_resnet50_[Classification]_End-to-End_Slicewise_LSTM_From_scratch/epoch_180_best_metric_model.pth")
model_dict = scratch.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
# pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
pretrained_dict = {k.replace('CNN_model.encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
scratch.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])


# Imagenet
check_point = torch.load('/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/[DownTASK]Hemo_resnet50_[Segmentation]_freeze_encoder_Conv3d_ImageNet/epoch_320_best_metric_model.pth')
model_dict = imagenet.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
imagenet.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])


# Model genesis
check_point = torch.load("/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/Downstream_ModelGenesis_Recon_Skip_SEG/epoch_335_checkpoint.pth")
model_dict = model_genesis.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
model_genesis.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])

# ablation pool
# 16x16 MAX + 16x16 AVG
check_point = torch.load("/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/[DownTASK]Hemo_resnet50_[Segmentation]_freeze_encoder_Conv3d_Consistency_16_16_GAP/epoch_440_best_metric_model.pth")
model_dict = max_avg.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
max_avg.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])
# GMP
check_point = torch.load("/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/[DownTASK]Hemo_resnet50_[Segmentation]_freeze_encoder_Conv3d_Consistency_Max_Aux/epoch_90_best_metric_model.pth")
model_dict = gmp.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
gmp.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])
# GAP
check_point = torch.load("/workspace/sunggu/1.Hemorrhage/models/MTL/Downstream/SEG/[DownTASK]Hemo_resnet50_[Segmentation]_freeze_encoder_Conv3d_Consistency_Avg_Aux/epoch_150_best_metric_model.pth")
model_dict = gap.state_dict()
print("이전 weight = ", model_dict['conv1.weight'][0])
pretrained_dict = {k.replace('encoder.', ''):v for k, v in check_point['model_state_dict'].items() if 'encoder.' in k}
model_dict.update(pretrained_dict) 
gap.load_state_dict(model_dict)
print("이후 weight = ", model_dict['conv1.weight'][0])

def run():
    image_list = []
    label_list = []

    only_cls_am_maps            = []
    only_seg_am_maps            = []
    only_rec_am_maps            = []

    cls_seg_am_maps             = []
    cls_rec_am_maps             = []
    seg_rec_am_maps             = []
    
    cls_seg_consist_am_maps     = []
    cls_seg_rec_am_maps         = []

    mr_net_am_maps              = []

    scratch_am_maps             = []
    imagenet_am_maps            = []
    model_genesis_am_maps       = []


    only_cls.to('cuda');          only_cls.eval();
    only_seg.to('cuda');          only_seg.eval();
    only_rec.to('cuda');          only_rec.eval();

    cls_seg.to('cuda');           cls_seg.eval();
    cls_rec.to('cuda');           cls_rec.eval();
    seg_rec.to('cuda');           seg_rec.eval();

    cls_seg_consist.to('cuda');   cls_seg_consist.eval();
    cls_seg_rec.to('cuda');       cls_seg_rec.eval();
    mr_net.to('cuda');            mr_net.eval();
    scratch.to('cuda');           scratch.eval();
    imagenet.to('cuda');          imagenet.eval();
    model_genesis.to('cuda');     model_genesis.eval();

    # for idx, loader in enumerate([physionet_loader]):        
    for idx, loader in enumerate([asan_loader]):        
        with torch.no_grad():
            for k, val_data in enumerate(loader):
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)

                for i in range(val_labels.shape[-1]):
                    image_list.append(val_inputs[..., i])
                    label_list.append(val_labels[..., i])
                    
                    only_cls_am_maps.append(only_cls(val_inputs[..., i])[-1])
                    only_seg_am_maps.append(only_seg(val_inputs[..., i])[-1])
                    only_rec_am_maps.append(only_rec(val_inputs[..., i])[-1])

                    cls_seg_am_maps.append(cls_seg(val_inputs[..., i])[-1])
                    cls_rec_am_maps.append(cls_rec(val_inputs[..., i])[-1])
                    seg_rec_am_maps.append(seg_rec(val_inputs[..., i])[-2])

                    cls_seg_consist_am_maps.append(cls_seg_consist(val_inputs[..., i])[-1])
                    cls_seg_rec_am_maps.append(cls_seg_rec(val_inputs[..., i])[-1])
                    mr_net_am_maps.append(mr_net(val_inputs[..., i])[-1])
                    
                    scratch_am_maps.append(scratch(val_inputs[..., i])[-1])
                    imagenet_am_maps.append(imagenet(val_inputs[..., i])[-1])
                    model_genesis_am_maps.append(model_genesis(val_inputs[..., i])[-1])

                if k == 28:
                    return only_cls_am_maps, only_seg_am_maps, only_rec_am_maps, cls_seg_am_maps, cls_rec_am_maps, seg_rec_am_maps, cls_seg_consist_am_maps, cls_seg_rec_am_maps, mr_net_am_maps, scratch_am_maps, imagenet_am_maps, model_genesis_am_maps, image_list, label_list


    return only_cls_am_maps, only_seg_am_maps, only_rec_am_maps, cls_seg_am_maps, cls_rec_am_maps, seg_rec_am_maps, cls_seg_consist_am_maps, cls_seg_rec_am_maps, mr_net_am_maps, scratch_am_maps, imagenet_am_maps, model_genesis_am_maps, image_list, label_list

            


def run_ablation():
    image_list = []
    label_list = []

    MaxAVG_am_maps  = []
    GAP_am_maps     = []
    GMP_am_maps     = []

    max_avg.to('cuda');    max_avg.eval();
    gap.to('cuda');        gap.eval();
    gmp.to('cuda');        gmp.eval();

    # for idx, loader in enumerate([physionet_loader]):        
    for idx, loader in enumerate([asan_loader]):        
        with torch.no_grad():
            for k, val_data in enumerate(loader):
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)

                for i in range(val_labels.shape[-1]):
                    image_list.append(val_inputs[..., i])
                    label_list.append(val_labels[..., i])
                    
                    MaxAVG_am_maps.append(max_avg(val_inputs[..., i])[-1])
                    GAP_am_maps.append(gap(val_inputs[..., i])[-1])
                    GMP_am_maps.append(gmp(val_inputs[..., i])[-1])
                    
                if k == 28:
                    return MaxAVG_am_maps, GAP_am_maps, GMP_am_maps, image_list, label_list


    return MaxAVG_am_maps, GAP_am_maps, GMP_am_maps, image_list, label_list

            
