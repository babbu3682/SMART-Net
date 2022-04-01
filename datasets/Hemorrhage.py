import re
import glob
import cv2
import functools
import torch
import numpy as np
import skimage
import random

import albumentations as albu
from monai.transforms import *
from monai.data import Dataset

import warnings
warnings.filterwarnings(action='ignore') 


# functions
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

def resize_keep_depths(x, size, mode):
    depths = x.shape[-1]  # (C, H, W, D)
    if mode == 'image':
        x = Resize(spatial_size=(size, size, depths), mode='trilinear', align_corners=True)(x)
    else :
        x = Resize(spatial_size=(size, size, depths), mode='nearest', align_corners=None)(x)
    return x

def pad(image, new_shape, border_mode="constant", value=0):
    '''
    image: [H, W, D, C] or [H, W, D]
    new_shape: [H, W, D]
    '''
    axes_not_pad = len(image.shape) - len(new_shape)

    old_shape = np.array(image.shape[:len(new_shape)])
    new_shape = np.array([max(new_shape[i], old_shape[i]) for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference - pad_below

    pad_list = [list(i) for i in zip(pad_below, pad_above)] + [[0, 0]] * axes_not_pad

    if border_mode == 'reflect':
        res = np.pad(image, pad_list, border_mode)
    elif border_mode == 'constant':
        res = np.pad(image, pad_list, border_mode, constant_values=value)
    else:
        raise ValueError

    return res

def crop(img, x1, y1, z1, x2, y2, z2):
    height, width, depth = img.shape[:3]
    if x2 <= x1 or y2 <= y1 or z2 <= z1:
        raise ValueError
    if x1 < 0 or y1 < 0 or z1 < 0:
        raise ValueError
    if x2 > height or y2 > width or z2 > depth:
        img = pad(img, (x2, y2, z2))
        warnings.warn('image size smaller than crop size, pad by default.', UserWarning)

    return img[x1:x2, y1:y2, z1:z2]

def Crop_Non_Empty_Mask_If_Exists(input, patch_size=(256, 256, 1)):
    # reference: https://github.com/ZFTurbo/volumentations
    image = input['image'].squeeze(0)
    mask  = input['label'].squeeze(0)
    
    patch_height, patch_width, patch_depth = patch_size
    mask_height,  mask_width,  mask_depth  = mask.shape

    if mask.sum() == 0:
        x_min = random.randint(0, mask_height - patch_height)
        y_min = random.randint(0, mask_width  - patch_width)
        z_min = random.randint(0, mask_depth  - patch_depth)

    else:
        non_zero = np.argwhere(mask)
        x, y, z  = random.choice(non_zero)
        x_min    = x - random.randint(0, patch_height - 1)
        y_min    = y - random.randint(0, patch_width  - 1)
        z_min    = z - random.randint(0, patch_depth  - 1)
        x_min    = np.clip(x_min, 0, mask_height - patch_height)
        y_min    = np.clip(y_min, 0, mask_width  - patch_width)
        z_min    = np.clip(z_min, 0, mask_depth  - patch_depth)

    x_max = x_min + patch_height
    y_max = y_min + patch_width
    z_max = z_min + patch_depth

    image = crop(image, x_min, y_min, z_min, x_max, y_max, z_max)
    mask  = crop(mask,  x_min, y_min, z_min, x_max, y_max, z_max)

    input['image'] = np.expand_dims(image, axis=0)
    input['label'] = np.expand_dims(mask, axis=0)
    
    return input

def clahe_keep_depths(image, clipLimit, tileGridSize):
    image = skimage.util.img_as_ubyte(image.squeeze(0))
    
    assert image.dtype == np.uint8
    assert len(image.shape) == 3  # 2d --> (H, W, 1) / 3d --> (H, W, D)

    clahe_mat   = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    stacked_img = np.stack([clahe_mat.apply(image[..., i]) for i in range(image.shape[-1])], axis=-1)
    
    stacked_img = skimage.util.img_as_float32(stacked_img)        

    return np.expand_dims(stacked_img, axis=0)

def delete_too_small_noise_keep_depths(mask, min_size, connectivity):
    mask  = skimage.morphology.remove_small_objects(mask.squeeze(0).astype('bool'), min_size=min_size, connectivity=connectivity) # for noise reduction
    
    return np.expand_dims(mask, axis=0).astype('float32')

def Albu_2D_Transform_Compose(input):
    '''
    reference: https://github.com/albumentations-team/albumentations
    We can conduct 2D-based Trasnform with keeping the depth slice, so depth slices are all composed of the same augmentation combination.
    1. 2D-based
        (H, W, 1)  -> (H, W, 1)
    2. 3D-based
        (H, W, D*)  -> (H, W, D*)
    '''
    image = input['image'].squeeze(0) 
    mask  = input['label'].squeeze(0)
    
    Trans = albu.Compose(
        [albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, interpolation=2, border_mode=4, p=0.5),
         albu.OneOf(
             [
                 albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
                 albu.RandomBrightness(limit=0.1, p=1),
                 albu.RandomContrast(limit=0.1, p=1),
             ], p=0.5
         ),
         
         albu.OneOf(
             [
                 albu.IAAAdditiveGaussianNoise(loc=0, scale=(0.005, 0.01), p=1),
                 albu.GaussNoise(var_limit=(0.001, 0.005), p=1),
             ], p=0.5
         ),         

         albu.OneOf(
             [
                 albu.Blur(blur_limit=3, p=1),
                 albu.MotionBlur(blur_limit=3, p=1),
                 albu.MedianBlur(blur_limit=3, p=1),
             ], p=0.5
         )])
    
    augment = Trans(image=image, mask=mask)
    image = augment['image']
    mask  = augment['mask']
    
    input['image'] = np.expand_dims(image, axis = 0)
    input['label'] = np.expand_dims(mask,  axis = 0)
    
    return input

def minmax_normalize(image, option=False):
    image -= image.min()
    image /= image.max() 
    if option:
        image = (image - 0.5) / 0.5                  # Range -1.0 ~ 1.0   @ We do not use -1~1 range becuase there is no Tanh act.
    return image.astype('float32')

def Filter_Zero_Depths(input):
    image = input['image'].squeeze(0)
    mask  = input['label'].squeeze(0)

    depths = np.argwhere(image.sum(axis=(0, 1)) != 0).squeeze()  # considering depths
    
    image = image[..., depths.min():depths.max()+1]
    mask  = mask[..., depths.min():depths.max()+1]

    input['image'] = np.expand_dims(image, axis=0)
    input['label'] = np.expand_dims(mask, axis=0)
    
    return input



# collate_fn for patient-level
def default_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def pad_collate_fn(batch):
    # batches is a list with the size of batch_size.
    # batches[i] == dataset[i]
    X          = [ sample['image'] for sample in batch ]
    Y          = [ sample['label'] for sample in batch ]
    img_list   = [ sample['image_meta_dict']['filename_or_obj'] for sample in batch ]
    mask_list  = [ sample['label_meta_dict']['filename_or_obj'] for sample in batch ]    
        
    depths     = torch.IntTensor([x.shape[-1] for x in X])
    
    stack_padded_image = []
    stack_padded_label = []
    
    for image, label in zip(X, Y):
        assert image.shape == label.shape # [C, H, W, D]
        
        ori_c, ori_x, ori_y, ori_z = image.shape

        if (depths.max()-ori_z != 0):
            pad = torch.zeros( (ori_c, ori_x, ori_y, depths.max()-ori_z) )
            stack_padded_image.append(torch.cat([image, pad], dim=-1))
            stack_padded_label.append(torch.cat([label, pad], dim=-1))
            
        else :
            stack_padded_image.append(image)
            stack_padded_label.append(label)
            
    batch = dict()
    batch['img_path']    = img_list
    batch['mask_path']   = mask_list
    batch['image']       = torch.stack(stack_padded_image, dim=0)
    batch['label']       = torch.stack(stack_padded_label, dim=0)
    batch['depths']      = depths

    return batch




# Dataset
    ## Up Task
def Hemo_Uptask_Dataset(mode, data_folder_dir="/workspace/sunggu/1.Hemorrhage/SMART-Net/samples"):
    if mode == 'train':
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/train/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob(data_folder_dir + "/train/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]        

        print("Train [Total]  number = ", len(img_list))
        print("Train [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Train [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                # Load nii data
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="PLS"),

                # Pre-processing
                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),                     # Windowing HU [min:0, max:80]
                Filter_Zero_Depths,                                                                                               # Remove the empty slice
                Lambdad(keys=["image"], func=functools.partial(resize_keep_depths, size=256, mode='image')),                      # Resize Image                
                Lambdad(keys=["label"], func=functools.partial(resize_keep_depths, size=256, mode='label')),                      # Resize Label                                
                Crop_Non_Empty_Mask_If_Exists,                                                                                    # Sampling one hemorrhage slice with high probability in patient level nii data for 2D-based manner
                Lambdad(keys=["image"], func=functools.partial(clahe_keep_depths, clipLimit=2.0, tileGridSize=(8, 8))),           # CLAHE for image contrast
                Lambdad(keys=["label"], func=functools.partial(delete_too_small_noise_keep_depths, min_size=3, connectivity=2)),  # Noise Reduction
                
                # Augmentation
                Albu_2D_Transform_Compose,                                            # 2D based Transform with keeping the depth slices
                RandFlipd(keys=["image", "label"], spatial_axis=[0, 1], prob=0.5),    # 3D based Transform 
                
                # Normalize
                Lambdad(keys=["image"], func=functools.partial(minmax_normalize, option=False)),
                ToTensord(keys=["image", "label"])
            ]
        )   

    else :
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/valid/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob(data_folder_dir + "/valid/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Valid [Total]  number = ", len(img_list))
        print("Valid [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Valid [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                # Load nii data
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="PLS"),

                # Pre-processing
                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),                     # Windowing HU [min:0, max:80]
                Filter_Zero_Depths,                                                                                               # Remove the empty slice
                Lambdad(keys=["image"], func=functools.partial(resize_keep_depths, size=256, mode='image')),                      # Resize Image                
                Lambdad(keys=["label"], func=functools.partial(resize_keep_depths, size=256, mode='label')),                      # Resize Label                                                
                Lambdad(keys=["image"], func=functools.partial(clahe_keep_depths, clipLimit=2.0, tileGridSize=(8, 8))),           # CLAHE for image contrast
                Lambdad(keys=["label"], func=functools.partial(delete_too_small_noise_keep_depths, min_size=3, connectivity=2)),  # Noise Reduction
                
                # Normalize
                Lambdad(keys=["image"], func=functools.partial(minmax_normalize, option=False)),
                ToTensord(keys=["image", "label"]),
            ]
        )

    return Dataset(data=data_dicts, transform=transforms), default_collate_fn

    ## Down Task
def Hemo_Downtask_Dataset(mode, data_folder_dir="/workspace/sunggu/1.Hemorrhage/SMART-Net/samples"):
    if mode == 'train':
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/train/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob(data_folder_dir + "/train/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]        

        print("Train [Total]  number = ", len(img_list))
        print("Train [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Train [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                # Load nii data
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="PLS"),

                # Pre-processing
                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),                     # Windowing HU [min:0, max:80]
                Filter_Zero_Depths,                                                                                               # Remove the empty slice
                Lambdad(keys=["image"], func=functools.partial(resize_keep_depths, size=256, mode='image')),                      # Resize Image                
                Lambdad(keys=["label"], func=functools.partial(resize_keep_depths, size=256, mode='label')),                      # Resize Label                                
                Lambdad(keys=["image"], func=functools.partial(clahe_keep_depths, clipLimit=2.0, tileGridSize=(8, 8))),           # CLAHE for image contrast
                Lambdad(keys=["label"], func=functools.partial(delete_too_small_noise_keep_depths, min_size=3, connectivity=2)),  # Noise Reduction
                
                # Augmentation
                Albu_2D_Transform_Compose,                                            # 2D based Transform with keeping the depth slices
                RandFlipd(keys=["image", "label"], spatial_axis=[0, 1], prob=0.5),    # 3D based Transform 
                
                # Normalize
                Lambdad(keys=["image"], func=functools.partial(minmax_normalize, option=False)),
                ToTensord(keys=["image", "label"]),
            ]
        )        

    else :
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/valid/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob(data_folder_dir + "/valid/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Valid [Total]  number = ", len(img_list))
        print("Valid [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Valid [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                # Load nii data
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="PLS"),

                # Pre-processing
                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),                     # Windowing HU [min:0, max:80]
                Filter_Zero_Depths,                                                                                               # Remove the empty slice
                Lambdad(keys=["image"], func=functools.partial(resize_keep_depths, size=256, mode='image')),                      # Resize Image                
                Lambdad(keys=["label"], func=functools.partial(resize_keep_depths, size=256, mode='label')),                      # Resize Label                                
                Lambdad(keys=["image"], func=functools.partial(clahe_keep_depths, clipLimit=2.0, tileGridSize=(8, 8))),           # CLAHE for image contrast
                Lambdad(keys=["label"], func=functools.partial(delete_too_small_noise_keep_depths, min_size=3, connectivity=2)),  # Noise Reduction
                
                # Normalize
                Lambdad(keys=["image"], func=functools.partial(minmax_normalize, option=False)),
                ToTensord(keys=["image", "label"]),
            ]
        )
        
    return Dataset(data=data_dicts, transform=transforms), pad_collate_fn


# TEST
def Hemo_TEST_Dataset(test_dataset_name, data_folder_dir="/workspace/sunggu/1.Hemorrhage/SMART-Net/samples"):
    if test_dataset_name == 'Custom':
        img_list     = list_sort_nicely(glob.glob(data_folder_dir + "/test/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob(data_folder_dir + "/test/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Test [Total]  number = ", len(img_list))
        print("Test [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Test [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                # Load nii data
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="PLS"),

                # Pre-processing
                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True),                     # Windowing HU [min:0, max:80]
                Filter_Zero_Depths,                                                                                               # Remove the empty slice
                Lambdad(keys=["image"], func=functools.partial(resize_keep_depths, size=256, mode='image')),                      # Resize Image                
                Lambdad(keys=["label"], func=functools.partial(resize_keep_depths, size=256, mode='label')),                      # Resize Label                                
                Lambdad(keys=["image"], func=functools.partial(clahe_keep_depths, clipLimit=2.0, tileGridSize=(8, 8))),           # CLAHE for image contrast
                Lambdad(keys=["label"], func=functools.partial(delete_too_small_noise_keep_depths, min_size=3, connectivity=2)),  # Noise Reduction
                                
                # Normalize
                Lambdad(keys=["image"], func=functools.partial(minmax_normalize, option=False)),
                ToTensord(keys=["image", "label"]),
            ]
        )

    else :
        raise Exception('Error, Dataset name')        
                  

    return Dataset(data=data_dicts, transform=transforms), pad_collate_fn




