import re
import glob
import cv2
import functools
import torch
import numpy as np
import skimage
import random
import pydicom
import SimpleITK as sitk
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from typing import Dict, Optional, Sequence, Tuple, Union, Callable

import warnings
warnings.filterwarnings(action='ignore') 


# functions
def list_sort_nicely(l):
    def convert(text): return int(text) if text.isdigit() else text
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def change_to_uint8(image, **kwargs):
    return skimage.util.img_as_ubyte(image)

def change_to_float32(image, **kwargs):
    return skimage.util.img_as_float32(image)

def squeeze_transpose(image, **kwargs):
    return image.squeeze(3).transpose(1, 2, 0)

def crop(img: np.ndarray, x_min: int, y_min: int, z_min: int, x_max: int, y_max: int, z_max: int):
    channel, height, width, depth = img.shape
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min}, x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max
            )
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height or z_min < 0 or z_max > depth:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes \n"
            "(x_min = {x_min}, y_min = {y_min}, z_min = {z_min}, x_max = {x_max}, y_max = {y_max}, z_max = {z_max}) \n"
            "height = {height}, width = {width}, depth = {depth})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max, height=height, width=width, depth=depth
            )
        )

    return img[:, y_min:y_max, x_min:x_max, z_min:z_max] # image is C, H, W, D

def windowing(image, window_center=40, window_width=80, **kwargs):
    lower_bound = window_center - window_width/2
    upper_bound = window_center + window_width/2
    image = (np.clip(image, lower_bound, upper_bound) - lower_bound) / window_width
    return image.astype(np.float32)

def fixed_clahe(image, **kwargs):
    clahe_mat = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = clahe_mat.apply(image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        image[:, :, 0] = clahe_mat.apply(image[:, :, 0])
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    return image

def minmax_normalize(image, option=False, **kwargs):
    if len(np.unique(image)) != 1:  # Sometimes it cause the nan inputs...
        image -= image.min()
        image /= image.max() 

    if option:
        image = (image - 0.5) / 0.5  # Range -1.0 ~ 1.0   @ We do not use -1~1 range becuase there is no Tanh act.

    return image.astype('float32')


class DualTransform_V2(DualTransform):
    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
            "idx": self.apply_to_idx,
            "pad_loc": self.apply_to_loc,
        }    
    def apply_to_idx(self, idx, **params):
        return idx

    def apply_to_loc(self, pad_loc, **params):
        return pad_loc

class SamplingSlice_3D_Pos_Crop(DualTransform_V2):
    def __init__(self, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0):
        super(SamplingSlice_3D_Pos_Crop, self).__init__(always_apply, p)

        if ignore_values is not None and not isinstance(ignore_values, list):
            raise ValueError("Expected `ignore_values` of type `list`, got `{}`".format(type(ignore_values)))
        if ignore_channels is not None and not isinstance(ignore_channels, list):
            raise ValueError("Expected `ignore_channels` of type `list`, got `{}`".format(type(ignore_channels)))

        self.depth  = 1
        self.width  = 256
        self.height = 256

        self.ignore_values   = ignore_values
        self.ignore_channels = ignore_channels

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0, **params):
        return crop(img, x_min, y_min, z_min, x_max, y_max, z_max)

    def apply_to_idx(self, idx, x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0, **params):
        return z_min

    def _preprocess_mask(self, mask):
        channel, mask_height, mask_width, mask_depth = mask.shape

        if self.ignore_values is not None:
            ignore_values_np = np.array(self.ignore_values)
            mask = np.where(np.isin(mask, ignore_values_np), 0, mask)

        if mask.ndim == 4 and self.ignore_channels is not None:
            target_channels = np.array([ch for ch in range(mask.shape[0]) if ch not in self.ignore_channels])
            mask = np.take(mask, target_channels, axis=0)

        if self.height > mask_height or self.width > mask_width or self.depth > mask_depth:
            raise ValueError("Crop size ({},{},{}) is larger than image ({},{},{})".format(self.height, self.width, self.depth, mask_height, mask_width, mask_depth))

        return mask

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)

        if "mask" in kwargs:
            mask = self._preprocess_mask(kwargs["mask"])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = kwargs["masks"]
            mask = self._preprocess_mask(np.copy(masks[0]))  # need copy as we perform in-place mod afterwards
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            raise RuntimeError("Can not find mask for CropNonEmptyMaskIfExists")

        channel, mask_height, mask_width, mask_depth = mask.shape

        if mask.any():
            mask = mask.sum(axis=0) if mask.ndim == 4 else mask
            non_zero_yxz = np.argwhere(mask)
            y, x, z = random.choice(non_zero_yxz)
            x_min = x - random.randint(0, self.width  - 1)
            y_min = y - random.randint(0, self.height - 1)
            z_min = z - random.randint(0, self.depth  - 1)
            x_min = np.clip(x_min, 0, mask_width  - self.width)
            y_min = np.clip(y_min, 0, mask_height - self.height)
            z_min = np.clip(z_min, 0, mask_depth  - self.depth)
        else:
            x_min = random.randint(0, mask_width  - self.width)
            y_min = random.randint(0, mask_height - self.height)
            z_min = random.randint(0, mask_depth  - self.depth)

        x_max = x_min + self.width
        y_max = y_min + self.height
        z_max = z_min + self.depth

        params.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "z_min": z_min, "z_max": z_max})
        return params

    def get_transform_init_args_names(self):
        return ("ignore_values", "ignore_channels")

class SamplingSlice_3D_Pos(DualTransform_V2):
    def __init__(self, ignore_values=None, ignore_channels=None, always_apply=False, p=1.0):
        super(SamplingSlice_3D_Pos, self).__init__(always_apply, p)

        if ignore_values is not None and not isinstance(ignore_values, list):
            raise ValueError("Expected `ignore_values` of type `list`, got `{}`".format(type(ignore_values)))
        if ignore_channels is not None and not isinstance(ignore_channels, list):
            raise ValueError("Expected `ignore_channels` of type `list`, got `{}`".format(type(ignore_channels)))

        self.depth  = 1

        self.ignore_values   = ignore_values
        self.ignore_channels = ignore_channels

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0, **params):
        return crop(img, x_min, y_min, z_min, x_max, y_max, z_max)

    def apply_to_idx(self, idx, x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0, **params):
        return z_min

    def _preprocess_mask(self, mask):
        channel, mask_height, mask_width, mask_depth = mask.shape

        if self.ignore_values is not None:
            ignore_values_np = np.array(self.ignore_values)
            mask = np.where(np.isin(mask, ignore_values_np), 0, mask)

        if mask.ndim == 4 and self.ignore_channels is not None:
            target_channels = np.array([ch for ch in range(mask.shape[0]) if ch not in self.ignore_channels])
            mask = np.take(mask, target_channels, axis=0)

        if self.height > mask_height or self.width > mask_width or self.depth > mask_depth:
            raise ValueError("Crop size ({},{},{}) is larger than image ({},{},{})".format(self.height, self.width, self.depth, mask_height, mask_width, mask_depth))

        return mask

    def update_params(self, params, **kwargs):
        super().update_params(params, **kwargs)

        C, H, W, D = kwargs["image"].shape
        self.height = H
        self.width  = W

        if "mask" in kwargs:
            mask = self._preprocess_mask(kwargs["mask"])
        elif "masks" in kwargs and len(kwargs["masks"]):
            masks = kwargs["masks"]
            mask = self._preprocess_mask(np.copy(masks[0]))  # need copy as we perform in-place mod afterwards
            for m in masks[1:]:
                mask |= self._preprocess_mask(m)
        else:
            raise RuntimeError("Can not find mask for CropNonEmptyMaskIfExists")

        channel, mask_height, mask_width, mask_depth = mask.shape

        if mask.any():
            mask = mask.sum(axis=0) if mask.ndim == 4 else mask
            non_zero_yxz = np.argwhere(mask)
            y, x, z = random.choice(non_zero_yxz)
            x_min = x - random.randint(0, self.width  - 1)
            y_min = y - random.randint(0, self.height - 1)
            z_min = z - random.randint(0, self.depth  - 1)
            x_min = np.clip(x_min, 0, mask_width  - self.width)
            y_min = np.clip(y_min, 0, mask_height - self.height)
            z_min = np.clip(z_min, 0, mask_depth  - self.depth)
        else:
            x_min = random.randint(0, mask_width  - self.width)
            y_min = random.randint(0, mask_height - self.height)
            z_min = random.randint(0, mask_depth  - self.depth)

        x_max = x_min + self.width
        y_max = y_min + self.height
        z_max = z_min + self.depth

        params.update({"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max, "z_min": z_min, "z_max": z_max})
        return params

    def get_transform_init_args_names(self):
        return ("ignore_values", "ignore_channels")

class WindowingTransform(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(WindowingTransform, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return windowing(image)
                  
    def get_transform_init_args_names(self):
        return ()
    
class ChangeToUint8(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(ChangeToUint8, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return change_to_uint8(image)        
                  
    def get_transform_init_args_names(self):
        return ()
    
class ChangeToFloat32(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(ChangeToFloat32, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return change_to_float32(image)            
              
    def get_transform_init_args_names(self):
        return ()
    
class FixedClahe(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(FixedClahe, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return fixed_clahe(image)  
              
    def get_transform_init_args_names(self):
        return ()
    
class MinmaxNormalize(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(MinmaxNormalize, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return minmax_normalize(image)   
             
    def get_transform_init_args_names(self):
        return ()

# waste
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
    
def clahe_norm(image):
    image = skimage.util.img_as_float32(image)
    
    if len(np.unique(image)) != 1:  # Sometimes it cause the nan inputs...
        image -= image.min()
        image /= image.max() 

    return image

class FixedCLAHE(ImageOnlyTransform):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), always_apply=False, p=0.5):
        super(FixedCLAHE, self).__init__(always_apply, p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image, **params):
        clahe_mat = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = clahe_mat.apply(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            image[:, :, 0] = clahe_mat.apply(image[:, :, 0])
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        return image

    def get_transform_init_args_names(self):
        return ("clip_limit", "tile_grid_size")


def get_array_pydicom(path):
    ds = pydicom.dcmread(path)
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = ds.pixel_array
    if ds.PixelRepresentation == 1:
        bit_shift = ds.BitsAllocated - ds.BitsStored
        dtype = pixel_array.dtype 
        pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
    
    intercept = float(ds.RescaleIntercept)
    slope     = float(ds.RescaleSlope)

    pixel_array = (pixel_array * slope) + intercept
    pixel_array = np.array(pixel_array, dtype=np.float32)
    return pixel_array

def extract_3D(x):
    dcm_list = list_sort_nicely(glob.glob(x + '/*.dcm'))
    # data_3d  = np.concatenate([sitk.GetArrayFromImage(sitk.ReadImage(dcm)) for dcm in dcm_list], axis=0)
    data_3d  = np.stack([get_array_pydicom(dcm) for dcm in dcm_list], axis=0)
    return data_3d


# collate_fn
def pad_collate_fn_512(batch):    
    depths = []
    labels = []
    patient_ids = []
    type_ids = []

    for sample in batch:
        labels.append(sample[1])
        depths.append(sample[2])
        patient_ids.append(sample[3])
        type_ids.append(sample[4])

    depths = torch.IntTensor(depths)
    labels = torch.stack(labels)

    stack_padded_image = torch.zeros( (len(batch), 1, depths.max(), 512, 512) )    
    for i, sample in enumerate(batch):
        ori_c, ori_z, ori_x, ori_y = sample[0].shape
        stack_padded_image[i, :ori_c, :ori_z, :ori_x, :ori_y] = sample[0]

    return stack_padded_image, labels, depths, patient_ids, type_ids


def get_transforms_2D_Crop(mode="train"):
    # medical augmentation
    if mode == "train":
        return A.Compose([
            # preprocessing
            SamplingSlice_3D_Pos_Crop(ignore_values=None, ignore_channels=None, always_apply=False, p=1.0),
            A.Lambda(image=windowing),
            A.Lambda(image=squeeze_transpose, mask=squeeze_transpose),
            A.Lambda(image=change_to_uint8, always_apply=True),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True),
            A.Lambda(image=change_to_float32, always_apply=True),
            A.Lambda(image=minmax_normalize, always_apply=True),

            # augmentation
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.15, rotate_limit=15, shift_limit=0.1, border_mode=0, p=1.0),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=1.0),
                A.RandomBrightness(limit=0.05, p=1.0),
                A.RandomContrast(limit=0.05, p=1.0),
                ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, sigma_limit=0, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.5),
            A.GaussNoise(var_limit=(0.00001, 0.00005), mean=0, per_channel=True, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.8),    # https://arxiv.org/pdf/2212.04690.pdf            
            
            # normalization
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0, p=1.0),
            ToTensorV2(transpose_mask=True)
        ], additional_targets={'image2':'image'})
    
    elif mode == "valid":
        return A.Compose([
            # preprocessing
            SamplingSlice_3D_Pos(ignore_values=None, ignore_channels=None, always_apply=False, p=1.0),
            A.Lambda(image=windowing),
            A.Lambda(image=squeeze_transpose, mask=squeeze_transpose),
            A.Lambda(image=change_to_uint8, always_apply=True),
            A.Lambda(image=fixed_clahe, always_apply=True),
            A.Lambda(image=change_to_float32, always_apply=True),
            A.Lambda(image=minmax_normalize, always_apply=True),

            # normalization
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0, p=1.0),
            ToTensorV2(transpose_mask=True)
        ], additional_targets={'image2':'image'})

def get_transforms_2D(mode="train"):
    # medical augmentation
    if mode == "train":
        return A.Compose([
            # preprocessing
            SamplingSlice_3D_Pos(ignore_values=None, ignore_channels=None, always_apply=False, p=1.0),
            A.Lambda(image=windowing),
            A.Lambda(image=squeeze_transpose, mask=squeeze_transpose),
            A.Lambda(image=change_to_uint8, always_apply=True),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), always_apply=True),
            A.Lambda(image=change_to_float32, always_apply=True),
            A.Lambda(image=minmax_normalize, always_apply=True),

            # augmentation
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.15, rotate_limit=15, shift_limit=0.1, border_mode=0, p=1.0),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=1.0),
                A.RandomBrightness(limit=0.05, p=1.0),
                A.RandomContrast(limit=0.05, p=1.0),
                ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=3, sigma_limit=0, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
                ], p=0.5),
            A.GaussNoise(var_limit=(0.00001, 0.00005), mean=0, per_channel=True, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.8),    # https://arxiv.org/pdf/2212.04690.pdf            
            
            # normalization
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0, p=1.0),
            ToTensorV2(transpose_mask=True)
        ], additional_targets={'image2':'image'})
    
    elif mode == "valid":
        return A.Compose([
            # preprocessing
            SamplingSlice_3D_Pos(ignore_values=None, ignore_channels=None, always_apply=False, p=1.0),
            A.Lambda(image=windowing),
            A.Lambda(image=squeeze_transpose, mask=squeeze_transpose),
            A.Lambda(image=change_to_uint8, always_apply=True),
            A.Lambda(image=fixed_clahe, always_apply=True),
            A.Lambda(image=change_to_float32, always_apply=True),
            A.Lambda(image=minmax_normalize, always_apply=True),

            # normalization
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0, p=1.0),
            ToTensorV2(transpose_mask=True)
        ], additional_targets={'image2':'image'})

def get_transforms_3D_2Dtransfer():
    return A.ReplayCompose([
        # preprocessing
        A.CenterCrop(height=512, width=512, always_apply=True, p=1.0),
        WindowingTransform(always_apply=True, p=1.0),
        ChangeToUint8(always_apply=True, p=1.0),
        FixedClahe(always_apply=True, p=1.0),
        ChangeToFloat32(always_apply=True, p=1.0),
        MinmaxNormalize(always_apply=True, p=1.0),

        # normalization
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0, p=1.0),
        ToTensorV2(transpose_mask=True)
    ], additional_targets={'image2':'image'})


# 2D
class HEMO_2D_MTL_CLS_SEG_REC_Dataset(BaseDataset):
    def __init__(self, mode="train"):
        self.root = '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets'
        self.mode = mode
        if mode == 'train':
            self.image_list = list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Train_nii/*_img.nii.gz'))  + list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Coreline_1350/NIFTI_IMG/*.nii.gz'))
            self.mask_list  = list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Train_nii/*_mask.nii.gz')) + list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Coreline_1350/NIFTI_SEGGT/*.nii.gz'))
        else:
            self.image_list = list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_img.nii.gz'))
            self.mask_list  = list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_mask.nii.gz'))
        
        self.transforms = get_transforms_2D(mode=mode)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # image
        image = sitk.ReadImage(self.image_list[index])
        image = sitk.GetArrayFromImage(image).transpose(1, 2, 0)  # H, W, D
        image = np.expand_dims(image, axis=0) # C, H, W, D

        # mask
        mask = sitk.ReadImage(self.mask_list[index])
        mask = sitk.GetArrayFromImage(mask).transpose(1, 2, 0)  # H, W, D
        mask = np.expand_dims(mask, axis=0) # C, H, W, D
  
        C, H, W, D = image.shape

        # augmentation
        sample = self.transforms(image=image, mask=mask, idx='slice_idx', pad_loc=('pad_top', 'pad_bottom', 'pad_left', 'pad_right'))
        image  = sample['image']
        mask   = sample['mask']
        
        # BG=0, B = 1, M = 2
        if mask.sum().item() > 0:
            label = torch.tensor([1.0])
        else :
            label = torch.tensor([0.0])

        # image [B, 1, 512, 512], mask [B, 1, 512, 512], label [B, 1]
        return image.float(), label.float(), mask.float()

class HEMO_2D_MTL_CLS_SEG_REC_Dataset_Crop(BaseDataset):
    def __init__(self, mode="train"):
        self.root = '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets'
        self.mode = mode
        if mode == 'train':
            self.image_list = list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Train_nii/*_img.nii.gz'))  + list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Coreline_1350/NIFTI_IMG/*.nii.gz'))
            self.mask_list  = list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Train_nii/*_mask.nii.gz')) + list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Coreline_1350/NIFTI_SEGGT/*.nii.gz'))
        else:
            self.image_list = list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_img.nii.gz'))
            self.mask_list  = list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_mask.nii.gz'))
        
        self.transforms = get_transforms_2D_Crop(mode=mode)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # image
        image = sitk.ReadImage(self.image_list[index])
        image = sitk.GetArrayFromImage(image).transpose(1, 2, 0)  # H, W, D
        image = np.expand_dims(image, axis=0) # C, H, W, D

        # mask
        mask = sitk.ReadImage(self.mask_list[index])
        mask = sitk.GetArrayFromImage(mask).transpose(1, 2, 0)  # H, W, D
        mask = np.expand_dims(mask, axis=0) # C, H, W, D
  
        C, H, W, D = image.shape

        # augmentation
        sample = self.transforms(image=image, mask=mask, idx='slice_idx', pad_loc=('pad_top', 'pad_bottom', 'pad_left', 'pad_right'))
        image  = sample['image']
        mask   = sample['mask']
        
        # BG=0, B = 1, M = 2
        if mask.sum().item() > 0:
            label = torch.tensor([1.0])
        else :
            label = torch.tensor([0.0])

        # image [B, 1, 512, 512], mask [B, 1, 512, 512], label [B, 1]
        return image.float(), label.float(), mask.float()


# 3D - 2D transfer
class HEMO_3D_CLS_Dataset_2Dtransfer(BaseDataset):
    def __init__(self):
        self.root = '/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets'
        # self.image_list = list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Coreline_TEST_NII/*/*/*.nii.gz'))
        self.image_list = list_sort_nicely(glob.glob('/workspace/sunggu/1.Hemorrhage/dataset/Coreline_TEST/*/*'))
        self.transforms = get_transforms_3D_2Dtransfer()
        self.excel_file = pd.read_excel('/workspace/sunggu/1.Hemorrhage/dataset/GAN_Brain 강릉아산 TEST명단_Revision_2023.12.19.xlsx', sheet_name='최종 데이터')
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # image
        # image = sitk.ReadImage(self.image_list[index])
        # image = sitk.GetArrayFromImage(image)  # D, H, W
        image = extract_3D(self.image_list[index]) # D, H, W

        D, H, W = image.shape

        # augmentation
        augmented_images = torch.empty((1, D, 512, 512), dtype=torch.float32)
        first_image_slice = image[0, :, :]
        sample = self.transforms(image=first_image_slice)
        augmented_images[:, 0, :, :] = sample['image']
        replay = sample['replay']
        for d in range(1, D):
            slice_image = image[d, :, :]
            sample = A.ReplayCompose.replay(replay, image=slice_image)
            augmented_images[:, d, :, :] = sample["image"]
        
        # patient info
        patient_id = self.image_list[index].split('/')[-1]
        type_id    = self.image_list[index].split('/')[-2]

        label = torch.tensor([self.excel_file[self.excel_file['익명화번호(논문사용)'] == patient_id]['label'].values[0]])

        # image [B, 1, 512, 512], label [B, 1]
        return augmented_images.float(), label.float(), D, patient_id, type_id


# 3D - 331
class TDSC_REAL3D_MTL_CLS_SEG_REC_Dataset_CLS(BaseDataset):
    def __init__(self, mode="train"):
        self.root = '/workspace/sunggu/0.Challenge/MICCAI2023_TDSC/dataset'
        self.mode = mode
        if mode == 'train':
            self.image_list = list_sort_nicely(glob.glob('/workspace/sunggu/0.Challenge/MICCAI2023_TDSC/dataset/DATA/train/*.nrrd'))
            self.mask_list  = list_sort_nicely(glob.glob('/workspace/sunggu/0.Challenge/MICCAI2023_TDSC/dataset/MASK/train/*.nrrd'))
        else:
            self.image_list = list_sort_nicely(glob.glob('/workspace/sunggu/0.Challenge/MICCAI2023_TDSC/dataset/DATA/valid/*.nrrd'))
            self.mask_list  = list_sort_nicely(glob.glob('/workspace/sunggu/0.Challenge/MICCAI2023_TDSC/dataset/MASK/valid/*.nrrd'))
        
        self.transforms = get_transforms_slice_3D(mode=mode)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        label_name = self.label[self.label['data_path'] == self.image_list[index].split('/')[-1]]['label'].values[0]

        # image
        image = sitk.ReadImage(self.image_list[index])
        image = sitk.GetArrayFromImage(image).transpose(1, 2, 0)  # H, W, D

        # mask
        mask  = sitk.ReadImage(self.mask_list[index])
        mask  = sitk.GetArrayFromImage(mask).transpose(1, 2, 0)  # H, W, D

        # Resize
        H, W, D = image.shape

        image = zoom(input=image, zoom=(0.5, 0.5, 0.5), order=1) # nearest
        mask  = zoom(input=mask,  zoom=(0.5, 0.5, 0.5), order=0) # Bilinear

        if self.mode == 'train':
            # preprocessing
            image = np.expand_dims(image, axis=0)  # C, H, W, D
            mask  = np.expand_dims(mask,  axis=0)  # C, H, W, D
            
            sample = A.OneOf([
                Crop_3D_Pos(height=256, width=256, depth=32, ignore_values=None, ignore_channels=None, always_apply=False, p=0.80),
                Crop_3D_Neg(height=256, width=256, depth=32, ignore_values=None, ignore_channels=None, always_apply=False, p=0.20)
                ], p=1.0)(image=image, mask=mask)

            image = sample['image'].squeeze(0) # H, W, D
            mask  = sample['mask'].squeeze(0)
    
            # augmentation
            augmented_images = torch.empty((1, 256, 256, 32), dtype=torch.float32)
            augmented_masks  = torch.empty((1, 256, 256, 32), dtype=torch.float32)
            first_image_slice = image[:, :, 0]
            first_mask_slice  = mask[:, :, 0]
            sample = self.transforms(image=first_image_slice, mask=first_mask_slice, pad_loc=('pad_top', 'pad_bottom', 'pad_left', 'pad_right'))
            augmented_images[:, :, :, 0] = sample['image']
            augmented_masks[:, :, :, 0]  = sample['mask']
            replay = sample['replay']
            for d in range(1, 32):
                slice_image = image[:, :, d]
                slice_mask  = mask[:, :, d]
                sample = A.ReplayCompose.replay(replay, image=slice_image, mask=slice_mask)
                augmented_images[:, :, :, d] = sample["image"]
                augmented_masks[:, :, :, d]  = sample["mask"]
        
        else:
            # preprocessing
            image = np.expand_dims(image, axis=0)  # C, H, W, D
            mask  = np.expand_dims(mask,  axis=0)  # C, H, W, D
            
            # augmentation
            augmented_images = torch.empty(image.shape, dtype=torch.float32)
            augmented_masks  = torch.empty(mask.shape, dtype=torch.float32)
            first_image_slice = image[0, :, :, 0]
            first_mask_slice  = mask[0, :, :, 0]
            sample = self.transforms(image=first_image_slice, mask=first_mask_slice, pad_loc=('pad_top', 'pad_bottom', 'pad_left', 'pad_right'))
            augmented_images[:, :, :, 0] = sample['image']
            augmented_masks[:, :, :, 0]  = sample['mask']
            replay = sample['replay']
            for d in range(1, image.shape[-1]):
                slice_image = image[0, :, :, d]
                slice_mask  = mask[0, :, :, d]
                sample = A.ReplayCompose.replay(replay, image=slice_image, mask=slice_mask)
                augmented_images[:, :, :, d] = sample["image"]
                augmented_masks[:, :, :, d]  = sample["mask"]

        # label
        # BG = 0, B = 1, M = 2
        if mask.sum().item() > 0:
            if label_name == 'B':
                label = torch.tensor(1).long()
            else: 
                label = torch.tensor(2).long()
        else :
            label = torch.tensor(0).long()

        return augmented_images.float(), augmented_masks.float(), label.long()


# Define the DataLoader
def get_dataloader(name, batch_size, num_workers):
    # 3D - 2D transfer
    if name == 'coreline_dataset_3d_2dtransfer':
        test_dataset = HEMO_3D_CLS_Dataset_2Dtransfer()
        print("TEST [Total] number = ", len(test_dataset))
        data_loader   = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=pad_collate_fn_512)

    return data_loader

