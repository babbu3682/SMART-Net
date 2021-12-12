import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
import skimage
from pydicom import dcmread
import albumentations as albu
from sklearn.model_selection import train_test_split
from skimage.morphology import disk, binary_dilation

from monai.transforms import *
from monai.data import Dataset
import volumentations
import albumentations as albu
import re
import glob
import cv2
import functools


import warnings
warnings.filterwarnings(action='ignore') 

label_csv = pd.read_csv('/workspace/sunggu/1.Hemorrhage/dataset/Orbital_Wall_Fracture/label.csv')

def get_label(x):
    name = x.split('/')[-2]
    return label_csv[label_csv['inspect_num'] == int(name)]['brain_injury_label'].values

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
    # 'morphology.remove_small_objects()' is better!!

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

def Preprocessing_inference(x):
    image = x['image'].squeeze(0)
    
    image = clahe_slice_wise(image)
        
    x['image'] = np.expand_dims(image, axis=0)
    
    return x

def Albu_2D_Transform(x):
    '''
    Albumentation follow numpy shape and must be 2D
    (H, W, C)
    '''
    image = x['image'].squeeze(0) 
    mask  = x['label'].squeeze(0)
    
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
         )
         ,])
    
    augment = Trans(image=image, mask=mask)
    image = augment['image']
    mask  = augment['mask']
    
    x['image'] = np.expand_dims(image, axis = 0)
    x['label'] = np.expand_dims(mask, axis = 0)
    
    return x

def RandCrop_individual(x):
    z = x['image'].shape[-1]    
    x = RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(224, 224, z), 
                               pos=1.0, neg=0.0, num_samples=1, image_key="image", image_threshold=0)(x)        
    return x

def Resized_individual(x):
    z = x['image'].shape[-1]    
    x = Resized(keys=["image"], spatial_size=(256, 256, z), mode=['trilinear'], align_corners=True)(x)
    x = Resized(keys=["label"], spatial_size=(256, 256, z), mode=['nearest'],   align_corners=None)(x)
    
    return x

def Resized_individual_inference(x):
    z = x['image'].shape[-1]    
    x = Resized(keys=["image"], spatial_size=(256, 256, z), mode=['trilinear'], align_corners=True)(x)
    
    return x
######################################################################################################################################################################
######################################################                    Down Task --- collate_fn            ########################################################
######################################################################################################################################################################

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

def pad_collate_fn_inference(batches):

    if isinstance(batches[0], (list, tuple)):
        X          = [ batch[0]['image'] for batch in batches ]
        Y          = torch.FloatTensor([ batch[0]['label'] for batch in batches ])
        img_list   = [ batch[0]['image_meta_dict']['filename_or_obj'] for batch in batches ]
        
    else : 
        X          = [ batch['image'] for batch in batches ]
        Y          = torch.FloatTensor([ batch['label'] for batch in batches ])
        img_list   = [ batch['image_meta_dict']['filename_or_obj'] for batch in batches ] 

    z_shapes = torch.IntTensor([x.shape[-1] for x in X])
    
    pad_image = []
    pad_label = []
    
    for img, label in zip(X, Y):
        
        if (z_shapes.max() - img.shape[3] != 0):
            pad = torch.zeros( (img.shape[0], img.shape[1], img.shape[2], z_shapes.max()-img.shape[3]) )
            pad_image.append(torch.cat([img, pad], dim=-1))
        else :
            pad_image.append(img)
            
    batch = dict()
    batch['img_path']    = img_list
    batch['image']       = torch.stack(pad_image, dim=0)
    batch['label']       = Y
    batch['z_shape']     = z_shapes

    
    return batch

def default_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


######################################################################################################################################################################
######################################################                    UP Task                             ########################################################
######################################################################################################################################################################

def Hemo_Uptask_Dataset(mode):
    if mode == 'train':
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Train_nii/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Train_nii/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]        

        print("Train [Total]  number = ", len(img_list))
        print("Train [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Train [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                # Common preprocessing
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),                

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,
                
                ##### Preprocessing
                Sampling_Z_axis_volumentation,
                Preprocessing,
                
                ##### Augmentation
                Albu_2D_Transform,
                RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),        
                RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),  
                
                ToTensord(keys=["image", "label"]),
            ]
        )        

    else :
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Valid [Total]  number = ", len(img_list))
        print("Valid [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Valid [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),                

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,     
                
                ##### Preprocessing
                Preprocessing,
                
                ToTensord(keys=["image", "label"]),
            ]
        )


    return Dataset(data=data_dicts, transform=transforms), default_collate_fn



######################################################################################################################################################################
######################################################                    UP Task  --Recon--                  ########################################################
######################################################################################################################################################################

from ModelsGenesis.pytorch.utils_model_genesis import *
### We only consider 2d slice-wise processing due to the large thinkness ###
def make_large_mask(label):
    cluster = skimage.measure.label(label)
    empty = np.zeros_like(label)
    x, y  = np.where(cluster!=0)
    
    if (x.sum() == 0) and (y.sum() == 0):
        return empty
    
    else:
        for i in np.unique(cluster)[1:]:
            x, y  = np.where(cluster==i)
            empty[ np.max([0, x.min()-20]):np.min([256, x.max()+21]), np.max([0, y.min()-20]):np.min([256, y.max()+21])] = 1.0
    
    return empty



def Model_Genesis_distortImages(x, recon_option):
    image = x['image'].squeeze(0)
    label = x['label'].squeeze(0)
    
    # Local Shuffle Pixel
    distort = np.stack([local_pixel_shuffling(image[..., i], prob=0.7) for i in range(image.shape[-1])], axis=-1).astype('float32')
    
    # Apply non-Linear transformation with an assigned probability
    distort = np.stack([nonlinear_transformation(distort[..., i], prob=0.7) for i in range(distort.shape[-1])], axis=-1).astype('float32')
    
    # Inpainting & Outpainting
    if random.random() < 0.9:  # paint_rate
        if random.random() < 0.7:  # inpaint_rate and outpaint_rate
            # Inpainting
            distort = np.stack([image_in_painting(distort[..., i]) for i in range(distort.shape[-1])], axis=-1).astype('float32')
            
        else:
            # Outpainting
            distort = np.stack([image_out_painting(distort[..., i]) for i in range(distort.shape[-1])], axis=-1).astype('float32') 
    

    if recon_option != 'original_model_genesis':
        if label.sum() > 0:
            tmp_mask = np.stack( [make_large_mask(label[..., i]) for i in range(label.shape[-1])], axis=-1)      
            distort[np.where(tmp_mask)] = image[np.where(tmp_mask)] 
    
    x['distort'] = np.expand_dims(distort, axis=0)

    return x

    # distort = np.stack([local_pixel_shuffling(image[..., i], prob=0.5) for i in range(image.shape[-1])], axis=-1).astype('float32')
    # distort = np.stack([nonlinear_transformation(distort[..., i], prob=0.5) for i in range(distort.shape[-1])], axis=-1).astype('float32')

    # if random.random() < 0.9:  # paint_rate 0.9
    # if random.random() < 0.2:  # inpaint_rate and outpaint_rate 0.2

    # if recon_option == 'except_mask_all':
    #     tmp_mask = np.stack( [binary_dilation(label[..., i], disk(15, dtype=bool)) for i in range(label.shape[-1])], axis=-1)      
    #     distort[np.where(tmp_mask)] = image[np.where(tmp_mask)] 
    #     x['distort'] = np.expand_dims(distort, axis=0)
    
    # else :
    #     tmp_mask_big   = np.stack( [binary_dilation(label[..., i], disk(15, dtype=bool)) for i in range(label.shape[-1])], axis=-1)      
    #     tmp_mask_small = np.stack( [binary_dilation(label[..., i], disk(5,  dtype=bool)) for i in range(label.shape[-1])], axis=-1)      
    #     tmp_mask = tmp_mask_big ^ tmp_mask_small  # ^ means -
    #     image[np.where(tmp_mask)] = distort[np.where(tmp_mask)]
    #     x['distort'] = np.expand_dims(image, axis=0)

    # return x   

def Hemo_Uptask_Recon_Dataset(mode, recon_option):
    if mode == 'train':
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Train_nii/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Train_nii/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]        

        print("Train [Total]  number = ", len(img_list))
        print("Train [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Train [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                # Common preprocessing
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,
                
                ##### Preprocessing
                Sampling_Z_axis_volumentation,
                Preprocessing,
                
                ##### Augmentation
                Albu_2D_Transform,
                RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),        
                RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),  
                functools.partial(Model_Genesis_distortImages, recon_option=recon_option),

                ToTensord(keys=["image", "label", "distort"]),
            ]
        )        

    else :
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Valid [Total]  number = ", len(img_list))
        print("Valid [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Valid [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,     
                
                ##### Preprocessing
                Preprocessing,
                functools.partial(Model_Genesis_distortImages, recon_option=recon_option),
                
                ToTensord(keys=["image", "label", "distort"]),
            ]
        )


    return Dataset(data=data_dicts, transform=transforms), default_collate_fn









######################################################################################################################################################################
######################################################                    Down Task                           ########################################################
######################################################################################################################################################################

def Hemo_Downtask_Dataset(mode):
    if mode == 'train':
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Train_nii/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Train_nii/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]        

        print("Train [Total]  number = ", len(img_list))
        print("Train [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Train [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                # Common preprocessing
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,
                
                ##### Preprocessing
                Preprocessing,
                
                ##### Augmentation
                Albu_2D_Transform,
                RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),        
                RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),  
                
                ToTensord(keys=["image", "label"]),
            ]
        )        

    else :
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Valid [Total]  number = ", len(img_list))
        print("Valid [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Valid [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,
  
                ##### Preprocessing
                Preprocessing,
                
                ToTensord(keys=["image", "label"]),
            ]
        )


    return Dataset(data=data_dicts, transform=transforms), pad_collate_fn




def Hemo_TEST_Dataset(test_data_name, recon_option):
    if test_data_name == 'Asan':
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Asan_internal/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Asan_internal/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Test [Total]  number = ", len(img_list))
        print("Test [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Test [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),                

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,

                ##### Preprocessing
                Preprocessing,
                # SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),   # for only previous 3d seg unet careful...!!!
                
                ToTensord(keys=["image", "label"]),
            ]
        )

    elif test_data_name == 'Pohang':
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Pohang_external/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Pohang_external/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Test [Total]  number = ", len(img_list))
        print("Test [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Test [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,

                ##### Preprocessing
                Preprocessing,
                # SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),   # for only previous 3d seg unet careful...!!!
                
                ToTensord(keys=["image", "label"]),
            ]
        )

    elif test_data_name == 'Ulji':
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Ulji_external/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Ulji_external/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Test [Total]  number = ", len(img_list))
        print("Test [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Test [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),                

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,

                ##### Preprocessing
                Preprocessing,
                # SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),   # for only previous 3d seg unet careful...!!!
                
                ToTensord(keys=["image", "label"]),
            ]
        )

    elif test_data_name == 'Physionet':
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Physionet_external/Images/*.nii"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Physionet_external/Labels/*.nii"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Test [Total]  number = ", len(img_list))

        transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,

                ##### Preprocessing
                Preprocessing,
                # SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),   # for only previous 3d seg unet careful...!!!
                
                ToTensord(keys=["image", "label"]),
            ]
        )

    elif test_data_name == 'Valid':
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Valid_nii/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Test [Total]  number = ", len(img_list))
        print("Test [Hemo]   number = ", len([i for i in img_list if "_hemo_" in i]))
        print("Test [Normal] number = ", len([i for i in img_list if "_normal_" in i]))

        transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,

                ##### Preprocessing
                Preprocessing,
                # SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),   # for only previous 3d seg unet careful...!!!
                
                ToTensord(keys=["image", "label"]),
            ]
        )


    elif test_data_name == 'Orbital_Wall_Fracture':
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Orbital_Wall_Fracture/nifti/*/*.nii.gz"))
        label_list   = list(map(get_label, img_list))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Test [Total]  number = ", len(img_list))

        transforms = Compose(
            [
                LoadNiftid(keys=["image"]),
                AddChanneld(keys=["image"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image"], spatial_axis=1),
                Rotate90d(keys=["image"], k=1, spatial_axes=(0, 1)),

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual_inference,

                ##### Preprocessing
                Preprocessing_inference,
                # SpatialPadd(keys=["image", "label"], spatial_size=(256, 256, 64)),   # for only previous 3d seg unet careful...!!!
                
                ToTensord(keys=["image"]),
            ]
        )

        return Dataset(data=data_dicts, transform=transforms), pad_collate_fn_inference

    else :
        img_list     = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Asan_internal/*_img.nii.gz"))
        label_list   = list_sort_nicely(glob.glob("/workspace/sunggu/1.Hemorrhage/dataset/Test_nii/Asan_internal/*_mask.nii.gz"))
        data_dicts   = [{"image": image_name, "label": label_name} for image_name, label_name in zip(img_list, label_list)]

        print("Test [Total]  number = ", len(img_list))

        transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                # Orientationd(keys=["image", "label"], axcodes="PLS"),
                Flipd(keys=["image", "label"], spatial_axis=1),
                Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 1)),

                ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=80.0, b_min=0.0, b_max=1.0, clip=True), 
                Resized_individual,     
                
                ##### Preprocessing
                Preprocessing,
                functools.partial(Model_Genesis_distortImages, recon_option=recon_option),
                
                ToTensord(keys=["image", "label", "distort"]),
            ]
        )

        return Dataset(data=data_dicts, transform=transforms), default_collate_fn
                  


    return Dataset(data=data_dicts, transform=transforms), pad_collate_fn
