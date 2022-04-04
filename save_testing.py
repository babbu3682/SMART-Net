# import glob
# from monai.transforms import *




# image_list = sorted(glob.glob('/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/Images/*.nii'))
# label_list = sorted(glob.glob('/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/Labels/*.nii'))



# image_saver = SaveImage(output_dir='/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/new_image', 
#                 output_postfix='Image', 
#                 output_ext='.nii.gz', 
#                 resample=True, 
#                 mode='bilinear', 
#                 squeeze_end_dims=True, 
#                 data_root_dir='', 
#                 separate_folder=False, 
#                 print_log=True)

# label_saver = SaveImage(output_dir='/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/new_label', 
#                 output_postfix='label', 
#                 output_ext='.nii.gz', 
#                 resample=True, 
#                 mode='nearest', 
#                 squeeze_end_dims=True, 
#                 data_root_dir='', 
#                 separate_folder=False, 
#                 print_log=True)



# for image_p, label_p in zip(image_list, label_list):

#     # Load nii data
#     image_p, img_meta = LoadImage()(image_p)
#     image_p = AddChannel()(image_p)
    
#     label_p, label_meta = LoadImage()(label_p)
#     label_p = AddChannel()(label_p)
    
#     if label_p.max() == 0:
#         img_meta['filename_or_obj']   = img_meta['filename_or_obj'].replace('.nii', '_normal_img.nii')
#         label_meta['filename_or_obj'] = label_meta['filename_or_obj'].replace('.nii', '_normal_mask.nii')
#     else :
#         img_meta['filename_or_obj']   = img_meta['filename_or_obj'].replace('.nii', '_hemo_img.nii')
#         label_meta['filename_or_obj'] = label_meta['filename_or_obj'].replace('.nii', '_hemo_mask.nii')
                
#     image_saver(image_p, img_meta)    # Note: image should be channel-first shape: [C,H,W,[D]].
#     label_saver(label_p.astype('bool').astype('float'), label_meta)    # Note: image should be channel-first shape: [C,H,W,[D]].
