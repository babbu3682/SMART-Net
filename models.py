import torch 
import torch.nn as nn
import torch.nn.functional as F

from arch.our_network import *

def get_model(args):
    
    # 2D: encoder with MTL
    if args.model == 'SMART-Net-2D':
        model = SMART_Net_2D(backbone=args.backbone, use_skip=args.use_skip, pool_type=args.pool_type, use_consist=args.use_consist, roi_size=args.roi_size)
    
    # 3D-CLS: 3D operator w/ 2D encoder
    elif args.model == 'SMART-Net-3D-CLS':
        model = SMART_Net_3D_CLS(transfer_pretrained=args.transfer_pretrained, 
                                 use_pretrained_encoder=args.use_pretrained_encoder,
                                 freeze_encoder=args.freeze_encoder, 
                                 roi_size=args.roi_size, 
                                 sw_batch_size=args.sw_batch_size, 
                                 spatial_dim=0, 
                                 backbone=args.backbone, 
                                 use_skip=args.use_skip, 
                                 pool_type=args.pool_type, 
                                 operator_3d=args.operator_3d)

    # 3D-SEG: 3D operator w/ 2D encoder
    elif args.model == 'SMART-Net-3D-SEG':
        model = SMART_Net_3D_SEG(transfer_pretrained=args.transfer_pretrained, 
                                 use_pretrained_encoder=args.use_pretrained_encoder,
                                 use_pretrained_decoder=args.use_pretrained_decoder,
                                 freeze_encoder=args.freeze_encoder,
                                 freeze_decoder=args.freeze_decoder,
                                 roi_size=args.roi_size, 
                                 sw_batch_size=args.sw_batch_size, 
                                 spatial_dim=0, 
                                 backbone=args.backbone, 
                                 use_skip=args.use_skip, 
                                 pool_type=args.pool_type, 
                                 operator_3d=args.operator_3d)
        

    # print number of learnable parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model
