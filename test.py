import os
import sys
# sys.path.append(os.path.abspath('/workspace/sunggu'))
# sys.path.append(os.path.abspath('/workspace/sunggu/MONAI'))
# sys.path.append(os.path.abspath('/workspace/sunggu/1.Hemorrhage'))
# sys.path.append(os.path.abspath('/workspace/sunggu/1.Hemorrhage/utils/FINAL_utils'))

import monai
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import random
import functools

from pathlib import Path

import utils
from create_model import create_model
from datasets.prepare_datasets import build_test_dataset
from engine import *
from losses import Uptask_Loss, Downtask_Loss


def lambda_rule(epoch, warm_up_epoch, start_decay_epoch, total_epoch):
    # Linear WarmUP
    if (epoch < warm_up_epoch):
        return max(0, epoch / warm_up_epoch)
    else :
        lr = 1.0 - max(0, epoch - start_decay_epoch) /(float(total_epoch) - start_decay_epoch)
    return lr

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Sunggu Deeplearning Train and Test script', add_help=False)

    # test name
    parser.add_argument('--backbone-name', default='resnet50', type=str, help='backbone-name')

    parser.add_argument('--test-name', default='Downstream_3d_seg_model1', type=str, help='test name')    
    parser.add_argument('--model-name', default='CLSTM_Previous_first', type=str, help='model name')
    parser.add_argument('--engine-option', default='previous', type=str, help='engine option')    
    parser.add_argument('--save-path', default='/workspace/sunggu/1.Hemorrhage/scripts study/predictions/Paper/', type=str, help='save path name')    
    parser.add_argument('--progressive-transfer',    type=str2bool, default="FALSE", help='progressive_transfer_learning')
    parser.add_argument('--decoder-transfer',    type=str2bool, default="FALSE", help='decoder_transfer_learning')
    parser.add_argument('--freeze-decoder',    type=str2bool, default="FALSE", help='freeze_decoder_learning')

    # Dataset parameters
    parser.add_argument('--data-set', default='CIFAR10', type=str, help='dataset name')    
    parser.add_argument('--test-data-name', default='Asan', type=str, help='test_data_name name')    

    # DataLoader setting
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem',    action='store_true', default=False, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Model parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--recon-option', default='except_mask_all', type=str, help='if you use recon task, select recon option')

    ## if training-mode is Upstream, Downstream
    parser.add_argument('--training-stream', default='Upstream', choices=['Upstream', 'Downstream'], type=str, help='training stream')  
    parser.add_argument('--training-mode', default='Upstream', choices=['only_cls', 'only_seg', 'only_recon_ae', 'cls+seg', 'cls+recon_ae', 'seg+recon_ae', 'cls+seg+consist', 'cls+seg+recon_ae', 'cls+seg+recon_ae+consist', 'cls+seg+recon_skip+consist', 'cls+seg+recon_skip+consist+patch_model_genesis', 'cls+seg+recon_ae+two_consist', 'model_genesis_skip', 
                                                                        '3d_cls', '3d_seg', 'other_cls', 'other_seg'], type=str, help='training mode')  

    # Continue Training
    parser.add_argument('--resume',     default='',   help='resume from checkpoint')  # '' = None
    parser.add_argument('--pretrained', default='',   help='pretrained from checkpoint')
    parser.add_argument('--end2end',    type=str2bool, default="FALSE", help='Downtask option end2end')

    # Validation setting
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--validate-every', default=2, type=int, help='validate and save the checkpoints every n epochs')  

    # Prediction and Save setting
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)

    # GPU setting 
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    return parser



# fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def main(args):
       
    device = torch.device(args.device)

    print("Loading dataset ....")
    dataset_valid, collate_fn_valid = build_test_dataset(args=args)
    
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=False, collate_fn=collate_fn_valid)

    # Select Model & Loss
    if args.training_stream == 'Upstream':
        criterion = Uptask_Loss(mode=args.training_mode)

    else :
        criterion = Downtask_Loss(mode=args.training_mode)


    print(f"Creating model: {args.model_name}")
    model = create_model(name=args.model_name, pretrained=args.pretrained, end2end=args.end2end, backbone_name=args.backbone_name, decoder_transfer=args.decoder_transfer, freeze_decoder=args.freeze_decoder)

    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        print("Loading... Resume")
        print("check == ", model.encoder.conv1.weight[0])
        if 'best_metric' in checkpoint:
            print("Epoch: ", checkpoint['epoch'], " Best Metric ==> ", checkpoint['best_metric'])

    model.to(device)        
    start_time = time.time()
    best_metric = best_metric1 = best_metric2 = 0.0

    # Test
    if args.training_stream == 'Upstream':
        if (args.training_mode == 'cls+seg+rec+consist'):
            valid_stats = test_Uptask_CLS_SEG_REC(model, criterion, data_loader_valid, device, test_name=args.test_name, save_path=args.save_path)  
            print(f"DICE of the network on the {len(dataset_valid)} valid images: {valid_stats['Dice']:.3f}")
            print(f"AUC of the network on the {len(dataset_valid)} valid images: {valid_stats['AUC']:.3f}")
            best_metric1  = max(best_metric1, valid_stats["Dice"])
            best_metric2  = max(best_metric2, valid_stats["AUC"])
            print(f'Max Dice: {best_metric1:.3f}, Max AUC: {best_metric2:.3f}')     
            best_metric = (best_metric2 + best_metric1) / 2.0

        else :
            print("Please Select train mode...! [only_cls, only_seg, mtl, mtl+consist, mtl+consist+recon]")

    # 핵심
    else:
        if args.training_mode == '3d_cls':
            valid_stats = test_Downtask_3dCls(model, criterion, data_loader_valid, device, test_name=args.test_name, save_path=args.save_path)            

            print(f"AUC of the network on the {len(dataset_valid)} valid images: {valid_stats['AUC']:.3f}%")
            best_metric1 = max(best_metric1, valid_stats["AUC"])
            print(f'Max AUC: {best_metric1:.3f}')
            best_metric = best_metric1

        elif args.training_mode == '3d_seg':
            valid_stats = test_Downtask_3dSeg(model, criterion, data_loader_valid, device, test_name=args.test_name, save_path=args.save_path)            

            print(f"DICE of the network on the {len(dataset_valid)} valid images: {valid_stats['Dice']:.3f}%")
            best_metric1 = max(best_metric1, valid_stats["Dice"])
            print(f'Max Dice: {best_metric1:.3f}')
            best_metric = best_metric1


    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sunggu Test script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    main(args)

