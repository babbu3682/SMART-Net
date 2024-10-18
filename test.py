import os
import argparse
import datetime
import numpy as np
import time
import torch
import json
import random

import utils
from dataset import get_dataloader_test
from models import get_model
from engine import *

def get_args_parser():
    parser = argparse.ArgumentParser('SMART-Net Framework Test script', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset',               default='ldctiqa',  type=str, help='dataset name')    
    parser.add_argument('--test-batch-size',       default=72, type=int)
    parser.add_argument('--test-num_workers',      default=10, type=int)
    
    # Model parameters
    parser.add_argument('--model',                  default='Unet',  type=str, help='model name')    
    parser.add_argument('--transfer-pretrained',    default=None,    type=str, help='Path to a pre-trained model for transfer learning')    
    parser.add_argument('--use-pretrained-encoder', default=True,    type=bool, help='Whether to use a pre-trained encoder (True or False)')    
    parser.add_argument('--use-pretrained-decoder', default=True,    type=bool, help='Whether to use a pre-trained decoder (True or False)')    
    parser.add_argument('--freeze-encoder',         default=True,    type=bool, help='Whether to freeze encoder layers during training (True or False)')    
    parser.add_argument('--freeze-decoder',         default=True,    type=bool, help='Whether to freeze decoder layers during training (True or False)')        
    parser.add_argument('--roi_size',               default=512,     type=int, help='Region of interest size for input images (e.g., 512)')    
    parser.add_argument('--sw_batch_size',          default=32,      type=int, help='Sliding window batch size for model inference')    
    parser.add_argument('--backbone',               default='resnet-50',  type=str, choices=['resnet-50', 'efficientnet-b7', 'maxvit-small', 'maxvit-xlarge'], help='Backbone model for feature extraction (e.g., "resnet-50")')    
    parser.add_argument('--use_skip',               default=True,    type=bool, help='Whether to use skip connections in the model (True or False)')
    parser.add_argument('--use_consist',            default=True,    type=bool, help='Whether to apply consistency regularization (True or False)')
    parser.add_argument('--pool_type',              default='gem',   type=str, help='Type of pooling to use in the model (e.g., "gem" for generalized mean pooling)')
    parser.add_argument('--operator_3d',            default='LSTM',  type=str, help='3D operator to be used in the model (e.g., "LSTM")')

    # Continue Training (Resume)
    parser.add_argument('--resume',           default='',  help='Resume training from a checkpoint')

    # Single GPU train
    parser.add_argument('--multi-gpu',        default='DataParallel', choices=['DDP', 'DataParallel', 'Single'], type=str, help='Mode for multi-GPU training (e.g., "DataParallel")')
    parser.add_argument('--device',           default='cuda', help='device to use for training / testing')
    
    # Save setting
    parser.add_argument('--save-dir',         default='', help='Directory where prediction outputs (e.g., PNG files) will be saved')
    parser.add_argument('--time',             default='', help='for log')
    parser.add_argument('--memo',             default='', help='Additional notes or comments for the script')

    return parser


# fix random seeds for reproducibility
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



# MAIN
def main(args):
    seed_everything()

    utils.print_args_test(args)
    device = torch.device(args.device)

    # Dataset
    test_loader = get_dataloader_test(name=args.dataset, batch_size=args.test_batch_size, num_workers=args.test_num_workers, roi_size=args.roi_size, operator_3d=args.operator_3d)

    # Model
    model = get_model(args)

    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint  = torch.load(args.resume, map_location='cpu')
        checkpoint['model_state_dict'] = {k.replace('module.', ''):v for k,v in checkpoint['model_state_dict'].items()} # fix loading multi-gpu 
        model.load_state_dict(checkpoint['model_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        target_epoch = checkpoint['epoch']

    # GPU & CUDA
    model = model.to(device)

    # Etc traing setting
    print(f"Start Test for {target_epoch} epoch")
    start_time = time.time()    

    # 2D
    if args.model == 'SMART-Net-2D':
        test_stats = test_smartnet_2d(test_loader, model, device, target_epoch, args.save_dir, args.use_consist)
        print("Averaged test_stats: ", test_stats)

    # 3D - CLS
    elif args.model == 'SMART-Net-3D-CLS':
        test_stats = test_smartnet_3d_2dtransfer_CLS(test_loader, model, device, target_epoch, args.save_dir)
        print("Averaged test_stats: ", test_stats)

    # Log text
    log_stats = {**{f'{k}': v for k, v in test_stats.items()}, 
                'epoch': target_epoch}

    with open(args.save_dir + "/logs/test_log_" + args.time + ".txt", "a") as f:
        f.write(json.dumps(log_stats) + "\n")

    # Finish
    total_time_str = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print('TEST time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
        
    # Make folder if not exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + "/args", exist_ok=True)
    os.makedirs(args.save_dir + "/logs", exist_ok=True)

    # Save args to json    
    the_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
    if not os.path.isfile(args.save_dir + "/args/test_args_" + the_time + ".json"):
        with open(args.save_dir + "/args/test_args_" + the_time + ".json", "w") as f:
            json.dump(args.__dict__, f, indent=2)

    args.time = the_time

    main(args)