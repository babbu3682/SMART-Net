import os
import argparse
import datetime
import numpy as np
import time
import torch
import json
import random

import utils
from dataset import get_dataloader
from models import test_get_model
from engine import *


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('SMART-Net Framework Train and Test script', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset',          default='ldctiqa',  type=str, help='dataset name')    
    parser.add_argument('--test-batch-size',       default=72, type=int)
    parser.add_argument('--test-num_workers',      default=10, type=int)
    
    # Model parameters
    parser.add_argument('--model',            default='Unet',  type=str, help='model name')    

    # Continue Training (Resume)
    parser.add_argument('--resume',           default='',  help='resume from checkpoint')  # '' = None

    # Single GPU train
    parser.add_argument('--device',           default='cuda', help='device to use for training / testing')

    # Save setting
    parser.add_argument('--checkpoint-dir',   default='', help='path where to save checkpoint or output')
    parser.add_argument('--save-dir',         default='', help='path where to prediction PNG save')
    parser.add_argument('--epoch',            default=10, type=int)
    parser.add_argument('--memo',             default='', help='memo for script')

    parser.add_argument('--T',                default=1.0, type=float)
    return parser


# fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
np.random.seed(random_seed)
random.seed(random_seed)

# MAIN
def main(args):
    utils.print_args_test(args)
    device = torch.device(args.device)

    # Dataset
    test_loader = get_dataloader(name=args.dataset, batch_size=args.test_batch_size, num_workers=args.test_num_workers)

    # Model
    model = test_get_model(name=args.model)

    # GPU & CUDA
    model = model.to(device)

    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint  = torch.load(args.resume, map_location='cpu')
        checkpoint['model_state_dict'] = {k.replace('module.', ''):v for k,v in checkpoint['model_state_dict'].items()} # fix loading multi-gpu 
        model.load_state_dict(checkpoint['model_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # Etc traing setting
    print(f"Start training for {args.epoch} epoch")
    start_time = time.time()    

    # 3D - 2D transfer
    if args.model == 'EfficientNetB7_LSTM' or args.model == 'MaxViT_LSTM':
        test_stats = test_smartnet_3d_2dtransfer_CLS(test_loader, model, device, args.save_dir, args.T)
        print("Averaged test_stats: ", test_stats)

    # Log text
    log_stats = {**{f'{k}': v for k, v in test_stats.items()}, 
                'epoch': args.epoch}

    with open(args.checkpoint_dir + "/test_log.txt", "a") as f:
        f.write(json.dumps(log_stats) + "\n")

    # Finish
    total_time_str = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print('TEST time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
        
    # Make folder if not exist
    os.makedirs(args.checkpoint_dir + "/args", exist_ok =True)
    os.makedirs(args.save_dir, exist_ok =True)

    # Save args to json
    if not os.path.isfile(args.checkpoint_dir + "/args/test_args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json"):
        with open(args.checkpoint_dir + "/args/test_args_" + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".json", "w") as f:
            json.dump(args.__dict__, f, indent=2)
       
    main(args)