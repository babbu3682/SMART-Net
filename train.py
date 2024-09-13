import os
import argparse
import datetime
import numpy as np
import time
import torch
import json
import random
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

import utils
from dataset import get_dataloader
from models import get_model
from schedulers import get_scheduler
from losses import get_loss
from optimizers import get_optimizer
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
    parser.add_argument('--train-batch-size',       default=72, type=int)
    parser.add_argument('--valid-batch-size',       default=72, type=int)
    parser.add_argument('--train-num-workers',      default=10, type=int)
    parser.add_argument('--valid-num-workers',      default=10, type=int)
    
    # Model parameters
    parser.add_argument('--model',                     default='Unet',  type=str, help='model name')    
    parser.add_argument('--transfer-pretrained',       default=None,    type=str, help='encoder-pretrained')    
    parser.add_argument('--use-pretrained-encoder',    default=True,    type=bool, help='model name')    
    parser.add_argument('--use-pretrained-decoder',    default=True,    type=bool, help='model name')    
    parser.add_argument('--freeze-encoder',            default=True,    type=bool, help='model name')    
    parser.add_argument('--freeze-decoder',            default=True,    type=bool, help='model name')    
    parser.add_argument('--roi_size',                  default=512,     type=int, help='model name')
    parser.add_argument('--sw_batch_size',             default=32,      type=int, help='model name')    
    parser.add_argument('--backbone',                  default='resnet-50',  type=str, choices=['resnet-50', 'efficientnet-b7', 'maxvit-xlarge'], help='model name')    
    parser.add_argument('--use_skip',                  default=True,    type=bool, help='model name')
    parser.add_argument('--use_consist',               default=True,    type=bool, help='model name')
    parser.add_argument('--pool_type',                 default='gem',   type=str, help='model name')
    parser.add_argument('--operator_3d',               default='LSTM',  type=str, help='model name')

    # Loss parameters
    parser.add_argument('--loss',             default='dice_loss',  type=str, help='loss name')

    # Training parameters - Optimizer, LR, Scheduler, Epoch
    parser.add_argument('--optimizer',        default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "AdamW"')
    parser.add_argument('--scheduler',        default='poly_lr', type=str, metavar='scheduler', help='scheduler (default: "poly_learning_rate"')
    parser.add_argument('--epochs',           default=1000, type=int, help='Upstream 1000 epochs, Downstream 500 epochs')  
    parser.add_argument('--lr',               default=5e-4, type=float, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--min-lr',           default=1e-5, type=float, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs',    default=10, type=int, metavar='N', help='epochs to warmup LR, if scheduler supports')    

    # Continue Training (Resume)
    parser.add_argument('--from-pretrained',  default='',  help='pre-trained from checkpoint')
    parser.add_argument('--resume',           default='',  help='resume from checkpoint')  # '' = None

    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode',   default='DataParallel', choices=['DataParallel', 'Single'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',           default='cuda', help='device to use for training / testing')

    # Save setting
    parser.add_argument('--save-dir',         default='', help='path where to prediction PNG save')
    parser.add_argument('--memo',             default='', help='memo for script')
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
import torch

# MAIN
def main(args):
    print(torch.__version__)
    print(torch.backends.cudnn.version())    
    start_epoch = 0
    utils.print_args(args)
    device = torch.device(args.device)

    # Dataset
    train_loader = get_dataloader(name=args.dataset, mode='train', batch_size=args.train_batch_size, num_workers=args.train_num_workers, roi_size=args.roi_size, operator_3d=args.operator_3d)
    valid_loader = get_dataloader(name=args.dataset, mode='valid', batch_size=args.valid_batch_size, num_workers=args.valid_num_workers, roi_size=args.roi_size, operator_3d=args.operator_3d)

    # Model
    model = get_model(args)

    # Pretrained
    if args.from_pretrained:
        print("Loading... Pretrained")
        checkpoint = torch.load(args.from_pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Multi-GPU & CUDA
    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model)         
        model = model.to(device)
    else :
        model = model.to(device)

    # Optimizer & LR Schedule & Loss
    optimizer = get_optimizer(name=args.optimizer, model=model, lr=args.lr)
    scheduler = get_scheduler(name=args.scheduler, optimizer=optimizer, warm_up_epoch=10, start_decay_epoch=args.epochs/10, total_epoch=args.epochs, min_lr=1e-6)
    criterion = get_loss(name=args.loss)

    # Resume
    if args.resume:
        print("Loading... Resume")
        start_epoch, model, optimizer, scheduler = utils.load_checkpoint(model, optimizer, scheduler, filename=args.resume)

    # Tensorboard
    tensorboard = SummaryWriter(args.save_dir + '/runs')

    # Etc traing setting
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()    

    # Whole Loop Train & Valid 
    for epoch in range(start_epoch, args.epochs):

        # 2D
        if args.model == 'SMART-Net-2D':
            train_stats = train_smartnet_2d(train_loader, model, criterion, optimizer, device, epoch, args.use_consist)
            print("Averaged train_stats: ", train_stats)
            for key, value in train_stats.items():
                tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)            
            valid_stats = valid_smartnet_2d(valid_loader, model, device, epoch, args.save_dir, args.use_consist)
            print("Averaged valid_stats: ", valid_stats)
            for key, value in valid_stats.items():
                tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)  

        # 3D - 2D transfer
        elif args.model == 'SMART-Net-3D-CLS':
            train_stats = train_smartnet_3d_2dtransfer_CLS(train_loader, model, criterion, optimizer, device, epoch)
            print("Averaged train_stats: ", train_stats)
            for key, value in train_stats.items():
                tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)            
            valid_stats = valid_smartnet_3d_2dtransfer_CLS(valid_loader, model, device, epoch, args.save_dir)
            print("Averaged valid_stats: ", valid_stats)
            for key, value in valid_stats.items():
                tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)                      

        elif args.model == 'SMART-Net-3D-SEG':
            train_stats = train_smartnet_3d_2dtransfer_SEG(train_loader, model, criterion, optimizer, device, epoch)
            print("Averaged train_stats: ", train_stats)
            for key, value in train_stats.items():
                tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)            
            valid_stats = valid_smartnet_3d_2dtransfer_SEG(valid_loader, model, device, epoch, args.save_dir)
            print("Averaged valid_stats: ", valid_stats)
            for key, value in valid_stats.items():
                tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)                                      

        # LR update
        scheduler.step()

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if args.multi_gpu_mode == 'DataParallel' else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, args.save_dir + '/weights/epoch_' + str(epoch) + '_checkpoint.pth')

        # Log text
        log_stats = {**{f'{k}': v for k, v in train_stats.items()}, 
                    **{f'{k}': v for k, v in valid_stats.items()}, 
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr']}

        with open(args.save_dir + "/log.txt", "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    # Finish
    tensorboard.close()
    total_time_str = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train script', parents=[get_args_parser()])
    args = parser.parse_args()


    # Make folder if not exist
    os.makedirs(args.save_dir, exist_ok =True)
    os.makedirs(args.save_dir + "/args", exist_ok =True)
    os.makedirs(args.save_dir + "/weights", exist_ok =True)
    os.makedirs(args.save_dir + "/predictions", exist_ok =True)
    os.makedirs(args.save_dir + "/runs", exist_ok =True)

    # Save args to json
    the_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
    if not os.path.isfile(args.save_dir + "/args/args_" + the_time + ".json"):
        with open(args.save_dir + "/args/args_" + the_time + ".json", "w") as f:
            json.dump(args.__dict__, f, indent=2)
       
    main(args)