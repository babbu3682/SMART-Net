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
from accelerate.utils import gather_object
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs


def get_args_parser():
    parser = argparse.ArgumentParser('SMART-Net Framework Train script', add_help=False)

    # Dataset parameters
    parser.add_argument('--dataset',           default='ldctiqa',  type=str, help='Name of the dataset to be used for training and validation (e.g., "ldctiqa")')    
    parser.add_argument('--train-batch-size',  default=72, type=int, help='Batch size for training data')
    parser.add_argument('--valid-batch-size',  default=72, type=int, help='Batch size for validation data')
    parser.add_argument('--train-num-workers', default=10, type=int, help='Number of workers for loading training data')
    parser.add_argument('--valid-num-workers', default=10, type=int, help='Number of workers for loading validation data')
    
    # Model parameters
    parser.add_argument('--model',                     default='Unet',  type=str, help='Model architecture to be used (e.g., "Unet")')    
    parser.add_argument('--transfer-pretrained',       default=None,    type=str, help='Path to a pre-trained model for transfer learning')    
    parser.add_argument('--use-pretrained-encoder',    default=True,    type=bool, help='Whether to use a pre-trained encoder (True or False)')    
    parser.add_argument('--use-pretrained-decoder',    default=True,    type=bool, help='Whether to use a pre-trained decoder (True or False)')    
    parser.add_argument('--freeze-encoder',            default=True,    type=bool, help='Whether to freeze encoder layers during training (True or False)')    
    parser.add_argument('--freeze-decoder',            default=True,    type=bool, help='Whether to freeze decoder layers during training (True or False)')    
    parser.add_argument('--roi_size',                  default=512,     type=int, help='Region of interest size for input images (e.g., 512)')    
    parser.add_argument('--sw_batch_size',             default=32,      type=int, help='Sliding window batch size for model inference')    
    parser.add_argument('--backbone',                  default='resnet-50',  type=str, choices=['resnet-50', 'efficientnet-b7', 'maxvit-small', 'maxvit-xlarge'], help='Backbone model for feature extraction (e.g., "resnet-50")')    
    parser.add_argument('--use_skip',                  default=True,    type=bool, help='Whether to use skip connections in the model (True or False)')
    parser.add_argument('--use_consist',               default=True,    type=bool, help='Whether to apply consistency regularization (True or False)')
    parser.add_argument('--pool_type',                 default='gem',   type=str, help='Type of pooling to use in the model (e.g., "gem" for generalized mean pooling)')
    parser.add_argument('--operator_3d',               default='lstm',  type=str, choices=['lstm', 'bert', '3d_cnn', '3d_vit'], help='3D operator to be used in the model (e.g., "LSTM")')

    # Loss parameters
    parser.add_argument('--loss',             default='dice_loss',  type=str, help='Loss function to be used during training (e.g., "dice_loss")')

    # Training parameters - Optimizer, LR, Scheduler, Epoch
    parser.add_argument('--optimizer',        default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer for training (e.g., "adamw")')
    parser.add_argument('--scheduler',        default='poly_lr', type=str, metavar='scheduler', help='Learning rate scheduler (e.g., "poly_lr" for polynomial learning rate)')
    parser.add_argument('--epochs',           default=1000, type=int, help='Number of epochs for training (e.g., 1000 for upstream training)')
    parser.add_argument('--lr',               default=5e-4, type=float, metavar='LR', help='Initial learning rate (default: 5e-4)')
    parser.add_argument('--min-lr',           default=1e-5, type=float, metavar='LR', help='Minimum learning rate for the scheduler (default: 1e-5)')
    parser.add_argument('--warmup-epochs',    default=10, type=int, metavar='N', help='Number of warmup epochs for learning rate')

    # Continue Training (Resume)
    parser.add_argument('--from-pretrained',  default='',  help='Path to pre-trained model checkpoint')
    parser.add_argument('--resume',           default='',  help='Resume training from a checkpoint')

    # DataParallel or Single GPU train
    parser.add_argument('--multi-gpu',        default='DataParallel', choices=['DDP', 'DataParallel', 'Single'], type=str, help='Mode for multi-GPU training (e.g., "DataParallel")')          
    parser.add_argument('--device',           default='cuda', help='Device to be used for training and testing (e.g., "cuda" for GPU)')

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

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    start_epoch = 0
    
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

    # Optimizer & LR Schedule & Loss
    optimizer = get_optimizer(name=args.optimizer, model=model, lr=args.lr)
    scheduler = get_scheduler(name=args.scheduler, optimizer=optimizer, warm_up_epoch=10, start_decay_epoch=args.epochs/10, total_epoch=args.epochs, min_lr=1e-6)
    criterion = get_loss(name=args.loss)

    # Resume
    if args.resume:
        print("Loading... Resume")
        start_epoch, model, optimizer, scheduler = utils.load_checkpoint(model, optimizer, scheduler, filename=args.resume)

    # Multi-GPU & CUDA
    if args.multi_gpu == 'DataParallel':
        device = torch.device(args.device)
        model  = torch.nn.DataParallel(model)
        model  = model.to(device)
    elif args.multi_gpu == 'DDP':
        device      = None
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        # accelerator.init_trackers("SMART-Net")
        train_loader, valid_loader, model, optimizer = accelerator.prepare(train_loader, 
                                                                           valid_loader, 
                                                                           model, 
                                                                           optimizer)
    elif args.multi_gpu == 'Single':
        model = model.to(device)

    # Tensorboard
    tensorboard = SummaryWriter(args.save_dir + '/runs')
    
    if accelerator.is_main_process and args.multi_gpu == 'DDP':
        print(torch.__version__)
        print(torch.backends.cudnn.version())    
        utils.print_args(args)
        print(f"Start training for {args.epochs} epochs")

    # Whole Loop Train & Valid 
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):

        # 2D
        if args.model == 'SMART-Net-2D':
            accelerator.wait_for_everyone()
            train_stats = train_smartnet_2d(train_loader, model, criterion, optimizer, device, epoch, args.use_consist, args.multi_gpu, accelerator)
            if accelerator.is_main_process and args.multi_gpu == 'DDP':
                print("Averaged train_stats: ", train_stats)
                for key, value in train_stats.items():
                    tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)
            accelerator.wait_for_everyone()
            valid_stats = valid_smartnet_2d(valid_loader, model, device, epoch, args.save_dir, args.use_consist, args.multi_gpu, accelerator)
            if accelerator.is_main_process and args.multi_gpu == 'DDP':
                print("Averaged valid_stats: ", valid_stats)
                for key, value in valid_stats.items():
                    tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)
        
        # 3D - 2D transfer
        elif args.model == 'SMART-Net-3D-CLS':
            train_stats = train_smartnet_3d_2dtransfer_CLS(train_loader, model, criterion, optimizer, device, epoch, args.multi_gpu, accelerator)
            if accelerator.is_main_process and args.multi_gpu == 'DDP':
                print("Averaged train_stats: ", train_stats)
                for key, value in train_stats.items():
                    tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)
            valid_stats = valid_smartnet_3d_2dtransfer_CLS(valid_loader, model, device, epoch, args.save_dir, args.multi_gpu, accelerator)
            if accelerator.is_main_process and args.multi_gpu == 'DDP':
                print("Averaged valid_stats: ", valid_stats)
                for key, value in valid_stats.items():
                    tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)

        elif args.model == 'SMART-Net-3D-SEG':
            train_stats = train_smartnet_3d_2dtransfer_SEG(train_loader, model, criterion, optimizer, device, epoch, args.multi_gpu, accelerator)
            print("Averaged train_stats: ", train_stats)
            for key, value in train_stats.items():
                tensorboard.add_scalar(f'Train Stats/{key}', value, epoch)            
            valid_stats = valid_smartnet_3d_2dtransfer_SEG(valid_loader, model, device, epoch, args.save_dir, args.multi_gpu, accelerator)
            print("Averaged valid_stats: ", valid_stats)
            for key, value in valid_stats.items():
                tensorboard.add_scalar(f'Valid Stats/{key}', value, epoch)                

        # LR update
        scheduler.step()

        # Save checkpoint
        if args.multi_gpu == 'Single':
            model_state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, args.save_dir + '/weights/epoch_' + str(epoch) + '_checkpoint.pth')
        
        elif args.multi_gpu == 'DataParallel':
            model_state_dict = model.module.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, args.save_dir + '/weights/epoch_' + str(epoch) + '_checkpoint.pth')
        
        elif args.multi_gpu == 'DDP':
            accelerator.wait_for_everyone()
            accelerator.save({
                'epoch': epoch,
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, args.save_dir + '/weights/epoch_' + str(epoch) + '_checkpoint.pth')

        else:
            raise ValueError(f"Invalid multi_gpu mode: {args.multi_gpu}")

        if accelerator.is_main_process and args.multi_gpu == 'DDP':
            # Log text
            log_stats = {**{f'{k}': v for k, v in train_stats.items()}, 
                        **{f'{k}': v for k, v in valid_stats.items()}, 
                        'epoch': epoch,
                        'lr': optimizer.param_groups[0]['lr']}

            with open(args.save_dir + "/logs/log_" + args.time + ".txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    # Finish
    tensorboard.close()
    total_time_str = str(datetime.timedelta(seconds=int(time.time()-start_time)))
    print('Training time {}'.format(total_time_str))
    accelerator.end_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train script', parents=[get_args_parser()])
    args = parser.parse_args()


    # Make folder if not exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + "/args", exist_ok=True)
    os.makedirs(args.save_dir + "/weights", exist_ok=True)
    os.makedirs(args.save_dir + "/predictions", exist_ok=True)
    os.makedirs(args.save_dir + "/runs", exist_ok=True)
    os.makedirs(args.save_dir + "/logs", exist_ok=True)

    # Save args to json
    the_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
    if not os.path.isfile(args.save_dir + "/args/args_" + the_time + ".json"):
        with open(args.save_dir + "/args/args_" + the_time + ".json", "w") as f:
            json.dump(args.__dict__, f, indent=2)

    args.time = the_time
       
    main(args)