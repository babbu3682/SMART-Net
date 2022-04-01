import os
import sys
# sys.path.append(os.path.abspath('/workspace/sunggu'))
# sys.path.append(os.path.abspath('/workspace/sunggu/MONAI'))
# sys.path.append(os.path.abspath('/workspace/sunggu/1.Hemorrhage'))
# sys.path.append(os.path.abspath('/workspace/sunggu/1.Hemorrhage/utils/FINAL_utils'))
from pathlib import Path
import argparse
import datetime
import numpy as np
import time
import torch
import json
import random
import functools


import utils
from create_model import create_model
from create_datasets.prepare_datasets import build_dataset
from engine import *
from losses import Uptask_Loss, Downtask_Loss
from optimizers import create_optim
from lr_schedulers import create_scheduler

# pretrained_dict = {k: v for k, v in check_point['model_state_dict'].items() if k in model_dict and 'encoder.' in k} # if문에 있는 param만 가져오기
# model_dict.update(pretrained_dict) 
# model.load_state_dict(model_dict)     


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
    parser.add_argument('--data-folder-dir', default="/workspace/sunggu/1.Hemorrhage/SMART-Net/datasets/samples", type=str, help='dataset folder dirname')    
    
    # Model parameters
    parser.add_argument('--model-name', default='SMART_Net', type=str, help='model name')

    # DataLoader setting
    parser.add_argument('--batch-size',  default=20, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem',    action='store_true', default=False, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Optimizer parameters
    parser.add_argument('--optimizer', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    
    # Learning rate and schedule and Epoch parameters
    parser.add_argument('--lr-scheduler', default='poly_lr', type=str, metavar='lr_scheduler', help='lr_scheduler (default: "poly_learning_rate"')
    parser.add_argument('--epochs', default=1000, type=int, help='Upstream 1000 epochs, Downstream 500 epochs')  
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    ## Setting Upstream, Downstream task
    parser.add_argument('--training-stream', default='Upstream', choices=['Upstream', 'Downstream'], type=str, help='training stream')  

    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode', default='DataParallel', choices=['DataParallel', 'Single'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',         default='cuda', help='device to use for training / testing')
    parser.add_argument('--cuda-device-order', default='PCI_BUS_ID', type=str, help='cuda_device_order')
    parser.add_argument('--cuda-visible-devices', default='0', type=str, help='cuda_visible_devices')

    # Option
    parser.add_argument('--gradual-unfreeze',    type=str2bool, default="TRUE", help='gradual unfreezing the encoder for Downstream Task')

    # Continue Training
    parser.add_argument('--resume',     default='',   help='resume from checkpoint')  # '' = None
    parser.add_argument('--pretrained', default='',   help='pretrained from checkpoint')
    parser.add_argument('--end2end',    type=str2bool, default="FALSE", help='Downtask option end2end')

    # Validation setting
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')

    # Prediction and Save setting
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)

    return parser


# Fix random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def main(args):
           
    utils.print_args(args)
    device = torch.device(args.device)

    print("Loading dataset ....")
    dataset_train, collate_fn_train = build_dataset(is_train=True,  args=args)   
    dataset_valid, collate_fn_valid = build_dataset(is_train=False, args=args)
    
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,  pin_memory=args.pin_mem, drop_last=True,  collate_fn=collate_fn_train)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1,               num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=False, collate_fn=collate_fn_valid)

    # Select Loss
    if args.training_stream == 'Upstream':
        criterion = Uptask_Loss(name=args.model_name)
    else :
        criterion = Downtask_Loss(name=args.model_name)

    # Select Model
    print(f"Creating model  : {args.model_name}")
    print(f"Pretrained model: {args.pretrained}")
    model = create_model(stream=args.training_stream, name=args.model_name)
    print(model)

    # Optimizer & LR Scheduler
    optimizer    = create_optim(name=args.optimizer, model=model, args=args)
    lr_scheduler = create_scheduler(name=args.lr_scheduler, optimizer=optimizer, args=args)


    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])        
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])        
        args.start_epoch = checkpoint['epoch'] + 1  
        try:
            log_path = os.path.dirname(args.resume)+'/log.txt'
            lines    = open(log_path,'r').readlines()
            val_loss_list = []
            for l in lines:
                exec('log_dict='+l.replace('NaN', '0'))
                val_loss_list.append(log_dict['valid_loss'])
            print("Epoch: ", np.argmin(val_loss_list), " Minimum Val Loss ==> ", np.min(val_loss_list))
        except:
            pass

        # Optimizer Error fix...!
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()


    # Pre-trained and End-to-End
    if args.from_pretrained:
        print("Loading... Pre-trained")       
        print("Check Before weight = ", model.state_dict()['encoder.conv1.weight'][0])
        checkpoint = torch.load(args.resume, map_location='cpu')
        if args.load_weight_type == 'full':
            model.load_state_dict(checkpoint['model_state_dict'])   
        elif args.load_weight_type == 'encoder':
            model.encoder.load_state_dict(checkpoint['model_state_dict'])   
        print("Check After weight = ", model.state_dict()['encoder.conv1.weight'][0])



    # Multi GPU
    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model)
        model.to(device)
    elif args.multi_gpu_mode == 'Single':
        model.to(device)
    else :
        raise Exception('Error...! args.multi_gpu_mode')    


    output_dir = Path(args.output_dir)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # Whole LOOP
    for epoch in range(args.start_epoch, args.epochs):

        # Train & Valid
        if args.training_stream == 'Upstream':

            if args.model_name == 'SMART_Net':
                train_stats = train_Up_SMART_Net(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                valid_stats = valid_Up_SMART_Net(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
            
            ## Dual    
            elif args.model_name == 'Up_SMART_Net_Dual_CLS_SEG':
                train_stats = train_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                valid_stats = valid_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
            elif args.model_name == 'Up_SMART_Net_Dual_CLS_REC':
                train_stats = train_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                valid_stats = valid_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
            elif args.model_name == 'Up_SMART_Net_Dual_SEG_REC':
                train_stats = train_Up_SMART_Net_Dual_SEG_REC(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                valid_stats = valid_Up_SMART_Net_Dual_SEG_REC(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)

            ## Single
            elif args.model_name == 'Up_SMART_Net_Single_CLS':
                train_stats = train_Up_SMART_Net_Single_CLS(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                valid_stats = valid_Up_SMART_Net_Single_CLS(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
            elif args.model_name == 'Up_SMART_Net_Single_SEG':
                train_stats = train_Up_SMART_Net_Single_SEG(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                valid_stats = valid_Up_SMART_Net_Single_SEG(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
            elif args.model_name == 'Up_SMART_Net_Single_REC':
                train_stats = train_Up_SMART_Net_Single_REC(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size)
                valid_stats = valid_Up_SMART_Net_Single_REC(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
            else : 
                raise KeyError("Wrong model name `{}`".format(args.model_name))     

        elif args.training_stream == 'Downstream':

            if args.model_name == 'Down_SMART_Net_CLS':
                train_stats = train_Down_SMART_Net_CLS(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size, args.gradual_unfreeze)
                valid_stats = valid_Down_SMART_Net_CLS(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
            elif args.model_name == 'Down_SMART_Net_SEG':
                train_stats = train_Down_SMART_Net_SEG(model, criterion, data_loader_train, optimizer, device, epoch, args.print_freq, args.batch_size, args.gradual_unfreeze)
                valid_stats = valid_Down_SMART_Net_SEG(model, criterion, data_loader_valid, device, args.print_freq, args.batch_size)
            else :
                raise KeyError("Wrong model name `{}`".format(args.model_name))     
        else :
            raise KeyError("Wrong training stream `{}`".format(args.training_stream))        



    # Save & Prediction png
        checkpoint_paths = output_dir + '/epoch_'+str(epoch)+'_checkpoint.pth'
        torch.save({
            'model_state_dict': model.state_dict() if args.multi_gpu_mode == 'Single' else model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_paths)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'valid_{k}': v for k, v in valid_stats.items()},
                    'epoch': epoch}
        
        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        lr_scheduler.step(epoch)


    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('SMART-Net Framework training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    os.environ["CUDA_DEVICE_ORDER"]     =  args.cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"]  =  args.cuda_visible_devices        
    
    main(args)
