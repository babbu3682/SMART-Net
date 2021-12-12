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
from datasets.prepare_datasets import build_dataset
from engine import *
from losses import Uptask_Loss, Downtask_Loss


def lambda_rule(epoch, warm_up_epoch, start_decay_epoch, total_epoch, min_lr):
    # Linear WarmUP
    if (epoch < warm_up_epoch):
        return max(0, epoch / warm_up_epoch)
    else :
        lr = 1.0 - max(0, epoch - start_decay_epoch) /(float(total_epoch) - start_decay_epoch)

        if lr <= min_lr:
            lr = min_lr

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

    # Dataset parameters
    parser.add_argument('--backbone-name', default='resnet50', type=str, help='backbone-name')

    parser.add_argument('--test-name', default='Downstream_3d_seg_model1', type=str, help='test name')    
    parser.add_argument('--data-set', default='CIFAR10', type=str, help='dataset name')    
    parser.add_argument('--model-name', default='CLSTM_Previous_first', type=str, help='model name')
    parser.add_argument('--uncert',    type=str2bool, default="FALSE", help='Uncert loss')
    parser.add_argument('--progressive-transfer',    type=str2bool, default="FALSE", help='progressive_transfer_learning')
    parser.add_argument('--decoder-transfer',    type=str2bool, default="FALSE", help='decoder_transfer_learning')
    parser.add_argument('--freeze-decoder',    type=str2bool, default="FALSE", help='freeze_decoder_learning')

    # DataLoader setting
    parser.add_argument('--batch-size',  default=72, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem',    action='store_true', default=False, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    # parser.set_defaults(pin_mem=True) # 뭔가 default로 설정하는 느낌?

    # Model parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--recon-option', default='except_mask_all', type=str, help='if you use recon task, select recon option')

    # Optimizer parameters
    parser.add_argument('--optimizer', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--optimizer-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--optimizer-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    
    # Learning rate and schedule and Epoch parameters
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')

    ## if training-mode is Upstream, Downstream
    parser.add_argument('--training-stream', default='Upstream', choices=['Upstream', 'Downstream'], type=str, help='training stream')  
    parser.add_argument('--training-mode', default='Upstream', choices=['only_cls', 'only_seg', 'only_recon_ae', 'cls+seg', 'cls+recon_ae', 'seg+recon_ae', 'cls+seg+consist', 'cls+seg+recon_ae', 'cls+seg+recon_ae+consist', 'cls+seg+recon_skip+consist', 'cls+seg+recon_skip+consist+patch_model_genesis', 'cls+seg+recon_ae+two_consist', 'model_genesis_skip', 
                                                                        '3d_cls', '3d_seg', 'other_cls', 'other_seg'], type=str, help='training mode')  

    # Distributed or DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode', default='DataParallel', choices=['DataParallel', 'DistributedDataParallel', 'Single'], type=str, help='multi-gpu-mode')          

    # Continue Training
    parser.add_argument('--resume',     default='',   help='resume from checkpoint')  # '' = None
    parser.add_argument('--pretrained', default='',   help='pretrained from checkpoint')
    parser.add_argument('--end2end',    type=str2bool, default="FALSE", help='Downtask option end2end')

    # Validation setting
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--validate-every', default=2, type=int, help='validate and save the checkpoints every n epochs')  


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Prediction and Save setting
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)
    

    # GPU setting 
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    # parser.add_argument('--gpus', default='0', type=str, help='Gpu index')  

    return parser

    # TO DO
    # parser.add_argument('--multiprocessing-distributed', action='store_true', help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. '
    #                                                                             'This is the fastest way to use PyTorch for either single node or multi node data parallel training')
    # parser.add_argument('--dist-url', default='tcp://192.168.45.53:3684', type=str, help='url used to set up distributed training')
    # parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    



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
    
    if (args.multi_gpu_mode):
        utils.init_distributed_mode(args)
       
    utils.print_args(args)
    device = torch.device(args.device)

    print("Loading dataset ....")
    dataset_train, collate_fn_train = build_dataset(is_train=True,  args=args)   
    dataset_valid, collate_fn_valid = build_dataset(is_train=False, args=args)
    
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,  pin_memory=args.pin_mem, drop_last=True,  collate_fn=collate_fn_train)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1,               num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=False, collate_fn=collate_fn_valid)

    # Select Model & Loss
    if args.training_stream == 'Upstream':
        criterion = Uptask_Loss(mode=args.training_mode, uncert=args.uncert)

    else :
        criterion = Downtask_Loss(mode=args.training_mode)


    print(f"Creating model  : {args.model_name}")
    print(f"Pretrained model: {args.pretrained}")
    # model = create_model(name=model_name, pretrained=args.pretrained, end2end=args.end2end)
    model = create_model(name=args.model_name, pretrained=args.pretrained, end2end=args.end2end, backbone_name=args.backbone_name, decoder_transfer=args.decoder_transfer, freeze_decoder=args.freeze_decoder)

    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        print("Loading... Resume")
        args.start_epoch = checkpoint['epoch'] + 1  # for finetuning!!!

        if 'best_metric' in checkpoint:
            print("Epoch: ", checkpoint['epoch'], " Best Metric ==> ", checkpoint['best_metric'])

    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model)
        model.to(device)

    # elif args.multi_gpu_mode == 'DistributedDataParallel':
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # model_without_ddp = model.module
        # model.to(device)
    
    else :
        model.to(device)


    if args.uncert:
        params       = list(model.parameters()) + [criterion.cls_weight] + [criterion.seg_weight] + [criterion.consist_weight] 
        optimizer    = torch.optim.Adam(params=params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else :
        optimizer    = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=functools.partial(lambda_rule, warm_up_epoch=args.warmup_epochs, start_decay_epoch=args.epochs/10, total_epoch=args.epochs, min_lr=args.min_lr))            

    if args.resume:  # for finetuning!!!
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


    output_dir = Path(args.output_dir)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_epoch = best_metric = best_metric1 = best_metric2 = 0.0
    best_mse = 10000


    # Whole LOOP
    for epoch in range(args.start_epoch, args.epochs):

        # Train
        if args.training_stream == 'Upstream':

            if (args.training_mode == 'cls+seg+rec+consist'):
                train_stats = train_Uptask_CLS_SEG_REC(model, criterion, data_loader_train, optimizer, device, epoch)               
            else : 
                print("Please Select train mode...! [only_cls, only_seg, mtl, mtl+consist, mtl+consist+recon]")

        else:
            if args.training_mode == '3d_cls':
                train_stats = train_Downtask_3dCls(model, criterion, data_loader_train, optimizer, device, epoch)
            
            elif args.training_mode == '3d_seg':
                train_stats = train_Downtask_3dSeg(model, criterion, data_loader_train, optimizer, device, epoch, args.progressive_transfer)

            
        # Valid
        if epoch % args.validate_every == 0:
            if args.training_stream == 'Upstream':
                if (args.training_mode == 'cls+seg+rec+consist'):
                    valid_stats = valid_Uptask_CLS_SEG_REC(model, criterion, data_loader_valid, device)
                    print(f"DICE of the network on the  {len(dataset_valid)} valid images: {valid_stats['Dice']:.3f}")
                    print(f"AUC  of the network on the  {len(dataset_valid)} valid images: {valid_stats['AUC']:.3f}")
                    print(f"F1   of the network on the  {len(dataset_valid)} valid images: {valid_stats['F1']:.3f}")

                    if (valid_stats["AUC"] + valid_stats["Dice"] + valid_stats["F1"]) / 3.0 > best_metric :    
                        best_metric1 = valid_stats["Dice"]
                        best_metric2 = valid_stats["AUC"]
                        best_metric3 = valid_stats["F1"]
                        best_metric  = (best_metric1 + best_metric2 + best_metric3) / 3.0
                        best_metric_epoch = epoch
                    
                    print(f'Max DICE: {best_metric1:.3f}')
                    print(f'Max AUC : {best_metric2:.3f}')
                    print(f'Max F1  : {best_metric3:.3f}')
                    print(f'Best Epoch: {best_metric_epoch:.3f}')  

                else :
                    print("Please Select train mode...! [only_cls, only_seg, mtl, mtl+consist, mtl+consist+recon]")

            else:
                if args.training_mode == '3d_cls':
                    valid_stats = valid_Downtask_3dCls(model, criterion, data_loader_valid, device)
                    print(f"AUC of the network on the {len(dataset_valid)} valid images: {valid_stats['AUC']:.3f}%")                
                    if valid_stats["AUC"] > best_metric1 :    
                        best_metric1 = valid_stats["AUC"]
                        best_metric = best_metric1
                        best_metric_epoch = epoch  
                    print(f'Max AUC: {best_metric:.3f}')
                    print(f'Best Epoch: {best_metric_epoch:.3f}')               

                elif args.training_mode == '3d_seg':
                    valid_stats = valid_Downtask_3dSeg(model, criterion, data_loader_valid, device)
                    print(f"DICE of the network on the {len(dataset_valid)} valid images: {valid_stats['Dice']:.3f}%")                
                    if valid_stats["Dice"] > best_metric1 :    
                        best_metric1 = valid_stats["Dice"]
                        best_metric = best_metric1
                        best_metric_epoch = epoch         
                    print(f'Max Dice: {best_metric:.3f}')    
                    print(f'Best Epoch: {best_metric_epoch:.3f}')     
                    

            # Save & Prediction png
            if epoch % args.validate_every == 0:
                save_name = 'epoch_' + str(epoch) + '_checkpoint.pth'
                checkpoint_paths = [output_dir / str(save_name)]

                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model_state_dict': model.state_dict() if args.multi_gpu_mode == 'Single' else model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'best_metric': best_metric,
                        'args': args,
                    }, checkpoint_path)

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
    parser = argparse.ArgumentParser('Sunggu training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    
    main(args)
