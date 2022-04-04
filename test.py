import os
from pathlib import Path
import argparse
import datetime
import numpy as np
import time
import torch
import json
import random
# import functools

import utils
from create_model import create_model
from create_datasets.prepare_datasets import build_test_dataset
from engine import *
from losses import Uptask_Loss, Downtask_Loss


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
    parser.add_argument('--test-dataset-name', default="Custom", type=str, help='test dataset name')    
    parser.add_argument('--slice-wise-manner', type=str2bool, default="True", help='stacking slices like slice-wise manner for patient-level testing for Upstream')
    
    # Model parameters
    parser.add_argument('--model-name', default='SMART_Net', type=str, help='model name')

    # DataLoader setting
    parser.add_argument('--num-workers', default=10, type=int)
    parser.add_argument('--pin-mem',    action='store_true', default=False, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    # Setting Upstream, Downstream task
    parser.add_argument('--training-stream', default='Upstream', choices=['Upstream', 'Downstream'], type=str, help='training stream')  

    # DataParrel or Single GPU train
    parser.add_argument('--multi-gpu-mode',       default='DataParallel', choices=['DataParallel', 'Single'], type=str, help='multi-gpu-mode')          
    parser.add_argument('--device',               default='cuda', help='device to use for training / testing')
    parser.add_argument('--cuda-device-order',    default='PCI_BUS_ID', type=str, help='cuda_device_order')
    parser.add_argument('--cuda-visible-devices', default='0', type=str, help='cuda_visible_devices')

    # Continue Training
    parser.add_argument('--resume', default='',  help='resume from checkpoint')  # '' = None

    # Validation setting
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')

    # Prediction and Save setting
    parser.add_argument('--output-dir', default='', help='path where to save, empty for no saving')

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
           
    utils.print_args_test(args)
    device = torch.device(args.device)

    print("Loading dataset ....")
    dataset_test, collate_fn_test = build_test_dataset(args=args)
    
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_mem, drop_last=False, collate_fn=collate_fn_test)


    # Select Loss
    if args.training_stream == 'Upstream':
        criterion = Uptask_Loss(name=args.model_name)
    else :
        criterion = Downtask_Loss(name=args.model_name)


    # Select Model
    print(f"Creating model  : {args.model_name}")
    model = create_model(stream=args.training_stream, name=args.model_name)
    print(model)


    # Resume
    if args.resume:
        print("Loading... Resume")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])        

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



    # Multi GPU
    if args.multi_gpu_mode == 'DataParallel':
        model = torch.nn.DataParallel(model)
        model.to(device)
    elif args.multi_gpu_mode == 'Single':
        model.to(device)
    else :
        raise Exception('Error...! args.multi_gpu_mode')    


    start_time = time.time()


    # TEST
    if args.training_stream == 'Upstream':

        if args.model_name == 'Up_SMART_Net':
            if args.slice_wise_manner:
                test_stats = test_Up_SMART_Net(model, criterion, data_loader_test, device, args.print_freq, 1)
            else :
                test_stats = test_Up_SMART_Net_Patient_Level(model, criterion, data_loader_test, device, args.print_freq, 1)
                

        ## Dual    
        elif args.model_name == 'Up_SMART_Net_Dual_CLS_SEG':
            test_stats = test_Up_SMART_Net_Dual_CLS_SEG(model, criterion, data_loader_test, device, args.print_freq, 1)
        elif args.model_name == 'Up_SMART_Net_Dual_CLS_REC':
            test_stats = test_Up_SMART_Net_Dual_CLS_REC(model, criterion, data_loader_test, device, args.print_freq, 1)
        elif args.model_name == 'Up_SMART_Net_Dual_SEG_REC':
            test_stats = test_Up_SMART_Net_Dual_SEG_REC(model, criterion, data_loader_test, device, args.print_freq, 1)

        ## Single
        elif args.model_name == 'Up_SMART_Net_Single_CLS':
            test_stats = test_Up_SMART_Net_Single_CLS(model, criterion, data_loader_test, device, args.print_freq, 1)
        elif args.model_name == 'Up_SMART_Net_Single_SEG':
            test_stats = test_Up_SMART_Net_Single_SEG(model, criterion, data_loader_test, device, args.print_freq, 1)
        elif args.model_name == 'Up_SMART_Net_Single_REC':
            test_stats = test_Up_SMART_Net_Single_REC(model, criterion, data_loader_test, device, args.print_freq, 1)

        else : 
            raise KeyError("Wrong model name `{}`".format(args.model_name))     

    elif args.training_stream == 'Downstream':

        if args.model_name == 'Down_SMART_Net_CLS':
            test_stats = test_Down_SMART_Net_CLS(model, criterion, data_loader_test, device, args.print_freq, 1)
        elif args.model_name == 'Down_SMART_Net_SEG':
            test_stats = test_Down_SMART_Net_SEG(model, criterion, data_loader_test, device, args.print_freq, 1)
        else :
            raise KeyError("Wrong model name `{}`".format(args.model_name))     
    
    else :
        raise KeyError("Wrong training stream `{}`".format(args.training_stream))        



    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
    
    if args.output_dir:
        with open(args.output_dir + "/test_log.txt", "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    print("Averaged test_stats: ", test_stats)

    # Finish
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('TEST time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('SMART-Net Framework training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    os.environ["CUDA_DEVICE_ORDER"]     =  args.cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"]  =  args.cuda_visible_devices        
    
    main(args)
