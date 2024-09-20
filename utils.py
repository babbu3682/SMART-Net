"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
from collections import defaultdict, OrderedDict
import os
import torch
import transformers

# Please make a code for AverageMeter. All indicators and losses are stored in dictionary form. Track a series of values and provide access to smoothed values over a window or the global series average.
from collections import defaultdict
import logging


class AverageMeter:
    def __init__(self, **kwargs):
        self.reset()

    def reset(self):
        self.data = defaultdict(lambda: {'sum': 0, 'count': 0})

    def update(self, key, value, n):
        self.data[key]['sum']   += value * n
        self.data[key]['count'] += n
    
    def average(self):
        return {k: v['sum'] / v['count'] for k, v in self.data.items()}

# Check the resume point
def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth'):
    start_epoch = 0
    best_loss   = 1000

    if os.path.isfile(filename):
        checkpoint  = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))

    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return start_epoch, model, optimizer, scheduler


def fix_optimizer(optimizer):
    # Optimizer Error fix...!
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

def str2bool(value):
    value = value.lower()
    if value in ['true', '1', 'yes', 'y', 'on']:
        return True
    elif value in ['false', '0', 'no', 'n', 'off']:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")

def check_checkpoint_if_wrapper(model_state_dict):
    if list(model_state_dict.keys())[0].startswith('module'):
        return OrderedDict({k.replace('module.', ''): v for k, v in model_state_dict.items()}) # 'module.' 제거
    else:
        return model_state_dict



def print_args(model_args, data_args, training_args):
    print('***********************************************')
    print('Dataset Name:                   ', data_args.dataset)
    print('---------- Model --------------')
    print('Model Name:                     ', model_args.model)
    print('Resume From:                    ', model_args.resume)
    print('Save To:                        ', training_args.output_dir)
    print('Available CPUs:                 ', os.cpu_count())
    print('---------- Optimizer ----------')
    print('Optimizer Name:                 ', training_args.optim)
    print('Learning Rate:                  ', training_args.learning_rate)
    print('Scheduler Name:                 ', training_args.lr_scheduler_type)
    print('Train Batchsize per device:     ', training_args.per_device_train_batch_size)
    print('Valid Batchsize per device:     ', training_args.per_device_eval_batch_size)
    print('Total Epoch:                    ', training_args.num_train_epochs)
    print('Torch version:                  ', torch.__version__)
    print('Torch cudnn version:            ', torch.backends.cudnn.version())    



def print_args_test(args):
    print('***********************************************')
    print('Dataset Name:   ', args.dataset)
    print('---------- Model --------------')
    print('Model Name:     ', args.model)
    print('Resume From:    ', args.resume)
    print('Save To:        ', args.save_dir)
    print('Available CPUs: ', os.cpu_count())
