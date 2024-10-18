"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""
from collections import defaultdict, OrderedDict
import os
import torch


# Please make a code for AverageMeter. All indicators and losses are stored in dictionary form. Track a series of values and provide access to smoothed values over a window or the global series average.
from collections import defaultdict
import matplotlib.pyplot as plt


# Setting...!
fn_denorm  = lambda x: (x * 0.5) + 0.5
fn_tonumpy = lambda x: x.detach().cpu().numpy()


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

def check_2d_data_device(multi_gpu, device, image, label, mask):
    if multi_gpu == 'DDP':
        image = image
        label = label
        mask  = mask
    else:
        image = image.to(device).float()
        label = label.to(device).float()
        mask  = mask.to(device).float()
    return image, label, mask

def check_3d_cls_data_device(multi_gpu, device, image, label):
    if multi_gpu == 'DDP':
        image = image
        label = label
    else:
        image = image.to(device).float()
        label = label.to(device).float()
    return image, label

def check_3d_seg_data_device(multi_gpu, device, image, mask):
    if multi_gpu == 'DDP':
        image = image
        mask  = mask
    else:
        image = image.to(device).float()
        mask  = mask.to(device).float()
    return image, mask

def collect_dict_all_processor(processor, average=True):
    merged_dict = defaultdict(list)
    for i in processor:
        for k, v in i.items():
            merged_dict[k].append(v)
    # Average
    if average:
        for k, v in merged_dict.items():
            merged_dict[k] = sum(v) / len(v)
    return merged_dict

def save_image_and_prediction(image, mask, pred_seg, pred_rec, save_dir, epoch, index=0):
    # SAVE SEG
    image_png = fn_tonumpy(fn_denorm(mask[index]))  # B, C, H, W
    plt.imsave(f'{save_dir}/predictions/epoch_{epoch}_mask.png', image_png[0], cmap='gray')

    pred_png = fn_tonumpy(fn_denorm(pred_seg[index])).round()
    plt.imsave(f'{save_dir}/predictions/epoch_{epoch}_pred_seg.png', pred_png[0], cmap='gray')

    # SAVE REC
    image_png = fn_tonumpy(fn_denorm(image[index]))  # B, C, H, W
    plt.imsave(f'{save_dir}/predictions/epoch_{epoch}_image.png', image_png[0], cmap='gray')

    pred_png = fn_tonumpy(fn_denorm(pred_rec[index]))
    plt.imsave(f'{save_dir}/predictions/epoch_{epoch}_pred_rec.png', pred_png[0], cmap='gray')


def print_args(args):
    print('***********************************************')
    print('Dataset Name:   ', args.dataset)
    print('---------- Model --------------')
    print('Model Name:     ', args.model)
    print('Resume From:    ', args.resume)
    print('Save To:        ', args.save_dir)
    print('Available CPUs: ', os.cpu_count())
    print('---------- Loss ---------------')
    print('Loss Name:      ', args.loss)
    print('---------- Optimizer ----------')
    print('Optimizer Name: ', args.optimizer)
    print('Learning Rate:  ', args.lr)
    print('Scheduler Name: ', args.scheduler)
    print('Train Batchsize:      ', args.train_batch_size)
    print('Valid Batchsize:      ', args.valid_batch_size)
    print('Total Epoch:    ', args.epochs)
    

def print_args_test(args):
    print('***********************************************')
    print('Dataset Name:   ', args.dataset)
    print('---------- Model --------------')
    print('Model Name:     ', args.model)
    print('Resume From:    ', args.resume)
    print('Save To:        ', args.save_dir)
    print('Available CPUs: ', os.cpu_count())
