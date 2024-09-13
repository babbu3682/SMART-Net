'''
Declares the Simple Optimizer & Scheduler for training.
'''

import torch
import functools

def poly_learning_rate(epoch, warm_up_epoch, start_decay_epoch, total_epoch, min_lr):
    # Linear Warmup
    if (epoch < warm_up_epoch):
        return max(0, epoch / warm_up_epoch)
    else :
        lr = 1.0 - max(0, epoch - start_decay_epoch) /(float(total_epoch) - start_decay_epoch)

        if lr <= min_lr:
            lr = min_lr

    return lr


def get_scheduler(name, optimizer, warm_up_epoch=10, start_decay_epoch=1000/10, total_epoch=1000, min_lr=1e-6):
    if name == 'poly_lr':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=functools.partial(poly_learning_rate, warm_up_epoch=warm_up_epoch, start_decay_epoch=start_decay_epoch, total_epoch=total_epoch, min_lr=min_lr))

    elif name == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    elif name == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)

    else :
        raise KeyError("Wrong scheduler name `{}`".format(name))        


    return lr_scheduler