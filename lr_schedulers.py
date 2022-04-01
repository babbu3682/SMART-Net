
import torch
import math
import functools
from torch.optim.lr_scheduler import _LRScheduler


def poly_learning_rate(epoch, warm_up_epoch, start_decay_epoch, total_epoch, min_lr):
    # Linear Warmup
    if (epoch < warm_up_epoch):
        return max(0, epoch / warm_up_epoch)
    else :
        lr = 1.0 - max(0, epoch - start_decay_epoch) /(float(total_epoch) - start_decay_epoch)

        if lr <= min_lr:
            lr = min_lr

    return lr


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



def create_scheduler(name, optimizer, args):
    if name == 'poly_lr':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=functools.partial(poly_learning_rate, warm_up_epoch=args.warmup_epochs, start_decay_epoch=args.epochs/10, total_epoch=args.epochs, min_lr=args.min_lr))            

    elif name == 'cosine_annealing_warm_restart':
        # optimizer에서 시작할 learning rate를 일반적으로 사용하는 learning rate가 아닌 0에 가까운 아주 작은 값을 입력해야 합니다.
        # ref: https://gaussian37.github.io/dl-pytorch-lr_scheduler/

        # 주기 100 epoch
        # 최고 max lr = eta_max
        # T_up -> warm_up시 필요한 epoch
        # gamma -> 몇 퍼센트 살릴지 0.6 이면, 다음 lr = 0.6 * 현재 lr
        # lr => 최소 learning rate 때문에 작게 설정해야한다!
        
        lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=0.001,  T_up=10, gamma=0.6)

    else :
        raise KeyError("Wrong scheduler name `{}`".format(name))        


    return lr_scheduler