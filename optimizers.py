'''
Declares the Resnet-50 model using either the torchvision or the huggingface package. Note that input is one-channel.
'''

import torch

def get_optimizer(name, model, lr=1e-4):
    if name == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)

    elif name == 'adamw':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)

    else :
        raise KeyError("Wrong optim name `{}`".format(name))        

    return optimizer