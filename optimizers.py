import torch


def create_optim(name, model, args):
    if name == 'adam':
        optimizer    = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)

    elif name == 'adamw':
        optimizer    = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)
    
    else :
        raise KeyError("Wrong optim name `{}`".format(name))        

    return optimizer