# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch

# spatial_transformer param eg. spatial_transformer.localisation_net.bn4.bias: torch.Size([1024])
def make_optimizer(optimizer_setting, model):
    # optimizer_setting = {'lr_start': Lr_Start,
    #                          'weight_decay': 0.0005,
    #                          'bias_lr_factor': 1,
    #                          'weight_decay_bias': 0.0005,
    #                          'optimizer_name': 'Adam',  # 'SGD'
    #                          'SGD_momentum': 0.9,
    #                          'center_lr': 0.5,
    #                          'STN_lr': STN_lr_start,
    #                          'OP_lr': OP_lr
    #                          }
    STN_param_idx = []
    OP_param_idx = []
    params = []
    for idx,(key, value) in enumerate(model.named_parameters()):
        if not value.requires_grad:
            continue
        lr = optimizer_setting['lr_start']
        weight_decay = optimizer_setting['weight_decay']
        if "bias" in key:
            lr = optimizer_setting['lr_start'] * optimizer_setting['bias_lr_factor']
            weight_decay = optimizer_setting['weight_decay_bias']
        if 'spatial_transformer' in key:
            lr = optimizer_setting['STN_lr']
            STN_param_idx.append(idx)
        elif 'orientation_predictor' in key or idx <= 32: # 32 layer1.2.bn3.bias: torch.Size([256])
            lr = optimizer_setting['OP_lr']
            OP_param_idx.append(idx)
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if optimizer_setting['optimizer_name'] == 'SGD':
        optimizer = getattr(torch.optim, optimizer_setting['optimizer_name'])(
                params, momentum=optimizer_setting['SGD_momentum'])
    else:
        optimizer = getattr(torch.optim, optimizer_setting['optimizer_name'])(params)
    # XCP
    print('STN parameter idx in optimizer: {}\n{}'.format(len(STN_param_idx),STN_param_idx)) # 14
    print('OP_Ex parameter idx in optimizer: {}\n{}'.format(len(OP_param_idx),OP_param_idx)) #
    return optimizer, set(STN_param_idx), set(OP_param_idx)


def make_optimizer_with_center(optimizer_setting, model, center_criterion):
    # optimizer_setting = {'lr_start': Lr_Start,
    #                          'weight_decay': 0.0005,
    #                          'bias_lr_factor': 1,
    #                          'weight_decay_bias': 0.0005,
    #                          'optimizer_name': 'Adam',  # 'SGD'
    #                          'SGD_momentum': 0.9,
    #                          'center_lr': 0.5,
    #                          'STN_lr': STN_lr_start,
    #                          'OP_lr': OP_lr
    #                          }
    STN_param_idx = []
    OP_param_idx = []
    params = []
    for idx,(key, value) in enumerate(model.named_parameters()):
        if not value.requires_grad:
            continue
        lr = optimizer_setting['lr_start']
        weight_decay = optimizer_setting['weight_decay']
        if "bias" in key:
            lr = optimizer_setting['lr_start'] * optimizer_setting['bias_lr_factor']
            weight_decay = optimizer_setting['weight_decay_bias']
        if 'spatial_transformer' in key:
            lr = optimizer_setting['STN_lr']
            STN_param_idx.append(idx)
        elif 'orientation_predictor' in key or idx <= 32: # 32 layer1.2.bn3.bias: torch.Size([256])
            lr = optimizer_setting['OP_lr']
            OP_param_idx.append(idx)
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if optimizer_setting['optimizer_name'] == 'SGD':
        optimizer = getattr(torch.optim, optimizer_setting['optimizer_name'])(
                params,momentum=optimizer_setting['SGD_momentum'])
    else:
        optimizer = getattr(torch.optim, optimizer_setting['optimizer_name'])(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=optimizer_setting['center_lr'])
    # XCP
    print('STN parameter idx in optimizer: {}\n{}'.format(len(STN_param_idx),STN_param_idx)) # 14
    print('OP_Ex parameter idx in optimizer: {}\n{}'.format(len(OP_param_idx),OP_param_idx)) #
    return optimizer, optimizer_center, set(STN_param_idx), set(OP_param_idx)