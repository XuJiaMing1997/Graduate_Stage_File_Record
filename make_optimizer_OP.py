import torch


def make_optimizer_OP(optimizer_setting, model):
    # optimizer_setting = {'lr_start': lr_start,
    #                      'weight_decay': 0.0005,
    #                      'bias_lr_factor': 1,
    #                      'weight_decay_bias': 0.0005,
    #                      'optimizer_name': optimizer_name,  # 'SGD' 'Adam'
    #                      'SGD_momentum': 0.9
    #                      }
    NoneOP_idx = []
    params = []
    for idx,(key, value) in enumerate(model.named_parameters()):
        if not value.requires_grad:
            continue
        lr = optimizer_setting['lr_start']
        weight_decay = optimizer_setting['weight_decay']
        if "bias" in key:
            lr = optimizer_setting['lr_start'] * optimizer_setting['bias_lr_factor']
            weight_decay = optimizer_setting['weight_decay_bias']
        if 'orientation_predictor' not in key:
            NoneOP_idx.append(idx)
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if optimizer_setting['optimizer_name'] == 'SGD':
        optimizer = getattr(torch.optim, optimizer_setting['optimizer_name'])(
                params, momentum=optimizer_setting['SGD_momentum'])
    else:
        optimizer = getattr(torch.optim, optimizer_setting['optimizer_name'])(params)
    print('----> None OP param idx:{}\n{}'.format(len(NoneOP_idx),NoneOP_idx))
    return optimizer, set(NoneOP_idx)