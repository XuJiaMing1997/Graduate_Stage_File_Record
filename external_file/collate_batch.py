# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import numpy as np # XCP


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    # return 4 tuple
    # print('len batch: {} type batch: {}'.format(len(batch),type(batch))) # 64 <class 'list'>
    # print('imgs_len: {} imgs_type: {} pids: {}'.format(len(imgs),type(imgs),np.shape(pids)))
    # # 64 <class 'tuple'> 1D[64,]
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids
