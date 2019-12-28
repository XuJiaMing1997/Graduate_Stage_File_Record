import torch
import numpy as np # XCP


def inference_collate_fn(batch):
    imgs, pids, camids, paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, paths
