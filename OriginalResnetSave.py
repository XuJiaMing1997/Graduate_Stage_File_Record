import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math
import PIL.Image as Image
import argparse

import torch as t
import torchvision as tv
import torch.nn as nn
import torchvision.transforms as trans
import torchvision.datasets as dsets
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToPILImage as tensor2PIL
from torch.utils.data import DataLoader



myModel = tv.models.resnet50(pretrained=True)
print(myModel)
for name,var in myModel.named_parameters():
    print('{}: {}'.format(name,var.shape))
# load pretrained model parameters
t.save(myModel.state_dict(), 'ResNet50_ImageNet.pth')

lenet = tv.models.resnet50(pretrained=False)
# lenet.load_state_dict(t.load('ResNet50_ImageNet.pth'), strict=True)
lenet.load_state_dict(t.load('resnet50-19c8e357.pth'), strict=True)
# False will ignore not match key between dict and model
recv_state_dict = t.load('resnet50-19c8e357.pth')
print('--------> state dict: resnet50-19c8e357.pth')
for idx,(k,v) in enumerate(recv_state_dict.items()):
    print('{} {}: {}'.format(idx,k,v.shape))
