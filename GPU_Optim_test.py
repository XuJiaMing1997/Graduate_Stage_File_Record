import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math
import PIL
import PIL.Image as Image
import argparse
import random

import torch as t
import torchvision as tv
import torch.nn as nn
import torchvision.transforms as trans
import torchvision.datasets as dsets
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToPILImage as tensor2PIL
from torch.utils.data import DataLoader

if t.cuda.is_available():
    use_GPU = True
    print('Use GPU')
else:
    use_GPU = False
    print('No GPU, Use CPU')
available_device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

a = t.randint(0,10,(3,5),dtype=t.float32,requires_grad=True) # leaf grad
b = t.randint(0,10,(3,5),dtype=t.float32,device=available_device,requires_grad=True) # leaf grad
print('a device: {} is_leaf: {}'.format(a.device,a.is_leaf))
print('b device: {} is_leaf: {}'.format(b.device,b.is_leaf))
a_gpu = a.to(available_device)
print('a_gpu device: {} is_leaf: {}'.format(a_gpu.device,a_gpu.is_leaf))
print('a device: {} is_leaf: {}'.format(a.device,a.is_leaf))






