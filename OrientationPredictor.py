# PASS
import torch
from torch import nn


import numpy as np
import time
import math
import PIL
import PIL.Image as Image


import torchvision as tv
import torch.nn as nn
import torchvision.transforms as trans
import torchvision.datasets as dsets
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToPILImage as tensor2PIL

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Use ResNet SAME padding for conv_same
class OrientationPredictor(nn.Module):
    def __init__(self, isDefaultImageSize = True, OP_train_dropout = False, dropout_rate = 0.6):
        # TODO: 'isDefaultImageSize' Not USE in new Res Structure!!!!!!
        super(OrientationPredictor, self).__init__()
        self.OP_train_dropout = OP_train_dropout

        # if isDefaultImageSize: # 56 * 56 * 256
        #     self.conv1 = nn.Conv2d(256, 128, kernel_size=5, stride=3, padding=0, bias=False) # VALID
        #     self.bn1 = nn.BatchNorm2d(128)
        #     self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=3, padding=2, bias=False) # SAME
        #     self.bn2 = nn.BatchNorm2d(256)
        #     self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False) # SAME
        #     self.bn3 = nn.BatchNorm2d(512)
        #     self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, bias=False) # VALID
        #     self.bn4 = nn.BatchNorm2d(1024)
        # else: # 64 * 32 * 256
        #     self.conv1 = nn.Conv2d(256, 128, kernel_size=5, stride=3, padding=2, bias=False) # SAME
        #     self.bn1 = nn.BatchNorm2d(128)
        #     self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=3, padding=2, bias=False) # SAME
        #     self.bn2 = nn.BatchNorm2d(256)
        #     self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False) # SAME
        #     self.bn3 = nn.BatchNorm2d(512)
        #     self.conv4 = nn.Conv2d(512, 1024, kernel_size=(4,2), stride=1, padding=0, bias=False) # VALID
        #     #  H - kernel[0]  W - kernel[1]
        #     self.bn4 = nn.BatchNorm2d(1024)

        self.net1 = self._make_layer(Bottleneck, 128, 4, 256, stride=2) # (B, 512, 28, 28)

        if self.OP_train_dropout:
            self.dropout = nn.Dropout(p=dropout_rate,inplace=False) # True will wrong 'variable require grad changed'!!

        # self.conv5 = nn.Conv2d(1024,3,kernel_size=1,bias=True) # No need BN layer
        self.net2 = nn.Conv2d(512, 3, kernel_size=1, bias=True)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1) # for full model use, to weight Orientation Branch
        self.logits = 0 # for training, by using nn.CrossEntropy

        # May use Sequential or List cover all Conv !!!!!!!
        self.kaiming_init()

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        # # print('after conv1: {}'.format(out.shape)) # ([B, 128, 18, 18])  ([B, 128, 22, 11])
        #
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # # print('after conv2: {}'.format(out.shape)) # ([B, 256, 6, 6])  ([B, 256, 8, 4])
        #
        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu(out)
        # # print('after conv3: {}'.format(out.shape)) # ([B, 512, 3, 3])  ([B, 512, 4, 2])
        #
        # out = self.conv4(out)
        # out = self.bn4(out)
        # out = self.relu(out)
        # # print('after conv4: {}'.format(out.shape)) # ([B, 1024, 1, 1])  ([B, 1024, 1, 1])

        out = self.net1(x)
        # print('after net1_layer2: {}'.format(out.shape)) # torch.Size([B, 512, 28, 28]) ([2, 512, 32, 16])

        out = self.gap(out)
        # print('after GAP: {}'.format(out.shape)) # torch.Size([B, 512, 1, 1])

        if self.OP_train_dropout:
            out = self.dropout(out)
            # print('apply dropout')

        # out = self.conv5(out)
        # # print('after conv5: {}'.format(out.shape)) # ([B, 3, 1, 1])

        out = self.net2(out)
        # print('after net2: {}'.format(out.shape)) # torch.Size([B, 3, 1, 1])

        out = out.reshape((out.shape[0],-1)) # flatten to 2D
        self.logits = out
        # print('logits shape: {}'.format(self.logits.shape)) # ([B, 3])
        out = self.softmax(out)
        return out

    def _make_layer(self, block, planes, blocks, inplanes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def kaiming_init(self,if_linear_init = False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m,nn.Linear) and if_linear_init:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                nn.init.constant_(m.bias, 0.0)


if __name__ == '__main__':
    synthesize_input_1 = torch.randn((2,256,56,56))
    synthesize_input_2 = torch.randn((2,256,64,32))
    my_model_1 = OrientationPredictor(isDefaultImageSize=True,OP_train_dropout=True)
    my_model_2 = OrientationPredictor(isDefaultImageSize=False,OP_train_dropout=False)
    print(20 * '-' + '>' + ' model 1 ' + '<' + 20 * '-')
    print(my_model_1)
    print(20 * '-' + '>' + ' model 2 ' + '<' + 20 * '-')
    print(my_model_2)
    print(20 * '-' + '>' + ' forward model 1 ' + '<' + 20 * '-')
    recv = my_model_1(synthesize_input_1)
    print('softmax res: {}'.format(recv))
    print(20 * '-' + '>' + ' forward model 2 ' + '<' + 20 * '-')
    recv = my_model_2(synthesize_input_2)
    print('softmax res: {}'.format(recv))
    print('logits res: {}'.format(my_model_2.logits))
    print(20 * '-' + '>' + ' model 2 parameters ' + '<' + 20 * '-')
    for idx,(name, param) in enumerate(my_model_2.named_parameters()):
        print('{} {}: {}'.format(idx,name,param.shape))

    print(20 * '-' + '>' + ' grad test ' + '<' + 20 * '-')
    # This prove OP grad is related to main_stream forward value, bigger value bigger grad
    synthesize_main_stream_1 = torch.rand((2,2048,16,8)) / 1e9 # (2,2048,8,4)
    synthesize_main_stream_2 = torch.zeros((2,2048,16,8)) # (2,2048,8,4)
    synthesize_main_stream_3 = torch.zeros((2,2048,16,8)) # (2,2048,8,4)
    weighted_1 = recv[:,0].reshape((-1,1,1,1)) * synthesize_main_stream_1
    weighted_2 = recv[:,1].reshape((-1,1,1,1)) * synthesize_main_stream_2
    weighted_3 = recv[:,2].reshape((-1,1,1,1)) * synthesize_main_stream_3
    fused = weighted_1 + weighted_2 + weighted_3
    loss = fused.mean()
    loss.backward()
    # weighted_1.mean().backward()
    # print('my_model_2.conv5.bias.grad: {}'.format(my_model_2.conv5.bias.grad))
    # print('my_model_2.conv1.weight.grad: {}'.format(my_model_2.conv1.weight.grad))
    print('my_model_2.net1[0].conv1.weight.grad: {}'.format(my_model_2.net1[0].conv1.weight.grad))
    print('my_model_2.net2.weight.grad: {}'.format(my_model_2.net2.weight.grad))















