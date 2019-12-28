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


from OrientationPredictor import OrientationPredictor


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


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)





def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class BaselineOP(nn.Module):
    def __init__(self, if_dropout = True, dropout_rate = 0.6, isDefaultImageSize = True,
                 ImageNet_Init = True, ImageNet_model_path = None):
        '''
        only Block1 & Predictor

        :param num_classes:
        :param ImageNet_Init:
        :param ImageNet_model_path:
        :return:
        '''
        super(BaselineOP, self).__init__()

        # Resnet backbone use
        Resnet50_layers=[3, 4, 6, 3]
        self.inplanes = 64 # ResNet backbone build use, later _make_layers will use !!!!!!!
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, Resnet50_layers[0],self.inplanes)

        # Orientation Predict use
        self.orientation_predictor = OrientationPredictor(isDefaultImageSize=isDefaultImageSize,
                                                          OP_train_dropout=if_dropout,dropout_rate=dropout_rate)
        self.OP_score = 0

        # initialization setting
        print('----> Random initialize Conv and BN layer')
        self.kaiming_init() # XCP ????? duplicate with above for Classifier Layer
        # first init for parameters which can not get pretrained data
        if ImageNet_Init:
            print('----> Loading pretrained ImageNet model......')
            self.load_param_ImageNet_OP(ImageNet_model_path)

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

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)    # add missed relu
        out = self.maxpool(out) # torch.Size([1, 64, 56, 56])

        out = self.layer1(out)
        # print('after block 1: {}'.format(out.shape)) # ([B, 256, 56, 56])

        orientation_score = self.orientation_predictor(out) # softmax results
        self.OP_score = orientation_score
        out = self.orientation_predictor.logits # logits to compute loss
        # print('Orientation Predictor logits: {}'.format(out.shape)) # ([B, 3])

        return out


    def load_param_ImageNet_OP(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            # print(i)
            if 'layer3' in i or 'layer4'in i or 'fc' in i :
                continue
            elif 'layer2' in i:
                OP_key = i.replace('layer2','orientation_predictor.net1')
                self.state_dict()[OP_key].copy_(param_dict[i])
            else:
                self.state_dict()[i].copy_(param_dict[i])

    def load_param_transfer(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

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
    # for single file check

    # load_state_dict = torch.load('./resnet50-19c8e357.pth')
    # for idx,(i,j) in enumerate(load_state_dict.items()):
    #     print('{} {}: {}'.format(idx,i,j.shape))

    # myModel = BaselineOP(if_dropout=False,dropout_rate=0.6,isDefaultImageSize=True,ImageNet_Init=True,
    #                      ImageNet_model_path='./resnet50-19c8e357.pth')
    myModel = BaselineOP(if_dropout=True,dropout_rate=0.6,isDefaultImageSize=False,ImageNet_Init=True,
                         ImageNet_model_path='./resnet50-19c8e357.pth')

    print(myModel)
    print('\n\n-----> var check ')
    for idx, (name, var) in enumerate(myModel.named_parameters()):
        print('{} {}: {}'.format(idx,name,var.shape))

    import torch as t
    # myInput = t.randn((4,3,224,224),dtype=t.float32) # batch must bigger than 1 or will cause BN_layer error!!!!!
    myInput = t.randn((4,3,256,128),dtype=t.float32) # batch must bigger than 1 or will cause BN_layer error!!!!!
    myModel.train()
    print('-----> train forward check')
    recv = myModel(myInput)
    print('recv shape: {}'.format(recv.shape))
    print('recv: {}'.format(recv))

    # grad test
    myModel.zero_grad()
    recv.mean().backward()
    # print('myModel.orientation_predictor.conv1.weight.grad: {}'.format(
    #         myModel.orientation_predictor.conv1.weight.grad))
    print('myModel.orientation_predictor.net1[0].conv1.weight.grad: {}'.format(
            myModel.orientation_predictor.net1[0].conv1.weight.grad))

    with t.no_grad():
        myModel.eval() # dropout switch test
        print('-----> eval forward check')
        recv = myModel(myInput)
        print('recv shape: {}'.format(recv.shape))
        print('recv {}'.format(recv))

