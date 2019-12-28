# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from external_file.resnet import ResNet, BasicBlock, Bottleneck
from external_file.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck


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


class Baseline(nn.Module):
    in_planes = 2048
    # TODO: May exist problem: change to self.in_planes = 2048

    def __init__(self, num_classes, last_stride, neck, eval_neck, model_name, ImageNet_Init, ImageNet_model_path):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)

        if ImageNet_Init:
            self.base.load_param(ImageNet_model_path)
            print('----> Loading pretrained ImageNet model......')
        else:
            self.base.random_init() # XCP
            print('----> Random initialize Conv and BN layer')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.eval_neck = eval_neck

        self.GAP_feature = 0
        self.BN_feature = 0
        self.FC_feature = 0

        if self.neck == False:
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == True:
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        # global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        global_feat = global_feat.reshape((global_feat.shape[0],-1)) # XCP (bs, 2048)
        self.GAP_feature = global_feat

        if self.neck == False:
            feature = global_feat
        elif self.neck == True:
            feature = self.bottleneck(global_feat)  # normalize for angular softmax
            self.BN_feature = feature

        if self.training:
            cls_score = self.classifier(feature)
            self.FC_feature = cls_score
            # return cls_score, global_feat  # global feature for triplet loss
            # print('train output: FC_feature and GAP_feature') # XCP
            return self.FC_feature, self.GAP_feature  # global feature for triplet loss
        else:
            if self.eval_neck == True:
                # print("Test with feature after BN -- BN_feature") # XCP
                # return feat
                assert self.neck == True
                return self.BN_feature
            else:
                # print("Test with feature before BN -- GAP_feature") # XCP
                # return global_feat
                return self.GAP_feature

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


if __name__ == '__main__':
    # for single file check
    myModel = Baseline(101,2,True,True,'resnet50',True,'../resnet50-19c8e357.pth')
    print(myModel)
    print('\n\n-----> var check ')
    for idx,(name, var) in enumerate(myModel.named_parameters()):
        print('{} {}: {}'.format(idx,name,var.shape))

    import torch as t
    myInput = t.randn((2,3,224,224),dtype=t.float32) # batch must bigger than 1 or will cause BN_layer error!!!!!
    myModel.train()
    print('-----> train forward check')
    recv0,recv1 = myModel(myInput)
    print('recv0 shape: {}\trecv1 shape: {}'.format(recv0.shape,recv1.shape))
    myModel.eval()
    print('-----> eval forward check')
    recv = myModel(myInput)
    print('recv shape: {}'.format(recv.shape))

