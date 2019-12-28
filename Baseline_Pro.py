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

from SpatialTransformer import SpatialTransformer
from OrientationPredictor import OrientationPredictor
from AttentionMaskNet import AttentionMaskNet


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


class BaselinePro(nn.Module):
    def __init__(self, num_classes, neck, eval_neck,
                 ImageNet_Init = True, ImageNet_model_path = None, last_stride=2,
                 addOrientationPart = True, addSTNPart = True, addAttentionMaskPart = True, addChannelReduce = True,
                 isDefaultImageSize = True, final_dim = 1024, mask_num = 8, dropout_rate = 0.6, use_GPU = False,
                 if_affine = False, if_OP_channel_wise = False, STN_fc_b_init = 1.0):
        """
        Train output: FC_feature, GAP_feature  &&  Test output: GAP_feature or BN_Feature

        final_dim only available when use ChannelReduce
        :param num_classes:
        :param neck:
        :param eval_neck:
        :param ImageNet_Init:
        :param ImageNet_model_path:
        :param last_stride:
        :param addOrientationPart:
        :param addSTNPart:
        :param addAttentionMaskPart:
        :param addChannelReduce:
        :param isDefaultImageSize:
        :param final_dim:
        :param mask_num:
        :param dropout_rate:
        :return:
        """
        super(BaselinePro, self).__init__()
        # classifier & BatchNormNeck use
        if not addAttentionMaskPart:
            self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.eval_neck = eval_neck
        self.if_OP_channel_wise = if_OP_channel_wise

        # Three Part switch
        self.addOrientationPart = addOrientationPart
        self.addSTNPart = addSTNPart
        self.addAttentionMaskPart = addAttentionMaskPart
        self.addChannelReduce = addChannelReduce

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
        if addOrientationPart:
            self.orientation_predictor = OrientationPredictor(isDefaultImageSize=isDefaultImageSize)

        # Resnet backbone use
        self.layer2 = self._make_layer(Bottleneck, 128, Resnet50_layers[1], self.inplanes, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, Resnet50_layers[2], self.inplanes, stride=2)

        # STN use
        if addSTNPart:
            self.spatial_transformer = SpatialTransformer(initial_parameter=[[STN_fc_b_init, 0., 0.],
                                                                             [0., STN_fc_b_init, 0.],
                                                                             [0., 0., 1.]],
                                                          use_GPU=use_GPU,if_affine=if_affine)

        if addOrientationPart:
            # Orientation Branch
            tem_inplanes = self.inplanes # Unknown but work
            self.layer4_front = self._make_layer(Bottleneck, 512, Resnet50_layers[3], tem_inplanes, stride=last_stride)
            self.layer4_back = self._make_layer(Bottleneck, 512, Resnet50_layers[3], tem_inplanes, stride=last_stride)
            self.layer4_side = self._make_layer(Bottleneck, 512, Resnet50_layers[3], tem_inplanes, stride=last_stride)
        else:
            self.layer4 = self._make_layer(Bottleneck, 512, Resnet50_layers[3], self.inplanes, stride=last_stride)


        # channel reduction use
        if addChannelReduce:
            if if_OP_channel_wise and addOrientationPart: # channel-wise concatenate
                self.channel_reduce_conv = nn.Conv2d(6144, final_dim, kernel_size=1, bias=False)
                # 'Deeply learned' use this, 'PSE' also use this which following classifier
                self.channel_reduce_bn = nn.BatchNorm2d(final_dim)
            else: # element-wise add
                # 'Deeply learned' use this, 'PSE' also use this which following classifier
                self.channel_reduce_conv = nn.Conv2d(2048, final_dim, kernel_size=1, bias=False)
                self.channel_reduce_bn = nn.BatchNorm2d(final_dim)

        else:
            if if_OP_channel_wise and addOrientationPart:
                final_dim = 6144
            else:
                final_dim = 2048

        # Attention Mask Branch use
        if addAttentionMaskPart:
            AM_in_planes = final_dim
            self.attention_mask_net = AttentionMaskNet(mask_num=mask_num,input_dim=AM_in_planes,
                                                       dropout_rate=dropout_rate)

        # Classifier Layer
        fc_in_planes = final_dim
        if neck == False:
            self.classifier = nn.Linear(fc_in_planes, self.num_classes) # not initialize
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif neck == True:
            self.bottleneck = nn.BatchNorm1d(fc_in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(fc_in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier) # ??????????????? Need or not duplicate with below


        # initialization setting
        print('----> Random initialize Conv and BN layer')
        self.kaiming_init() # XCP ????? duplicate with above for Classifier Layer
        # first init for parameters which can not get pretrained data
        if ImageNet_Init:
            print('----> Loading pretrained ImageNet model......')
            self.load_param_ImageNet(ImageNet_model_path)


        # end_points record
        self.STN_transform_parameters = 0 # [Batch, 9]
        self.orientation_predict_score = 0 # [Batch, 3] Softmax results, for weight 3 block4 branch
        self.orientation_predict_logits = 0 # [Batch, 3] Not activated results, for Train Predictor
        self.masks = [] # list[tensor,tensor,..] [mask_num=8,(B,C=1,H,W)]

        self.GAP_feature = 0
        self.BN_feature = 0
        self.FC_feature = 0


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
        # print('after block 1: {}'.format(out.shape)) # ([B, 256, 56, 56]) ([B, 256, 64, 32])

        # Orientation Predict
        if self.addOrientationPart:
            orientation_score = self.orientation_predictor(out) # softmax results
            self.orientation_predict_score = orientation_score
            self.orientation_predict_logits = self.orientation_predictor.logits
            # print('Orientation Predictor out: {}'.format(orientation_score.shape)) # ([B, 3])
            # print('Orientation Predictor logits: {}'.format(self.orientation_predict_logits.shape)) # ([B, 3])

        out = self.layer2(out)
        # print('after Block 2: {}'.format(out.shape)) # ([4, 512, 28, 28]) ([B, 512, 32, 16])
        out = self.layer3(out)
        # print('after Block 3: {}'.format(out.shape)) # ([4, 1024, 14, 14]) ([B, 1024, 16, 8])

        # STN
        if self.addSTNPart:
            out = self.spatial_transformer(out)
            self.STN_transform_parameters = self.spatial_transformer.transform_parameters
            # print('after STN: {}'.format(out.shape)) # ([4, 1024, 16, 8])

        # Orientation Branch
        if self.addOrientationPart:
            front_fmap = self.layer4_front(out)
            # print('after Block 4 front: {}'.format(front_fmap.shape)) # ([B, 2048, 16, 8])
            back_fmap = self.layer4_back(out)
            # print('after Block 4 back: {}'.format(back_fmap.shape)) # ([B, 2048, 16, 8])
            side_fmap = self.layer4_side(out)
            # print('after Block 4 side: {}'.format(side_fmap.shape)) # ([B, 2048, 16, 8])

            weighted_front_fmap = orientation_score[:,0].reshape((-1,1,1,1)) * front_fmap
            weighted_back_fmap = orientation_score[:,1].reshape((-1,1,1,1)) * back_fmap
            weighted_side_fmap = orientation_score[:,2].reshape((-1,1,1,1)) * side_fmap
            # print('weighted_side_fmap: {}'.format(weighted_side_fmap.shape)) # ([B, 2048, 16, 8])

            if self.if_OP_channel_wise:
                out = torch.cat((weighted_front_fmap,weighted_back_fmap,weighted_side_fmap),dim=1)
                # print('3 Orientation Branch Channel-Wise concatenate: {}'.format(out.shape)) # ([B, 6144, 16, 8])
            else:
                out = weighted_front_fmap + weighted_back_fmap + weighted_side_fmap
                # print('3 Orientation Branch Element-Wise add: {}'.format(out.shape)) # ([B, 2048, 16, 8])

        else:
            out = self.layer4(out)
            # print('after Block 4: {}'.format(out.shape)) # ([B, 2048, 16, 8])

        # channel reduce
        if self.addChannelReduce:
            out = self.channel_reduce_conv(out)
            out = self.channel_reduce_bn(out)
            out = self.relu(out)
            # print('after Channel Reduce: {}'.format(out.shape)) # ([B, 2048, 16, 8])

        # Attention Mask Net
        if self.addAttentionMaskPart:
            out = self.attention_mask_net(out) # output 2D [Batch, final_dim]
            # print('after Attention Mask Net: {}'.format(out.shape)) # ([B, 2048])
            self.masks = self.attention_mask_net.masks
            # print('Received Mask: {}'.format(len(self.masks))) # (Mask_num, )
            # for i in self.masks:
            #     print(i.shape) # ([B, 1, 7, 7])
        else:
            out = self.gap(out)
            # print('after GAP: {}'.format(out.shape)) # ([B, 2048, 1, 1])
            out = out.reshape((out.shape[0],-1))
            # print('after Flatten: {}'.format(out.shape)) # ([B, 2048])

        # BatchNormNeck & ClassifierLayer
        self.GAP_feature = out
        # print('GAP feature shape: {}'.format(self.GAP_feature.shape)) # ([B, 2048])

        if self.neck == False:
            feature = out
        elif self.neck == True:
            feature = self.bottleneck(out)  # normalize for angular softmax
            self.BN_feature = feature
            # print('BN_feature shape: {}'.format(self.BN_feature.shape)) # ([B, 2048])

        if self.training:
            cls_score = self.classifier(feature)
            self.FC_feature = cls_score
            # print('FC_feature shape: {}'.format(self.FC_feature.shape)) # ([B, class_num])
            # return cls_score, global_feat  # global feature for triplet loss
            # print('-----> Train Output\nFC_feature: {} GAP_feature: {}'.format(self.FC_feature.shape,
            #                                                                    self.GAP_feature.shape))
            return self.FC_feature, self.GAP_feature  # global feature for triplet loss

        else:
            if self.eval_neck == True:
                # print("Test with feature after BN -- BN_feature") # XCP
                # return feat
                assert self.neck == True
                # print('-----> Evaluate Output: BN_feature: {}'.format(self.BN_feature.shape))
                return self.BN_feature
            else:
                # print("Test with feature before BN -- GAP_feature") # XCP
                # return global_feat
                # print('-----> Evaluate Output: GAP_feature: {}'.format(self.GAP_feature.shape))
                return self.GAP_feature

    def load_param_ImageNet(self, model_path):
        '''
        Only available when MyModel is bigger than ImageNet model

        :param model_path:
        :return:
        '''
        param_dict = torch.load(model_path)
        for i in param_dict:
            # print(i)
            if 'fc' in i:
                continue
            elif 'layer4' in i:
                if self.addOrientationPart:
                    front_key = i.replace('layer4','layer4_front')
                    back_key = i.replace('layer4','layer4_back')
                    side_key = i.replace('layer4','layer4_side')
                    self.state_dict()[front_key].copy_(param_dict[i])
                    self.state_dict()[back_key].copy_(param_dict[i])
                    self.state_dict()[side_key].copy_(param_dict[i])
                else:
                    self.state_dict()[i].copy_(param_dict[i])
                if self.addSTNPart:
                    STN_key = i.replace('layer4','spatial_transformer.localisation_net.net1')
                    self.state_dict()[STN_key].copy_(param_dict[i])
            elif 'layer2' in i:
                self.state_dict()[i].copy_(param_dict[i])
                if self.addOrientationPart:
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

    def load_STN_from_SBP_model_layer4(self, SBP_model_path):
        assert self.addSTNPart == True, 'Must have STN Part'
        param_dict = torch.load(SBP_model_path)
        for i in param_dict:
            if 'layer4' in i:
                if 'base.' in i:
                    STN_key = i.replace('base.layer4', 'spatial_transformer.localisation_net.net1')
                else:
                    STN_key = i.replace('layer4', 'spatial_transformer.localisation_net.net1')
                self.state_dict()[STN_key].copy_(param_dict[i])

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
    import torch as t
    # t.manual_seed(520)

    # load_state_dict = torch.load('./resnet50-19c8e357.pth')
    # for idx,(i,j) in enumerate(load_state_dict.items()):
    #     print('{} {}: {}'.format(idx,i,j.shape))

    # myModel = BaselinePro(101,True,False,True,ImageNet_model_path='./resnet50-19c8e357.pth',last_stride=1,
    #                       addAttentionMaskPart=True,addOrientationPart=False,addSTNPart=False,
    #                       addChannelReduce=True,isDefaultImageSize=True)
    myModel = BaselinePro(101,neck=True,eval_neck=True,ImageNet_Init=True,
                          ImageNet_model_path='./resnet50-19c8e357.pth',last_stride=1,
                          addOrientationPart=True,addSTNPart=True,addAttentionMaskPart=True,
                          addChannelReduce=True,isDefaultImageSize=False,final_dim=1024,dropout_rate=0.6,
                          use_GPU=True, if_affine=False, if_OP_channel_wise=True,STN_fc_b_init=0.9)
    myModel.cuda()
    print(myModel)
    print('\n\n-----> var check ')
    for idx, (name, var) in enumerate(myModel.named_parameters()):
        print('{} {}: {}'.format(idx,name,var.shape))


    # myInput = t.randn((4,3,224,224),dtype=t.float32) # batch must bigger than 1 or will cause BN_layer error!!!!!
    myInput = t.randn((4,3,256,128),dtype=t.float32) # batch must bigger than 1 or will cause BN_layer error!!!!!
    myInput = myInput.cuda()
    myModel.train()
    print('-----> train forward check')
    myModel.zero_grad()
    recv0,recv1 = myModel(myInput)
    print('recv0 shape: {}\trecv1 shape: {}'.format(recv0.shape,recv1.shape))
    print('-----> mask check')
    print('{}'.format(len(myModel.masks))) # 8
    print('{}'.format(type(myModel.masks))) # <class 'list'>
    for mask in myModel.masks:
        print('{}'.format(mask.shape)) # torch.Size([4, 1, 16, 8])

    print('-----> grad check')
    # print('myModel.spatial_transformer.localisation_net.fc_W.grad Before BP: {}'.format(
    #         myModel.spatial_transformer.localisation_net.fc_W.grad))
    print('myModel.conv1.weight.grad Before BP: {}'.format(myModel.conv1.weight.grad))
    print('layer4_front.0.conv1.weight Before BP: {}'.format(myModel.layer4_front[0].conv1.weight.grad))
    # print('myModel.orientation_predictor.conv5.bias.grad Before BP: {}'.format(
    #         myModel.orientation_predictor.conv5.bias.grad))
    # print('myModel.orientation_predictor.conv5.weight.grad Before BP: {}'.format(
    #         myModel.orientation_predictor.conv5.weight.grad))
    print('myModel.orientation_predictor.net1[0].conv1.weight.grad: {}'.format(
        myModel.orientation_predictor.net1[0].conv1.weight.grad))
    print('myModel.attention_mask_net.mask_gen_list[6][0].weight.grad Before BP {}'.format(
            myModel.attention_mask_net.mask_gen_list[6][0].weight.grad))
    (recv0.mean() + recv1.mean()).backward()
    # myModel.anchor.mean().backward()
    # recv1.mean().backward()
    # print('myModel.spatial_transformer.localisation_net.fc_W.grad After BP: {}'.format(
    #         myModel.spatial_transformer.localisation_net.fc_W.grad))
    print('myModel.conv1.weight.grad After BP: {}'.format(myModel.conv1.weight.grad))
    print('layer4_front.0.conv1.weight After BP: {}'.format(myModel.layer4_front[0].conv1.weight.grad))
    # print('myModel.orientation_predictor.conv5.bias.grad After BP: {}'.format(
    #         myModel.orientation_predictor.conv5.bias.grad))
    # print('myModel.orientation_predictor.conv5.weight.grad Before BP: {}'.format(
    #         myModel.orientation_predictor.conv5.weight.grad))
    print('myModel.orientation_predictor.net1[0].conv1.weight.grad: {}'.format(
        myModel.orientation_predictor.net1[0].conv1.weight.grad))
    print('myModel.attention_mask_net.mask_gen_list[6][0].weight.grad After BP {}'.format(
            myModel.attention_mask_net.mask_gen_list[6][0].weight.grad))

    # print('front map value: {}'.format(myModel.anchor))

    with t.no_grad():
        myModel.eval()
        print('-----> eval forward check')
        recv = myModel(myInput)
        print('recv shape: {}'.format(recv.shape))

