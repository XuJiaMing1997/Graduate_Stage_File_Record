# PASS
import numpy as np
import time
import math
import PIL
import PIL.Image as Image
import argparse


import torch
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

class LocalisationNet(nn.Module):
    def __init__(self, initial_parameter, if_manual_test = False, last_stride = 1, use_GPU = False):
        '''
        input 14x14x1024 / 16x8x1024

        :param initial_parameter:
        :param if_manual_test:
        :param last_stride:
        :param use_GPU:
        :return:
        '''
        super(LocalisationNet, self).__init__()
        self.if_manual_test = if_manual_test
        self.use_GPU = use_GPU
        # initial_parameter: [3,3] nparray
        self.transfer_parameter = np.reshape(initial_parameter,(-1,)).astype(dtype=np.float32)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1,1))

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

        self.net1 = self._make_layer(Bottleneck, 512, 3, 1024, stride=last_stride)
        self.net2 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

        if not if_manual_test:
            input_size = 128
        else:
            input_size = 3
        out_size = np.size(self.transfer_parameter)
        self.fc_W = nn.Parameter(torch.zeros(input_size, out_size)) # is included in model.named_parameters()
        self.fc_b = nn.Parameter(torch.tensor(self.transfer_parameter)) # is included in model.named_parameters()
        # init loc_fc_W, loc_fc_b finish

        self.kaiming_init()

    def forward(self, x):
        # input_map: 4D tensor 14x14x1024 / 16x8x1024
        if not self.if_manual_test:
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
            # print('after net1_layer4: {}'.format(out.shape)) # torch.Size([B, 2048, 16, 8])

            out = self.net2(out)
            # print('after net2_Conv: {}'.format(out.shape)) # torch.Size([B, 128, 16, 8])

            out = self.gap(out)
            # print('after GAP: {}'.format(out.shape)) # torch.Size([B, 128, 1, 1])

            out = out.reshape((x.shape[0],-1))
            # print('flatten: {}'.format(out.shape)) # torch.Size([B, 128])

            out = torch.matmul(out,self.fc_W) + self.fc_b
            # print('final parameters shape: {}'.format(out.shape)) # torch.Size([B, 9]) / torch.Size([B, 6])

        else:
            batch_size = x.shape[0]
            tem_map = torch.zeros((batch_size,3))
            if self.use_GPU:
                tem_map = tem_map.cuda()
            out = torch.matmul(tem_map,self.fc_W) + self.fc_b

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


class SpatialTransformer(nn.Module):
    '''
    input: B,C,H,W
    output: B,C,H,W
    '''
    def __init__(self, initial_parameter = [[1.0,0.,0.],[0.,1.0,0.],[0.,0.,1.0]], out_dim = None,
                 if_manual_test = False, use_GPU = False, if_affine = False):
        '''
        input 14x14x1024 / 16x8x1024

        :param initial_parameter:
        :param out_dim:
        :param if_manual_test:
        :param use_GPU:
        :param if_affine:
        :param loc_last_stride:
        :return:
        '''
        super(SpatialTransformer, self).__init__()
        self.out_dim = out_dim
        self.use_GPU = use_GPU
        self.if_affine = if_affine
        if if_affine:
            self.initial_parameter = initial_parameter[:2][:]
        else:
            self.initial_parameter = initial_parameter
        self.localisation_net = LocalisationNet(self.initial_parameter, if_manual_test=if_manual_test,
                                                last_stride=1, use_GPU=use_GPU)

        # endpoints
        self.transform_parameters = 0

        self.kaiming_init()

    def forward(self, input_fmap):
        Height = input_fmap.shape[2] # (B, C, H, W)
        Width = input_fmap.shape[3] # (B, C, H, W)
        fmap = input_fmap.float()

        affine_parameter = self.localisation_net(fmap)
        self.transform_parameters = affine_parameter
        # return 2-D Batch * 9/6

        if self.if_affine:
            if self.out_dim is not None:
                x_s, y_s = self.sample_grid_generator_affine(self.out_dim[0], self.out_dim[1], affine_parameter,
                                                             self.use_GPU)
            else:
                x_s, y_s = self.sample_grid_generator_affine(Height, Width, affine_parameter, self.use_GPU)
        else:
            if self.out_dim is not None:
                x_s, y_s = self.sample_grid_generator(self.out_dim[0], self.out_dim[1], affine_parameter, self.use_GPU)
            else:
                x_s, y_s = self.sample_grid_generator(Height, Width, affine_parameter, self.use_GPU)


        out_fmap = self.bilinear_sampler(fmap, x_s, y_s)

        return out_fmap


    def sample_grid_generator(self, Height, Width, transfer_parameter, use_GPU):
        # PASS
        batch_num = transfer_parameter.shape[0]
        # reshape theta to (B, 3, 3)
        transfer_parameter = transfer_parameter.reshape((-1, 3, 3))

        # create normalized 2D grid
        x = torch.linspace(-1.0, 1.0, Width)
        y = torch.linspace(-1.0, 1.0, Height)
        y_t, x_t = torch.meshgrid((y, x)) # here !!!!!!!! different return order as TF
        # NEED t.meshgrid((y, x)) not t.meshgrid((x, y))  Only For Pytorch Version!!!!!!!!

        # flatten
        x_t_flat = x_t.reshape((-1,))
        y_t_flat = y_t.reshape((-1,))

        # reshape to [x_t, y_t , 1] - (homogeneous form)
        ones = torch.ones_like(x_t_flat)
        sampling_grid = torch.stack([x_t_flat, y_t_flat, ones]) # concatenate along new 0 axis -- shape torch.Size([3, H*W])

        # repeat grid num_batch times
        sampling_grid = sampling_grid.repeat((batch_num,1,1)) # shape torch.Size([B, 3, H*W])

        # cast to float32 (required for matmul)
        theta = transfer_parameter.float()
        sampling_grid = sampling_grid.float()

        if use_GPU:
            sampling_grid = sampling_grid.cuda()

        # transform the sampling grid - batch multiply
        batch_grids = torch.matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 3, H*W)

        # reshape to (num_batch, 3, H, W)
        batch_grids = batch_grids.reshape([batch_num, 3, Height, Width])

        # generate sample grid location list
        x_s = batch_grids[:, 0, :, :] / (batch_grids[:, 2, :, :] + 1e-12)
        y_s = batch_grids[:, 1, :, :] / (batch_grids[:, 2, :, :] + 1e-12)

        return x_s, y_s


    def sample_grid_generator_affine(self, Height, Width, transfer_parameter, use_GPU):
        batch_num = transfer_parameter.shape[0]
        # reshape theta to (B, 2, 3)
        transfer_parameter = transfer_parameter.reshape((-1, 2, 3))

        # create normalized 2D grid
        x = torch.linspace(-1.0, 1.0, Width)
        y = torch.linspace(-1.0, 1.0, Height)
        y_t, x_t = torch.meshgrid((y, x)) # here !!!!!!!! different return order as TF
        # NEED t.meshgrid((y, x)) not t.meshgrid((x, y))  Only For Pytorch Version!!!!!!!!

        # flatten
        x_t_flat = x_t.reshape((-1,))
        y_t_flat = y_t.reshape((-1,))

        # reshape to [x_t, y_t , 1] - (homogeneous form)
        ones = torch.ones_like(x_t_flat)
        sampling_grid = torch.stack([x_t_flat, y_t_flat, ones]) # concatenate along new 0 axis -- shape torch.Size([3, H*W])

        # repeat grid num_batch times
        sampling_grid = sampling_grid.repeat((batch_num,1,1)) # shape torch.Size([B, 3, H*W])

        # cast to float32 (required for matmul)
        theta = transfer_parameter.float()
        sampling_grid = sampling_grid.float()

        if use_GPU:
            sampling_grid = sampling_grid.cuda()

        # transform the sampling grid - batch multiply
        batch_grids = torch.matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 2, H*W)

        # reshape to (num_batch, 2, H, W)
        batch_grids = batch_grids.reshape([batch_num, 2, Height, Width])

        # generate sample grid location list
        x_s = batch_grids[:, 0, :, :]
        y_s = batch_grids[:, 1, :, :]

        return x_s, y_s


    def bilinear_sampler(self, input_fmap, x, y):
        def get_pixel_value(img, x, y):
            """
            Utility function to get pixel value for coordinate
            vectors x and y from a  4D tensor image.

            Input
            -----
            - img: tensor of shape (B, H, W, C) in TF, (B, C, H, W) in Pytorch
            - x: tensor of shape (B, Ht, Wt)
            - y: tensor of shape (B, Ht, Wt)

            Returns
            -------
            - output: tensor of shape (B, H, W, C) in TF, (B, C, H, W) in Pytorch
            """
            shape = x.shape
            batch_size = shape[0]
            height = shape[1]
            width = shape[2]

            batch_idx = torch.arange(0, batch_size,dtype=torch.int64)
            batch_idx = batch_idx.reshape(batch_size, 1, 1)
            b = batch_idx.repeat((1, height, width)) # shape (B,H,W)

            x = x.long() # indices must be Long type
            y = y.long()

            # when use stack need this -- (B,H,W,C) type
            # indices = t.stack((b,y,x),dim=3)
            # idx1, idx2, idx3 = indices.chunk(3,dim=3)
            # res = img[idx1,idx2,idx3].squeeze(3)
            # return res

            # not use stack -- (B,H,W,C) type
            # idx1, idx2, idx3 = b, y, x
            # res = img[idx1,idx2,idx3]

            # not use stack -- (B,C,H,W) type #Success
            idx1, idx2, idx3 = b, y, x
            res = img[idx1, :, idx2, idx3] # !!!!! torch.Size([B, H, W, C])

            # not use stack -- (B,C,H,W) type
            # idx1, idx2, idx3 = b, y, x
            # img = img.permute((0,2,3,1))
            # res = img[idx1, idx2,idx3]
            # res = res.permute((0,3,1,2))

            # print('pixel sampler res: {}'.format(res.shape)) # torch.Size([4, 600, 700, 3])
            return res


        """
        Performs bilinear sampling of the input images according to the
        normalized coordinates provided by the sampling grid. Note that
        the sampling is done identically for each channel of the input.

        To test if the function works properly, output image should be
        identical to input image when theta is initialized to identity
        transform.

        Input
        -----
        - img: batch of images in (B, H, W, C) layout.
        - grid: x, y which is the output of affine_grid_generator. -- 3D tensor (B, H, W)

        Returns
        -------
        - out: interpolated images according to grids. Same size as grid.
        """
        H = input_fmap.shape[2] # Pytorch (B, C, H, W)
        W = input_fmap.shape[3] # Pytorch (B, C, H, W)
        max_y = H - 1
        max_x = W - 1

        # rescale x and y to [0, W-1/H-1]
        x = x.float()
        y = y.float()
        x = 0.5 * ((x + 1.0) * float(max_x-1))
        y = 0.5 * ((y + 1.0) * float(max_y-1))

        # grab 4 nearest corner points for each (x_i, y_i)
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        # get pixel value at corner coords
        Ia = get_pixel_value(input_fmap, x0, y0)
        Ib = get_pixel_value(input_fmap, x0, y1)
        Ic = get_pixel_value(input_fmap, x1, y0)
        Id = get_pixel_value(input_fmap, x1, y1)

        # recast as float for delta calculation
        x0 = x0.float()
        x1 = x1.float()
        y0 = y0.float()
        y1 = y1.float()

        # calculate deltas
        wa = (x1-x) * (y1-y)
        wb = (x1-x) * (y-y0)
        wc = (x-x0) * (y1-y)
        wd = (x-x0) * (y-y0)

        # add dimension for addition
        wa = wa.unsqueeze(3)
        wb = wb.unsqueeze(3)
        wc = wc.unsqueeze(3)
        wd = wd.unsqueeze(3)

        # compute output
        out = wa*Ia+wb*Ib+wc*Ic+wd*Id # !!!!!  torch.Size([B, H, W, C])

        # pytorch transpose
        out = out.permute((0,3,1,2)) # torch.Size([B, C, H, W])

        # print('STN output: {}'.format(out.shape))
        return out


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


def STN_Visualization(batch_img_path, batch_parameters, img_size, if_affine, use_GPU):
    # batch_img_path: list[B,]
    # batch_parameters: tensor[B,9/6]
    # return: trans_list(np,uint8), raw_img_list(np,uint8)
    def torch_array2img_array(x):
        # input: 3D float torch image
        # return [0,255] uint8
        x = x.cpu().numpy()
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
        return x.astype('uint8')

    def img2array(data_path, desired_size=None, expand=False, view=False):
        """Loads an RGB image as a 3D or 4D [0,1] float numpy array."""
        # Good Code !!!!!
        img = Image.open(data_path)
        img = img.convert('RGB')
        if desired_size:
            img = img.resize((desired_size[1], desired_size[0]))
        if view:
            img.show()
        x = np.asarray(img, dtype='float32')
        if expand:
            x = np.expand_dims(x, axis=0)
        x /= 255.0
        return x

    batch_img = []
    for img_path in batch_img_path:
        img = img2array(img_path,img_size,True)
        img = torch.tensor(img,dtype=torch.float32).permute((0,3,1,2)) # [1,3,H,W]
        batch_img.append(img)

    STN_model = SpatialTransformer(if_manual_test=True,use_GPU=use_GPU,if_affine=if_affine)
    if if_affine:
        batch_parameters = batch_parameters[:,:6]
    if use_GPU:
        batch_img = [i.cuda() for i in batch_img]
        batch_parameters = batch_parameters.cuda()
        STN_model.to(torch.device('cuda:0'))
    STN_model.eval()

    with torch.no_grad():
        raw_list = []
        trans_list = []
        for idx,img in enumerate(batch_img):
            parameters = batch_parameters[idx]
            STN_model.localisation_net.fc_b.data = parameters
            recv_trans_img = STN_model(img) # float tensor([1,3,H,W])
            trans_img = torch_array2img_array(recv_trans_img.squeeze(0).permute(1,2,0))
            raw_img = torch_array2img_array(img.squeeze(0).permute(1,2,0))
            trans_list.append(trans_img)
            raw_list.append(raw_img)
    return trans_list, raw_list



if __name__ == '__main__':

    import os
    import matplotlib.pyplot as plt

    Data_dir = '../Market/bounding_box_train'

    def array2img_array(x):
        # x = np.asarray(x) # tf
        # x = x.numpy() # when input is tensor with requires_grad # manual test open this, real test cancel this !!!
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
        return x.astype('uint8')

    def img2array(data_path, desired_size=None, expand=False, view=False):
        """Loads an RGB image as a 3D or 4D [0,1] float numpy array."""
        # Good Code !!!!!
        img = Image.open(data_path)
        img = img.convert('RGB')
        if desired_size:
            img = img.resize((desired_size[1], desired_size[0]))
        if view:
            img.show()
        x = np.asarray(img, dtype='float32')
        if expand:
            x = np.expand_dims(x, axis=0)
        x /= 255.0
        return x

    def stn_test():
        degree = 45
        parameter_rotate = np.array([
            [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
            [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0],
            [0.,0.,1.]
        ])
        parameter_identity = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        parameter_scale = np.array([[0.5,0.,0.],[0.,0.5,0.],[0.,0.,1.]])
        parameter_project = np.array([[1.,0.,0.],[0.,1.,0.],[0.5,0.3,1.]])
        parameter_scale_project = np.array([[1.5,0.,0.],[0.,1.5,0.],[0.5,-0.2,1.]])
        parameter_shear = np.array([[1.,0.,0.],[0.5,1.,0.],[0.,0.,1.]])

        # img_1 = img2array(os.path.join(Data_dir,'cat1.jpg'),(600,700),True)
        # img_2 = img2array(os.path.join(Data_dir,'cat2.jpg'),(600,700),True)
        # img_3 = img2array(os.path.join(Data_dir,'cat3.jpg'),(600,700),True)
        # img_4 = img2array(os.path.join(Data_dir,'cat4.jpg'),(600,700),True) # Problems here !!!! must same HW

        img_1 = img2array(os.path.join(Data_dir, '0002_c1s1_000776_01.jpg'), (256, 128), True)
        img_2 = img2array(os.path.join(Data_dir, '0002_c2s1_000801_01.jpg'), (256, 128), True)
        img_3 = img2array(os.path.join(Data_dir, '0007_c2s3_071052_01.jpg'), (256, 128), True)
        img_4 = img2array(os.path.join(Data_dir, '0011_c6s4_002502_01.jpg'), (256, 128), True)

        image_all = np.concatenate((img_1,img_2,img_3,img_4),axis=0)
        print('image_all shape: {0}'.format(image_all.shape))


        random_input = torch.tensor(image_all,dtype=torch.float32)
        random_label = np.array([1,2,3,4])
        # to pytorch (B,C,H,W)
        random_input = random_input.permute((0,3,1,2))
        print('transposed input shape: {0}'.format(random_input.shape))

        myModel_affine = SpatialTransformer(initial_parameter=[[0.9,0.,0.],[0.,0.9,0.],[0.,0.,1.0]],
                                            if_manual_test=True) # ,[100,150]
        myModel_project = SpatialTransformer(parameter_project,if_manual_test=True) # ,[100,150]

        # Redundancy ?????  Optional ?????
        myModel_project.apply(weights_init_kaiming)
        myModel_affine.apply(weights_init_kaiming)

        with torch.no_grad():
            result_affine = myModel_affine(random_input)
            result_project = myModel_project(random_input)

            for i in range(4):
                fig = plt.figure(figsize = (12,6))
                subfig = fig.add_subplot(1,3,1)
                subfig.imshow(image_all[i])
                subfig.set_title('Raw img')
                subfig = fig.add_subplot(1,3,2)
                recv_img = array2img_array(result_affine[i].permute((1,2,0)))
                subfig.imshow(recv_img)
                subfig.set_title('Affine img')
                subfig = fig.add_subplot(1,3,3)
                recv_img = array2img_array(result_project[i].permute((1,2,0)))
                subfig.imshow(recv_img)
                subfig.set_title('Project img')

                plt.show()


            for count in range(4):
                # random_select = np.random.randint(0,len(x_train) + 1,5)
                # random_input = x_train[random_select]
                # random_label = y_train[random_select]
                #
                # result = spatial_transformer_project(random_input[:,:,:,np.newaxis],parameter_scale)
                # random_input = t.tensor(image_all)
                # random_label = np.array([1,2,3,4])

                result = myModel_project(random_input)

                fig = plt.figure(figsize = (10,6))
                for i in range(4):
                    subfig = fig.add_subplot(2,4,i+1)
                    subfig.imshow(image_all[i])
                    subfig.set_title('raw_{0}'.format(random_label[i]))
                    subfig = fig.add_subplot(2,4,i+1+4)
                    # transfer = np.squeeze(result[i])
                    # transfer = array2img_array(transfer)
                    transfer = array2img_array(result[i].permute((1,2,0)))
                    subfig.imshow(transfer)
                    subfig.set_title('transformed_{0}'.format(random_label[i]))

                plt.show()

    def stn_real_situation_test():
        degree = 45
        parameter_rotate = np.array([
            [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
            [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0],
            [0.,0.,1.]
        ])
        parameter_identity = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        parameter_scale = np.array([[0.5,0.,0.],[0.,0.5,0.],[0.,0.,1.]])
        parameter_project = np.array([[1.,0.,0.],[0.,1.,0.],[0.5,0.3,1.]])
        parameter_scale_project = np.array([[1.5,0.,0.],[0.,1.5,0.],[0.5,-0.2,1.]])
        parameter_shear = np.array([[1.,0.,0.],[0.5,1.,0.],[0.,0.,1.]])

        # img_1 = img2array(os.path.join(Data_dir,'cat1.jpg'),(600,700),True)
        # img_2 = img2array(os.path.join(Data_dir,'cat2.jpg'),(600,700),True)
        # img_3 = img2array(os.path.join(Data_dir,'cat3.jpg'),(600,700),True)
        # img_4 = img2array(os.path.join(Data_dir,'cat4.jpg'),(600,700),True)

        img_1 = img2array(os.path.join(Data_dir,'0002_c1s1_000776_01.jpg'),(16,8),True)
        img_2 = img2array(os.path.join(Data_dir,'0002_c2s1_000801_01.jpg'),(16,8),True)
        img_3 = img2array(os.path.join(Data_dir,'0007_c2s3_071052_01.jpg'),(16,8),True)
        img_4 = img2array(os.path.join(Data_dir,'0011_c6s4_002502_01.jpg'),(16,8),True)

        # img_1 = img2array(os.path.join(Data_dir,'cat1.jpg'),(56,56),True)
        # img_2 = img2array(os.path.join(Data_dir,'cat2.jpg'),(56,56),True)
        # img_3 = img2array(os.path.join(Data_dir,'cat3.jpg'),(56,56),True)
        # img_4 = img2array(os.path.join(Data_dir,'cat4.jpg'),(56,56),True)


        image_all = np.concatenate((img_1,img_2,img_3,img_4),axis=0)
        print('image_all shape: {0}'.format(image_all.shape))


        random_input = torch.tensor(image_all,dtype=torch.float32)
        random_label = np.array([1,2,3,4])

        random_input = random_input.permute((0,3,1,2)) # to pytorch (B,C,H,W)
        print('transposed input shape: {0}'.format(random_input.shape))

        myModel_affine = SpatialTransformer(initial_parameter=[[0.9,0.,0.],[0.,0.9,0.],[0.,0.,1.0]],
                                            if_manual_test=False,out_dim=[14,14],
                                            use_GPU=True,if_affine=False)
        myModel_project = SpatialTransformer(parameter_scale_project,if_manual_test=False,
                                             use_GPU=True,if_affine=True) # ,[100,150]
        myModel_affine.cuda()
        myModel_project.cuda()
        print(myModel_project)
        for idx,(name, param) in enumerate(myModel_project.named_parameters()):
            print('{} {}: {}'.format(idx,name,param.shape))

        # Redundancy ?????  Optional ?????
        # myModel_project.apply(weights_init_kaiming)
        # myModel_affine.apply(weights_init_kaiming)

        # with torch.no_grad():
        synthesize_input = random_input.repeat((1,342,1,1))
        synthesize_input = synthesize_input[:,:1024,:,:]
        synthesize_input = synthesize_input.cuda()
        result_affine = myModel_affine(synthesize_input)
        result_project = myModel_project(synthesize_input)

        result_affine.mean().backward()
        result_project.mean().backward()

        print('fc_W grad: {}'.format(myModel_project.localisation_net.fc_W.grad))
        print('fc_b grad: {}'.format(myModel_project.localisation_net.fc_b.grad))

        for i in range(4):
            fig = plt.figure(figsize = (12,6))
            subfig = fig.add_subplot(1,3,1)
            subfig.imshow(image_all[i])
            subfig.set_title('Raw img')
            subfig = fig.add_subplot(1,3,2)
            recv_img = array2img_array(result_affine[i].permute((1,2,0))[:,:,:3].cpu().detach().numpy())
            subfig.imshow(recv_img)
            subfig.set_title('Affine img')
            subfig = fig.add_subplot(1,3,3)
            recv_img = array2img_array(result_project[i].permute((1,2,0))[:,:,:3].cpu().detach().numpy())
            subfig.imshow(recv_img)
            subfig.set_title('Project img')

            plt.show()

    def STN_Visualization_test():
        degree = 45
        parameter_rotate = np.array([
            [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
            [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0],
            [0.,0.,1.]
        ])
        parameter_identity = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        parameter_scale = np.array([[0.5,0.,0.],[0.,0.5,0.],[0.,0.,1.]])
        parameter_project = np.array([[1.,0.,0.],[0.,1.,0.],[0.5,0.3,1.]])
        parameter_scale_project = np.array([[1.5,0.,0.],[0.,1.5,0.],[0.5,-0.2,1.]])
        parameter_shear = np.array([[1.,0.,0.],[0.5,1.,0.],[0.,0.,1.]])

        batch_path_list = [os.path.join(Data_dir,'0002_c1s1_000776_01.jpg'),
                           os.path.join(Data_dir,'0002_c2s1_000801_01.jpg'),
                           os.path.join(Data_dir,'0007_c2s3_071052_01.jpg'),
                           os.path.join(Data_dir,'0011_c6s4_002502_01.jpg')]
        batch_parameters = [parameter_rotate,parameter_shear,
                            parameter_project,parameter_scale_project]
        batch_parameters = torch.tensor(batch_parameters,dtype=torch.float32).reshape((4,-1))

        recv_trans, recv_raw = STN_Visualization(batch_path_list,batch_parameters,[256,128],
                                                 if_affine=False,use_GPU=True)

        fig = plt.figure(figsize=(16,6))
        for idx, (trans, raw, path) in enumerate(zip(recv_trans,recv_raw,batch_path_list)):
            subfig = fig.add_subplot(2,4,idx+1)
            subfig.set_title(path)
            subfig.imshow(raw)
            subfig = fig.add_subplot(2,4,idx+1+4)
            subfig.imshow(trans)
        plt.show()


    # stn_test() # test sample logic
    stn_real_situation_test() # test localization logic
    # STN_Visualization_test()



