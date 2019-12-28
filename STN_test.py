import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math
import PIL
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


Data_dir = '../ViewInvariantNet/data'

def array2img_array(x):
    # x = np.asarray(x) # tf
    x = x.numpy()
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


def localisation_part(input_fmap, initial_parameter):
    # initial_parameter: [2,3] or [3,3] nparray
    # input_map: 4D tensor
    Batch = input_fmap.shape[0]
    transfer_parameter = np.reshape(initial_parameter,(-1,)).astype(dtype=np.float32)

    # #######################
    #
    # stack conv layer on input_fmap
    # need squeeze for final conv layer !!!!!!
    #
    # #######################

    # for simulate test !!!!!
    net = t.zeros((Batch,100))

    # final layer
    in_size = net.shape[1]
    out_size = np.size(transfer_parameter)
    W_loc = t.tensor(t.zeros((in_size, out_size)),dtype=t.float32,requires_grad=True)
    b_loc = t.tensor(transfer_parameter,dtype=t.float32,requires_grad=True)

    loc = t.matmul(net,W_loc) + b_loc
    return loc

def sample_grid_generator_affine(Height, Width, transfer_parameter):
    # PASS
    batch_num = transfer_parameter.shape[0]
    # reshape theta to (B, 2, 3)
    transfer_parameter = transfer_parameter.reshape((-1, 2, 3))

    # create normalized 2D grid
    x = t.linspace(-1.0, 1.0, Width)
    y = t.linspace(-1.0, 1.0, Height)
    # y_t, x_t = t.meshgrid((x, y)) # here !!!!!!!! different return order as TF && When H != W occur Wrong!!!!
    y_t, x_t = t.meshgrid((y, x))

    # flatten
    x_t_flat = x_t.reshape((-1,))
    y_t_flat = y_t.reshape((-1,))

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = t.ones_like(x_t_flat)
    sampling_grid = t.stack([x_t_flat, y_t_flat, ones]) # concatenate along new 0 axis -- shape torch.Size([3, H*W])

    # repeat grid num_batch times
    sampling_grid = sampling_grid.repeat((batch_num,1,1)) # shape torch.Size([B, 3, H*W])

    # cast to float32 (required for matmul)
    theta = transfer_parameter.float()
    sampling_grid = sampling_grid.float()

    # transform the sampling grid - batch multiply
    batch_grids = t.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, 2, Ht, Wt)
    batch_grids = batch_grids.reshape([batch_num, 2, Height, Width])

    # generate sample grid location list
    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    return x_s, y_s


def bilinear_sampler(input_fmap,x,y):
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

        batch_idx = t.arange(0, batch_size,dtype=t.int64)
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
        res = img[idx1, :, idx2, idx3]

        # not use stack -- (B,C,H,W) type
        # idx1, idx2, idx3 = b, y, x
        # img = img.permute((0,2,3,1))
        # res = img[idx1, idx2,idx3]
        # res = res.permute((0,3,1,2))
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
    # H = input_fmap.shape[1] # (B, H, W, C)
    # W = input_fmap.shape[2] # (B, H, W, C)
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
    x0 = t.floor(x).int()
    x1 = x0 + 1
    y0 = t.floor(y).int()
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = t.clamp(x0, 0, max_x)
    x1 = t.clamp(x1, 0, max_x)
    y0 = t.clamp(y0, 0, max_y)
    y1 = t.clamp(y1, 0, max_y)

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
    out = wa*Ia+wb*Ib+wc*Ic+wd*Id

    return out



# ########################## Affine Part
def spatial_transformer_affine(input_fmap, initial_parameter, out_dim = None):
    '''

    Input feature map  must be float32 type !!!!!!!!!!!!
    :param input_fmap:
    :param initial_affine_parameter:
    :param out_dim:
    :return:
    '''
    # Height = input_fmap.shape[1] # (B, H, W, C)
    # Width = input_fmap.shape[2] # (B, H, W, C)
    Height = input_fmap.shape[2] # (B, C, H, W)
    Width = input_fmap.shape[3] # (B, C, H, W)
    fmap = input_fmap.float()

    # return 2-D Batch * 6/9
    affine_parameter = localisation_part(fmap, initial_parameter)
    print(affine_parameter)
    # Pass

    if out_dim:
        x_s, y_s = sample_grid_generator_affine(out_dim[0], out_dim[1], affine_parameter)
    else:
        x_s, y_s = sample_grid_generator_affine(Height, Width, affine_parameter)

    out_fmap = bilinear_sampler(fmap, x_s, y_s)

    return out_fmap

def affine_model_test():
    degree = 45
    parameter_rotate = np.array([
        [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
        [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0]
    ])
    parameter_identity = np.array([[1.,0.,0.],[0.,1.,0.]])
    parameter_scale = np.array([[0.5,0.,0.],[0.,0.5,0.]])

    img_1 = img2array(os.path.join(Data_dir,'cat1.jpg'),(600,700),True)
    img_2 = img2array(os.path.join(Data_dir,'cat2.jpg'),(600,700),True)
    img_3 = img2array(os.path.join(Data_dir,'cat3.jpg'),(600,700),True)
    img_4 = img2array(os.path.join(Data_dir,'cat4.jpg'),(600,700),True)

    image_all = np.concatenate((img_1,img_2,img_3,img_4),axis=0)

    # pytorch (B,C,H,W)
    image_all = np.transpose(image_all,(0,3,1,2))
    print('image_all transposed shape: {0}'.format(image_all.shape))

    with t.no_grad():
        for count in range(4):
            # random_select = np.random.randint(0,len(x_train) + 1,5)
            # random_input = x_train[random_select]
            # random_label = y_train[random_select]
            #
            # result = spatial_transformer_affine(random_input[:,:,:,np.newaxis],parameter_rotate)
            random_input = t.tensor(image_all,dtype=t.float32)
            random_label = np.array([1,2,3,4])
            # result = spatial_transformer_affine(random_input,parameter_rotate,[100,100]) # looks work
            result = spatial_transformer_affine(random_input,parameter_rotate,[150,100]) # occur problem !!!!

            fig = plt.figure(figsize = (10,6))
            for i in range(4):
                subfig = fig.add_subplot(2,4,i+1)
                # pytorch transpose
                raw_input = np.transpose(image_all[i],(1,2,0))
                subfig.imshow(raw_input)
                # subfig.imshow(image_all[i])
                subfig.set_title('raw_{0}'.format(random_label[i]))
                subfig = fig.add_subplot(2,4,i+1+4)
                # transfer = np.squeeze(result[i])
                # transfer = array2img_array(transfer)
                transfer = array2img_array(result[i])
                print(transfer.shape) # (H, W, C)

                # pytorch transpose
                # transfer = np.transpose(transfer,(1,2,0)) #
                subfig.imshow(transfer)
                subfig.set_title('transformed_{0}'.format(random_label[i]))

            plt.show()

# affine_model_test()
# PASS
# ########################## Affine Part




# ########################## Project Part

def sample_grid_generator_project(Height, Width, transfer_parameter):
    # PASS
    batch_num = transfer_parameter.shape[0]
    # reshape theta to (B, 3, 3)
    transfer_parameter = transfer_parameter.reshape((-1, 3, 3))

    # create normalized 2D grid
    x = t.linspace(-1.0, 1.0, Width)
    y = t.linspace(-1.0, 1.0, Height)
    y_t, x_t = t.meshgrid((y, x))

    # flatten
    x_t_flat = x_t.reshape((-1,))
    y_t_flat = y_t.reshape((-1,))

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = t.ones_like(x_t_flat)
    sampling_grid = t.stack([x_t_flat, y_t_flat, ones]) # concatenate along new 0 axis -- shape torch.Size([3, H*W])

    # repeat grid num_batch times
    sampling_grid = sampling_grid.repeat((batch_num,1,1)) # shape torch.Size([B, 3, H*W])

    # cast to float32 (required for matmul)
    theta = transfer_parameter.float()
    sampling_grid = sampling_grid.float()

    # transform the sampling grid - batch multiply
    batch_grids = t.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, 2, H, W)
    batch_grids = batch_grids.reshape([batch_num, 3, Height, Width])

    # generate sample grid location list
    x_s = batch_grids[:, 0, :, :] / batch_grids[:,2,:,:]
    y_s = batch_grids[:, 1, :, :] / batch_grids[:,2,:,:]

    return x_s, y_s

def spatial_transformer_project(input_fmap, initial_parameter, out_dim = None):
    '''

    Input feature map  must be float32 type !!!!!!!!!!!!
    :param input_fmap:
    :param initial_affine_parameter:
    :param out_dim:
    :return:
    '''
    # Height = input_fmap.shape[1] # (B, H, W, C)
    # Width = input_fmap.shape[2] # (B, H, W, C)
    Height = input_fmap.shape[2] # (B, C, H, W)
    Width = input_fmap.shape[3] # (B, C, H, W)
    fmap = input_fmap.float()

    # return 2-D Batch * 6/9
    project_parameter = localisation_part(fmap, initial_parameter)
    print(project_parameter)
    # Pass

    if out_dim:
        x_s, y_s = sample_grid_generator_project(out_dim[0], out_dim[1], project_parameter)
    else:
        x_s, y_s = sample_grid_generator_project(Height, Width, project_parameter)

    out_fmap = bilinear_sampler(fmap, x_s, y_s)

    return out_fmap

def project_model_test():
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

    img_1 = img2array(os.path.join(Data_dir,'cat1.jpg'),(600,700),True)
    img_2 = img2array(os.path.join(Data_dir,'cat2.jpg'),(600,700),True)
    img_3 = img2array(os.path.join(Data_dir,'cat3.jpg'),(600,700),True)
    img_4 = img2array(os.path.join(Data_dir,'cat4.jpg'),(600,700),True)

    image_all = np.concatenate((img_1,img_2,img_3,img_4),axis=0)

    # pytorch (B,C,H,W)
    image_all = np.transpose(image_all,(0,3,1,2))
    print('image_all shape: {0}'.format(image_all.shape))

    with t.no_grad():
        for count in range(4):
            # random_select = np.random.randint(0,len(x_train) + 1,5)
            # random_input = x_train[random_select]
            # random_label = y_train[random_select]
            #
            # result = spatial_transformer_affine(random_input[:,:,:,np.newaxis],parameter_rotate)
            random_input = t.tensor(image_all,dtype=t.float32)
            random_label = np.array([1,2,3,4])
            result = spatial_transformer_project(random_input,parameter_project,[150,200])

            fig = plt.figure(figsize = (10,6))
            for i in range(4):
                subfig = fig.add_subplot(2,4,i+1)
                # pytorch transpose
                raw_input = np.transpose(image_all[i],(1,2,0))
                subfig.imshow(raw_input)
                subfig.set_title('raw_{0}'.format(random_label[i]))
                subfig = fig.add_subplot(2,4,i+1+4)
                # transfer = np.squeeze(result[i])
                # transfer = array2img_array(transfer)
                transfer = array2img_array(result[i])
                print(transfer.shape) # (H, W, C)

                # pytorch transpose
                # transfer = np.transpose(transfer,(1,2,0)) #
                subfig.imshow(transfer)
                subfig.set_title('transformed_{0}'.format(random_label[i]))

            plt.show()

# project_model_test()

# ########################## Project Part



# ########################## General Part

def sample_grid_generator(Height, Width, transfer_parameter):
    # PASS
    batch_num = transfer_parameter.shape[0]
    # reshape theta to (B, 3, 3)
    transfer_parameter = transfer_parameter.reshape((-1, 3, 3))

    # create normalized 2D grid
    x = t.linspace(-1.0, 1.0, Width)
    y = t.linspace(-1.0, 1.0, Height)
    y_t, x_t = t.meshgrid((y, x)) # here !!!!!!!! different return order as TF
    # NEED t.meshgrid((y, x)) not t.meshgrid((x, y))  Only For Pytorch Version!!!!!!!!

    # flatten
    x_t_flat = x_t.reshape((-1,))
    y_t_flat = y_t.reshape((-1,))

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = t.ones_like(x_t_flat)
    sampling_grid = t.stack([x_t_flat, y_t_flat, ones]) # concatenate along new 0 axis -- shape torch.Size([3, H*W])

    # repeat grid num_batch times
    sampling_grid = sampling_grid.repeat((batch_num,1,1)) # shape torch.Size([B, 3, H*W])

    # cast to float32 (required for matmul)
    theta = transfer_parameter.float()
    sampling_grid = sampling_grid.float()

    # transform the sampling grid - batch multiply
    batch_grids = t.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 3, H*W)

    # reshape to (num_batch, 3, H, W)
    batch_grids = batch_grids.reshape([batch_num, 3, Height, Width])

    # generate sample grid location list
    x_s = batch_grids[:, 0, :, :] / (batch_grids[:, 2, :, :] + 1e-12)
    y_s = batch_grids[:, 1, :, :] / (batch_grids[:, 2, :, :] + 1e-12)

    return x_s, y_s

def spatial_transformer(input_fmap, initial_parameter, out_dim = None):
    '''
    :param input_fmap: must input float32 type
    :param initial_parameter: 2D
    :param out_dim:
    :return:
    '''
    Height = input_fmap.shape[2] # (B, C, H, W)
    Width = input_fmap.shape[3] # (B, C, H, W)
    fmap = input_fmap.float()

    # return 2-D Batch * 9
    transfer_parameter = localisation_part(fmap, initial_parameter)
    print(transfer_parameter)
    # Pass

    if out_dim:
        x_s, y_s = sample_grid_generator(out_dim[0], out_dim[1], transfer_parameter)
    else:
        x_s, y_s = sample_grid_generator(Height, Width, transfer_parameter)

    out_fmap = bilinear_sampler(fmap, x_s, y_s)

    return out_fmap


def stn_model_test():
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

    img_1 = img2array(os.path.join(Data_dir,'cat1.jpg'),(600,700),True)
    img_2 = img2array(os.path.join(Data_dir,'cat2.jpg'),(600,700),True)
    img_3 = img2array(os.path.join(Data_dir,'cat3.jpg'),(600,700),True)
    img_4 = img2array(os.path.join(Data_dir,'cat4.jpg'),(600,700),True) # Problems here !!!! must same HW

    image_all = np.concatenate((img_1,img_2,img_3,img_4),axis=0)
    print('image_all shape: {0}'.format(image_all.shape))


    random_input = t.tensor(image_all,dtype=t.float32)
    random_label = np.array([1,2,3,4])
    # to pytorch (B,C,H,W)
    random_input = random_input.permute((0,3,1,2))
    print('transposed input shape: {0}'.format(random_input.shape))

    with t.no_grad():
        result_project = spatial_transformer(random_input,parameter_project)
        result_affine = spatial_transformer(random_input,parameter_scale)

        for i in range(4):
            fig = plt.figure(figsize = (12,6))
            subfig = fig.add_subplot(1,3,1)
            subfig.imshow(image_all[i])
            subfig.set_title('Raw img')
            subfig = fig.add_subplot(1,3,2)
            recv_img = array2img_array(result_affine[i])
            subfig.imshow(recv_img)
            subfig.set_title('Affine img')
            subfig = fig.add_subplot(1,3,3)
            recv_img = array2img_array(result_project[i])
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

            result = spatial_transformer(random_input,parameter_project)

            fig = plt.figure(figsize = (10,6))
            for i in range(4):
                subfig = fig.add_subplot(2,4,i+1)
                subfig.imshow(image_all[i])
                subfig.set_title('raw_{0}'.format(random_label[i]))
                subfig = fig.add_subplot(2,4,i+1+4)
                # transfer = np.squeeze(result[i])
                # transfer = array2img_array(transfer)
                transfer = array2img_array(result[i])
                subfig.imshow(transfer)
                subfig.set_title('transformed_{0}'.format(random_label[i]))

            plt.show()


# stn_model_test()

# ########################## General Part
# #######################################################################################




# Pytorch Module Version

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


class LocalisationNet(nn.Module):
    def __init__(self, initial_parameter):
        super(LocalisationNet, self).__init__()
        # initial_parameter: [3,3] nparray
        transfer_parameter = np.reshape(initial_parameter,(-1,)).astype(dtype=np.float32)

        # #######################################
        #
        #  Define NN Conv Layers here!!!!!!!!
        #
        # #######################################

        input_size = 200
        out_size = np.size(transfer_parameter)
        self.loc_fc_W = nn.Parameter(t.zeros(input_size, out_size))
        self.loc_fc_b = nn.Parameter(t.tensor(transfer_parameter))
        # init loc_fc_W, loc_fc_b finish

        self.kaiming_init()

    def forward(self, input_featuremap):
        # input_map: 4D tensor

        # #######################################
        #
        #  Stack NN Conv Layers here!!!!!!!!
        #
        # #######################################

        batch_size = input_featuremap.shape[0]
        tem_map = t.zeros((batch_size,200))
        res = t.matmul(tem_map,self.loc_fc_W) + self.loc_fc_b
        return res


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
    def __init__(self, initial_parameter, out_dim = None):
        super(SpatialTransformer, self).__init__()
        self.out_dim = out_dim
        self.initial_parameter = initial_parameter
        self.localisation_net = LocalisationNet(self.initial_parameter)

        self.kaiming_init()

    def forward(self, input_fmap):
        Height = input_fmap.shape[2] # (B, C, H, W)
        Width = input_fmap.shape[3] # (B, C, H, W)
        fmap = input_fmap.float()

        affine_parameter = self.localisation_net(fmap)
        # return 2-D Batch * 9

        if self.out_dim is not None:
            x_s, y_s = self.sample_grid_generator(self.out_dim[0], self.out_dim[1], affine_parameter)
        else:
            x_s, y_s = self.sample_grid_generator(Height, Width, affine_parameter)

        out_fmap = self.bilinear_sampler(fmap, x_s, y_s)

        return out_fmap


    def sample_grid_generator(self, Height, Width, transfer_parameter):
        # PASS
        batch_num = transfer_parameter.shape[0]
        # reshape theta to (B, 3, 3)
        transfer_parameter = transfer_parameter.reshape((-1, 3, 3))

        # create normalized 2D grid
        x = t.linspace(-1.0, 1.0, Width)
        y = t.linspace(-1.0, 1.0, Height)
        y_t, x_t = t.meshgrid((y, x)) # here !!!!!!!! different return order as TF
        # NEED t.meshgrid((y, x)) not t.meshgrid((x, y))  Only For Pytorch Version!!!!!!!!

        # flatten
        x_t_flat = x_t.reshape((-1,))
        y_t_flat = y_t.reshape((-1,))

        # reshape to [x_t, y_t , 1] - (homogeneous form)
        ones = t.ones_like(x_t_flat)
        sampling_grid = t.stack([x_t_flat, y_t_flat, ones]) # concatenate along new 0 axis -- shape torch.Size([3, H*W])

        # repeat grid num_batch times
        sampling_grid = sampling_grid.repeat((batch_num,1,1)) # shape torch.Size([B, 3, H*W])

        # cast to float32 (required for matmul)
        theta = transfer_parameter.float()
        sampling_grid = sampling_grid.float()

        # transform the sampling grid - batch multiply
        batch_grids = t.matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 3, H*W)

        # reshape to (num_batch, 3, H, W)
        batch_grids = batch_grids.reshape([batch_num, 3, Height, Width])

        # generate sample grid location list
        x_s = batch_grids[:, 0, :, :] / (batch_grids[:, 2, :, :] + 1e-12)
        y_s = batch_grids[:, 1, :, :] / (batch_grids[:, 2, :, :] + 1e-12)

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

            batch_idx = t.arange(0, batch_size,dtype=t.int64)
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
            res = img[idx1, :, idx2, idx3]

            # not use stack -- (B,C,H,W) type
            # idx1, idx2, idx3 = b, y, x
            # img = img.permute((0,2,3,1))
            # res = img[idx1, idx2,idx3]
            # res = res.permute((0,3,1,2))
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
        x0 = t.floor(x).int()
        x1 = x0 + 1
        y0 = t.floor(y).int()
        y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = t.clamp(x0, 0, max_x)
        x1 = t.clamp(x1, 0, max_x)
        y0 = t.clamp(y0, 0, max_y)
        y1 = t.clamp(y1, 0, max_y)

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
        out = wa*Ia+wb*Ib+wc*Ic+wd*Id

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


# myModel.apply(weights_init_kaiming)
# This can cover more situations than use self isinstance version

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

    img_1 = img2array(os.path.join(Data_dir,'cat1.jpg'),(600,700),True)
    img_2 = img2array(os.path.join(Data_dir,'cat2.jpg'),(600,700),True)
    img_3 = img2array(os.path.join(Data_dir,'cat3.jpg'),(600,700),True)
    img_4 = img2array(os.path.join(Data_dir,'cat4.jpg'),(600,700),True) # Problems here !!!! must same HW

    image_all = np.concatenate((img_1,img_2,img_3,img_4),axis=0)
    print('image_all shape: {0}'.format(image_all.shape))


    random_input = t.tensor(image_all,dtype=t.float32)
    random_label = np.array([1,2,3,4])
    # to pytorch (B,C,H,W)
    random_input = random_input.permute((0,3,1,2))
    print('transposed input shape: {0}'.format(random_input.shape))

    myModel_affine = SpatialTransformer(parameter_identity) # ,[100,150]
    myModel_project = SpatialTransformer(parameter_project) # ,[100,150]

    with t.no_grad():
        result_affine = myModel_affine(random_input)
        result_project = myModel_project(random_input)

        for i in range(4):
            fig = plt.figure(figsize = (12,6))
            subfig = fig.add_subplot(1,3,1)
            subfig.imshow(image_all[i])
            subfig.set_title('Raw img')
            subfig = fig.add_subplot(1,3,2)
            recv_img = array2img_array(result_affine[i])
            subfig.imshow(recv_img)
            subfig.set_title('Affine img')
            subfig = fig.add_subplot(1,3,3)
            recv_img = array2img_array(result_project[i])
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
                transfer = array2img_array(result[i])
                subfig.imshow(transfer)
                subfig.set_title('transformed_{0}'.format(random_label[i]))

            plt.show()

stn_test()

# ########################################################################
# For view STN effect, input batch parameters and apply on batch images
class SpatialTransformerView(nn.Module):
    def __init__(self, transform_parameters, out_dim = None):
        super(SpatialTransformerView, self).__init__()
        self.out_dim = out_dim
        self.transform_parameters = transform_parameters # B * 9


    def forward(self, input_fmap):
        Height = input_fmap.shape[2] # (B, C, H, W)
        Width = input_fmap.shape[3] # (B, C, H, W)
        fmap = input_fmap.float()

        affine_parameter = self.transform_parameters
        # return 2-D Batch * 9

        if self.out_dim is not None:
            x_s, y_s = self.sample_grid_generator(self.out_dim[0], self.out_dim[1], affine_parameter)
        else:
            x_s, y_s = self.sample_grid_generator(Height, Width, affine_parameter)

        out_fmap = self.bilinear_sampler(fmap, x_s, y_s)

        return out_fmap


    def sample_grid_generator(self, Height, Width, transfer_parameter):
        # PASS
        batch_num = transfer_parameter.shape[0]
        # reshape theta to (B, 3, 3)
        transfer_parameter = transfer_parameter.reshape((-1, 3, 3))

        # create normalized 2D grid
        x = t.linspace(-1.0, 1.0, Width)
        y = t.linspace(-1.0, 1.0, Height)
        y_t, x_t = t.meshgrid((y, x)) # here !!!!!!!! different return order as TF
        # NEED t.meshgrid((y, x)) not t.meshgrid((x, y))  Only For Pytorch Version!!!!!!!!

        # flatten
        x_t_flat = x_t.reshape((-1,))
        y_t_flat = y_t.reshape((-1,))

        # reshape to [x_t, y_t , 1] - (homogeneous form)
        ones = t.ones_like(x_t_flat)
        sampling_grid = t.stack([x_t_flat, y_t_flat, ones]) # concatenate along new 0 axis -- shape torch.Size([3, H*W])

        # repeat grid num_batch times
        sampling_grid = sampling_grid.repeat((batch_num,1,1)) # shape torch.Size([B, 3, H*W])

        # cast to float32 (required for matmul)
        theta = transfer_parameter.float()
        sampling_grid = sampling_grid.float()

        # transform the sampling grid - batch multiply
        batch_grids = t.matmul(theta, sampling_grid)
        # batch grid has shape (num_batch, 3, H*W)

        # reshape to (num_batch, 3, H, W)
        batch_grids = batch_grids.reshape([batch_num, 3, Height, Width])

        # generate sample grid location list
        x_s = batch_grids[:, 0, :, :] / (batch_grids[:, 2, :, :] + 1e-12)
        y_s = batch_grids[:, 1, :, :] / (batch_grids[:, 2, :, :] + 1e-12)

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

            batch_idx = t.arange(0, batch_size,dtype=t.int64)
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
            res = img[idx1, :, idx2, idx3]

            # not use stack -- (B,C,H,W) type
            # idx1, idx2, idx3 = b, y, x
            # img = img.permute((0,2,3,1))
            # res = img[idx1, idx2,idx3]
            # res = res.permute((0,3,1,2))
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
        x0 = t.floor(x).int()
        x1 = x0 + 1
        y0 = t.floor(y).int()
        y1 = y0 + 1

        # clip to range [0, H-1/W-1] to not violate img boundaries
        x0 = t.clamp(x0, 0, max_x)
        x1 = t.clamp(x1, 0, max_x)
        y0 = t.clamp(y0, 0, max_y)
        y1 = t.clamp(y1, 0, max_y)

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
        out = wa*Ia+wb*Ib+wc*Ic+wd*Id

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
