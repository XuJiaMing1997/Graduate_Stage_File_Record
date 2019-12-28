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
import matplotlib.pyplot as plt

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


class AttentionMaskNet(nn.Module):
    def __init__(self, mask_num = 8, input_dim = 1024, dropout_rate = 0.6):
        '''
        Notice input_dim must // mask_num !!!!!!
        :param mask_num:
        :param input_dim:
        :param dropout_rate:
        :return:
        '''
        super(AttentionMaskNet, self).__init__()
        self.mask_num = mask_num
        self.sigmoid = nn.Sigmoid()
        self.mask_gen_list = nn.ModuleList()
        self.masked_conv_list = nn.ModuleList()
        assert input_dim % mask_num == 0 # input_dim must // mask_num !!!!!!
        self.branch_dim = input_dim // mask_num

        for branch in range(mask_num):
            self.mask_gen_list.append(
                    nn.Sequential(
                        nn.Conv2d(input_dim,1,kernel_size=1,bias=True),
                        nn.Sigmoid()
                    )
            )

            # adaptive pool version
            self.masked_conv_list.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Conv2d(input_dim,self.branch_dim,kernel_size=1,bias=True),
                    nn.ReLU(inplace=True)
                )
            )

        self.dropout = nn.Dropout(p=dropout_rate, inplace=False) # True will wrong 'variable require grad changed'!!
        self.masks = [0 for _ in range(mask_num)] # for mask map visualization
        self.kaiming_init()

    def forward(self, x):
        final_feature = []
        for branch in range(self.mask_num):
            # print('branch {}'.format(branch+1))

            out = self.mask_gen_list[branch](x)
            self.masks[branch] = out
            # print('mask shape: {}'.format(out.shape)) # torch.Size([B, 1, H, W])

            out = out.repeat((1,x.shape[1],1,1))
            # print('mask repeated shape: {}'.format(out.shape)) # torch.Size([B, C, H, W])

            out = x * out
            # print('element-wise multiply shape: {}'.format(out.shape)) # torch.Size([B, C, H, W])

            out = self.masked_conv_list[branch](out)
            # print('reduced fmap_size and fmap_dim shape: {}'.format(out.shape)) # torch.Size([B, part_dim, 1, 1])

            out = out.reshape((x.shape[0],-1))
            # print('final part feature shape: {}'.format(out.shape)) # torch.Size([B, part_dim])

            out = self.dropout(out)
            final_feature.append(out)

        out = torch.cat(final_feature,dim=1)
        # print('final feature shape: {}'.format(out.shape)) # torch.Size([B, final_dim])
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





# ############################################################
# mask visualization part
def generate_PIL_black_mask(target_size,mask_alpha):
    '''
    add 2-D alpha to a Black Image as 'RGBA'type
    :param target_size: [Height, Width]
    :param mask_alpha: 2-D array
    :return: !!!!!!!!!!  PIL Image uint8 type  !!!!!!!!!!!!!!
    '''

    # TODO: Not Apply Scale, Use Raw Softmax Value to Check Mask!!!!    If Paper use can Scale !!!!
    # mask_alpha = mask_alpha - np.minimum(0.,np.min(mask_alpha))
    # mask_alpha = mask_alpha / np.max(mask_alpha)

    mask_alpha = 1 - mask_alpha
    mask_alpha = mask_alpha * 255
    mask_alpha = np.array(mask_alpha,dtype='uint8')[:,:,np.newaxis]

    balck_mask = np.zeros([mask_alpha.shape[0],mask_alpha.shape[1],3],dtype='uint8')
    mask = np.concatenate((balck_mask,mask_alpha),axis=2)

    mask = Image.fromarray(mask)
    mask = mask.resize((target_size[1],target_size[0]),resample=PIL.Image.BILINEAR)
    return mask


def mask_visualization(mask2D, target_size = [256,128], raw_img_dir = '../ViewInvariantNet/CUHK_part/0001_c1_1.png'):
    # return img(PIL,(H,W,3)), mask(PIL,(H,W,4)), fuse_img(PIL,(H,W,3))
    img = Image.open(raw_img_dir)
    img = img.resize((target_size[1],target_size[0]),resample=PIL.Image.BILINEAR)

    mask = generate_PIL_black_mask(target_size,mask2D)
    # return 255 type

    img_channel = img.split() # RGB
    mask_channel = mask.split() #RGBA
    alpha = np.array(mask_channel[-1]) / 255. # (0,1) style
    fuse_img = np.zeros(np.shape(img))
    for i in range(3):
        fuse_img[:,:,i] = img_channel[i] * (1-alpha)
        fuse_img[:,:,i] += mask_channel[i] * alpha
    # prevent overflow
    fuse_img = fuse_img / np.max(fuse_img)
    fuse_img = np.array(fuse_img * 255,dtype='uint8')
    fuse_img = Image.fromarray(fuse_img) # XCP new add
    return img, mask, fuse_img


# TODO: Need get img path when use dataloader, new collate_fn !!!!!!
def masks_visualization(masks, target_size = [256,128],
                        raw_img_dir = '../ViewInvariantNet/CUHK_part/0001_c1_1.png',
                        check_threshold = 0.7, use_GPU = False):
    # Only for one image !!!!!!
    # First Need import matplotlib.pyplot as plt
    # assign one image dir, given masks from BaselinePro with respect to this one image, matplot masks visualization
    # masks [MaskNum=8,B=1,C=1,H,W] tensor
    def binarylization(matrix, threshold = 0.5, use_GPU = False):
        zero_fill = torch.zeros_like(matrix)
        if use_GPU:
            zero_fill = zero_fill.cuda()
        out = torch.where(matrix < threshold, zero_fill, matrix) # fill zero
        out_data = out.data.reshape((-1,))
        for idx in range(len(out_data)):
            if out_data[idx] > 0:
                out_data[idx] = 1 # without fill 1 may bring uncertain optimization target
        return out

    binary_masks = binarylization(masks,threshold=check_threshold,use_GPU=use_GPU) # tensor [N,1,1,H,W]
    # compute check_threshold
    # percentage_2 = [len(np.where(m.cpu().numpy() >= check_threshold)[0]) / m.numel() for m in masks]
    # or this
    percentage = [1. - len(np.where(bm.cpu().numpy() == 0.)[0]) / bm.numel() for bm in binary_masks]

    img = 0
    alpha_mask_list = []
    binary_mask_list = []
    fused_img_list = []
    for mask,b_mask in zip(masks,binary_masks):
        img, recv_mask, recv_fused = mask_visualization(mask.squeeze().cpu().numpy(), target_size, raw_img_dir)
        img = img
        alpha_mask_list.append(recv_mask)
        fused_img_list.append(recv_fused)
        recv_b_mask = generate_PIL_black_mask(target_size,b_mask.squeeze().cpu().numpy())
        binary_mask_list.append(recv_b_mask)

    col_num = len(masks) + 1
    fig = plt.figure()
    fig.suptitle('{}'.format(raw_img_dir))
    subfig = fig.add_subplot(3,col_num,1)
    subfig.imshow(img)
    subfig.set_xticks([])
    subfig.set_yticks([])
    subfig = fig.add_subplot(3,col_num,1+col_num)
    subfig.imshow(img)
    subfig.set_xticks([])
    subfig.set_yticks([])
    subfig = fig.add_subplot(3,col_num,1+col_num+col_num)
    subfig.imshow(img)
    subfig.set_xticks([])
    subfig.set_yticks([])
    for i in range(len(masks)):
        subfig = fig.add_subplot(3,col_num,1 + i + 1)
        subfig.imshow(alpha_mask_list[i])
        subfig.set_xticks([])
        subfig.set_yticks([])

        subfig = fig.add_subplot(3,col_num,1 + i + 1 + col_num)
        subfig.imshow(binary_mask_list[i])
        subfig.set_xticks([])
        subfig.set_yticks([])
        subfig.set_title('if Th{}-{:.3%}'.format(check_threshold,percentage[i]))

        subfig = fig.add_subplot(3,col_num,1 + i + 1 + col_num + col_num)
        subfig.imshow(fused_img_list[i])
        subfig.set_xticks([])
        subfig.set_yticks([])
    plt.show()
    return img, alpha_mask_list, fused_img_list





if __name__ == '__main__':
    synthesize_input_1 = torch.randn((2,12,14,14))
    synthesize_input_2 = torch.randn((2,12,16,8))
    my_model_1 = AttentionMaskNet(4,12)
    my_model_2 = AttentionMaskNet(4,12)
    print(20 * '-' + '>' + ' model 1 ' + '<' + 20 * '-')
    print(my_model_1)
    print(20 * '-' + '>' + ' model 2 ' + '<' + 20 * '-')
    print(my_model_2)
    print(20 * '-' + '>' + ' forward model 1 ' + '<' + 20 * '-')
    recv = my_model_1(synthesize_input_1)
    print('--------> self.masks list')
    for i in my_model_1.masks:
        print(i.shape)
    print(my_model_1.masks[0])
    print(20 * '-' + '>' + ' forward model 2 ' + '<' + 20 * '-')
    recv = my_model_2(synthesize_input_2)
    print(20 * '-' + '>' + ' model 2 parameters ' + '<' + 20 * '-')
    for idx,(name, param) in enumerate(my_model_2.named_parameters()):
        print('{} {}: {}'.format(idx,name,param.shape))


    print(20 * '-' + '>' + ' mask visualization test ' + '<' + 20 * '-')
    # import matplotlib.pyplot as plt

    # choose this ????
    synthesize_masks_1 = torch.rand((4,1,1,14,14),dtype=torch.float32)
    synthesize_masks_2 = torch.rand((4,1,1,14,14),dtype=torch.float32)

    # or choose this ??????
    synthesize_masks_1 = [] # 14*14 [MaskNum=8,B=1,C=1,H,W]
    synthesize_masks_2 = [] # 16*8 [MaskNum=8,B=1,C=1,H,W]
    synthesize_masks_1 = torch.randn((4,1,1,14,14),dtype=torch.float32)
    synthesize_masks_1 -= np.minimum(0.,synthesize_masks_1.min()).float()
    synthesize_masks_1 = synthesize_masks_1 / synthesize_masks_1.max()
    synthesize_masks_2 = torch.randn((4,1,1,14,14),dtype=torch.float32)
    synthesize_masks_2 -= np.minimum(0.,synthesize_masks_2.min()).float()
    synthesize_masks_2 = synthesize_masks_2 / synthesize_masks_2.max()

    with torch.no_grad():
        my_model_1 = AttentionMaskNet(4,12)
        my_model_2 = AttentionMaskNet(4,12)
        person_dir = '../Dataset_256x128/CUHK03/bounding_box_train/0001_c1_1.png'
        synthesize_input_1 = Image.open(person_dir)
        synthesize_input_1 = synthesize_input_1.resize((14,14),resample=PIL.Image.BILINEAR)
        synthesize_input_1 = torch.tensor(np.array(synthesize_input_1)/255.,
                                          dtype=torch.float32).permute(2,0,1).repeat(1,4,1,1)
        recv = my_model_1(synthesize_input_1)
        print('masks[0]:\n{}'.format(my_model_1.masks[0]))
        masks_visualization(torch.stack(my_model_1.masks,dim=0).cuda(),raw_img_dir=person_dir,
                            check_threshold=0.6,use_GPU=True)
        # my_model_1.masks: list[tensor,tensor,..] [mask_num=8,(B,C=1,H,W)]

        synthesize_input_2 = Image.open(person_dir)
        synthesize_input_2 = synthesize_input_2.resize((8,16),resample=PIL.Image.BILINEAR)
        synthesize_input_2 = torch.tensor(np.array(synthesize_input_2)/255.,
                                          dtype=torch.float32).permute(2,0,1).repeat(1,4,1,1)
        recv = my_model_2(synthesize_input_2)
        print('masks[0]:\n{}'.format(my_model_2.masks[0]))
        masks_visualization(torch.stack(my_model_2.masks, dim=0).cuda(),raw_img_dir=person_dir,
                            check_threshold=0.55,use_GPU=True)
        # masks_visualization(synthesize_masks_2,raw_img_dir=person_dir)
















