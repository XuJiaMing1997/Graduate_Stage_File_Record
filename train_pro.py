import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math
import PIL
import PIL.Image as Image
import argparse
import random

from matplotlib import cm
from sklearn.manifold import TSNE

import torch as t
import torchvision as tv
import torch.nn as nn
import torchvision.transforms as trans
import torchvision.datasets as dsets
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToPILImage as tensor2PIL
from torch.utils.data import DataLoader

# from external_file.build import build_transforms as my_trans
from no_resize_transform import build_transforms as my_trans
from no_resize_transform import build_transforms_OP as my_trans_op
from external_file.dataset_loader import ImageDataset
from dataset_loader_OP import ImageDatasetOP
from external_file.triplet_sampler import RandomIdentitySampler
from external_file.triplet_sampler import RandomIdentitySampler_alignedreid
from Baseline_Pro import BaselinePro
from Baseline_OP import BaselineOP
import external_file.optimizer_build as optimizer_build
from external_file.triplet_loss import TripletLoss, CrossEntropyLabelSmooth, normalize
from external_file.center_loss import CenterLoss
from external_file.collate_batch import train_collate_fn, val_collate_fn
from collate_batch_OP import collate_fn_OP_1, collate_fn_OP_2
from inference_collate_batch_MIE import inference_collate_fn
from external_file.re_rank import re_ranking
# from MaskLoss import LocLoss as MaskLoss
from MaskLoss import AreaLoss as MaskLoss
from AttentionMaskNet import masks_visualization
from SpatialTransformer import STN_Visualization
from make_optimizer_OP import make_optimizer_OP


class MIE(object):
    def __init__(self,
                 target_dataset_name='CUHK03',
                 pre_dataset_loc='../Dataset',
                 save_dir='./MIE/model_save',
                 save_name='pytorch_MIE',
                 net_setting_dir='./MIE/structure.txt',
                 loss_lr_dir='./MIE/loss_lr.txt',
                 acc_dir='./MIE/acc.txt',
                 independent_eval_dir='./MIE/independent_acc.txt',
                 history_log_dir='./MIE/history.txt',
                 isDefaultImageSize=False):
        self.target_dataset_name = target_dataset_name
        self.save_dir = save_dir
        self.save_name = save_name
        self.net_setting_dir = net_setting_dir
        self.loss_lr_dir = loss_lr_dir
        self.acc_dir = acc_dir
        self.independent_eval_dir = independent_eval_dir
        self.history_log_dir = history_log_dir
        self.isDefaultImageSize = isDefaultImageSize

        folder_tag = '224x224' if isDefaultImageSize else '256x128'
        self.dataset_loc = pre_dataset_loc + '_' + folder_tag
        print('---------> load image from {} folder'.format(folder_tag))

        Dataset_dir_name = {}
        Dataset_dir_name['MARKET'] = 'Market'
        Dataset_dir_name['DUKE'] = 'Duke'
        Dataset_dir_name['CUHK03'] = 'CUHK03'
        Dataset_dir_name['MSMT17'] = 'MSMT17'

        train_dir = {}
        train_dir['MARKET'] = '{0}/{1}/bounding_box_train'.format(self.dataset_loc,
                                                                  Dataset_dir_name[self.target_dataset_name])
        train_dir['DUKE'] = '{0}/{1}/bounding_box_train'.format(self.dataset_loc,
                                                                Dataset_dir_name[self.target_dataset_name])
        train_dir['CUHK03'] = '{0}/{1}/bounding_box_train'.format(self.dataset_loc,
                                                                  Dataset_dir_name[self.target_dataset_name])
        train_dir['MSMT17'] = '{0}/{1}/bounding_box_train'.format(self.dataset_loc,
                                                                  Dataset_dir_name[self.target_dataset_name])

        query_dir = {}
        query_dir['MARKET'] = '{0}/{1}/query'.format(self.dataset_loc,
                                                     Dataset_dir_name[self.target_dataset_name])
        query_dir['DUKE'] = '{0}/{1}/query'.format(self.dataset_loc,
                                                   Dataset_dir_name[self.target_dataset_name])
        query_dir['CUHK03'] = '{0}/{1}/query'.format(self.dataset_loc,
                                                     Dataset_dir_name[self.target_dataset_name])
        query_dir['MSMT17'] = '{0}/{1}/query'.format(self.dataset_loc,
                                                     Dataset_dir_name[self.target_dataset_name])

        test_dir = {}
        test_dir['MARKET'] = '{0}/{1}/bounding_box_test'.format(self.dataset_loc,
                                                                Dataset_dir_name[self.target_dataset_name])
        test_dir['DUKE'] = '{0}/{1}/bounding_box_test'.format(self.dataset_loc,
                                                              Dataset_dir_name[self.target_dataset_name])
        test_dir['CUHK03'] = '{0}/{1}/bounding_box_test'.format(self.dataset_loc,
                                                                Dataset_dir_name[self.target_dataset_name])
        test_dir['MSMT17'] = '{0}/{1}/bounding_box_test'.format(self.dataset_loc,
                                                                Dataset_dir_name[self.target_dataset_name])

        self.train_dir = train_dir[self.target_dataset_name]
        self.query_dir = query_dir[self.target_dataset_name]
        self.test_dir = test_dir[self.target_dataset_name]

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists('./MIE'):
            os.makedirs('./MIE')

    def lr_check(self, optimizer):
        random_target = np.random.randint(0, len(optimizer.param_groups) + 1, 3)
        print('----> lr check -- num {}'.format(len(random_target)))
        for i in random_target:
            print('param {}: {}'.format(i, optimizer.param_groups[i]['lr']))

    def NormalizedTensorImage2Numpy(self, input_tensor_3D, pixel_mean, pixel_stddev):
        img = input_tensor_3D.permute(1, 2, 0)
        img = img.cpu().numpy()
        img = img * pixel_stddev + \
              pixel_mean
        img = img * 255.
        # img_recovery = img.type(t.uint8).numpy()
        img_recovery = img.astype(np.uint8)
        return img_recovery

    def forward_BP_logic_check_MIE(self,
                                   Epoch=2,
                                   train_batch_size=64,
                                   eval_batch_size=128,
                                   if_center_loss=True,
                                   if_triplet_loss=True,
                                   center_loss_weight=0.0005):
        if t.cuda.is_available():
            use_GPU = True
        else:
            use_GPU = False
        available_device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

        augmentation_preprocess = {'image_size': [256, 128],
                                   'pixel_mean': [0.485, 0.456, 0.406],
                                   'pixel_stddev': [0.229, 0.224, 0.225],
                                   'flip_probability': 0.5,
                                   'padding_size': 10,
                                   'random_eras_probability': 0.5,
                                   's_ratio_min': 0.02,
                                   's_ratio_max': 0.4,
                                   'aspect_ratio_min': 0.3
                                   }
        # Random Erase use pixel mean fill, may cause performance deteriorate !!!!!!
        dataloader_setting = {'if_triplet_loss': if_triplet_loss,
                              'K_num': 4,
                              'if_center_loss': if_center_loss,
                              'num_workers': 8
                              }
        model_settting = {'last_stride': 2,
                          'neck': True,
                          'eval_neck': False,
                          'model_name': 'resnet50',
                          'ImageNet_Init': True,
                          'ImageNet_model_path': './resnet50-19c8e357.pth'
                          }
        optimizer_setting = {'lr_start': 0.1,
                             'weight_decay': 0.0005,
                             'bias_lr_factor': 1,
                             'weight_decay_bias': 0.0005,
                             'optimizer_name': 'SGD',  # 'Adam'
                             'SGD_momentum': 0.9,
                             'center_lr': 0.0001
                             }
        loss_setting = {'eval_feature_dim': 2048,
                        'if_triplet_loss': if_triplet_loss,
                        'triplet_margin': 0.2,  # 0.1-0.3 Zheng suggest
                        'label_smooth_rate': 0.1  # 0. original softmax
                        }
        train_dataloader, train_img_num, train_person_num = self.construct_dataset_dataloader_MIE(
            self.train_dir, augmentation_preprocess, dataloader_setting,
            if_reset_id=True, is_train=True, batch_size=train_batch_size)

        print('train iteration num per epoch: {}'.format(len(train_dataloader)))

        # construct model and initialize weights
        my_model = self.construct_model_MIE(train_person_num, model_settting, use_GPU)
        my_model.to(available_device)
        # if use_GPU:
        #     my_model.cuda()
        print(my_model)

        print('-----> var check')
        for idx, (p_name, param) in enumerate(my_model.named_parameters()):
            print('{} {}: {}'.format(idx, p_name, param.shape))

        # construct optimizer
        if if_center_loss:
            loss_calculator, center_criterion = self.build_loss_structure(person_num=train_person_num,
                                                                          if_center_loss=True,
                                                                          loss_setting=loss_setting,
                                                                          use_GPU=use_GPU)
            # loss_cal param: fc_logits, GAP_feature, label, if_triplet_loss, if_center_loss, center_loss_weight
            my_optimizer, center_optimizer, STN_param_idx_set, OP_param_idx_set = self.generate_center_optimizer_MIE(
                optimizer_setting,
                my_model,
                center_criterion)
        else:
            my_optimizer, STN_param_idx_set, OP_param_idx_set = self.generate_optimizer_MIE(optimizer_setting, my_model)
            loss_calculator = self.build_loss_structure(person_num=train_person_num,
                                                        if_center_loss=False,
                                                        loss_setting=loss_setting,
                                                        use_GPU=use_GPU)
            # loss_cal param: fc_logits, GAP_feature, label, if_triplet_loss, if_center_loss, center_loss_weight

        print('STN_param_idx_set: {}'.format(STN_param_idx_set))

        # train and eval
        time.clock()
        my_model.train()
        for ep in range(Epoch):
            print('--------->  Start epoch {}'.format(ep))
            time_start = time.clock()
            avg_loss = []
            avg_acc = []
            for iter, data in enumerate(train_dataloader):
                batch_image, batch_label = data
                batch_image = batch_image.to(available_device)
                batch_label = batch_label.to(available_device)
                # if use_GPU:
                #     batch_image = batch_image.cuda()
                #     batch_label = batch_label.cuda()

                #
                fig = plt.figure()
                for idx, (img, lb) in enumerate(zip(batch_image, batch_label)):
                    if idx == 8:
                        break
                    subfig = fig.add_subplot(2, 4, idx + 1)
                    img_recovery = self.NormalizedTensorImage2Numpy(img, augmentation_preprocess['pixel_mean']
                                                                    , augmentation_preprocess['pixel_stddev'])
                    subfig.imshow(img_recovery)
                    subfig.set_title('new_pid: {}'.format(lb))
                plt.show()
                #

                my_optimizer.zero_grad()
                if if_center_loss:
                    center_optimizer.zero_grad()

                fc_logits, GAP_feature = my_model(batch_image)
                loss = loss_calculator(fc_logits, GAP_feature, batch_label,
                                       if_triplet_loss, if_center_loss, center_loss_weight)
                # need param: fc_logits, GAP_feature, label, if_triplet_loss, if_center_loss, center_loss_weight
                # loss = [tensor_loss_all, item_loss_all, item_loss_s, item_loss_t, item_loss_c]

                loss[0].backward()
                my_optimizer.step()

                if if_center_loss:
                    # self.centers_update()
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / center_loss_weight)
                    center_optimizer.step()

                acc = (fc_logits.max(1)[1] == batch_label).float().mean().item()  # transfer tensor to float
                print('iter {}\tacc {}\tloss {}\ts_loss {}\tt_loss {}\tc_loss {}\tlr {}'.format(
                    iter, acc, loss[1], loss[2], loss[3], loss[4], 0))
                avg_loss.append(loss[1:])
                avg_acc.append(acc)

    def inference_check_MIE(self,
                            image_size=[256, 128],
                            if_OP_check=False,
                            if_Mask_check=False,
                            Mask_threshold = 0.7,
                            mask_target = 0,
                            if_STN_check=True,
                            batch_size=1,
                            target_model_path=None,
                            model_setting_path=None,
                            dataset_folder_dir=None
                            ):
        # batch_size=1
        # Pay Attention to Parameter Dict !!!!!!!! Need keep same with train  !!!!!!!
        if t.cuda.is_available():
            use_GPU = True
            print('Use GPU')
        else:
            use_GPU = False
            print('No GPU, Use CPU')
        available_device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        if target_model_path is None:
            fname_list = os.listdir(self.save_dir)
            fname_list.sort()
            target_model_path = os.path.join(self.save_dir, fname_list[-1])
        if model_setting_path is None:
            model_setting_path = self.net_setting_dir
        if dataset_folder_dir is None:
            dataset_folder_dir = self.query_dir

        pixel_mean = [0.485, 0.456, 0.406]
        pixel_stddev = [0.229, 0.224, 0.225]

        my_dataloader = self.construct_inference_dataloader_MIE(dataset_folder_dir,[256,128],
                                                                pixel_mean=pixel_mean,pixel_stddev=pixel_stddev,
                                                                PK_sample=False,if_shuffle=False,
                                                                batch_size=batch_size)
        # return 3 in one iter: 4D batch_imgs(tensor), 1D pids(tuple), 1D paths(tuple)

        model_setting = self.net_setting_load(model_setting_path)
        # compatible with older version
        if 'if_OP_channel_wise' not in model_setting:
            model_setting['if_OP_channel_wise'] = False
            print('Add model_setting[\'if_OP_channel_wise\'] = False')
        if 'if_affine' not in model_setting:
            model_setting['if_affine'] = False
            print('Add model_setting[\'if_affine\'] = False')

        my_model = self.construct_model_MIE(2, model_setting, use_GPU)  # random class_num
        my_model.to(available_device)

        if if_OP_check:
            assert model_setting['addOrientationPart'] == True, 'Must have Orientation Part When check OP'
        if if_Mask_check:
            assert model_setting['addAttentionMaskPart'] == True, 'Must have Attention Mask Part When check Mask'
        if if_STN_check:
            assert model_setting['addSTNPart'] == True, 'Must have STN Part When check STN'

        # Need exclude classifier in state_dict
        # Method 1
        # loaded_state_dict = t.load(target_model_path)
        # exclude = []
        # for key in loaded_state_dict:
        #     if 'classifier' in key:
        #         exclude.append(key)
        # for key in exclude:
        #     loaded_state_dict.pop(key)
        # my_model.load_state_dict(loaded_state_dict, strict=True)

        # Method 2
        my_model.load_param_transfer(target_model_path)

        print('-------> start check {}'.format(target_model_path))
        my_model.eval()
        with t.no_grad():
            for data in my_dataloader:
                image, pids, paths = data
                image = image.to(available_device)
                feature = my_model(image)

                if if_OP_check:
                    check_tag = []
                    for i in range(batch_size):
                        softmax_score = my_model.orientation_predict_score[i]
                        OP_tag = 'F:{:.3%}'.format(softmax_score[0]) + ' B:{:.3%}'.format(softmax_score[1]) + \
                                 ' S:{:.3%}'.format(softmax_score[2])
                        check_tag.append(OP_tag)

                    fig = plt.figure(figsize=(16, 6))
                    fig.suptitle('{}'.format(target_model_path))
                    for idx in range(batch_size):
                        subfig = fig.add_subplot(2, int(math.ceil(batch_size / 2)), idx + 1)
                        subfig.imshow(
                                self.NormalizedTensorImage2Numpy(image[idx].cpu(), pixel_mean, pixel_stddev))
                        subfig.set_title('ID{}\n{}'.format(pids[idx], check_tag[idx]))
                    plt.show()

                if if_Mask_check:
                    masks_list = my_model.masks # list[tensor,tensor,..] [mask_num=8,(B,C=1,H,W)]
                    masks = t.stack(masks_list,dim=0) # [N8,B,C1,H,W]
                    for i in range(batch_size):
                        print('image{} in batch_size {}'.format(i+1,batch_size))

                        tem_mask = masks[mask_target, i, :, :, :].squeeze(0)  # [H,W]
                        print('-----> mask {}:\n{}'.format(mask_target, tem_mask))
                        print('Max activated: {} Min activated: {} Mean activated: {}'.format(
                            t.max(tem_mask), t.min(tem_mask), t.mean(tem_mask)))
                        masked_num = len(np.where(tem_mask.cpu().numpy() >= Mask_threshold)[0])
                        percentage = masked_num / tem_mask.numel()
                        print('Mask{} Threshold {} Percentage: {:.3%}'.format(mask_target,Mask_threshold, percentage))
                        # Max activated: 0.9411682486534119
                        # Min activated: 0.0024127261713147163
                        # Mean activated: 0.3069003224372864

                        img, alpha_mask_list, fused_img_list = masks_visualization(masks[:,i,:,:,:],
                                                                                   image_size,paths[i],
                                                                                   Mask_threshold,use_GPU=use_GPU)

                if if_STN_check:
                    batch_param = my_model.STN_transform_parameters # [B,9/6]
                    trans_list, raw_list = STN_Visualization(paths,batch_param,image_size,
                                                             if_affine=model_setting['if_affine'],
                                                             use_GPU=use_GPU)
                    print('fcW: {}'.format(my_model.spatial_transformer.localisation_net.fc_W))
                    # 1.0711e-06 --> -2.7907e-05
                    fig = plt.figure(figsize=(16,6))
                    for idx,(id,trans,raw) in enumerate(zip(pids, trans_list, raw_list)):
                        subfig = fig.add_subplot(2,batch_size,idx+1)
                        subfig.set_title(id)
                        subfig.imshow(raw)
                        subfig.set_xticks([])
                        subfig.set_yticks([])
                        subfig = fig.add_subplot(2,batch_size,idx+1+batch_size)
                        subfig.imshow(trans)
                        subfig.set_xticks([])
                        subfig.set_yticks([])
                    plt.show()


    def inference_check_OP(self,
                           target_model_path,
                           model_setting_path,
                           if_use_reid_dataset,
                           reid_list_dir,
                           dataset_folder_dir,
                           image_size=[256, 128],
                           batch_size=16,
                           only_side=True
                           ):
        if t.cuda.is_available():
            use_GPU = True
            print('Use GPU')
        else:
            use_GPU = False
            print('No GPU, Use CPU')
        available_device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

        augmentation_preprocess = {'image_size': image_size,
                                   'pixel_mean': [0.485, 0.456, 0.406],
                                   'pixel_stddev': [0.229, 0.224, 0.225],
                                   'flip_probability': 0.5,
                                   'padding_size': 10,
                                   'zoom_out_pad_prob': 0.3,
                                   'random_eras_probability': 0.5,
                                   's_ratio_min': 0.02,
                                   's_ratio_max': 0.4,
                                   'aspect_ratio_min': 0.3
                                   }

        # TODO: BUG Here!!! When No list folder use ---- set if_use_reid_dataset = False
        train_dataloader, train_img_num, test_dataloader, test_img_num = self.construct_dataset_dataloader_OP(
                dataset_folder_dir, if_use_reid_dataset, reid_list_dir, augmentation_preprocess,
                train_proportion=0., test_batch_size=batch_size)

        model_setting = self.net_setting_load(Dir=model_setting_path)
        my_model = self.construct_model_OP(model_setting)
        my_model.to(available_device)

        my_model.load_state_dict(t.load(target_model_path), strict=True)

        my_model.eval()
        with t.no_grad():
            print('----------> Start check {}'.format(dataset_folder_dir))
            for idx, data in enumerate(test_dataloader):
                image, label = data
                image = image.to(available_device)
                logits = my_model(image)

                if only_side:
                    available_num_idx = np.where(np.array(label) == 2)[0]
                    available_num = len(available_num_idx)
                else:
                    available_num_idx = np.arange(batch_size)
                    available_num = len(available_num_idx)

                check_tag = []
                for i in available_num_idx:
                    softmax_score = my_model.OP_score[i]
                    OP_tag = 'F:{:.3%}'.format(softmax_score[0]) + ' B:{:.3%}'.format(softmax_score[1]) + \
                             ' S:{:.3%}'.format(softmax_score[2])
                    check_tag.append(OP_tag)

                fig = plt.figure(figsize=(16, 6))
                fig.suptitle('{}'.format(target_model_path))
                count = 0
                for idx in available_num_idx:
                    subfig = fig.add_subplot(2, int(math.ceil(available_num / 2)), count+1)
                    subfig.imshow(
                        self.NormalizedTensorImage2Numpy(image[idx].cpu(), augmentation_preprocess['pixel_mean'],
                                                         augmentation_preprocess['pixel_stddev']))
                    subfig.set_title('label:{}\n{}'.format(label[idx], check_tag[count]))
                    count += 1
                plt.show()

    def net_setting_save(self, net_setting, save_dir=None):
        type2str = {int: 'int',
                    float: 'float',
                    bool: 'bool',
                    str: 'str'}
        if save_dir is None:
            save_dir = self.net_setting_dir
        f = open(save_dir, 'w')
        for p_name, param in net_setting.items():
            f.write('{} {} {}\n'.format(type2str[type(param)], p_name, param))
        f.close()
        print('Write {} net_setting to {}'.format(len(net_setting), save_dir))
        return net_setting

    def net_setting_load(self, Dir=None):
        if Dir is None:
            f = open(self.net_setting_dir, 'r')
        else:
            f = open(Dir, 'r')
        f_lines = f.readlines()
        f.close()

        line_split = []
        for line in f_lines:
            tem_line_split = line.strip('\n').split(' ')
            line_split.append(tem_line_split)

        def str2bool(a):
            return True if a.lower() == 'true' else False

        str2type = {'int': int,
                    'float': float,
                    'bool': str2bool,
                    'str': str}

        net_setting = {}
        for line in line_split:
            net_setting[line[1]] = str2type[line[0]](line[2])

        print('load {} net_setting from {}'.format(len(net_setting), self.net_setting_dir))
        # XCP
        for p_name, param in net_setting.items():
            print('{0}: {1}'.format(p_name, param))
        return net_setting

    def plot_acc_MIE(self, eval_dir=None, ifSave=True, SaveDir='./MIE/independent_acc.jpg'):
        # acc file:  'epoch: {0}\t''1: {0}\t5: {1}\t10: {2}\t20: {3}\tmAP: {4}\t''Time: {0}:{1}:{2}\n'
        if eval_dir is None:
            eval_dir = self.acc_dir

        f_eval = open(eval_dir, 'r')
        eval_lines = f_eval.readlines()
        f_eval.close()

        eval_epoch = []
        rank = []
        mAP = []
        for line in eval_lines:
            tem_rank = []
            line_split = line.strip('\n').split('\t')
            eval_epoch.append(int(line_split[0].split(' ')[1]))
            tem_rank.append(float(line_split[1].split(' ')[1]))
            tem_rank.append(float(line_split[2].split(' ')[1]))
            tem_rank.append(float(line_split[3].split(' ')[1]))
            tem_rank.append(float(line_split[4].split(' ')[1]))
            mAP.append(float(line_split[5].split(' ')[1]))
            rank.append(np.array(tem_rank))

        rank = np.array(rank)
        eval_x_axis = np.arange(len(eval_epoch))
        eval_x_label = ['{0}\n{1}'.format(ep, cnt) for cnt, ep in zip(eval_x_axis, eval_epoch)]

        fig = plt.figure(figsize=(16, 6))
        subfig = fig.add_subplot(1, 2, 1)
        subfig.set_title('rank-n')
        subfig.set_xlabel('epoch-No.')
        subfig.set_ylabel('percentage')
        subfig.set_xticks(eval_x_axis)

        subfig.set_xticklabels(eval_x_label)

        rank_1_line, = subfig.plot(eval_x_axis, rank[:, 0], color='blue', linewidth=1,
                                   linestyle='-', label='rank_1')
        rank_5_line, = subfig.plot(eval_x_axis, rank[:, 1], color='green', linewidth=1,
                                   linestyle='-', label='rank_5')
        rank_10_line, = subfig.plot(eval_x_axis, rank[:, 2], color='yellow', linewidth=1,
                                    linestyle='-', label='rank_10')
        rank_20_line, = subfig.plot(eval_x_axis, rank[:, 3], color='red', linewidth=1,
                                    linestyle='-', label='rank_20')

        subfig.legend(handles=[rank_1_line, rank_5_line, rank_10_line, rank_20_line],
                      labels=['rank_1', 'rank_5', 'rank_10', 'rank_20'], loc='best')

        subfig = fig.add_subplot(1, 2, 2)
        subfig.set_title('mAP')
        subfig.set_xlabel('epoch-No.')
        subfig.set_ylabel('value')
        subfig.set_xticks(eval_x_axis)

        subfig.set_xticklabels(eval_x_label)

        mAP_line, = subfig.plot(eval_x_axis, mAP, color='red', linewidth=1,
                                linestyle='-', label='mAP')

        subfig.legend(handles=[mAP_line, ],
                      labels=['mAP'], loc='best')

        if ifSave:
            fig.savefig(SaveDir)
        plt.show()

        # report max value
        max_rank_1_idx = np.argmax(rank[:, 0])  # may occur two value error XCP
        max_rank_1_value = rank[:, 0][max_rank_1_idx]
        max_rank_1_ep = eval_epoch[max_rank_1_idx]

        max_mAP_idx = np.argmax(mAP)  # may occur two value error XCP
        max_mAP_value = mAP[max_mAP_idx]
        max_mAP_ep = eval_epoch[max_mAP_idx]

        print('MAX rank_1: {} EP: {}'.format(max_rank_1_value, max_rank_1_ep))
        print('MAX mAP: {} EP: {}'.format(max_mAP_value, max_mAP_ep))
        return max_rank_1_ep, max_mAP_ep

    def plot_acc_OP(self, eval_dir, ifSave=True, SaveDir='./OP/independent_acc.jpg'):
        # acc:  'epoch: {0}\tAvg_acc: {}\tAvg_f_acc: {}\tAvg_b_acc: {}\tAvg_s_acc: {}\tTime: {0}:{1}:{2}\n'
        accf = open(eval_dir, 'r')
        acc_lines = accf.readlines()
        accf.close()

        acc_ep = []
        acc_list = []
        f_acc_list = []
        b_acc_list = []
        s_acc_list = []

        for line in acc_lines:
            line_split = line.split('\t')
            acc_ep.append(int(line_split[0].split(' ')[1]))
            acc_list.append(float(line_split[1].split(' ')[1]))
            f_acc_list.append(float(line_split[2].split(' ')[1]))
            b_acc_list.append(float(line_split[3].split(' ')[1]))
            s_acc_list.append(float(line_split[4].split(' ')[1]))

        eval_x_axis = np.arange(len(acc_ep))
        eval_x_label = ['{0}\n{1}'.format(ep, cnt) for cnt, ep in zip(eval_x_axis, acc_ep)]

        fig = plt.figure()
        subfig = fig.add_subplot(1, 1, 1)
        subfig.set_title('test acc')
        subfig.set_xlabel('epoch-No.')
        subfig.set_ylabel('percentage')
        subfig.set_xticks(eval_x_axis)

        subfig.set_xticklabels(eval_x_label)

        total_acc_line, = subfig.plot(eval_x_axis, acc_list, color='blue', linewidth=1,
                                      linestyle='-', label='acc')
        f_acc_line, = subfig.plot(eval_x_axis, f_acc_list, color='green', linewidth=1,
                                  linestyle='-', label='f_acc')
        b_acc_line, = subfig.plot(eval_x_axis, b_acc_list, color='yellow', linewidth=1,
                                  linestyle='-', label='b_acc')
        s_acc_line, = subfig.plot(eval_x_axis, s_acc_list, color='red', linewidth=1,
                                  linestyle='-', label='s_acc')

        subfig.legend(handles=[total_acc_line, f_acc_line, b_acc_line, s_acc_line],
                      labels=['total_acc', 'front_acc', 'back_acc', 'side_acc'], loc='best')

        if ifSave:
            fig.savefig(SaveDir)
        plt.show()

        # report max value
        max_acc_value = np.max(acc_list)
        max_acc_idx = acc_list.index(max_acc_value)
        max_acc_ep = acc_ep[max_acc_idx]

        print('MAX acc: {} EP: {}'.format(max_acc_value, max_acc_ep))
        return max_acc_ep

    def plot_loss_acc_lr_MIE(self, eval_dir=None, ifSave=True, SaveDir='./MIE/loss_acc_lr.jpg', if_show=True):
        # loss file: 'epoch {} train_loss {} softmax_loss {} triplet_loss {} center_loss {} '
        #            'mask_loss {} lr {} time {}:{}:{}\n'
        # acc file:  'epoch: {0}\t''1: {0}\t5: {1}\t10: {2}\t20: {3}\tmAP: {4}\t''Time: {0}:{1}:{2}\n'
        if eval_dir is None:
            eval_dir = self.acc_dir

        f_loss = open(self.loss_lr_dir, 'r')
        f_eval = open(eval_dir, 'r')
        loss_lines = f_loss.readlines()
        eval_lines = f_eval.readlines()
        f_loss.close()
        f_eval.close()

        loss_epoch = []
        loss = []
        s_loss = []
        t_loss = []
        c_loss = []
        m_loss = []
        lr = []
        for line in loss_lines:
            line_split = line.strip('\n').split(' ')
            loss_epoch.append(int(line_split[1]))
            loss.append(float(line_split[3]))
            s_loss.append(float(line_split[5]))
            t_loss.append(float(line_split[7]))
            c_loss.append(float(line_split[9]))
            m_loss.append(float(line_split[11]))
            lr.append(float(line_split[13]))

        eval_epoch = []
        rank = []
        mAP = []
        for line in eval_lines:
            tem_rank = []
            line_split = line.strip('\n').split('\t')
            eval_epoch.append(int(line_split[0].split(' ')[1]))
            tem_rank.append(float(line_split[1].split(' ')[1]))
            tem_rank.append(float(line_split[2].split(' ')[1]))
            tem_rank.append(float(line_split[3].split(' ')[1]))
            tem_rank.append(float(line_split[4].split(' ')[1]))
            mAP.append(float(line_split[5].split(' ')[1]))
            rank.append(np.array(tem_rank))

        rank = np.array(rank)
        loss_x_axis = np.arange(len(loss_epoch))

        eval_x_axis = np.arange(len(eval_epoch))
        eval_x_label = ['{0}\n{1}'.format(ep, cnt) for cnt, ep in zip(eval_x_axis, eval_epoch)]

        fig = plt.figure(figsize=(16, 6))
        subfig = fig.add_subplot(1, 4, 1)
        subfig.set_title('train loss')
        subfig.set_xlabel('epoch')
        subfig.set_ylabel('value')
        subfig.set_xticks(loss_x_axis)

        train_loss_line, = subfig.plot(loss_x_axis, loss,
                                       color='red', linewidth=1.2, linestyle='-', label='train_loss')
        softmax_loss_line, = subfig.plot(loss_x_axis, s_loss,
                                         color='yellow', linewidth=1.0, linestyle='--',
                                         label='softmax_loss')
        triplet_loss_line, = subfig.plot(loss_x_axis, t_loss,
                                         color='green', linewidth=1.0, linestyle='--', label='triplet_loss')
        center_loss_line, = subfig.plot(loss_x_axis, c_loss,
                                        color='blue', linewidth=1.0, linestyle='--', label='center_loss')
        mask_loss_line, = subfig.plot(loss_x_axis, m_loss,
                                      color='black', linewidth=1.0, linestyle='--', label='mask_loss')

        subfig.legend(handles=[train_loss_line, softmax_loss_line, triplet_loss_line, center_loss_line, mask_loss_line],
                      labels=['train_loss', 'softmax_loss', 'triplet_loss', 'center_loss', 'mask_loss'],
                      loc='best')

        subfig = fig.add_subplot(1, 4, 2)
        subfig.set_title('rank-n')
        subfig.set_xlabel('epoch-No.')
        subfig.set_ylabel('percentage')
        subfig.set_xticks(eval_x_axis)

        subfig.set_xticklabels(eval_x_label)

        rank_1_line, = subfig.plot(eval_x_axis, rank[:, 0], color='blue', linewidth=1,
                                   linestyle='-', label='rank_1')
        rank_5_line, = subfig.plot(eval_x_axis, rank[:, 1], color='green', linewidth=1,
                                   linestyle='-', label='rank_5')
        rank_10_line, = subfig.plot(eval_x_axis, rank[:, 2], color='yellow', linewidth=1,
                                    linestyle='-', label='rank_10')
        rank_20_line, = subfig.plot(eval_x_axis, rank[:, 3], color='red', linewidth=1,
                                    linestyle='-', label='rank_20')

        subfig.legend(handles=[rank_1_line, rank_5_line, rank_10_line, rank_20_line],
                      labels=['rank_1', 'rank_5', 'rank_10', 'rank_20'], loc='best')

        subfig = fig.add_subplot(1, 4, 3)
        subfig.set_title('mAP')
        subfig.set_xlabel('epoch-No.')
        subfig.set_ylabel('value')
        subfig.set_xticks(eval_x_axis)

        subfig.set_xticklabels(eval_x_label)

        mAP_line, = subfig.plot(eval_x_axis, mAP, color='red', linewidth=1,
                                linestyle='-', label='mAP')

        subfig.legend(handles=[mAP_line, ],
                      labels=['mAP'], loc='best')

        subfig = fig.add_subplot(1, 4, 4)
        subfig.set_title('learning rate')
        subfig.set_xlabel('epoch')
        subfig.set_ylabel('value')

        lr_line, = subfig.plot(loss_epoch, lr, color='red', linewidth=1, linestyle='-', label='learning_rate')

        subfig.legend(handles=[lr_line, ],
                      labels=['learning_rate'], loc='best')

        if ifSave:
            fig.savefig(SaveDir)
        if if_show:
            plt.show()

        # report max value
        max_rank_1_idx = np.argmax(rank[:, 0])  # only return first max value index
        max_rank_1_value = rank[:, 0][max_rank_1_idx]
        max_rank_1_ep = eval_epoch[max_rank_1_idx]

        max_mAP_idx = np.argmax(mAP)  # only return first max value index
        max_mAP_value = mAP[max_mAP_idx]
        max_mAP_ep = eval_epoch[max_mAP_idx]

        print('MAX rank_1: {} EP: {}'.format(max_rank_1_value, max_rank_1_ep))
        print('MAX mAP: {} EP: {}'.format(max_mAP_value, max_mAP_ep))
        return max_rank_1_ep, max_mAP_ep

    def plot_loss_acc_lr_OP(self, loss_lr_dir, eval_dir, ifSave=True, SaveDir='./OP/loss_acc_lr.jpg', if_show=True):
        # loss_lr:  'epoch {} loss {} acc {} lr {} time {}:{}:{}\n'
        # acc:  'epoch: {0}\tAvg_acc: {}\tAvg_f_acc: {}\tAvg_b_acc: {}\tAvg_s_acc: {}\tTime: {0}:{1}:{2}\n'

        lossf = open(loss_lr_dir, 'r')
        accf = open(eval_dir, 'r')
        loss_lines = lossf.readlines()
        acc_lines = accf.readlines()
        lossf.close()
        accf.close()

        loss_ep = []
        loss_list = []
        train_acc_list = []
        lr = []
        acc_ep = []
        acc_list = []
        f_acc_list = []
        b_acc_list = []
        s_acc_list = []

        for line in loss_lines:
            line_split = line.split(' ')
            loss_ep.append(int(line_split[1]))
            loss_list.append(float(line_split[3]))
            train_acc_list.append(float(line_split[5]))
            lr.append(float(line_split[7]))
        for line in acc_lines:
            line_split = line.split('\t')
            acc_ep.append(int(line_split[0].split(' ')[1]))
            acc_list.append(float(line_split[1].split(' ')[1]))
            f_acc_list.append(float(line_split[2].split(' ')[1]))
            b_acc_list.append(float(line_split[3].split(' ')[1]))
            s_acc_list.append(float(line_split[4].split(' ')[1]))

        loss_x_axis = np.arange(len(loss_ep))
        eval_x_axis = np.arange(len(acc_ep))
        eval_x_label = ['{0}\n{1}'.format(ep, cnt) for cnt, ep in zip(eval_x_axis, acc_ep)]

        fig = plt.figure(figsize=(16, 6))
        subfig = fig.add_subplot(1, 4, 1)
        subfig.set_title('train loss')
        subfig.set_xlabel('epoch')
        subfig.set_ylabel('value')
        subfig.set_xticks(loss_x_axis)

        train_loss_line, = subfig.plot(loss_x_axis, loss_list,
                                       color='red', linewidth=1.2, linestyle='-', label='train_loss')

        subfig.legend(handles=[train_loss_line, ], labels=['train_loss', ], loc='best')

        subfig = fig.add_subplot(1, 4, 2)
        subfig.set_title('train acc')
        subfig.set_xlabel('epoch')
        subfig.set_ylabel('value')
        subfig.set_xticks(loss_x_axis)

        train_acc_line, = subfig.plot(loss_x_axis, train_acc_list,
                                      color='red', linewidth=1.2, linestyle='-', label='train_acc')

        subfig.legend(handles=[train_acc_line, ], labels=['train_acc', ], loc='best')

        subfig = fig.add_subplot(1, 4, 3)
        subfig.set_title('test acc')
        subfig.set_xlabel('epoch-No.')
        subfig.set_ylabel('percentage')
        subfig.set_xticks(eval_x_axis)

        subfig.set_xticklabels(eval_x_label)

        total_acc_line, = subfig.plot(eval_x_axis, acc_list, color='blue', linewidth=1,
                                      linestyle='-', label='acc')
        f_acc_line, = subfig.plot(eval_x_axis, f_acc_list, color='green', linewidth=1,
                                  linestyle='-', label='f_acc')
        b_acc_line, = subfig.plot(eval_x_axis, b_acc_list, color='yellow', linewidth=1,
                                  linestyle='-', label='b_acc')
        s_acc_line, = subfig.plot(eval_x_axis, s_acc_list, color='red', linewidth=1,
                                  linestyle='-', label='s_acc')

        subfig.legend(handles=[total_acc_line, f_acc_line, b_acc_line, s_acc_line],
                      labels=['total_acc', 'front_acc', 'back_acc', 'side_acc'], loc='best')

        subfig = fig.add_subplot(1, 4, 4)
        subfig.set_title('learning rate')
        subfig.set_xlabel('epoch')
        subfig.set_ylabel('value')

        lr_line, = subfig.plot(loss_ep, lr, color='red', linewidth=1, linestyle='-', label='learning_rate')

        subfig.legend(handles=[lr_line, ], labels=['learning_rate'], loc='best')

        if ifSave:
            fig.savefig(SaveDir)
            print('save plot to {}'.format(SaveDir))
        if if_show:
            plt.show()

        # report max value
        max_acc_value = np.max(acc_list)
        max_acc_idx = acc_list.index(max_acc_value)
        max_acc_ep = acc_ep[max_acc_idx]

        print('MAX acc: {} EP: {}'.format(max_acc_value, max_acc_ep))
        return max_acc_ep

    def compute_pixel_mean_stddev(self, image_array):
        # image_all: [img_all,256,128,3]   index_group: [Pall,K?]   id_group: 1D [0,Pall]
        pixel_mean = np.mean(image_array, axis=(0, 1, 2))
        # pixel_stddev = np.sqrt(np.mean(np.square(image_array - pixel_mean),axis=[0,1,2]))
        pixel_stddev = np.std(image_array, axis=(0, 1, 2))
        print('pixel mean: {0}'.format(pixel_mean))
        print('pixel stddev: {0}'.format(pixel_stddev))
        return pixel_mean, pixel_stddev

    def compute_dataset_train_pixel_mean_stddev(self, DatasetTrainDir=None):
        if DatasetTrainDir is None:
            train_set_dir = self.train_dir
        else:
            train_set_dir = DatasetTrainDir

        img_list = []
        fname_list = os.listdir(train_set_dir)
        for fname in fname_list:
            if fname == 'Thumbs.db':
                continue
            img = Image.open(os.path.join(train_set_dir, fname))
            img = img.resize((128, 256), resample=PIL.Image.BILINEAR)  # PIL.Image.NEAREST
            img = np.array(img)
            img = img / 255.  # [0,1] float type
            img_list.append(img)
        print('DatasetTrainFoler: {}'.format(train_set_dir))
        print('train set image num: {}'.format(len(img_list)))
        mean, stddev = self.compute_pixel_mean_stddev(img_list)
        return mean, stddev

    def compute_reID_mean_image_OP(self, ModelDir, NetSettingDir, if_show=False, SaveDir='./', DatasetDir=None):
        if DatasetDir is None:
            query_set_dir = self.query_dir
        else:
            query_set_dir = DatasetDir

        img_list = []
        fname_list = os.listdir(query_set_dir)
        for fname in fname_list:
            if fname == 'Thumbs.db':
                continue
            img = Image.open(os.path.join(query_set_dir, fname))
            img = img.resize((128, 256), resample=PIL.Image.BILINEAR)  # PIL.Image.NEAREST
            img = np.array(img)
            img = img / 255.  # [0,1] float type
            img_list.append(img)
        print('DatasetFoler: {}'.format(query_set_dir))
        print('image num: {}'.format(len(img_list)))
        mean, stddev = self.compute_pixel_mean_stddev(img_list)
        print('mean: {} stddev: {}'.format(mean, stddev))

        # mean image generation
        if t.cuda.is_available():
            use_GPU = True
            print('Use GPU')
        else:
            use_GPU = False
            print('No GPU, Use CPU')
        available_device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        # available_device = t.device("cpu")


        augmentation_preprocess = {'image_size': [256, 128],
                                   'pixel_mean': [0.485, 0.456, 0.406],
                                   'pixel_stddev': [0.229, 0.224, 0.225],
                                   'flip_probability': 0.5,
                                   'padding_size': 10,
                                   'zoom_out_pad_prob': 0.3,
                                   'random_eras_probability': 0.5,
                                   's_ratio_min': 0.02,
                                   's_ratio_max': 0.4,
                                   'aspect_ratio_min': 0.3
                                   }

        train_dataloader, train_img_num, test_dataloader, test_img_num = self.construct_dataset_dataloader_OP(
                query_set_dir, False, './', augmentation_preprocess,
                train_proportion=0.)
        # return 4D batch_imgs (tensor), 1D labels (tuple)

        model_setting = self.net_setting_load(Dir=NetSettingDir)
        my_model = self.construct_model_OP(model_setting)
        my_model.to(available_device)

        my_model.load_state_dict(t.load(ModelDir), strict=True)

        print('-------> start inference by {}'.format(ModelDir))
        my_model.eval()
        front_list = []
        back_list = []
        side_list = []
        with t.no_grad():
            for data in test_dataloader:
                images, _ = data
                images = images.to(available_device)
                logits = my_model(images)
                pred_np = logits.max(1)[1].cpu().numpy()  # [B,]

                front_idx = np.where(pred_np == 0)
                back_idx = np.where(pred_np == 1)
                side_idx = np.where(pred_np == 2)
                # 4D [B,C,H,W] normalized_tensor

                front_list.append(images[front_idx])
                back_list.append(images[back_idx])
                side_list.append(images[side_idx])

            front_images = t.cat(front_list, dim=0)
            back_images = t.cat(back_list, dim=0)
            side_images = t.cat(side_list, dim=0)
            # 4D [B, C, H, W] normalized_tensor

            print('Front image: {}'.format(front_images.shape))
            print('Back image: {}'.format(back_images.shape))
            print('Side image: {}'.format(side_images.shape))

            # #
            # fig = plt.figure()
            # subfig = fig.add_subplot(1,1,1)
            # subfig.imshow(front_images.permute(0, 2, 3, 1)[0])
            # plt.show()
            # # PASS
            #
            # tem = self.NormalizedTensorImage2Numpy(front_images[0],
            #                                        augmentation_preprocess['pixel_mean'],
            #                                        augmentation_preprocess['pixel_stddev'])
            # fig = plt.figure()
            # subfig = fig.add_subplot(1, 1, 1)
            # subfig.imshow(tem)
            # plt.show()
            # # PASS
            # #

            def tensor4Dimage_2_numpy(images4D, pixel_mean, pixel_stddev):
                img = images4D.permute(0, 2, 3, 1)
                img = img.cpu().numpy()
                img = img * pixel_stddev + \
                      pixel_mean
                img = img * 255.
                img_recovery = img.astype(np.uint8)
                return img_recovery

            recovery_front_images = tensor4Dimage_2_numpy(front_images,
                                                          augmentation_preprocess['pixel_mean'],
                                                          augmentation_preprocess['pixel_stddev'])
            t.cuda.empty_cache()
            del front_images
            recovery_back_images = tensor4Dimage_2_numpy(back_images,
                                                         augmentation_preprocess['pixel_mean'],
                                                         augmentation_preprocess['pixel_stddev'])
            t.cuda.empty_cache()
            del back_images
            recovery_side_images = tensor4Dimage_2_numpy(side_images,
                                                         augmentation_preprocess['pixel_mean'],
                                                         augmentation_preprocess['pixel_stddev'])
            t.cuda.empty_cache()
            del side_images
            # 4D [B, H, W, C] tensor uint8

        # #
        # fig = plt.figure()
        # subfig = fig.add_subplot(1, 3, 1)
        # subfig.imshow(recovery_front_images[0])
        # subfig = fig.add_subplot(1, 3, 2)
        # subfig.imshow(recovery_back_images[0])
        # subfig = fig.add_subplot(1, 3, 3)
        # subfig.imshow(recovery_side_images[0])
        # plt.show()
        # # PASS
        # #

        front_mean = recovery_front_images.mean(axis=0).astype(np.uint8)
        back_mean = recovery_back_images.mean(axis=0).astype(np.uint8)
        side_mean = recovery_side_images.mean(axis=0).astype(np.uint8)
        # side_mean = recovery_side_images.mean(axis=0, dtype=np.uint8) # WRONG All 0 ????

        orient_dict = {0:'front',1:'back', 2:'side'}
        fig = plt.figure()
        for idx, img in enumerate([front_mean, back_mean, side_mean]):
            subfig = fig.add_subplot(1,3,idx+1)
            subfig.imshow(img)
            subfig.set_title('{}'.format(orient_dict[idx]))
            subfig.set_xticks([])
            subfig.set_yticks([])

        fig.savefig(SaveDir)
        if if_show:
            plt.show()

    def load_from_folder_MIE(self, FolderDr='../CTL/dataset/CUHK03/bounding_box_train', if_reset_id=True):
        '''
        return: 1D image_path_list, id_list, cam_list; int person_num

        :param FolderDr:
        :param if_reset_id:
        :return:
        '''
        image_path_list = []
        id_list = []
        cam_list = []
        fname_list = os.listdir(FolderDr)
        fname_list.sort()
        if 'MSMT17'in FolderDr:
            for fname in fname_list:
                if fname == 'Thumbs.db':
                    continue
                image_path_list.append(str(os.path.join(FolderDr, fname)))
                fname_split = fname.split('_')
                pid = int(fname_split[0])
                cam = int(fname_split[2])
                id_list.append(pid)
                cam_list.append(cam)
        else:
            for fname in fname_list:
                if fname == 'Thumbs.db':
                    continue
                image_path_list.append(str(os.path.join(FolderDr, fname)))
                fname_split = fname.split('_')
                pid = int(fname_split[0])
                cam = int(fname_split[1][1])
                id_list.append(pid)
                cam_list.append(cam)

        if if_reset_id:
            pre_id_set = sorted(set(id_list), key=id_list.index)
            id_dict = {i: idx for idx, i in enumerate(pre_id_set)}  # dict has no sort problem!!!!
            # new_id_list = np.array([id_dict[i] for i in id_list])  # have sorted
            new_id_list = [id_dict[i] for i in id_list]  # have sorted -- list type
        else:
            new_id_list = id_list

        person_num = len(set(id_list))
        # XCP
        print('image_path_list {0}\tid_list {1}\tcam_list {2}\tperson_num {3}'.format(
            np.shape(image_path_list), np.shape(new_id_list), np.shape(cam_list), person_num
        ))
        print('Type:\nimage_path_list {}\tid_list {}\tcam_list {}\tperson_num {}'.format(
            type(image_path_list[0]), type(new_id_list[0]), type(cam_list[0]), type(person_num)
        ))

        return image_path_list, new_id_list, cam_list, person_num

    def load_from_folder_OP(self, FolderDir='../RAP_256x128'):
        '''
        return file_path_list, label_list

        RAP folder fname: imgid_pid_cam_orient  eg. 1_-2_01_0.jpg
        orientation label [0,1,2]
        :param FolderDir:
        :return:
        '''
        image_path_list = []
        orient_list = []
        fname_list = os.listdir(FolderDir)
        fname_list.sort()
        for fname in fname_list:
            if fname == 'Thumbs.db':
                continue
            image_path_list.append(str(os.path.join(FolderDir, fname)))
            fname_split = fname.split('_')
            orient_list.append(int(fname_split[3][0]))

        # XCP
        print('image_path_list {0}\torientation_list {1}'.format(
            np.shape(image_path_list), np.shape(orient_list)))
        print('load list type: path {}\torientation {}'.format(type(image_path_list[0]), type(orient_list[0])))

        label = np.array(orient_list)  # or np.where will not work !!!!!
        count_front = len(np.where(label == 0)[0])
        count_back = len(np.where(label == 1)[0])
        count_side = len(np.where(label == 2)[0])
        print('image num: {0}'.format(len(label)))
        print('percentage for each view:')
        print('Front: {0} Back: {1} Side: {2}'.format(count_front, count_back, count_side))
        print('Front: {0:.2%} Back: {1:.2%} Side: {2:.2%}'.format(
            count_front / float(len(label)), count_back / float(len(label)), count_side / float(len(label))
        ))

        return image_path_list, orient_list

    def load_from_reid_folder_list_OP(self, folder_dir = './', list_dir = './Market_train_OP_label.list'):
        '''
        return image_path_list(str), label_list(int)

        list file format: image_name label
        :param folder_dir:
        :param list_dir:
        :return:
        '''

        # double verify
        folder_fname_list = os.listdir(folder_dir)
        folder_fname_list.sort()
        if 'Thumbs.db' in folder_fname_list:
            folder_fname_list.remove('Thumbs.db')
            print('remove \'Thumbs.db\'')

        list_fname_list = []
        image_path_list = []
        orient_list = []
        listf = open(list_dir,'r')
        lines = listf.readlines()
        listf.close()
        for line in lines:
            line.strip()
            line_list = line.split(' ')
            list_fname_list.append(line_list[0])
            orient_list.append(int(line_list[1]))
            image_path_list.append(str(os.path.join(folder_dir, line_list[0])))

        folder_fname_set = set(folder_fname_list)
        list_fname_set = set(list_fname_list)
        assert folder_fname_set == list_fname_set, 'Folder fname Must Be Equal with List fname'

        # XCP
        print('image_path_list {0}\torientation_list {1}'.format(
                np.shape(image_path_list), np.shape(orient_list)))
        print('load list type: path {}\torientation {}'.format(type(image_path_list[0]), type(orient_list[0])))

        label = np.array(orient_list)  # or np.where will not work !!!!!
        count_front = len(np.where(label == 0)[0])
        count_back = len(np.where(label == 1)[0])
        count_side = len(np.where(label == 2)[0])
        print('image num: {0}'.format(len(label)))
        print('percentage for each view:')
        print('Front: {0} Back: {1} Side: {2}'.format(count_front, count_back, count_side))
        print('Front: {0:.2%} Back: {1:.2%} Side: {2:.2%}'.format(
                count_front / float(len(label)), count_back / float(len(label)), count_side / float(len(label))
        ))

        return image_path_list, orient_list

    def construct_dataset_dataloader_MIE(self,
                                         dataset_dir,
                                         augmentation_preprocess,
                                         dataloader_setting,
                                         if_reset_id=False,
                                         is_train=False,
                                         batch_size=128):
        # dataloader_setting = {'if_triplet_loss': if_triplet_loss,
        #                       'K_num': 4,
        #                       'if_center_loss': if_center_loss,
        #                       'num_workers': 8
        #                       }

        # construct dataset
        transforms = my_trans(augmentation_preprocess, is_train=is_train)

        img_path_array, id_array, cam_array, person_num = self.load_from_folder_MIE(dataset_dir,
                                                                                    if_reset_id=if_reset_id)
        # return: 1D image_path_list, id_list, cam_list; int person_num

        # dataset_array = np.concatenate((img_path_array[:,np.newaxis],id_array[:,np.newaxis],cam_array[:,np.newaxis]),
        #                               axis=1)# wrong !!!! type different !!!!!
        # wrong !!!! type different can not concatenate!!!!!

        dataset_list = [(img_path, pid, cam) for img_path, pid, cam in zip(img_path_array, id_array, cam_array)]
        # list [(path,pid,cam),...]


        my_dataset = ImageDataset(dataset_list, transforms)
        # each element in dataset including 4: transformed_image(tesnor), new_id(int), cam(int), img_path(str)


        # construct dataloader
        if not is_train:
            my_loader = DataLoader(
                my_dataset, batch_size=batch_size, shuffle=False,
                num_workers=dataloader_setting['num_workers'], collate_fn=val_collate_fn
            )
            # return 3 in one iter: 4D batch_imgs(tensor), 1D pids(int), 1D camids(int)
        else:
            # PK sample or shuffle sample
            if dataloader_setting['if_triplet_loss'] or dataloader_setting['if_center_loss']:
                my_loader = DataLoader(
                    my_dataset, batch_size=batch_size,
                    sampler=RandomIdentitySampler(dataset_list, batch_size, dataloader_setting['K_num']),
                    # sampler=RandomIdentitySampler_alignedreid(dataset_list, dataloader_setting['K_num']),  # new add by gu
                    num_workers=dataloader_setting['num_workers'], collate_fn=train_collate_fn
                )
                # return 2 in one iter: 4D batch_imgs(tensor), 1D pids(tensor)
            else:
                # softmax shuffle
                my_loader = DataLoader(
                    my_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=dataloader_setting['num_workers'], collate_fn=train_collate_fn
                )
                # return 2 in one iter: 4D batch_imgs(tensor), 1D pids(tensor)

        return my_loader, len(dataset_list), person_num

    def construct_inference_dataloader_MIE(self,
                                           dataset_dir,
                                           image_size,
                                           pixel_mean = [0.485, 0.456, 0.406],
                                           pixel_stddev = [0.229, 0.224, 0.225],
                                           PK_sample=False,
                                           if_shuffle=False,
                                           batch_size=8):
        augmentation_preprocess = {'image_size': image_size,
                                   'pixel_mean': pixel_mean,
                                   'pixel_stddev': pixel_stddev,
                                   'flip_probability': 0.5,
                                   'padding_size': 10,
                                   'random_eras_probability': 0.5,
                                   's_ratio_min': 0.02,
                                   's_ratio_max': 0.4,
                                   'aspect_ratio_min': 0.3
                                   }

        # construct dataset
        transforms = my_trans(augmentation_preprocess, is_train=False)

        img_path_array, id_array, cam_array, person_num = self.load_from_folder_MIE(dataset_dir, if_reset_id=False)
        # return: 1D image_path_list, id_list, cam_list; int person_num

        # dataset_array = np.concatenate((img_path_array[:,np.newaxis],id_array[:,np.newaxis],cam_array[:,np.newaxis]),
        #                               axis=1)# wrong !!!! type different !!!!!
        # wrong !!!! type different can not concatenate!!!!!

        dataset_list = [(img_path, pid, cam) for img_path, pid, cam in zip(img_path_array, id_array, cam_array)]
        # list [(path,pid,cam),...]


        my_dataset = ImageDataset(dataset_list, transforms)
        # each element in dataset including 4: transformed_image(tesnor), new_id(int), cam(int), img_path(str)


        # construct dataloader
        if PK_sample:
            # PK sample or shuffle sample
            my_loader = DataLoader(
                    my_dataset, batch_size=batch_size,
                    sampler=RandomIdentitySampler(dataset_list, batch_size, 4),
                    # sampler=RandomIdentitySampler_alignedreid(dataset_list, dataloader_setting['K_num']),  # new add by gu
                    num_workers=8, collate_fn=inference_collate_fn
            )
            # return 3 in one iter: 4D batch_imgs(tensor), 1D pids(tuple), 1D paths(tuple)
        else:
            # softmax shuffle
            my_loader = DataLoader(
                    my_dataset, batch_size=batch_size, shuffle=if_shuffle,
                    num_workers=8, collate_fn=inference_collate_fn
            )
            # return 3 in one iter: 4D batch_imgs(tensor), 1D pids(tuple), 1D paths(tuple)

        return my_loader

    def construct_dataset_dataloader_OP(self,
                                        folder_dir,
                                        if_use_reid_dataset,
                                        reid_list_dir,
                                        augmentation_preprocess,
                                        train_proportion=0.8,
                                        if_random_erase=False,
                                        train_batch_size = 64,
                                        test_batch_size = 128
                                        ):
        # transformation
        if not if_random_erase:
            # use rotation && random zoom out
            print('use random rotation && random zoom out')
            train_transformation = my_trans_op(augmentation_preprocess, is_train=True)
            test_transformation = my_trans_op(augmentation_preprocess, is_train=False)
        else:
            # use random erase, no rotation, no random zoom out
            print('use random erase, no rotation, no random zoom out')
            augmentation_preprocess['if_REA'] = True # fix BUG when use same trans with MIE
            train_transformation = my_trans(augmentation_preprocess, is_train=True)
            test_transformation = my_trans(augmentation_preprocess, is_train=False)

        if if_use_reid_dataset:
            image_path_list, label_list = self.load_from_reid_folder_list_OP(folder_dir, reid_list_dir)
        else:
            image_path_list, label_list = self.load_from_folder_OP(folder_dir)

        dataset_list = [(img_path, label) for img_path, label in zip(image_path_list, label_list)]
        # list [(path,label),...]
        random.shuffle(dataset_list)

        split_point = int(train_proportion * len(dataset_list))
        train_list = dataset_list[:split_point]
        test_list = dataset_list[split_point:]
        print('image num train: {}  test: {} train proportion: {:.2%}'.format(len(train_list),
                                                                              len(test_list),
                                                                              len(train_list) / len(dataset_list)))

        my_train_dataset = ImageDatasetOP(train_list, train_transformation)
        my_test_dataset = ImageDatasetOP(test_list, test_transformation)
        # each element in dataset including 3: transformed_image(tesnor), label(int), img_path(str)

        # construct dataloader
        my_train_loader = DataLoader(
            my_train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn_OP_1
        )
        # return 2 in one iter: 4D batch_imgs (tensor), 1D labels (tuple)

        # softmax shuffle
        my_test_loader = DataLoader(
            my_test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn_OP_2
        )
        # return 2 in one iter: 4D batch_imgs (tensor), 1D labels (tuple)

        return my_train_loader, len(train_list), my_test_loader, len(test_list)

    def construct_model_MIE(self, num_class, model_setting, use_GPU):
        # model_setting = {'last_stride': 1,
        #                  'neck': True,
        #                  'eval_neck': True,
        #                  'ImageNet_Init': True,
        #                  'ImageNet_model_path': './resnet50-19c8e357.pth',
        #                  'addOrientationPart': addOrientationPart,
        #                  'addSTNPart': addSTNPart,
        #                  'addAttentionMaskPart': addAttentionMaskPart,
        #                  'addChannelReduce': addChannelReduce,
        #                  'isDefaultImageSize': self.isDefaultImageSize,
        #                  'final_dim': final_dim,
        #                  'mask_num': mask_num,
        #                  'dropout_rate': 0.,  # 0.6 whether have influence on triplet loss??
        #                  'if_affine': if_affine,
        #                  'if_OP_channel_wise': if_OP_channel_wise,
        #                  'STN_fc_b_init': STN_init_value
        #                  }
        model = BaselinePro(num_class, model_setting['neck'], model_setting['eval_neck'],
                            model_setting['ImageNet_Init'], model_setting['ImageNet_model_path'],
                            model_setting['last_stride'], model_setting['addOrientationPart'],
                            model_setting['addSTNPart'], model_setting['addAttentionMaskPart'],
                            model_setting['addChannelReduce'], model_setting['isDefaultImageSize'],
                            model_setting['final_dim'], model_setting['mask_num'], model_setting['dropout_rate'],
                            use_GPU, model_setting['if_affine'],
                            model_setting['if_OP_channel_wise'], model_setting['STN_fc_b_init'])
        return model

    def construct_model_OP(self, model_setting):
        model = BaselineOP(if_dropout=model_setting['if_dropout'], dropout_rate=model_setting['dropout_rate'],
                           isDefaultImageSize=model_setting['isDefaultImageSize'],
                           ImageNet_Init=model_setting['ImageNet_Init'],
                           ImageNet_model_path=model_setting['ImageNet_model_path'])
        return model

    def build_loss_structure(self, person_num, if_center_loss, loss_setting, use_GPU=True):
        # loss_setting = {'eval_feature_dim': final_feature_dim,
        #                 'if_triplet_loss': if_triplet_loss,
        #                 'triplet_margin': 0.3,  # 0.1-0.3 Zheng suggest, LuoHao 0.3
        #                 'label_smooth_rate': 0.1,  # 0. original softmax
        #                 'if_mask_loss': if_mask_loss,
        #                 'binary_threshold': binary_threshold,
        #                 'area_constrain_proportion': area_constrain_proportion
        #                 }
        # GAP_feature -- raw not apply l2 normalization
        # strange????  local class with fn return, can be accessed in other place??????
        if if_center_loss:
            center_criterion = CenterLoss(num_classes=person_num, feat_dim=loss_setting['eval_feature_dim'],
                                          use_gpu=use_GPU)
            # center loss
        if loss_setting['if_triplet_loss']:
            triplet = TripletLoss(loss_setting['triplet_margin'])  # triplet loss

        if loss_setting['if_mask_loss']:
            mask = MaskLoss(binary_threshold=loss_setting['binary_threshold'],
                            area_constrain_proportion=loss_setting['area_constrain_proportion'], use_GPU=use_GPU)

        xent = CrossEntropyLabelSmooth(num_classes=person_num, epsilon=loss_setting['label_smooth_rate'],
                                       use_gpu=use_GPU)

        # label smooth softmax

        # input not one-hot label
        def loss_func(fc_logits, eval_feature, label, if_triplet_loss, if_center_loss, center_loss_weight,
                      use_GPU=False, if_mask_loss=False, masks_list=None, mask_loss_weight=0.0001):
            softmax_loss = xent(fc_logits, label)
            if if_triplet_loss:
                triplet_loss = triplet(eval_feature, label)[0]
            else:
                triplet_loss = t.tensor(0., dtype=t.float32)
                if use_GPU:
                    triplet_loss = triplet_loss.cuda()
            if if_center_loss:
                center_loss = center_criterion(eval_feature, label) * center_loss_weight  # scale tensor
            else:
                center_loss = t.tensor(0., dtype=t.float32)
                if use_GPU:
                    center_loss = center_loss.cuda()
            if if_mask_loss:
                mask_loss = mask(masks_list)[0] * mask_loss_weight
            else:
                mask_loss = t.tensor(0., dtype=t.float32)
                if use_GPU:
                    mask_loss = mask_loss.cuda()

            # # XCP
            # print(isinstance(softmax_loss,t.cuda.FloatTensor))
            # print(isinstance(triplet_loss,t.cuda.FloatTensor))
            # print(isinstance(center_loss,t.cuda.FloatTensor))
            # print(isinstance(mask_loss,t.cuda.FloatTensor))

            final_loss = softmax_loss + triplet_loss + center_loss + mask_loss
            return final_loss, final_loss.item(), \
                   softmax_loss.item(), triplet_loss.item(), center_loss.item(), mask_loss.item()

        if if_center_loss:
            return loss_func, center_criterion
        else:
            return loss_func

    def lr_scheduler_MIE(self, optimizer, current_epoch, scheduler_setting, STN_param_idx_set, OP_param_idx_set):
        # scheduler_setting = {'lr_start': Lr_Start,
        #                      'STN_lr': STN_lr_start,
        #                      'OP_lr': OP_lr_start,
        #                      'STN_freeze_end_ep': STN_freeze_end_ep,
        #                      'OP_freeze_end_ep': OP_freeze_end_ep,
        #                      'addOrientationPart': addOrientationPart,
        #                      'decay_rate': 10,
        #                      'warmup_rate': 100
        #                      }
        # one epoch call once
        # TODO: Now STN learning rate scheduler is decay version (decay with main lr), OP Fix Lr
        # TODO: when need change STN_lr scheduler strategy(such as fix lr), apply here !!!!!!
        if current_epoch == 0:
            factor = 1. / scheduler_setting['warmup_rate']
        elif current_epoch <= 10:
            factor = float(current_epoch) / 10
        elif current_epoch <= 40:
            factor = 1
        elif current_epoch <= 80:
            factor = 1. / scheduler_setting['decay_rate']
        else:
            factor = 1. / float(scheduler_setting['decay_rate'] ** 2)

        if scheduler_setting['STN_freeze_end_ep'] <= 40:
            STN_factor = factor
        elif scheduler_setting['STN_freeze_end_ep'] <= 80:
            STN_factor = factor * scheduler_setting['decay_rate']
        else:
            STN_factor = 1.

        if scheduler_setting['OP_freeze_end_ep'] <= 40:
            OP_factor = factor
        elif scheduler_setting['OP_freeze_end_ep'] <= 80:
            OP_factor = factor * scheduler_setting['decay_rate']
        else:
            OP_factor = 1.


        STN_lr_record = []
        OP_lr_record = []
        for idx, param_group in enumerate(optimizer.param_groups):
            if idx in STN_param_idx_set:
                if current_epoch >= scheduler_setting['STN_freeze_end_ep']:
                    if scheduler_setting['addOrientationPart']:
                        if idx in [143,144]: # fc_W, fc_b
                            param_group['lr'] = scheduler_setting['STN_lr'] * STN_factor # decay Version
                            # param_group['lr'] = scheduler_setting['STN_lr'] # Fix Version
                        else:
                            param_group['lr'] = scheduler_setting['lr_start'] * factor
                    else:
                        if idx in [129,130]: # fc_W, fc_b
                            param_group['lr'] = scheduler_setting['STN_lr'] * STN_factor # decay Version
                            # param_group['lr'] = scheduler_setting['STN_lr'] # Fix Version
                        else:
                            param_group['lr'] = scheduler_setting['lr_start'] * factor
                else:
                    param_group['lr'] = 0.

                STN_lr_record.append(param_group['lr'])
            elif idx in OP_param_idx_set: # if use re-id set trained OP, then no need train !!!!!
                if current_epoch >= scheduler_setting['OP_freeze_end_ep']:
                    param_group['lr'] = scheduler_setting['OP_lr'] * OP_factor # decay Version
                    # param_group['lr'] = scheduler_setting['OP_lr'] # Fix Version
                else:
                    param_group['lr'] = 0.

                OP_lr_record.append(param_group['lr'])
            else:
                param_group['lr'] = scheduler_setting['lr_start'] * factor
        print('----> apply lr scheduler & basic lr: {}'.format(scheduler_setting['lr_start'] * factor))
        print('----> STN lr: {}'.format(STN_lr_record))
        print('----> OP lr: {}'.format(OP_lr_record))
        return scheduler_setting['lr_start'] * factor

    def lr_scheduler_OP(self, optimizer, current_epoch, scheduler_setting, NoneOP_idx_set, if_load_SBP):
        # scheduler_setting = {'lr_start':0.1,
        #                      'decay_rate':10,
        #                      'warmup_rate':100
        #                      }
        # one epoch call once
        # !!!!!!!!!  STN independent lr may occur problems for UnKnow which var is belong to STN !!!!!!!!!!
        if current_epoch == 0:
            factor = 1. / scheduler_setting['warmup_rate']
        elif current_epoch <= 10:
            factor = float(current_epoch) / 10
        elif current_epoch <= 20:
            factor = 1
        elif current_epoch <= 40:
            factor = 1. / scheduler_setting['decay_rate']
        else:
            factor = 1. / float(scheduler_setting['decay_rate'] ** 2)

        NoneOP_lr_record = []
        for idx,param_group in enumerate(optimizer.param_groups):
            if idx in NoneOP_idx_set and if_load_SBP:
                param_group['lr'] = 0.
                NoneOP_lr_record.append(param_group['lr'])
            else:
                param_group['lr'] = scheduler_setting['lr_start'] * factor

        print('----> apply lr scheduler & basic lr: {}'.format(scheduler_setting['lr_start'] * factor))
        print('----> None OP Part freeze lr:\n{}'.format(NoneOP_lr_record))
        return scheduler_setting['lr_start'] * factor

    def generate_optimizer_MIE(self, optimizer_setting, model):
        optimizer, STN_param_idx_set, OP_param_idx_set = optimizer_build.make_optimizer(optimizer_setting, model)
        # make one var one group !!!
        # notice weight decay for bias item and for BN layer !!!!!!!
        return optimizer, STN_param_idx_set, OP_param_idx_set

    def generate_center_optimizer_MIE(self, optimizer_setting, model, center_criterion):
        optimizer, center_optimzier, STN_param_idx_set, OP_param_idx_set = optimizer_build.make_optimizer_with_center(
                optimizer_setting, model, center_criterion)
        # one var one group !!!
        return optimizer, center_optimzier, STN_param_idx_set, OP_param_idx_set

    def generate_optimizer_OP(self, optimizer_setting, model):
        optimizer, NoneOP_idx_set = make_optimizer_OP(optimizer_setting, model)
        # make one var one group !!!
        # notice weight decay for bias item and for BN layer !!!!!!!
        return optimizer, NoneOP_idx_set

    def calculate_topn_mAP(self, query_feature, query_label, test_feature, test_label,
                           if_rerank=False, if_Euclidean=False, if_sqrt=False):
        '''
        4D feature(tensor), [Num,2] label (list)

        :param query_feature:
        :param query_label:
        :param test_feature:
        :param test_label:
        :param if_rerank:
        :param if_Euclidean:
        :param if_sqrt:
        :return:
        '''

        def EuclideanDistanceMatrix(M1, M2, if_sqrt=False):
            # Useless!!!!!
            # eplison = 1e-12
            # if not t.equal(t.norm(M1,2,1),t.ones((M1.shape[0],))):
            #     print('EuclideanDistanceMatrix input M1 not L2 normalized tensor !!!!')
            #     M1 = M1 / (t.norm(M1,2,1,keepdim=True).expand_as(M1) + eplison)
            # if not t.equal(t.norm(M2,2,1),t.ones((M2.shape[0],))):
            #     print('EuclideanDistanceMatrix input M2 not L2 normalized tensor !!!!')
            #     M2 = M2 / (t.norm(M2,2,1,keepdim=True).expand_as(M2) + eplison)

            m, n = M1.shape[0], M2.shape[0]
            dists = t.pow(M1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    t.pow(M2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            dists.addmm_(1, -2, M1, M2.t())
            if if_sqrt:
                dists.sqrt_()
            return dists

        TEST_NUM = len(test_label)
        QUERY_NUM = len(query_label)

        # L2 normalization for Cosine or Euclidean distance or rerank
        query_feature_norm = normalize(query_feature, axis=-1)
        test_feature_norm = normalize(test_feature, axis=-1)

        match = []
        junk = []

        print('compute match-junk')
        for q_index, (qp, qc) in enumerate(query_label):
            tmp_match = []
            tmp_junk = []
            for t_index, (tp, tc) in enumerate(test_label):
                if tp == qp and qc != tc:
                    tmp_match.append(t_index)
                elif tp == qp or tp == -1:
                    tmp_junk.append(t_index)

            # match.append(tmp_match)
            # junk.append(tmp_junk)
            # XCP - Speed Up
            match.append(set(tmp_match))
            junk.append(set(tmp_junk))

        print('start matmul')
        if if_rerank:
            # k-reciprocal rerank
            # Smaller is similar, [0,...]
            print('use re-rank')
            similarity_matrix = re_ranking(query_feature_norm, test_feature_norm,
                                           k1=20, k2=6, lambda_value=0.3)
            # return nparray!!!!!
        elif if_Euclidean:
            # Euclidean Distance
            # Smaller is similar: [0,...]
            print('use Euclidean Distance')
            similarity_matrix = EuclideanDistanceMatrix(query_feature_norm, test_feature_norm, if_sqrt=if_sqrt)
            similarity_matrix = similarity_matrix.cpu().numpy()
        else:
            # Cosine Distance
            # Similarity Matrix -- Query_num * Test_num
            # Bigger is similar: [-1,1]
            print('use Cosine Distance')
            similarity_matrix = t.matmul(query_feature_norm, test_feature_norm.t())
            similarity_matrix = similarity_matrix.cpu().numpy()

        result_argsort = np.argsort(similarity_matrix, axis=1)
        print('end matmul')

        if if_rerank or if_Euclidean:
            test_idx_list = list(range(0, TEST_NUM))
        else:
            test_idx_list = list(reversed(range(0, TEST_NUM)))

        mAP = 0.0
        CMC = np.zeros([len(query_label), len(test_label)])
        for idx in range(len(query_label)):
            # XCP
            if idx % 100 == 0:
                print('person: {}'.format(idx))

            recall = 0.0
            precision = 1.0
            hit = 0.0
            cnt = 0
            ap = 0.0
            YES = match[idx]
            IGNORE = junk[idx]

            for i in test_idx_list:
                k = result_argsort[idx][i]
                if k in IGNORE:
                    continue
                else:
                    cnt += 1
                    if k in YES:
                        CMC[idx, cnt - 1:] = 1
                        hit += 1

                    tmp_recall = float(hit) / len(YES)
                    tmp_precision = float(hit) / cnt
                    ap = ap + (tmp_recall - recall) * ((precision + tmp_precision) / 2.)
                    recall = tmp_recall
                    precision = tmp_precision
                if hit == len(YES):
                    break
            mAP += ap

        rank_1 = np.mean(CMC[:, 0])
        rank_5 = np.mean(CMC[:, 4])
        rank_10 = np.mean(CMC[:, 9])
        rank_20 = np.mean(CMC[:, 19])
        mAP /= QUERY_NUM

        rank = [rank_1, rank_5, rank_10, rank_20]
        return rank, mAP

    def independent_evaluate_MIE(self,
                                 image_size=[224, 224],
                                 if_rerank=False,
                                 if_Euclidean=False,
                                 batch_size=128,
                                 target_model_path=None,
                                 model_setting_path=None):
        # Pay Attention to Parameter Dict !!!!!!!! Need keep same with train  !!!!!!!
        if t.cuda.is_available():
            use_GPU = True
            print('Use GPU')
        else:
            use_GPU = False
            print('No GPU, Use CPU')
        available_device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        if target_model_path is None:
            fname_list = os.listdir(self.save_dir)
            fname_list.sort()
            target_model_path = os.path.join(self.save_dir, fname_list[-1])
        if model_setting_path is None:
            model_setting_path = self.net_setting_dir

        augmentation_preprocess = {'image_size': image_size,
                                   'pixel_mean': [0.485, 0.456, 0.406],
                                   'pixel_stddev': [0.229, 0.224, 0.225],
                                   'flip_probability': 0.5,
                                   'padding_size': 10,
                                   'random_eras_probability': 0.5,
                                   's_ratio_min': 0.02,
                                   's_ratio_max': 0.4,
                                   'aspect_ratio_min': 0.3,
                                   'if_REA':False
                                   }
        dataloader_setting = {'if_triplet_loss': False,
                              'K_num': 4,
                              'if_center_loss': False,
                              'num_workers': 8
                              }

        query_dataloader, query_img_num, query_person_num = self.construct_dataset_dataloader_MIE(
            self.query_dir, augmentation_preprocess, dataloader_setting,
            if_reset_id=False, is_train=False, batch_size=batch_size)

        test_dataloader, test_img_num, test_person_num = self.construct_dataset_dataloader_MIE(
            self.test_dir, augmentation_preprocess, dataloader_setting,
            if_reset_id=False, is_train=False, batch_size=batch_size)

        model_setting = self.net_setting_load(Dir=model_setting_path)
        my_model = self.construct_model_MIE(2, model_setting, use_GPU)  # random class_num
        my_model.to(available_device)

        # Need exclude classifier in state_dict
        # Method 1
        # loaded_state_dict = t.load(target_model_path)
        # exclude = []
        # for key in loaded_state_dict:
        #     if 'classifier' in key:
        #         exclude.append(key)
        # for key in exclude:
        #     loaded_state_dict.pop(key)
        # my_model.load_state_dict(loaded_state_dict, strict=True)

        # Method 2
        my_model.load_param_transfer(target_model_path)

        print('-------> start evaluate {}'.format(target_model_path))
        eval_time_start = time.clock()
        my_model.eval()
        query_feature = []
        query_label = []
        test_feature = []
        test_label = []
        with t.no_grad():
            print('extract query set feature')
            for data in query_dataloader:
                image, pid, cam = data
                image = image.to(available_device)
                feature = my_model(image)
                query_feature.append(feature)

                pid_array = np.array(pid)[:, np.newaxis]
                cam_array = np.array(cam)[:, np.newaxis]
                tem_label = np.concatenate((pid_array, cam_array), axis=1).tolist()  # [B,2]
                query_label.extend(tem_label)

            query_feature = t.cat(query_feature, dim=0)
            print('query_feature: {}\tquery_label: {}'.format(query_feature.shape, np.shape(query_label)))

            print('extract test set feature')
            for data in test_dataloader:
                image, pid, cam = data
                image = image.to(available_device)
                feature = my_model(image)
                test_feature.append(feature)

                pid_array = np.array(pid)[:, np.newaxis]
                cam_array = np.array(cam)[:, np.newaxis]
                tem_label = np.concatenate((pid_array, cam_array), axis=1).tolist()  # [B,2]
                test_label.extend(tem_label)

            test_feature = t.cat(test_feature, dim=0)
            print('test_feature: {}\ttest_label: {}'.format(test_feature.shape, np.shape(test_label)))

            rank, mAP = self.calculate_topn_mAP(query_feature, query_label, test_feature, test_label,
                                                if_rerank=if_rerank, if_Euclidean=if_Euclidean, if_sqrt=False)

            print('Epoch {} Dataset {}'.format(0, self.target_dataset_name))
            print('1: %f\t5: %f\t10: %f\t20: %f\tmAP: %f' % (rank[0], rank[1], rank[2], rank[3], mAP))

            time_current = time.clock() - eval_time_start
            time_elapsed_hour = int(time_current // 3600)
            time_elapsed_minute = int(time_current // 60 - 60 * time_elapsed_hour)
            time_elapsed_second = int(time_current - 3600 * time_elapsed_hour - 60 * time_elapsed_minute)
            print('Time elapsed: {0}:{1}:{2}'.format(time_elapsed_hour, time_elapsed_minute, time_elapsed_second))

            acc_f = open(self.independent_eval_dir, 'a')
            acc_f.write('epoch: {0}\t'.format(0))
            acc_f.write('1: {0}\t5: {1}\t10: {2}\t20: {3}\tmAP: {4}\t'.format(
                rank[0], rank[1], rank[2], rank[3], mAP))
            acc_f.write('Time: {0}:{1}:{2}\n'.format(
                time_elapsed_hour, time_elapsed_minute, time_elapsed_second))
            acc_f.close()

    def independent_evaluate_all_MIE(self,
                                     image_size=[224, 224],
                                     batch_size=128,
                                     if_rerank=False, if_Euclidean=False,
                                     model_setting_path=None):
        f = open(self.independent_eval_dir, 'w')
        f.close()
        if model_setting_path is None:
            model_setting_path = self.net_setting_dir
        fname_list = os.listdir(self.save_dir)
        for fname in fname_list:
            target_model_path = os.path.join(self.save_dir, fname)
            self.independent_evaluate_MIE(image_size=image_size, if_rerank=if_rerank,
                                          if_Euclidean=if_Euclidean, batch_size=batch_size,
                                          target_model_path=target_model_path,
                                          model_setting_path=model_setting_path)

        self.plot_acc_MIE(eval_dir=self.independent_eval_dir, ifSave=True,
                          SaveDir='./MIE/independent_MIE_acc.jpg')


    def tSNE_MIE(self,
                 image_size=[256, 128],
                 P_num=20,
                 K_num = 10,
                 target_model_path=None,
                 model_setting_path=None,
                 if_show = True,
                 if_save = True,
                 save_name = 'tSNE.eps'):

        def tSNE_visualize(feature, label, if_show = True, if_save = True, save_name = 'tSNE.eps'):
            print('-------> start tSNE')
            def plot_with_labels(lowDWeights, labels, if_show = True, if_save = True, SaveName = 'tSNE.eps'):
                # visualize scale
                # lowDWeights = (lowDWeights - lowDWeights.min(axis=0)) / \
                #               (lowDWeights.max(axis=0) - lowDWeights.min(axis=0))

                # visualize normalize
                # lowDWeights = (lowDWeights - lowDWeights.mean(axis=0)) / lowDWeights.std(axis=0)

                plt.cla()
                X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
                for x, y, s in zip(X, Y, labels):
                    c = cm.rainbow(int(255 * s / 19))
                    # plt.text(x, y, s,backgroundcolor=c, fontsize=9)
                    plt.scatter([x], [y], s=20, c=c)
                plt.xlim(X.min(), X.max())
                plt.ylim(Y.min(), Y.max())
                # plt.title('Visualize last layer')
                plt.xticks([])
                plt.yticks([])
                if if_save:
                    plt.savefig(SaveName,dpi=1200)
                if if_show:
                    plt.show()
                    # plt.pause(0.01)

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

            feature_np = feature.cpu().data.numpy()

            # feature standarlization
            # f_mean = feature_np.mean(axis = 0)
            # f_std = feature_np.std(axis = 0)
            # feature_np = (feature_np - f_mean) / (f_std + 1e-12)


            low_dim_embs = tsne.fit_transform(feature_np)
            # labels = label.cpu().numpy()

            # save_name = 'MNist_tSNE_{}.eps'.format(save_name)
            plot_with_labels(low_dim_embs, labels, if_show=if_show, if_save=if_save, SaveName=save_name)

        # Pay Attention to Parameter Dict !!!!!!!! Need keep same with train  !!!!!!!
        if t.cuda.is_available():
            use_GPU = True
            print('Use GPU')
        else:
            use_GPU = False
            print('No GPU, Use CPU')
        available_device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        if target_model_path is None:
            fname_list = os.listdir(self.save_dir)
            fname_list.sort()
            target_model_path = os.path.join(self.save_dir, fname_list[-1])
        if model_setting_path is None:
            model_setting_path = self.net_setting_dir

        augmentation_preprocess = {'pixel_mean': [0.485, 0.456, 0.406],
                                   'pixel_stddev': [0.229, 0.224, 0.225],
                                   }

        # trans
        def tem_transforms(aug_setting):
            normalize_transform = trans.Normalize(mean=aug_setting['pixel_mean'], std=aug_setting['pixel_stddev'])
            transform = trans.Compose([
                trans.ToTensor(),
                normalize_transform
            ])
            return transform
        tem_trans = tem_transforms(augmentation_preprocess)


        # dataset
        img_path_array, id_array, cam_array, person_num = self.load_from_folder_MIE(self.test_dir,
                                                                                    if_reset_id=False)
        dataset_list = [(img_path, pid, cam) for img_path, pid, cam in zip(img_path_array, id_array, cam_array)]
        my_dataset = ImageDataset(dataset_list, tem_trans)


        # dataloader
        my_dataloader = DataLoader(
                my_dataset, batch_size=P_num * K_num,
                sampler=RandomIdentitySampler(dataset_list, P_num * K_num, K_num),
                # sampler=RandomIdentitySampler_alignedreid(dataset_list, dataloader_setting['K_num']),  # new add by gu
                num_workers=8, collate_fn=val_collate_fn
        )
        # return 2 in one iter: 4D batch_imgs(tensor), 1D pids(tensor)

        print('iteration num per epoch: {}'.format(len(my_dataloader)))
        print('iter images num: {} dataset image: {}'.format(len(my_dataloader) * P_num * K_num,
                                                             len(dataset_list)))
        print('including person: {}'.format(person_num))

        model_setting = self.net_setting_load(Dir=model_setting_path)
        my_model = self.construct_model_MIE(2, model_setting, use_GPU)  # random class_num
        my_model.to(available_device)

        my_model.load_param_transfer(target_model_path)

        print('-------> start evaluate {}'.format(target_model_path))
        my_model.eval()
        with t.no_grad():
            print('extract test set feature')
            for data in my_dataloader:
                image, pid, cam = data
                image = image.to(available_device)
                feature = my_model(image)

                pid_set = set(pid)
                new_id_dict = {i:idx for idx,i in enumerate(list(pid_set))}
                new_id_list = [new_id_dict[i] for i in pid]
                labels = np.array(new_id_list)
                tSNE_visualize(feature, labels, if_show, if_save, save_name)



    def independent_evaluate_OP(self,
                                folder_dir,
                                if_use_reid_dataset,
                                reid_list_dir,
                                target_model_path,
                                model_setting_dir='./OP/model_setting.txt',
                                independent_acc_dir='./OP/independent_acc.txt',
                                train_proportion=0.):
        if t.cuda.is_available():
            use_GPU = True
            print('Use GPU')
        else:
            use_GPU = False
            print('No GPU, Use CPU')
        available_device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

        image_size = [224, 224] if self.isDefaultImageSize else [256, 128]
        augmentation_preprocess = {'image_size': image_size,
                                   'pixel_mean': [0.485, 0.456, 0.406],
                                   'pixel_stddev': [0.229, 0.224, 0.225],
                                   'flip_probability': 0.5,
                                   'padding_size': 10,
                                   'zoom_out_pad_prob': 0.3,
                                   'random_eras_probability': 0.5,
                                   's_ratio_min': 0.02,
                                   's_ratio_max': 0.4,
                                   'aspect_ratio_min': 0.3
                                   }

        train_dataloader, train_img_num, test_dataloader, test_img_num = self.construct_dataset_dataloader_OP(
                folder_dir, if_use_reid_dataset, reid_list_dir, augmentation_preprocess,
                train_proportion=train_proportion)

        model_setting = self.net_setting_load(Dir=model_setting_dir)
        my_model = self.construct_model_OP(model_setting)
        my_model.to(available_device)

        my_model.load_state_dict(t.load(target_model_path), strict=True)

        eval_time_start = time.clock()
        my_model.eval()
        with t.no_grad():
            avg_front_acc = []
            avg_back_acc = []
            avg_side_acc = []
            avg_total_acc = []
            for idx, data in enumerate(test_dataloader):
                batch_image, batch_label = data
                batch_image = batch_image.to(available_device)
                batch_label = batch_label.to(available_device)

                logits = my_model(batch_image)

                total_acc = (logits.max(1)[1] == batch_label).float().mean().item()

                # single orientation acc
                label_np = batch_label.cpu().numpy()
                pred_np = logits.max(1)[1].cpu().numpy()  # [B,]
                front_idx = np.where(label_np == 0)[0]
                back_idx = np.where(label_np == 1)[0]
                side_idx = np.where(label_np == 2)[0]

                front_label = label_np[front_idx]
                back_label = label_np[back_idx]
                side_label = label_np[side_idx]

                front_pred = pred_np[front_idx]
                back_pred = pred_np[back_idx]
                side_pred = pred_np[side_idx]

                front_acc = (front_pred == front_label).astype(np.float32).mean()
                back_acc = (back_pred == back_label).astype(np.float32).mean()
                side_acc = (side_pred == side_label).astype(np.float32).mean()

                avg_total_acc.append(total_acc)
                avg_front_acc.append(front_acc)
                avg_back_acc.append(back_acc)
                avg_side_acc.append(side_acc)

            avg_total_acc = np.mean(avg_total_acc, axis=0)
            avg_front_acc = np.mean(avg_front_acc, axis=0)
            avg_back_acc = np.mean(avg_back_acc, axis=0)
            avg_side_acc = np.mean(avg_side_acc, axis=0)
            print('Avg_acc: {}\tAvg_f_acc: {}\tAvg_b_acc: {}\tAvg_s_acc: {}'.format(
                avg_total_acc, avg_front_acc, avg_back_acc, avg_side_acc))

            time_current = time.clock() - eval_time_start
            time_elapsed_hour = int(time_current // 3600)
            time_elapsed_minute = int(time_current // 60 - 60 * time_elapsed_hour)
            time_elapsed_second = int(time_current - 3600 * time_elapsed_hour - 60 * time_elapsed_minute)
            print('Time elapsed: {0}:{1}:{2}'.format(time_elapsed_hour, time_elapsed_minute,
                                                     time_elapsed_second))

            acc_f = open(independent_acc_dir, 'a')
            acc_f.write('epoch: {0}\t'.format(0))
            acc_f.write('Avg_acc: {}\tAvg_f_acc: {}\tAvg_b_acc: {}\tAvg_s_acc: {}\t'.format(
                avg_total_acc, avg_front_acc, avg_back_acc, avg_side_acc))
            acc_f.write('Time: {0}:{1}:{2}\n'.format(
                time_elapsed_hour, time_elapsed_minute, time_elapsed_second))
            acc_f.close()

    def independent_evaluate_all_OP(self,
                                    dataset_folder_dir,
                                    model_save_dir='./OP/model_save',
                                    model_setting_dir='./OP/model_setting.txt',
                                    independent_acc_dir='./OP/independent_acc.txt',
                                    train_proportion=0.):
        f = open(independent_acc_dir, 'w')
        f.close()
        fname_list = os.listdir(model_save_dir)
        for fname in fname_list:
            target_model_path = os.path.join(model_save_dir, fname)
            self.independent_evaluate_OP(dataset_folder_dir, target_model_path,
                                         model_setting_dir, independent_acc_dir, train_proportion)

        self.plot_acc_OP(independent_acc_dir, ifSave=True, SaveDir='./OP/independent_acc.jpg')

    def train_OP(self,
                 lr_start=0.0001,
                 Epoch=60,
                 save_step=1,
                 eval_step=1,
                 train_proportion=0.8,
                 if_dropout=True,
                 dropout_rate=0.6,
                 optimizer_name='Adam',
                 if_random_erase=False,
                 zoom_out_pad_prob=0.3,
                 if_weight_softmax=False,
                 weight_softmax_value = [2.,2.,1.],
                 if_load_SBP=False,
                 SBP_dir='./SBP/Cosine_Success/model_save/pytorch_SBP_119.pth',
                 if_use_reid_dataset=False,
                 reid_list_dir='./Market_train_OP_label.list',
                 folder_dir='../RAP_Dataset',
                 model_save_dir='./OP/model_save',
                 model_save_name='OP',
                 loss_lr_dir='./OP/loss_lr.txt',
                 acc_dir='./OP/acc.txt',
                 model_setting_dir='./OP/model_setting.txt',
                 history_dir='./OP/history.txt',
                 plot_dir='./OP/loss_acc_lr.png',
                 if_show_plot=True):

        size_tag = '224x224' if self.isDefaultImageSize else '256x128'
        if not if_use_reid_dataset:
            folder_dir = folder_dir + '_' + size_tag
            if not os.path.exists(folder_dir):
                print('{} not exist, call \'RAP_dataset_preprocess\' First'.format(folder_dir))
                return
            else:
                print('-----> Use RAP dataset train OP')
        else:
            if not os.path.exists(folder_dir):
                print('folder dir \'{}\' not exist'.format(folder_dir))
                return
            elif not os.path.isfile(reid_list_dir):
                print('list file \'{}\' not exist'.format(reid_list_dir))
                return
            else:
                print('-----> Use reid dataset train OP\ndataset:{}\nlist:{}'.format(folder_dir,
                                                                                     reid_list_dir))

        fname_list = os.listdir(folder_dir)
        print('{} include files num: {}'.format(folder_dir, len(fname_list)))

        model_save_name = model_save_name + '_' + size_tag

        if not os.path.exists('./OP'):
            os.makedirs('./OP')
            print('create \'./OP\'')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            print('create \'{}\''.format(model_save_dir))

        # clean log file first
        ckpt_list = os.listdir(model_save_dir)
        for i in ckpt_list:
            os.remove(os.path.join(model_save_dir, i))

        loss_f = open(loss_lr_dir, 'w')
        loss_f.close()

        acc_f = open(acc_dir, 'w')
        acc_f.close()

        net_setting_f = open(model_setting_dir, 'w')
        net_setting_f.close()

        if t.cuda.is_available():
            use_GPU = True
            print('Use GPU')
        else:
            use_GPU = False
            print('No GPU, Use CPU')
        available_device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

        image_size = [224, 224] if self.isDefaultImageSize else [256, 128]
        augmentation_preprocess = {'image_size': image_size,
                                   'pixel_mean': [0.485, 0.456, 0.406],
                                   'pixel_stddev': [0.229, 0.224, 0.225],
                                   'flip_probability': 0.5,
                                   'padding_size': 10,
                                   'zoom_out_pad_prob': zoom_out_pad_prob,
                                   'random_eras_probability': 0.5,
                                   's_ratio_min': 0.02,
                                   's_ratio_max': 0.4,
                                   'aspect_ratio_min': 0.3
                                   }
        model_setting = {'if_dropout': if_dropout,
                         'dropout_rate': dropout_rate,
                         'ImageNet_Init': True,
                         'ImageNet_model_path': './resnet50-19c8e357.pth',
                         'isDefaultImageSize': self.isDefaultImageSize,
                         }
        optimizer_setting = {'lr_start': lr_start,
                             'weight_decay': 0.0005,
                             'bias_lr_factor': 1,
                             'weight_decay_bias': 0.0005,
                             'optimizer_name': optimizer_name,  # 'SGD' 'Adam'
                             'SGD_momentum': 0.9
                             }
        scheduler_setting = {'lr_start': lr_start,
                             'decay_rate': 10,
                             'warmup_rate': 100
                             }

        # write history
        history_f = open(history_dir, 'a')
        history_f.write('\n\n')
        history_f.write('Time: {0}\n'.format(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))))
        history_f.write('Image Size:\t{}\n'.format(size_tag))
        history_f.write('Model Save Step in epoch:\t{0}\n'.format(save_step))
        history_f.write('Evaluate Model Step in epoch:\t{0}\n'.format(eval_step))
        history_f.write('Use reid dataset:\t{}\n'.format(if_use_reid_dataset))
        history_f.write('reid dataset:\t{}\tlist file:{}\n'.format(
                folder_dir, reid_list_dir) if if_use_reid_dataset else '')
        history_f.write('->->->->->-> Net_Setting:\n')
        history_f.write('is Default Image Size?\t{}\n'.format(model_setting['isDefaultImageSize']))
        history_f.write('if dropout?\t{}\tdropout rate:\t{}\n'.format(model_setting['if_dropout'],
                                                                      model_setting['dropout_rate']))
        history_f.write('ImageNet Initialization\t{}\n'.format(model_setting['ImageNet_Init']))
        history_f.write('->->->->->-> Augmentation Preprocess:\n')
        history_f.write('Random rotation degree:\t{}\n'.format(10))
        history_f.write('Image Size:\t{}\n'.format(augmentation_preprocess['image_size']))
        history_f.write('Pixel Mean:\t{}\tPixel Stddev:\t{}\n'.format(augmentation_preprocess['pixel_mean'],
                                                                      augmentation_preprocess['pixel_stddev']))
        history_f.write('Random Flip Probability:\t{}\n'.format(augmentation_preprocess['flip_probability']))
        history_f.write('Random Pad Crop -- Pad:\t{}\tCrop:\t{}\n'.format(augmentation_preprocess['padding_size'],
                                                                          augmentation_preprocess['image_size']))
        history_f.write('Random Zoom Out Pad Probability:\t{}\n'.format(
                augmentation_preprocess['zoom_out_pad_prob']))
        history_f.write('Random Erase Probability\t{}\n'.format(
            augmentation_preprocess['random_eras_probability']))
        history_f.write('random erase H/W aspect ratio min:\t{}\tmax:\t{}\n'.format(
            augmentation_preprocess['aspect_ratio_min'], 1. / augmentation_preprocess['aspect_ratio_min']))
        history_f.write('random erase Se/S ratio min:\t{}\tmax:\t{}\n'.format(
            augmentation_preprocess['s_ratio_min'], augmentation_preprocess['s_ratio_max']))
        history_f.write('->->->->->-> Train Setting:\n')
        history_f.write('Lr_Start:\t{}\n'.format(lr_start))
        history_f.write('Epoch:\t{}\n'.format(Epoch))
        history_f.write('Train images proportion:\t{}\n'.format(train_proportion))
        history_f.write('Use random erase and discard rotation?\t{}\n'.format(if_random_erase))
        history_f.write('Use weight softmax?\t{}\t'.format(if_weight_softmax))
        history_f.write('Weight value:\t{}\n'.format(weight_softmax_value))
        history_f.write('->->->->->-> Optimizer Setting:\n')
        history_f.write('Optimizer:\t{}\n'.format(optimizer_setting['optimizer_name']))
        history_f.write('Weight Decay:\t{}\tWeight Decay Bias:\t{}\tBias Lr Factor:\t{}\n'.format(
            optimizer_setting['weight_decay'], optimizer_setting['weight_decay_bias'],
            optimizer_setting['bias_lr_factor']))
        history_f.write('SGD Momentum:\t{}\n'.format(optimizer_setting['SGD_momentum']))
        history_f.write('->->->->->-> Scheduler Setting:\n')
        history_f.write('Decay Rate:\t{}\tWarmup Rate:\t{}\n'.format(scheduler_setting['decay_rate'],
                                                                     scheduler_setting['warmup_rate']))
        history_f.close()

        # write net setting
        self.net_setting_save(model_setting, save_dir=model_setting_dir)

        train_dataloader, train_img_num, test_dataloader, test_img_num = self.construct_dataset_dataloader_OP(
                folder_dir, if_use_reid_dataset, reid_list_dir, augmentation_preprocess,
                train_proportion=train_proportion, if_random_erase=if_random_erase)

        print('train iteration num per epoch: {}'.format(len(train_dataloader)))

        # construct model and initialize weights
        my_model = self.construct_model_OP(model_setting)
        if if_load_SBP:
            print('----> load Block 1 from {}'.format(SBP_dir))
            my_model.load_state_dict(t.load(SBP_dir),strict=False)
        my_model.to(available_device)

        # XCP
        print(my_model)
        for idx,(name,param) in enumerate(my_model.named_parameters()):
            print('{} {}: {}'.format(idx,name,param.shape))

        # construct loss
        if if_weight_softmax:
            print('use weighted softmax')
            loss_weight = t.tensor(weight_softmax_value, dtype=t.float32)
            loss_weight = loss_weight.to(available_device)
            my_loss = nn.CrossEntropyLoss(weight=loss_weight)
        else:
            print('use original softmax')
            my_loss = nn.CrossEntropyLoss()  # may set weight to tensor(2,2,1) ---- front,back,side

        # construct optimizer
        my_optimizer, NoneOP_idx_set = self.generate_optimizer_OP(optimizer_setting, my_model)

        t0 = time.clock()
        for ep in range(Epoch):
            print('--------->  Start epoch {}'.format(ep))
            time_start = time.clock()
            my_model.train()
            avg_loss = []
            avg_train_acc = []

            current_basic_lr = self.lr_scheduler_OP(my_optimizer, ep, scheduler_setting,
                                                    NoneOP_idx_set, if_load_SBP)

            for iter, data in enumerate(train_dataloader):
                batch_image, batch_label = data
                batch_image = batch_image.to(available_device)
                batch_label = batch_label.to(available_device)


                # orient_dict={0:'Front',1:'Back',2:'Side'}
                # fig = plt.figure(figsize=(16,4))
                # for idx, (img, lb) in enumerate(zip(batch_image, batch_label)):
                #     if idx == 16:
                #         break
                #     subfig = fig.add_subplot(2, 8, idx + 1)
                #     img_recovery = self.NormalizedTensorImage2Numpy(img, augmentation_preprocess['pixel_mean']
                #                                                     , augmentation_preprocess['pixel_stddev'])
                #     subfig.imshow(img_recovery)
                #     subfig.set_title('orient: {}'.format(orient_dict[lb.item()]))
                # plt.show()


                my_optimizer.zero_grad()

                logits = my_model(batch_image)
                loss = my_loss(logits, batch_label)
                loss.backward()
                my_optimizer.step()

                total_acc = (logits.max(1)[1] == batch_label).float().mean().item()
                print('iter {}\tacc {}\tloss {}\tlr {}'.format(iter, total_acc, loss.item(), current_basic_lr))
                avg_loss.append(loss.item())
                avg_train_acc.append(total_acc)

            print('-------> Epoch {} End'.format(ep))
            avg_loss = np.mean(avg_loss, axis=0)
            avg_train_acc = np.mean(avg_train_acc, axis=0)
            print('Avg_loss: {}\tAvg_acc: {}\tLr: {}'.format(avg_loss, avg_train_acc, current_basic_lr))

            # write loss lr
            time_current = time.clock() - time_start
            time_elapsed_hour = int(time_current // 3600)
            time_elapsed_minute = int(time_current // 60 - 60 * time_elapsed_hour)
            time_elapsed_second = int(time_current - 3600 * time_elapsed_hour - 60 * time_elapsed_minute)

            loss_f = open(loss_lr_dir, 'a')
            loss_f.write('epoch {} loss {} acc {} lr {} time {}:{}:{}\n'.format(ep, avg_loss, avg_train_acc,
                                                                                current_basic_lr,
                                                                                time_elapsed_hour, time_elapsed_minute,
                                                                                time_elapsed_second))
            loss_f.close()

            if ep % save_step == 0 or ep + 1 == Epoch:
                print('-------> save model')
                t.save(my_model.state_dict(), os.path.join(model_save_dir, '{}_{}.pth'.format(model_save_name, ep)))

            if ep % eval_step == 0 or ep + 1 == Epoch:
                print('-------> start evaluate Epoch{} model'.format(ep))
                eval_time_start = time.clock()
                my_model.eval()
                with t.no_grad():
                    avg_front_acc = []
                    avg_back_acc = []
                    avg_side_acc = []
                    avg_total_acc = []
                    for idx, data in enumerate(test_dataloader):
                        batch_image, batch_label = data
                        batch_image = batch_image.to(available_device)
                        batch_label = batch_label.to(available_device)

                        logits = my_model(batch_image)

                        total_acc = (logits.max(1)[1] == batch_label).float().mean().item()

                        # single orientation acc
                        label_np = batch_label.cpu().numpy()
                        pred_np = logits.max(1)[1].cpu().numpy()  # [B,]
                        front_idx = np.where(label_np == 0)[0]
                        back_idx = np.where(label_np == 1)[0]
                        side_idx = np.where(label_np == 2)[0]

                        front_label = label_np[front_idx]
                        back_label = label_np[back_idx]
                        side_label = label_np[side_idx]

                        front_pred = pred_np[front_idx]
                        back_pred = pred_np[back_idx]
                        side_pred = pred_np[side_idx]

                        front_acc = (front_pred == front_label).astype(np.float32).mean()
                        back_acc = (back_pred == back_label).astype(np.float32).mean()
                        side_acc = (side_pred == side_label).astype(np.float32).mean()

                        avg_total_acc.append(total_acc)
                        avg_front_acc.append(front_acc)
                        avg_back_acc.append(back_acc)
                        avg_side_acc.append(side_acc)

                    avg_total_acc = np.mean(avg_total_acc, axis=0)
                    avg_front_acc = np.mean(avg_front_acc, axis=0)
                    avg_back_acc = np.mean(avg_back_acc, axis=0)
                    avg_side_acc = np.mean(avg_side_acc, axis=0)
                    print('Epoch {}'.format(ep))
                    print('Avg_acc: {}\tAvg_f_acc: {}\tAvg_b_acc: {}\tAvg_s_acc: {}'.format(
                        avg_total_acc, avg_front_acc, avg_back_acc, avg_side_acc))

                    time_current = time.clock() - eval_time_start
                    time_elapsed_hour = int(time_current // 3600)
                    time_elapsed_minute = int(time_current // 60 - 60 * time_elapsed_hour)
                    time_elapsed_second = int(time_current - 3600 * time_elapsed_hour - 60 * time_elapsed_minute)
                    print('Time elapsed: {0}:{1}:{2}'.format(time_elapsed_hour, time_elapsed_minute,
                                                             time_elapsed_second))

                    acc_f = open(acc_dir, 'a')
                    acc_f.write('epoch: {0}\t'.format(ep))
                    acc_f.write('Avg_acc: {}\tAvg_f_acc: {}\tAvg_b_acc: {}\tAvg_s_acc: {}\t'.format(
                        avg_total_acc, avg_front_acc, avg_back_acc, avg_side_acc))
                    acc_f.write('Time: {0}:{1}:{2}\n'.format(
                        time_elapsed_hour, time_elapsed_minute, time_elapsed_second))
                    acc_f.close()

        time_current = time.clock() - t0
        time_elapsed_hour = int(time_current // 3600)
        time_elapsed_minute = int(time_current // 60 - 60 * time_elapsed_hour)
        time_elapsed_second = int(time_current - 3600 * time_elapsed_hour - 60 * time_elapsed_minute)

        history_f = open(history_dir, 'a')
        history_f.write('-------------> Success Finish <--------------')
        history_f.write('Avg_acc: {}\tAvg_f_acc: {}\tAvg_b_acc: {}\tAvg_s_acc: {}\n'.format(
            avg_total_acc, avg_front_acc, avg_back_acc, avg_side_acc))
        history_f.write('Final: epoch {} train_loss {} lr {}\n'.format(ep, avg_loss, current_basic_lr))
        history_f.write('Time Elapsed Overall {}:{}:{}\n'.format(time_elapsed_hour, time_elapsed_minute,
                                                                 time_elapsed_second))
        history_f.close()

        self.plot_loss_acc_lr_OP(loss_lr_dir=loss_lr_dir, eval_dir=acc_dir, ifSave=True,
                                 SaveDir=plot_dir, if_show=if_show_plot)

    # TODO: Do Not Change Default value, Fix to SBP setting !!!!!!!
    def train_MIE(self,
                  Lr_Start=0.00035,
                  STN_lr_start=0.0000001,
                  OP_lr_start=0.00035,
                  STN_freeze_end_ep=60,
                  OP_freeze_end_ep=130,
                  Epoch=120,
                  train_batch_size=64,
                  eval_batch_size=128,
                  optim_name='Adam',
                  if_REA = True,
                  if_triplet_loss=True,
                  if_center_loss=True,
                  center_loss_weight=0.0005,
                  if_mask_loss=False,
                  binary_threshold=0.7,
                  area_constrain_proportion=0.3,
                  mask_loss_weight=0.0001,  # m_loss: 0.06
                  addOrientationPart=False,
                  addSTNPart=False,
                  addAttentionMaskPart=False,
                  addChannelReduce=False,
                  final_dim=1024,
                  if_affine=False,
                  if_OP_channel_wise=False,
                  mask_num=8,
                  STN_init_value = 1.0,
                  save_step=1,  # 4
                  eval_step=1,  # 4
                  if_rerank=False,
                  if_Euclidean=False,
                  if_load_OP_model=False,
                  OP_model_dir='./OP/model_save/OP_256x128_100.pth',
                  if_load_STN_SBP_model = False,
                  SBP_model_dir='./MIE/MIE_SBP/Market_SBP/model_save-CS-MARKET/pytorch_MIE_104.pth',
                  if_check_OP_grad=False,
                  if_show_plot=False,
                  plot_dir='./MIE/loss_acc_lr.jpg'
                  ):

        if t.cuda.is_available():
            use_GPU = True
            print('Use GPU')
        else:
            use_GPU = False
            print('No GPU, Use CPU')
        available_device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

        # clean log file first
        ckpt_list = os.listdir(self.save_dir)
        for i in ckpt_list:
            os.remove(os.path.join(self.save_dir, i))

        loss_f = open(self.loss_lr_dir, 'w')
        loss_f.close()

        acc_f = open(self.acc_dir, 'w')
        acc_f.close()

        net_setting_f = open(self.net_setting_dir, 'w')
        net_setting_f.close()

        if if_mask_loss:
            assert addAttentionMaskPart == True, 'Use Mask Loss Must Have Attention Mask Part'

        if if_OP_channel_wise and addOrientationPart:
            final_feature_dim = final_dim if addChannelReduce else 6144
        else:
            final_feature_dim = final_dim if addChannelReduce else 2048

        image_size = [224, 224] if self.isDefaultImageSize else [256, 128]
        augmentation_preprocess = {'image_size': image_size,
                                   'pixel_mean': [0.485, 0.456, 0.406],
                                   'pixel_stddev': [0.229, 0.224, 0.225],
                                   'flip_probability': 0.5,
                                   'padding_size': 10,
                                   'random_eras_probability': 0.5,
                                   's_ratio_min': 0.02,
                                   's_ratio_max': 0.4,
                                   'aspect_ratio_min': 0.3,
                                   'if_REA':if_REA
                                   }
        # Random Erase use pixel mean fill, may cause performance deteriorate !!!!!!
        dataloader_setting = {'if_triplet_loss': if_triplet_loss,
                              'K_num': 4,
                              'if_center_loss': if_center_loss,
                              'num_workers': 8
                              }

        model_setting = {'last_stride': 1,
                         'neck': True,
                         'eval_neck': True,
                         'ImageNet_Init': True,
                         'ImageNet_model_path': './resnet50-19c8e357.pth',
                         'addOrientationPart': addOrientationPart,
                         'addSTNPart': addSTNPart,
                         'addAttentionMaskPart': addAttentionMaskPart,
                         'addChannelReduce': addChannelReduce,
                         'isDefaultImageSize': self.isDefaultImageSize,
                         'final_dim': final_dim,
                         'mask_num': mask_num,
                         'dropout_rate': 0.,  # 0.6 whether have influence on triplet loss??
                         'if_affine': if_affine,
                         'if_OP_channel_wise': if_OP_channel_wise,
                         'STN_fc_b_init': STN_init_value
                         }
        # when not addChannelReduce, final_dim will not available in Model build. Except switch on addChannelReduce
        optimizer_setting = {'lr_start': Lr_Start,
                             'weight_decay': 0.0005,
                             'bias_lr_factor': 1, # This can not work for Lr_Scheduler existing !!!!!!!
                             'weight_decay_bias': 0.0005, # 0.
                             'optimizer_name': optim_name,  # 'SGD'
                             'SGD_momentum': 0.9,
                             'center_lr': 0.5,
                             'STN_lr': STN_lr_start,
                             'OP_lr': OP_lr_start
                             }
        scheduler_setting = {'lr_start': Lr_Start,
                             'STN_lr': STN_lr_start,
                             'OP_lr': OP_lr_start,
                             'STN_freeze_end_ep': STN_freeze_end_ep,
                             'OP_freeze_end_ep': OP_freeze_end_ep,
                             'addOrientationPart': addOrientationPart,
                             'decay_rate': 10,
                             'warmup_rate': 100
                             }
        loss_setting = {'eval_feature_dim': final_feature_dim,
                        'if_triplet_loss': if_triplet_loss,
                        'triplet_margin': 0.3,  # 0.1-0.3 Zheng suggest, LuoHao 0.3
                        'label_smooth_rate': 0.1,  # 0. original softmax
                        'if_mask_loss': if_mask_loss,
                        'binary_threshold': binary_threshold,
                        'area_constrain_proportion': area_constrain_proportion
                        }

        # write history
        history_f = open(self.history_log_dir, 'a')
        history_f.write('\n\n')
        history_f.write('Time: {0}\n'.format(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))))
        history_f.write('Target Dataset: {0}\n'.format(self.target_dataset_name))
        history_f.write('->->->->->-> Net_Setting:\n')
        history_f.write('Last Stride\t{}\n'.format(model_setting['last_stride']))
        history_f.write('Use BatchNormNeck?\t{}\tEvaluate BatchNormNeck?\t{}\n'.format(model_setting['neck'],
                                                                                       model_setting['eval_neck']))
        history_f.write('ImageNet Initialization\t{}\n'.format(model_setting['ImageNet_Init']))
        history_f.write('add Orientation Part?\t{}\t'.format(model_setting['addOrientationPart']))
        history_f.write('Use Channel Wise Concatenate?\t{}\n'.format(model_setting['if_OP_channel_wise']))
        history_f.write('add STN Part?\t{}\tfc_b init:\t{}\t'.format(model_setting['addSTNPart'],
                                                           model_setting['STN_fc_b_init']))
        history_f.write('Use affine STN?\t{}\n'.format(model_setting['if_affine']))
        history_f.write('Use Attention Mask Part?\t{}\n'.format(model_setting['addAttentionMaskPart']))
        history_f.write('add Channel Reduce?\t{}\tFinal Channel:\t{}\n'.format(
                model_setting['addChannelReduce'], model_setting['final_dim']))
        history_f.write('is Default Image Size?\t{}\n'.format(model_setting['isDefaultImageSize']))
        history_f.write('mask_num:\t{}\tdropout_rate:\t{}\n'.format(
                model_setting['mask_num'], model_setting['dropout_rate']))
        history_f.write('load OP model?\t{}'.format(if_load_OP_model))
        history_f.write('\tmodel name\t{}\n'.format(OP_model_dir) if if_load_OP_model else '\n')
        history_f.write('load SBP model layer4 for STN?\t{}'.format(if_load_STN_SBP_model))
        history_f.write('\tmodel name\t{}\n'.format(SBP_model_dir) if if_load_STN_SBP_model else '\n')
        history_f.write('->->->->->-> Augmentation Preprocess:\n')
        history_f.write('Image Size:\t{}\n'.format(augmentation_preprocess['image_size']))
        history_f.write('Pixel Mean:\t{}\tPixel Stddev:\t{}\n'.format(augmentation_preprocess['pixel_mean'],
                                                                      augmentation_preprocess['pixel_stddev']))
        history_f.write('Random Flip Probability:\t{}\n'.format(augmentation_preprocess['flip_probability']))
        history_f.write('Random Pad Crop -- Pad:\t{}\tCrop:\t{}\n'.format(augmentation_preprocess['padding_size'],
                                                                          augmentation_preprocess['image_size']))
        history_f.write('Use Random Erase Augmentation?\t{}\n'.format(
                augmentation_preprocess['if_REA']))
        history_f.write('Random Erase Probability\t{}\n'.format(
            augmentation_preprocess['random_eras_probability']))
        history_f.write('random erase H/W aspect ratio min:\t{}\tmax:\t{}\n'.format(
            augmentation_preprocess['aspect_ratio_min'], 1. / augmentation_preprocess['aspect_ratio_min']))
        history_f.write('random erase Se/S ratio min:\t{}\tmax:\t{}\n'.format(
            augmentation_preprocess['s_ratio_min'], augmentation_preprocess['s_ratio_max']))
        history_f.write('->->->->->-> Train Setting:\n')
        history_f.write('Triplet Loss?\t{}\tMargin:\t{}K_num:\t{}\n'.format(if_triplet_loss,
                                                                            loss_setting['triplet_margin'],
                                                                            dataloader_setting['K_num']))
        history_f.write('Center Loss?\t{}\tWeight:\t{}\n'.format(if_center_loss,
                                                                 center_loss_weight))
        history_f.write('Mask Loss?\t{}\tBinary Threshold:\t{}\tArea Constrain Proportion:\t{}\t'.format(
                if_mask_loss,binary_threshold,area_constrain_proportion))
        history_f.write('Mask Loss Weight:\t{}\n'.format(mask_loss_weight))
        history_f.write('DataLoader Worker Num:\t{}\n'.format(dataloader_setting['num_workers']))
        history_f.write('->->->->->-> Optimizer Setting:\n')
        history_f.write('Lr_Start:\t{}\n'.format(Lr_Start))
        history_f.write('Optimizer:\t{}\n'.format(optimizer_setting['optimizer_name']))
        history_f.write('Weight Decay:\t{}\tWeight Decay Bias:\t{}\tBias Lr Factor:\t{}\n'.format(
            optimizer_setting['weight_decay'], optimizer_setting['weight_decay_bias'],
            optimizer_setting['bias_lr_factor']))
        history_f.write('SGD Momentum:\t{}\n'.format(optimizer_setting['SGD_momentum']))
        history_f.write('Center Lr:\t{}\n'.format(optimizer_setting['center_lr']))
        history_f.write('STN Lr:\t{}\n'.format(optimizer_setting['STN_lr']))
        history_f.write('OP Lr:\t{}\n'.format(optimizer_setting['OP_lr']))
        history_f.write('->->->->->-> Scheduler Setting:\n')
        history_f.write('STN freeze util epoch\t{}\tOP freeze util epoch\t{}\n'.format(
                scheduler_setting['STN_freeze_end_ep'],scheduler_setting['OP_freeze_end_ep']))
        history_f.write('Decay Rate:\t{}\tWarmup Rate:\t{}\n'.format(scheduler_setting['decay_rate'],
                                                                     scheduler_setting['warmup_rate']))
        history_f.write('Softmax Label Smooth Rate: {}'.format(loss_setting['label_smooth_rate']))

        history_f.write('Epoch:\t{}\tTrain Batch Size:\t{}\n'.format(Epoch, train_batch_size))
        history_f.write('Model Save Step in epoch:\t{0}\n'.format(save_step))
        history_f.write('Evaluate Model Step in epoch:\t{0}\n'.format(eval_step))
        history_f.write('->->->->->-> Evaluate Setting:\n')
        history_f.write('Evaluate batch size:\t{0}\n'.format(eval_batch_size))
        history_f.write('Use rerank:\t{0}\n'.format(if_rerank))
        history_f.write('Use Euclidean Distance:\t{0}\n'.format(if_Euclidean))
        history_f.close()

        # write net setting
        self.net_setting_save(model_setting)

        train_dataloader, train_img_num, train_person_num = self.construct_dataset_dataloader_MIE(
            self.train_dir, augmentation_preprocess, dataloader_setting,
            if_reset_id=True, is_train=True, batch_size=train_batch_size)

        query_dataloader, query_img_num, query_person_num = self.construct_dataset_dataloader_MIE(
            self.query_dir, augmentation_preprocess, dataloader_setting,
            if_reset_id=False, is_train=False, batch_size=eval_batch_size)

        test_dataloader, test_img_num, test_person_num = self.construct_dataset_dataloader_MIE(
            self.test_dir, augmentation_preprocess, dataloader_setting,
            if_reset_id=False, is_train=False, batch_size=eval_batch_size)

        print('train iteration num per epoch: {}'.format(len(train_dataloader)))
        print('train iter images num: {} dataset image: {}'.format(len(train_dataloader) * train_batch_size,
                                                                   train_img_num))

        # construct model and initialize weights
        my_model = self.construct_model_MIE(train_person_num, model_setting, use_GPU)
        if if_load_OP_model:
            assert model_setting['addOrientationPart'] == True  # no OP part can not load
            OP_model_image_size = OP_model_dir.split('_')[-2].split('x')
            # TODO: change depend on real situation
            assert int(OP_model_image_size[0]) == image_size[0]  # save model accept image size not fit now
            assert int(OP_model_image_size[1]) == image_size[1]  # save model accept image size not fit now
            print('-----> load OP model {}'.format(OP_model_dir))
            my_model.load_param_transfer(OP_model_dir)
        if if_load_STN_SBP_model:
            assert model_setting['addSTNPart'] == True, 'Must have STN Part'
            print('-----> load SBP model: {} Layer 4 for STN initialize'.format(SBP_model_dir))
            my_model.load_STN_from_SBP_model_layer4(SBP_model_dir)
        my_model.to(available_device)

        # XCP
        print(my_model)
        for idx, (name, param) in enumerate(my_model.named_parameters()):
            print('{} {}: {}'.format(idx, name, param.shape))

        # construct optimizer
        if if_center_loss:
            loss_calculator, center_criterion = self.build_loss_structure(person_num=train_person_num,
                                                                          if_center_loss=True,
                                                                          loss_setting=loss_setting,
                                                                          use_GPU=use_GPU)
            # loss_cal param: fc_logits, GAP_feature, label, if_triplet_loss, if_center_loss, center_loss_weight
            my_optimizer, center_optimizer, STN_param_idx_set, OP_param_idx_set = self.generate_center_optimizer_MIE(
                    optimizer_setting, my_model, center_criterion)
        else:
            my_optimizer, STN_param_idx_set, OP_param_idx_set = self.generate_optimizer_MIE(optimizer_setting, my_model)
            loss_calculator = self.build_loss_structure(person_num=train_person_num,
                                                        if_center_loss=False,
                                                        loss_setting=loss_setting,
                                                        use_GPU=use_GPU)
            # loss_cal param: fc_logits, GAP_feature, label, if_triplet_loss, if_center_loss, center_loss_weight

        # fix OP idx BUG, when not addOrientationPart
        if not addOrientationPart:
            OP_param_idx_set = set()

        # train and eval
        t0 = time.clock()
        for ep in range(Epoch):
            print('--------->  Start epoch {}'.format(ep))
            my_model.train()
            time_start = time.clock()
            avg_loss = []
            avg_acc = []
            current_basic_lr = self.lr_scheduler_MIE(my_optimizer, ep, scheduler_setting,
                                                     STN_param_idx_set, OP_param_idx_set)
            for iter, data in enumerate(train_dataloader):
                batch_image, batch_label = data
                batch_image = batch_image.to(available_device)
                batch_label = batch_label.to(available_device)

                my_optimizer.zero_grad()
                if if_center_loss:
                    center_optimizer.zero_grad()

                fc_logits, GAP_feature = my_model(batch_image)
                loss = loss_calculator(fc_logits, GAP_feature, batch_label,
                                       if_triplet_loss, if_center_loss, center_loss_weight,
                                       use_GPU, if_mask_loss, my_model.masks, mask_loss_weight)
                # need param: fc_logits, GAP_feature, label, if_triplet_loss, if_center_loss, center_loss_weight
                #             if_mask_loss = False, masks_list = None, mask_loss_weight = 0.001
                # loss = [tensor_loss_all, item_loss_all, item_loss_s, item_loss_t, item_loss_c, item_loss_m]
                loss[0].backward()

                # for OP grad check
                if iter % 50 == 0 and if_check_OP_grad and model_setting['addOrientationPart'] == True:
                    print('------> OP grad check')
                    print('my_model.orientation_predictor.conv1.grad: {}'.format(
                        my_model.orientation_predictor.conv1.weight.grad[0,0,:,:]))  # 3.4513e-09 --> 1.4550e-04
                    print('my_model.orientation_predictor.conv5.grad: {}'.format(
                        my_model.orientation_predictor.conv5.weight.grad[0,:3,:,:]))  # -2.5189e-08 --> -1.2460e-03
                    print('----> Layer4_Front[-1].conv1.weight.grad: {}'.format(
                        my_model.layer4_front[-1].conv1.weight.grad[0,:3,:,:]))  #
                    print('----> Layer4_Back[-1].conv1.weight.grad: {}'.format(
                        my_model.layer4_back[-1].conv1.weight.grad[0,:3,:,:]))  #
                    print('----> Layer4_Side[-1].conv1.weight.grad: {}'.format(
                        my_model.layer4_side[-1].conv1.weight.grad[0,:3,:,:]))  #
                    print('my_model.layer2[0].conv1.weight.grad: {}'.format(
                        my_model.layer2[0].conv1.weight.grad[0,:3,:,:]))  # 0.0342 --> same 0.0114
                    print('my_model.layer1[0].conv1.weight.grad: {}'.format(
                        my_model.layer1[0].conv1.weight.grad[0, :3, :, :]))  #
                # TODO: del later, implicate Need independent Lr for OP, like STN does

                my_optimizer.step()
                if if_center_loss:
                    # self.centers_update()
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / center_loss_weight)  # recovery grad weighted by CenterLossWeight
                    center_optimizer.step()

                acc = (fc_logits.max(1)[1] == batch_label).float().mean().item()
                print('iter {}\tacc {}\tloss {}\ts_loss {}\tt_loss {}\tc_loss {}\tm_loss {}\tlr {}'.format(
                    iter, acc, loss[1], loss[2], loss[3], loss[4], loss[5], current_basic_lr))
                avg_loss.append(loss[1:])
                avg_acc.append(acc)

            print('-------> Epoch {} End'.format(ep))
            avg_loss = np.mean(avg_loss, axis=0)
            avg_acc = np.mean(avg_acc, axis=0)
            print('Avg_acc: {}\tAvg_loss: {}\tAvg_s_loss: {}\tAvg_t_loss: {}\tAvg_c_loss: {}\t'
                  'Avg_m_loss: {}\tLr: {}'.format(
                avg_acc, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], avg_loss[4], current_basic_lr))

            # write loss lr
            time_current = time.clock() - time_start
            time_elapsed_hour = int(time_current // 3600)
            time_elapsed_minute = int(time_current // 60 - 60 * time_elapsed_hour)
            time_elapsed_second = int(time_current - 3600 * time_elapsed_hour - 60 * time_elapsed_minute)

            loss_f = open(self.loss_lr_dir, 'a')
            loss_f.write('epoch {} train_loss {} softmax_loss {} triplet_loss {} center_loss {} '
                         'mask_loss {} lr {} time {}:{}:{}\n'.format(ep, avg_loss[0], avg_loss[1],
                                                                     avg_loss[2], avg_loss[3], avg_loss[4],
                                                                     current_basic_lr,
                                                                     time_elapsed_hour, time_elapsed_minute,
                                                                     time_elapsed_second))
            loss_f.close()

            if ep % save_step == 0 or ep + 1 == Epoch:
                print('-------> save model')
                t.save(my_model.state_dict(), os.path.join(self.save_dir, '{}_{}.pth'.format(self.save_name, ep)))

            if ep % eval_step == 0 or ep + 1 == Epoch:
                print('-------> start evaluate Epoch{} model'.format(ep))
                # if use_GPU:
                #     t.cuda.empty_cache()
                eval_time_start = time.clock()
                my_model.eval()
                query_feature = []
                query_label = []
                test_feature = []
                test_label = []
                with t.no_grad():
                    print('extract query set feature')
                    for data in query_dataloader:
                        image, pid, cam = data
                        image = image.to(available_device)
                        feature = my_model(image)
                        query_feature.append(feature)

                        pid_array = np.array(pid)[:, np.newaxis]
                        cam_array = np.array(cam)[:, np.newaxis]
                        tem_label = np.concatenate((pid_array, cam_array), axis=1).tolist()  # [B,2]
                        query_label.extend(tem_label)

                    query_feature = t.cat(query_feature, dim=0)
                    print('query_feature: {}\tquery_label: {}'.format(query_feature.shape, np.shape(query_label)))

                    print('extract test set feature')
                    for data in test_dataloader:
                        image, pid, cam = data
                        image = image.to(available_device)
                        feature = my_model(image)
                        test_feature.append(feature)

                        pid_array = np.array(pid)[:, np.newaxis]
                        cam_array = np.array(cam)[:, np.newaxis]
                        tem_label = np.concatenate((pid_array, cam_array), axis=1).tolist()  # [B,2]
                        test_label.extend(tem_label)

                    test_feature = t.cat(test_feature, dim=0)
                    print('test_feature: {}\ttest_label: {}'.format(test_feature.shape, np.shape(test_label)))

                    rank, mAP = self.calculate_topn_mAP(query_feature, query_label, test_feature, test_label,
                                                        if_rerank=if_rerank, if_Euclidean=if_Euclidean,
                                                        if_sqrt=False)

                    print('Epoch {} Dataset {}'.format(ep, self.target_dataset_name))
                    print('1: %f\t5: %f\t10: %f\t20: %f\tmAP: %f' % (rank[0], rank[1], rank[2], rank[3], mAP))

                    time_current = time.clock() - eval_time_start
                    time_elapsed_hour = int(time_current // 3600)
                    time_elapsed_minute = int(time_current // 60 - 60 * time_elapsed_hour)
                    time_elapsed_second = int(time_current - 3600 * time_elapsed_hour - 60 * time_elapsed_minute)
                    print(
                        'Time elapsed: {0}:{1}:{2}'.format(time_elapsed_hour, time_elapsed_minute,
                                                           time_elapsed_second))

                    acc_f = open(self.acc_dir, 'a')
                    acc_f.write('epoch: {0}\t'.format(ep))
                    acc_f.write('1: {0}\t5: {1}\t10: {2}\t20: {3}\tmAP: {4}\t'.format(
                        rank[0], rank[1], rank[2], rank[3], mAP))
                    acc_f.write('Time: {0}:{1}:{2}\n'.format(
                        time_elapsed_hour, time_elapsed_minute, time_elapsed_second))
                    acc_f.close()

        time_current = time.clock() - t0
        time_elapsed_hour = int(time_current // 3600)
        time_elapsed_minute = int(time_current // 60 - 60 * time_elapsed_hour)
        time_elapsed_second = int(time_current - 3600 * time_elapsed_hour - 60 * time_elapsed_minute)

        history_f = open(self.history_log_dir, 'a')
        history_f.write('-------------> Success Finish <--------------')
        history_f.write('Acc: 1: %f\t5: %f\t10: %f\t20: %f\tmAP: %f\n' % (rank[0], rank[1], rank[2], rank[3], mAP))
        history_f.write('Final: epoch {} train_loss {} softmax_loss {} triplet_loss {} center_loss {} '
                        'mask_loss {} lr {}\n'.format(ep, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3],
                                                      avg_loss[4], current_basic_lr))
        history_f.write('Time Elapsed Overall {}:{}:{}\n'.format(time_elapsed_hour, time_elapsed_minute,
                                                                 time_elapsed_second))
        history_f.close()

        self.plot_loss_acc_lr_MIE(SaveDir=plot_dir, if_show=if_show_plot)


if __name__ == '__main__':
    # Inference Chek
    # myModel = MIE()
    # myModel.inference_check_MIE(if_OP_check=False,
    #                             if_Mask_check=True,
    #                             Mask_threshold=0.8,
    #                             mask_target=0,
    #                             if_STN_check=False,
    #                             batch_size=4,
    #                             target_model_path='./MIE/model_save-CS-OP(Ly2)-STN(3-ImageNet1.0-SBP)-AF-AM8-CR2048-SGD-REA-TR-CT-MK-0.8-0.2-0.001-OPEXlr(decay)0.001-OPEXfz0-STNlr(wb_decay)1e-05-STNfz90-4080-CH-OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pthMARKET/pytorch_MIE_119.pth',
    #                             model_setting_path='./MIE/structure-CS-OP(Ly2)-STN(3-ImageNet1.0-SBP)-AF-AM8-CR2048-SGD-REA-TR-CT-MK-0.8-0.2-0.001-OPEXlr(decay)0.001-OPEXfz0-STNlr(wb_decay)1e-05-STNfz90-4080-CH-OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pthMARKET.txt',
    #                             dataset_folder_dir='../Dataset_256x128/Market/bounding_box_train')
    # myModel.inference_check_MIE(if_OP_check=True,
    #                             if_Mask_check=False,
    #                             Mask_threshold=0.6,
    #                             if_STN_check=False,
    #                             batch_size=4,
    #                             target_model_path='./MIE/model_save-CS-OP-SGD-REA-TR-CT-OPEXlr(decay)0.001-OPEXfz0-4080-EM-OP-lr_0.0001-RE-DP0.6-ZP0.0-Market_train_OP_label.list_256x128_59.pth/pytorch_MIE_119.pth',
    #                             model_setting_path='./MIE/structure-CS-OP-SGD-REA-TR-CT-OPEXlr(decay)0.001-OPEXfz0-4080-EM-OP-lr_0.0001-RE-DP0.6-ZP0.0-Market_train_OP_label.list_256x128_59.pth.txt',
    #                             dataset_folder_dir='../Dataset_256x128/Market/query')
    # myModel.inference_check_OP(target_model_path='./OP/model_save-lr_0.0001-RE-DP0.6-ZP0.0-Market_train_OP_label.list/OP-lr_0.0001-RE-DP0.6-ZP0.0-Market_train_OP_label.list_256x128_59.pth',
    #                            model_setting_path='./OP/model_setting-lr_0.0001-RE-DP0.6-ZP0.0-Market_train_OP_label.list.txt',
    #                            if_use_reid_dataset=False, reid_list_dir='./Market_train_OP_label.list',
    #                            dataset_folder_dir='../Dataset_256x128/Market/query',
    #                            batch_size=8,only_side=False)


    # tSNE plot
    target_dataset_name = 'MARKET'
    myModel = MIE(target_dataset_name=target_dataset_name)
    myModel.tSNE_MIE(P_num=20, K_num=20,
                     # target_model_path='./tSNE_use/model_save-CS-OP(Ly2)-STN(3-ImageNet1.0-SBP)-AF-AM8-CR2048-SGD-REA-TR-CT-OPEXlr(decay)0.001-OPEXfz0-STNlr(wb_decay)1e-05-STNfz90-4080-CH-OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pthMARKET/pytorch_MIE_112.pth',
                     target_model_path='./MIE/model_save-CS-AM2-SGD-REA-TR-CT-4080MARKET/pytorch_MIE_104.pth',
                     model_setting_path='./MIE/structure-CS-AM2-SGD-REA-TR-CT-4080MARKET.txt',
                     if_show=True, if_save=True, save_name='./tSNE/tSNE_{}.eps'.format(target_dataset_name))

    # Market
    # FULL: ./tSNE_use/model_save-CS-OP(Ly2)-STN(3-ImageNet1.0-SBP)-AF-AM8-CR2048-SGD-REA-TR-CT-OPEXlr(decay)0.001-OPEXfz0-STNlr(wb_decay)1e-05-STNfz90-4080-CH-OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pthMARKET/pytorch_MIE_112.pth ?? 119
    # AM2: model_save-CS-AM2-SGD-REA-TR-CT-4080MARKET/pytorch_MIE_104.pth
    # EM_CR1024: model_save-CS-OP(Ly2)-CR-SGD-REA-TR-CT-OPEXlr(decay)0.001-OPEXfz0-4080-EM-OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pthMARKET/pytorch_MIE_112.pth
    # FAIL: Proj_1.0: model_save-CS-STN(3-ImageNet1.0-SBP)-SGD-REA-TR-CT-STNlr(wb_decay)1e-05-STNfz60-4080MARKET/pytorch_MIE_118.pth ???  119
    # FAIL: AF_1.0: model_save-CS-STN(3-ImageNet1.0-SBP)-AF-SGD-REA-TR-CT-STNlr(wb_decay)1e-05-STNfz60-4080MARKET/pytorch_MIE_119.pth
    # Baseline: model_save-CS-SGD/pytorch_MIE_119.pth

    # Duke

    # CUHK03



    # RK test
    # target_dataset_name = 'MARKET'
    # myModel = MIE(target_dataset_name=target_dataset_name)
    # myModel.independent_evaluate_MIE(image_size=[256,128],
    #                                  if_rerank=True,
    #                                  if_Euclidean=False,
    #                                  target_model_path='./MIE/model_save-CS-SGD/pytorch_MIE_119.pth',
    #                                  model_setting_path='./MIE/structure-CS-SGD.txt')

    # Market
    # MIE_Baseline: model_save-CS-SGD/pytorch_MIE_119.pth
    # Full: model_save-CS-OP(Ly2)-STN(3-ImageNet1.0-SBP)-AF-AM8-CR2048-SGD-REA-TR-CT-OPEXlr(decay)0.001-OPEXfz0-STNlr(wb_decay)1e-05-STNfz90-4080-CH-OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pthMARKET/pytorch_MIE_100.pth ?? 119

    # Duke
    # MIE_Baseline: model_save-CS-SGD-REA-TR-CT-4080DUKE/pytorch_MIE_116.pth
    # Full: model_save-CS-OP(Ly2)-STN(3-ImageNet1.0-SBP)-AF-AM8-CR2048-SGD-REA-TR-CT-MK-0.8-0.2-0.001-OPEXlr(decay)0.001-OPEXfz0-STNlr(wb_decay)1e-05-STNfz90-4080-CH-OP-DUKE_SBP_100_256x128_52.pthDUKE/pytorch_MIE_110.pth  ?? 116

    # CUHK03
    # MIE_Baseline: model_save-CS-Adam-REA-TR-CT-4080CUHK03/pytorch_MIE_114.pth ??? 112
    # Full: model_save-CS-OP(Ly2)-STN(3-ImageNet1.0-SBP)-AF-AM8-CR2048-Adam-REA-TR-CT-MK-0.8-0.2-0.001-OPEXlr(decay)1e-05-OPEXfz0-STNlr(wb_decay)1e-07-STNfz90-4080-CH-OP_CUHK03_SBP_112_256x128_56.pthCUHK03/pytorch_MIE_119.pth





    # myModel = MIE()
    # myModel.compute_reID_mean_image_OP(ModelDir='./OP/model_save-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2/OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pth',
    #                                    NetSettingDir='./OP/model_setting-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2.txt',
    #                                    if_show=True,
    #                                    SaveDir='./mean_image.png',
    #                                    DatasetDir='../Dataset_256x128/Market/query')

    # ###############################################
    # # parse
    # # !!!! Only for OP use !!!!!
    # parser = argparse.ArgumentParser(description='Orientation Predictor(OP) Train Setting')
    # parser.add_argument('--lr_start', type=float, default=0.0001, metavar='LR',
    #  help='learning rate (default: 0.0001)')
    # parser.add_argument('--Epoch', type=int, default=60, metavar='N',
    #  help='number of epochs to train (default: 60)')
    # parser.add_argument('--save_step', type=int, default=1, metavar='N',
    #  help='number of epoch interval to save model (default: 1)')
    # parser.add_argument('--eval_step', type=int, default=1, metavar='N',
    #  help='number of epoch interval to evaluate model (default: 1)')
    # parser.add_argument('--train_proportion', type=float, default=0.8, metavar='M',
    #  help='train image proportion in RAP folder (default: 0.8)')
    # parser.add_argument('--dropout_switch', action='store_false', default=True,
    #  help='disables dropout') # action means no input var
    # parser.add_argument('--dropout_rate', type=float, default=0.6, metavar='M',
    #  help='drop out rate for OP (default: 0.6)')
    # parser.add_argument('--optimizer_name', type=str, default='Adam', metavar='Name',
    #  help='optimizer\'s name for training (default: Adam)')  # No Need \' ... \' !!!!!
    # parser.add_argument('--if_random_erase', action='store_true', default=False,
    #                     help='use random erase, no rotation')  # action means no input var
    # parser.add_argument('--if_weight_softmax', action='store_true', default=False,
    #                     help='use weighted softmax')  # action means no input var
    # parser.add_argument('--zoom_out_pad_prob', type=float, default=0.3, metavar='P',
    #                     help='Random Zoom Out Pad Probability (default: 0.3)')
    # parser.add_argument('--weight_softmax_value', type=float, default=2.0, metavar='W',
    #                     help='softmax weight for Front and Back (default: 2.0)')
    # parser.add_argument('--if_load_SBP', action='store_true', default=False,
    #                     help='load && freeze SBP Block 1')  # action means no input var
    # parser.add_argument('--if_use_reid_dataset', action='store_true', default=False,
    #                         help='use reid dataset OP label')  # action means no input var
    # parser.add_argument('--reid_list_dir', type=str, default='./Market_train_OP_label.list', metavar='Dir',
    #                     help='list file dir')  # No Need \' ... \' !!!!!
    # parser.add_argument('--folder_dir', type=str, default='../RAP_Dataset', metavar='Dir',
    #                     help='dataset folder dir')  # No Need \' ... \' !!!!!
    # parser.add_argument('--SBP_dir', type=str, default='./SBP/Cosine_Success/model_save/pytorch_SBP_119.pth', metavar='Dir',
    #                     help='SBP pretrain dir')  # No Need \' ... \' !!!!!
    #
    #
    # args = parser.parse_args()
    # # ###############################################
    #
    #
    #
    # myModel = MIE()
    #
    # # myModel.compute_dataset_train_pixel_mean_stddev('../ViewInvariantNet/CUHK_part')
    # # myModel.compute_dataset_train_pixel_mean_stddev()
    #
    #
    # if_RE_tag = '-RE' if args.if_random_erase else ''
    # if_WT_tag = '-WT{}'.format(args.weight_softmax_value) if args.if_weight_softmax else ''
    # DP_rate_tag = '-DP{}'.format(args.dropout_rate) if args.dropout_switch else ''
    # ZP_Prob_tag = '-ZP{}'.format(args.zoom_out_pad_prob)
    # SBP_load_tag = '-SBP({})'.format(args.SBP_dir.replace('./','-').replace('/','-')) if args.if_load_SBP else ''
    # reid_tag = '-{}'.format(args.reid_list_dir.split('/')[-1]) if args.if_use_reid_dataset else ''
    # train_tag = if_RE_tag + if_WT_tag + DP_rate_tag + ZP_Prob_tag + SBP_load_tag + reid_tag + 'Ly2'
    # # TODO: use args, NEED cancel plot !!!!!!
    # # TODO: Need Add save dir control or will clean dir each train !!!!!!
    # # TODO: ALL independent Dir must start from './OP/' !!!!!!!
    # # TODO: use default 'history.txt'
    # myModel.train_OP(lr_start=args.lr_start,Epoch=args.Epoch,save_step=args.save_step,
    #                  eval_step=args.eval_step,train_proportion=args.train_proportion,
    #                  if_dropout=args.dropout_switch,dropout_rate=args.dropout_rate,
    #                  optimizer_name=args.optimizer_name, if_random_erase=args.if_random_erase,
    #                  zoom_out_pad_prob=args.zoom_out_pad_prob, if_weight_softmax=args.if_weight_softmax,
    #                  weight_softmax_value=[args.weight_softmax_value,args.weight_softmax_value,1.],
    #                  if_load_SBP=args.if_load_SBP, SBP_dir=args.SBP_dir,
    #                  if_use_reid_dataset=args.if_use_reid_dataset, reid_list_dir=args.reid_list_dir,
    #                  folder_dir=args.folder_dir,
    #                  model_save_dir='./OP/model_save-lr_{}{}'.format(args.lr_start,
    #                                                                  train_tag),
    #                  model_save_name='OP-lr_{}{}'.format(args.lr_start,train_tag),
    #                  loss_lr_dir='./OP/loss_lr-lr_{}{}.txt'.format(args.lr_start,train_tag),
    #                  acc_dir='./OP/acc-lr_{}{}.txt'.format(args.lr_start,train_tag),
    #                  model_setting_dir='./OP/model_setting-lr_{}{}.txt'.format(args.lr_start,
    #                                                                            train_tag),
    #                  plot_dir='./OP/loss_acc_lr-lr_{}{}.png'.format(args.lr_start,train_tag),
    #                  if_show_plot=False)


    # myModel.train_OP(folder_dir='../RAP_Dataset')
    #
    # myModel.independent_evaluate_OP('../RAP_Dataset_256x128',
    #                                 target_model_path='./OP/model_save-lr_6e-05_Adam/OP-lr_6e-05_Adam_256x128_50.pth',
    #                                 model_setting_dir='./OP/model_setting-lr_6e-05_Adam.txt',
    #                                 independent_acc_dir='./OP/independent_acc.txt',
    #                                 train_proportion=0.)

    # myModel.plot_loss_acc_lr_OP(loss_lr_dir='./OP/loss_lr.txt',eval_dir='./OP/acc.txt',ifSave=False)
    # myModel.plot_acc_OP(eval_dir='./OP/acc.txt',ifSave=False)



    # ########################################## OP ##############################################
    # ##############################################################################################
    # ##############################################################################################
    # ######################################### MIE ################################################


    # manual MIE
    # myModel = MIE(target_dataset_name='MARKET')
    # myModel.train_MIE(train_batch_size=32,addOrientationPart=True,addSTNPart=True,addAttentionMaskPart=True,
    #                   addChannelReduce=False,if_rerank=False,if_Euclidean=False,if_mask_loss=False,
    #                   if_check_OP_grad=True,if_show_plot=True)
    # train_size = 64 will out of memory !!!!!!

    # # manual MIE_SBP
    # myModel = MIE()
    # myModel.train_MIE(if_show_plot=True,if_rerank=True,if_Euclidean=False)

    # # ###############################################
    # # for MIE use
    # # Need change Dir for each run (save dir, model dir, plot dir, not show )
    # # TODO: Must keep same with method parameter list
    # parser = argparse.ArgumentParser(description='MIE Train Setting')
    # parser.add_argument('--Lr_Start', type=float, default=0.00035, metavar='LR',
    #                     help='learning rate (default: 0.00035)')
    # parser.add_argument('--STN_lr_start', type=float, default=0.0000001, metavar='LR',
    #                     help='STN learning rate (default: 0.0000001)')
    # parser.add_argument('--OP_lr_start', type=float, default=0.00035, metavar='LR',
    #                     help='OP learning rate (default: 0.00035)')
    # parser.add_argument('--STN_freeze_end_ep', type=int, default=60, metavar='N',
    #                     help='No. of epoch to stop STN freeze (default: 60)')
    # parser.add_argument('--OP_freeze_end_ep', type=int, default=130, metavar='N',
    #                     help='No. of epoch to stop OP freeze (default: 130)')
    # parser.add_argument('--Epoch', type=int, default=120, metavar='N',
    #                     help='number of epochs to train (default: 120)')
    # parser.add_argument('--optim_name', type=str, default='Adam', metavar='Name',
    #                     help='optimizer name (default: Adam)')
    # parser.add_argument('--REA_switch', action='store_false', default=True,
    #                     help='disables Random Erase Augmentation')  # action means no input var
    # parser.add_argument('--triplet_loss_switch', action='store_false', default=True,
    #                     help='disables triplet loss')  # action means no input var
    # parser.add_argument('--center_loss_switch', action='store_false', default=True,
    #                     help='disables center loss')  # action means no input var
    # parser.add_argument('--mask_loss_switch', action='store_true', default=False,
    #                     help='enables mask loss')  # action means no input var
    # parser.add_argument('--binary_threshold', type=float, default=0.7, metavar='Th',
    #                     help='mask binarylization threshold (default: 0.7)')
    # parser.add_argument('--area_constrain_proportion', type=float, default=0.3, metavar='Th',
    #                     help='mask area constrain proportion threshold (default: 0.3)')
    # parser.add_argument('--mask_loss_weight', type=float, default=0.0001, metavar='We',
    #                     help='mask loss weight (default: 0.0001)')
    # parser.add_argument('--addOrientationPart', action='store_true', default=False,
    #                     help='add Orientation Part')  # action means no input var
    # parser.add_argument('--addSTNPart', action='store_true', default=False,
    #                     help='add STN Part')  # action means no input var
    # parser.add_argument('--addAttentionMaskPart', action='store_true', default=False,
    #                     help='add Attention Mask Part')  # action means no input var
    # parser.add_argument('--addChannelReduce', action='store_true', default=False,
    #                     help='add Channel Reduce Part')  # action means no input var
    # parser.add_argument('--final_dim', type=int, default=1024, metavar='N',
    #                     help='final output dim for Channel Reduce Part (default: 1024)')
    # parser.add_argument('--if_affine', action='store_true', default=False,
    #                     help='use 6 affine parameters in STN')  # action means no input var
    # parser.add_argument('--if_OP_channel_wise', action='store_true', default=False,
    #                     help='use channel wise concatenate for 3 Layer4')  # action means no input var
    # parser.add_argument('--mask_num', type=int, default=8, metavar='N',
    #                     help='number of attention masks (default: 8)')
    # parser.add_argument('--STN_init_value', type=float, default=1.0, metavar='F',
    #                     help='init value for STN_Loc_fc_b (default: 1.0)')
    # parser.add_argument('--save_step', type=int, default=1, metavar='N',
    #                     help='number of epoch interval to save model (default: 1)')
    # parser.add_argument('--eval_step', type=int, default=1, metavar='N',
    #                     help='number of epoch interval to evaluate model (default: 1)')
    # parser.add_argument('--if_rerank', action='store_true', default=False,
    #                     help='enable rerank')  # action means no input var
    # parser.add_argument('--if_Euclidean', action='store_true', default=False,
    #                     help='use Euclidean Distance to evaluate')  # action means no input var
    # parser.add_argument('--if_load_OP_model', action='store_true', default=False,
    #                     help='load OP model to initialize')  # action means no input var
    # parser.add_argument('--OP_model_dir', type=str, default='./OP/SUGGEST/OP-lr_6e-05_Adam_256x128_50.pth',
    #                     metavar='Dir', help='target OP pre_trained model (default: No REA Version)')
    # parser.add_argument('--if_load_STN_SBP_model', action='store_true', default=False,
    #                     help='load SBP model to initialize STN')  # action means no input var
    # parser.add_argument('--SBP_model_dir', type=str, default='./MIE/MIE_SBP/Market_SBP/model_save-CS-MARKET/pytorch_MIE_104.pth',
    #                     metavar='Dir', help='target SBP pre_trained model')
    # parser.add_argument('--target_dataset_name', type=str, default='MARKET', metavar='Name',
    #                     help='target dataset name (default: MARKET)')
    #
    # args = parser.parse_args()
    #
    # # ###############################################
    #
    # # TODO: Need Add save dir control or will clean dir each train !!!!!!
    # # TODO: ALL independent Dir must start from './MIE/' !!!!!!!
    # # TODO: Use different tag for different shell 'testing aim' --- only show variable in name !!!!!!
    # net_OP_tag = '-OP(Ly2)' if args.addOrientationPart else ''
    # net_STN_tag = '-STN(3-ImageNet{}{})'.format(args.STN_init_value, '-SBP' if args.if_load_STN_SBP_model else ''
    #                                             ) if args.addSTNPart else ''
    # net_AF_tag = '-AF' if args.if_affine else ''
    # net_AM_tag = '-AM{}'.format(args.mask_num) if args.addAttentionMaskPart else ''
    # net_CR_tag = '-CR{}'.format(args.final_dim) if args.addChannelReduce else ''
    # net_OM_tag = '-{}'.format(args.optim_name)
    # net_REA_tag = '-REA' if args.REA_switch else ''
    # net_setting_tag = net_OP_tag + net_STN_tag + net_AF_tag + net_AM_tag + net_CR_tag + net_OM_tag + net_REA_tag
    # eval_RK_tag = '-RK' if args.if_rerank else ''
    # eval_EC_tag = '-EC' if args.if_Euclidean else ''
    # eval_CS_tag = '-CS' if not args.if_rerank and not args.if_Euclidean else ''
    # eval_tag = eval_RK_tag + eval_EC_tag + eval_CS_tag
    # loss_TR_tag = '-TR' if args.triplet_loss_switch else ''
    # loss_CT_tag = '-CT' if args.center_loss_switch else ''
    # loss_MK_tag = '-MK-{}-{}-{}'.format(args.binary_threshold, args.area_constrain_proportion,
    #                                     args.mask_loss_weight) if args.mask_loss_switch else ''
    # loss_tag = loss_TR_tag + loss_CT_tag + loss_MK_tag
    # OP_Mix_tag = '{}'.format('CH' if args.if_OP_channel_wise else 'EM')
    # OP_tag = '-{}-{}'.format(OP_Mix_tag, args.OP_model_dir.split('/')[-1]) if args.if_load_OP_model else ''
    # scheduler_OP_tag = '-OPEXlr(decay){}'.format(args.OP_lr_start) if args.addOrientationPart else ''
    # scheduler_OP_freeze_tag = '-OPEXfz{}'.format(args.OP_freeze_end_ep) if args.addOrientationPart else ''
    # scheduler_STN_tag = '-STNlr(wb_decay){}'.format(args.STN_lr_start) if args.addSTNPart else ''
    # scheduler_STN_freeze_tag = '-STNfz{}'.format(args.STN_freeze_end_ep) if args.addSTNPart else ''
    # scheduler_tag = scheduler_OP_tag + scheduler_OP_freeze_tag + scheduler_STN_tag + scheduler_STN_freeze_tag + '-4080'
    #
    # # TODO: Use different tag for different shell 'testing aim' --- only show variable in name !!!!!!
    # dataset_name = args.target_dataset_name.strip()
    # myModel = MIE(target_dataset_name=dataset_name, pre_dataset_loc='../Dataset',
    #               save_dir='./MIE/model_save{}{}{}{}{}{}'.format(eval_tag, net_setting_tag,loss_tag,
    #                                                            scheduler_tag, OP_tag, dataset_name),
    #               save_name='pytorch_MIE',
    #               net_setting_dir='./MIE/structure{}{}{}{}{}{}.txt'.format(eval_tag, net_setting_tag,loss_tag,
    #                                                                    scheduler_tag, OP_tag, dataset_name),
    #               loss_lr_dir='./MIE/loss_lr{}{}{}{}{}{}.txt'.format(eval_tag, net_setting_tag, loss_tag,
    #                                                              scheduler_tag, OP_tag, dataset_name),
    #               acc_dir='./MIE/acc{}{}{}{}{}{}.txt'.format(eval_tag, net_setting_tag, loss_tag, scheduler_tag,
    #                                                          OP_tag, dataset_name),
    #               independent_eval_dir='./MIE/independent_acc{}{}{}{}{}{}.txt'.format(eval_tag, net_setting_tag, loss_tag,
    #                                                                                   scheduler_tag, OP_tag,
    #                                                                                   dataset_name),
    #               history_log_dir='./MIE/history.txt', isDefaultImageSize=False)
    #
    # # TODO: use args, NEED cancel plot !!!!!!
    # # evaluate batch128 STN out of memory??????
    # myModel.train_MIE(Lr_Start=args.Lr_Start, STN_lr_start=args.STN_lr_start, OP_lr_start=args.OP_lr_start,
    #                   STN_freeze_end_ep=args.STN_freeze_end_ep, OP_freeze_end_ep=args.OP_freeze_end_ep,
    #                   Epoch=args.Epoch, train_batch_size=32, eval_batch_size=128,
    #                   optim_name=args.optim_name, if_REA=args.REA_switch,
    #                   if_triplet_loss=args.triplet_loss_switch, if_center_loss=args.center_loss_switch,
    #                   center_loss_weight=0.0005, if_mask_loss=args.mask_loss_switch,
    #                   binary_threshold=args.binary_threshold, area_constrain_proportion=args.area_constrain_proportion,
    #                   mask_loss_weight=args.mask_loss_weight,
    #                   addOrientationPart=args.addOrientationPart, addSTNPart=args.addSTNPart,
    #                   addAttentionMaskPart=args.addAttentionMaskPart, addChannelReduce=args.addChannelReduce,
    #                   final_dim=args.final_dim, if_affine=args.if_affine, if_OP_channel_wise=args.if_OP_channel_wise,
    #                   mask_num=args.mask_num, STN_init_value=args.STN_init_value,
    #                   save_step=args.save_step, eval_step=args.eval_step,
    #                   if_rerank=args.if_rerank, if_Euclidean=args.if_Euclidean, if_load_OP_model=args.if_load_OP_model,
    #                   OP_model_dir=args.OP_model_dir,if_load_STN_SBP_model=args.if_load_STN_SBP_model,
    #                   SBP_model_dir=args.SBP_model_dir,if_check_OP_grad=False,if_show_plot=False,
    #                   plot_dir='./MIE/loss_acc_lr{}{}{}{}{}{}.jpg'.format(eval_tag, net_setting_tag,loss_tag,
    #                                                                       scheduler_tag, OP_tag,
    #                                                                       dataset_name)
    #                   )
