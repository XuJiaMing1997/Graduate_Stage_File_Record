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
from torch.utils.data import DataLoader


from external_file.build import build_transforms as my_trans
from external_file.dataset_loader import ImageDataset
from external_file.triplet_sampler import RandomIdentitySampler
from external_file.triplet_sampler import RandomIdentitySampler_alignedreid
from external_file.baseline import Baseline
import external_file.optimizer_build as optimizer_build
from external_file.lr_scheduler import WarmupMultiStepLR
from external_file.triplet_loss import TripletLoss, CrossEntropyLabelSmooth, normalize
from external_file.center_loss import CenterLoss
from external_file.collate_batch import train_collate_fn, val_collate_fn
from external_file.re_rank import re_ranking


class pytorch_SBP(object):
    def __init__(self,
                 target_dataset_name='MARKET',
                 dataset_loc='../',
                 save_dir='./SBP/model_save',
                 save_name='pytorch_SBP',
                 net_setting_dir='./SBP/structure.txt',
                 loss_lr_dir='./SBP/loss_lr.txt',
                 acc_dir='./SBP/acc.txt',
                 independent_eval_dir='./SBP/independent_acc.txt',
                 history_log_dir='./SBP/history.txt'):
        self.target_dataset_name = target_dataset_name
        self.dataset_loc = dataset_loc
        self.save_dir = save_dir
        self.save_name = save_name
        self.net_setting_dir = net_setting_dir
        self.loss_lr_dir = loss_lr_dir
        self.acc_dir = acc_dir
        self.independent_eval_dir = independent_eval_dir
        self.history_log_dir = history_log_dir

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
        if not os.path.exists('./SBP'):
            os.makedirs('./SBP')

    def lr_check(self, optimizer):
        random_target = np.random.randint(0, len(optimizer.param_groups) + 1, 3)
        print('----> lr check -- num {}'.format(len(random_target)))
        for i in random_target:
            print('param {}: {}'.format(i, optimizer.param_groups[i]['lr']))

    def NormalizedTensorImage2Numpy(self, input_tensor_3D, pixel_mean, pixel_stddev):
        img = input_tensor_3D.permute(1, 2, 0)
        img = img * t.tensor(pixel_stddev, dtype=t.float32) + \
              t.tensor(pixel_mean, dtype=t.float32)
        img = img * t.tensor(255., dtype=t.float32)
        img_recovery = img.type(t.int).numpy()
        return img_recovery

    def forward_BP_logic_check(self,
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
                          'eval_neck': True,
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
        train_dataloader, train_img_num, train_person_num = self.construct_dataset_dataloader(
                self.train_dir, augmentation_preprocess, dataloader_setting,
                if_reset_id=True, is_train=True, batch_size=train_batch_size)

        test_dataloader, test_img_num, test_person_num = self.construct_dataset_dataloader(
                self.test_dir, augmentation_preprocess, dataloader_setting,
                if_reset_id=True, is_train=False, batch_size=128)

        print('train iteration num per epoch: {}'.format(len(train_dataloader)))
        # When use triplet or center loss, make sure person all in one epoch
        # When only use softmax loss, make sure image all in one epoch
        print('train iter images num: {} dataset image: {}'.format(len(train_dataloader) * train_batch_size,
                                                                   train_img_num))
        print('test iteration num per epoch: {}'.format(len(test_dataloader)))
        # make sure image all in one epoch
        print('test iter images num: {} dataset image: {}'.format(len(test_dataloader) * 128, test_img_num))

        # construct model and initialize weights
        my_model = self.construct_model(train_person_num, model_settting)
        my_model.to(available_device)
        print(my_model)

        print('-----> var check')
        for idx, (p_name, param) in enumerate(my_model.named_parameters()):
            print('{} {}: {}'.format(idx, p_name, param.shape))
        # 162 parameters

        # construct optimizer
        if if_center_loss:
            loss_calculator, center_criterion = self.build_loss_structure(person_num=train_person_num,
                                                                          if_center_loss=True,
                                                                          loss_setting=loss_setting,
                                                                          use_GPU=use_GPU)
            # loss_cal param: fc_logits, GAP_feature, label, if_triplet_loss, if_center_loss, center_loss_weight
            my_optimizer, center_optimizer = self.generate_center_optimizer(optimizer_setting, my_model,
                                                                            center_criterion)
        else:
            my_optimizer = self.generate_optimizer(optimizer_setting, my_model)
            loss_calculator = self.build_loss_structure(person_num=train_person_num,
                                                        if_center_loss=False,
                                                        loss_setting=loss_setting,
                                                        use_GPU=use_GPU)
            # loss_cal param: fc_logits, GAP_feature, label, if_triplet_loss, if_center_loss, center_loss_weight

        print('----------->  optimizer param group check')
        print('param group num: {}'.format(len(my_optimizer.param_groups)))
        for idx, gp in enumerate(my_optimizer.param_groups):
            print('{} {}'.format(idx,gp))
        # 161 parameters (total 162), BNNeck's /beta(bias) do not need grad, so not in optimize list

        print('----------->  center optimizer param group check')
        print('param group num: {}'.format(len(center_optimizer.param_groups)))
        for idx, gp in enumerate(center_optimizer.param_groups):
            print('{} {}'.format(idx,gp))
        # 1 parameters, independent lr


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

                #
                print('batch_image: {}'.format(batch_image.shape))
                print('batch_label: {}'.format(batch_label.shape))
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

    def plot_SBP_loss_acc_lr(self, eval_dir=None, ifSave=True, SaveDir='./SBP/loss_acc_lr.jpg', if_show = True):
        # loss file: 'epoch {} train_loss {} softmax_loss {} triplet_loss {} center_loss {} '
        #            'lr {} time {}:{}:{}\n'
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
        lr = []
        for line in loss_lines:
            line_split = line.strip('\n').split(' ')
            loss_epoch.append(int(line_split[1]))
            loss.append(float(line_split[3]))
            s_loss.append(float(line_split[5]))
            t_loss.append(float(line_split[7]))
            c_loss.append(float(line_split[9]))
            lr.append(float(line_split[11]))

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

        subfig.legend(handles=[train_loss_line, softmax_loss_line, triplet_loss_line, center_loss_line],
                      labels=['train_loss', 'softmax_loss', 'triplet_loss', 'center_loss'], loc='best')

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
        max_rank_1_idx = np.argmax(rank[:, 0]) # only return first max value index
        max_rank_1_value = rank[:, 0][max_rank_1_idx]
        max_rank_1_ep = eval_epoch[max_rank_1_idx]

        max_mAP_idx = np.argmax(mAP) # only return first max value index
        max_mAP_value = mAP[max_mAP_idx]
        max_mAP_ep = eval_epoch[max_mAP_idx]

        print('MAX rank_1: {} EP: {}'.format(max_rank_1_value,max_rank_1_ep))
        print('MAX mAP: {} EP: {}'.format(max_mAP_value, max_mAP_ep))
        return max_rank_1_ep, max_mAP_ep


    def compute_pixel_mean_stddev(self, image_array):
        # image_all: [img_all,256,128,3]   index_group: [Pall,K?]   id_group: 1D [0,Pall]
        pixel_mean = np.mean(image_array, axis=(0, 1, 2))
        # pixel_stddev = np.sqrt(np.mean(np.square(image_array - pixel_mean),axis=[0,1,2]))
        pixel_stddev = np.std(image_array, axis=(0, 1, 2))
        print('pixel mean: {0}'.format(pixel_mean))
        print('pixel stddev: {0}'.format(pixel_stddev))
        return pixel_mean, pixel_stddev

    def compute_dataset_train_pixel_mean_stddev(self, target_size = [224,224],
                                                DatasetTrainDir = None):
        if DatasetTrainDir is None:
            train_set_dir = self.train_dir
        else:
            train_set_dir = DatasetTrainDir

        img_list = []
        fname_list = os.listdir(train_set_dir)
        for fname in fname_list:
            if fname == 'Thumbs.db':
                continue
            img = Image.open(os.path.join(train_set_dir,fname))
            img = img.resize((target_size[1], target_size[0]), resample=PIL.Image.BILINEAR) # PIL.Image.NEAREST
            img = np.array(img)
            img = img / 255. # [0,1] float type
            img_list.append(img)
        print('DatasetTrainFoler: {}'.format(train_set_dir))
        print('train set image num: {}'.format(len(img_list)))
        mean, stddev = self.compute_pixel_mean_stddev(img_list)
        return mean, stddev

    def load_from_folder(self, FolderDr='../CTL/dataset/CUHK03/bounding_box_train', if_reset_id=True):
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
        print('Type:\timage_path_list {}\tid_list {}\tcam_list {}\tperson_num {}'.format(
                type(image_path_list[0]), type(new_id_list[0]), type(cam_list[0]), type(person_num)
        ))

        return image_path_list, new_id_list, cam_list, person_num

    def construct_dataset_dataloader(self,
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

        img_path_array, id_array, cam_array, person_num = self.load_from_folder(dataset_dir, if_reset_id=if_reset_id)
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

    def construct_model(self, num_class, model_setting):
        # model_setting = {'last_stride': 2,
        #                  'neck': True,
        #                  'eval_neck': False,
        #                  'model_name': 'resnet50',
        #                  'ImageNet_Init': True,
        #                  'ImageNet_model_path': './resnet50-19c8e357.pth'
        #                  }
        model = Baseline(num_class, model_setting['last_stride'], model_setting['neck'],
                         model_setting['eval_neck'], model_setting['model_name'],
                         model_setting['ImageNet_Init'], model_setting['ImageNet_model_path'])
        return model

    def build_loss_structure(self, person_num, if_center_loss, loss_setting, use_GPU=True):
        # loss_setting = {'eval_feature_dim': 2048,
        #                 'if_triplet_loss': if_triplet_loss,
        #                 'triplet_margin': 0.2,  # 0.1-0.3 Zheng suggest
        #                 'label_smooth_rate': 0.1  # 0. original softmax
        #                 }
        # GAP_feature -- raw not apply l2 normalization
        # strange????  local class with fn return, can be accessed in other place??????
        if if_center_loss:
            center_criterion = CenterLoss(num_classes=person_num, feat_dim=loss_setting['eval_feature_dim'],
                                          use_gpu=use_GPU)
            # center loss
        if loss_setting['if_triplet_loss']:
            triplet = TripletLoss(loss_setting['triplet_margin'])  # triplet loss
        xent = CrossEntropyLabelSmooth(num_classes=person_num, epsilon=loss_setting['label_smooth_rate'],
                                       use_gpu=use_GPU)
        # label smooth softmax

        # input not one-hot label
        def loss_func(fc_logits, eval_feature, label, if_triplet_loss, if_center_loss, center_loss_weight):
            softmax_loss = xent(fc_logits, label)
            if if_triplet_loss:
                triplet_loss = triplet(eval_feature, label)[0]
            else:
                triplet_loss = t.tensor(0., dtype=t.float32)
            if if_center_loss:
                center_loss = center_criterion(eval_feature, label) * center_loss_weight  # scale tensor
            else:
                center_loss = t.tensor(0., dtype=t.float32)

            final_loss = softmax_loss + triplet_loss + center_loss
            return final_loss, final_loss.item(), softmax_loss.item(), triplet_loss.item(), center_loss.item()

        if if_center_loss:
            return loss_func, center_criterion
        else:
            return loss_func

    def lr_scheduler(self, optimizer, current_epoch, scheduler_setting):
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
        elif current_epoch <= 40:
            factor = 1
        elif current_epoch <= 70:
            factor = 1. / scheduler_setting['decay_rate']
        else:
            factor = 1. / float(scheduler_setting['decay_rate'] ** 2)

        for param_group in optimizer.param_groups:
            param_group['lr'] = scheduler_setting['lr_start'] * factor
        print('----> apply lr scheduler & basic lr: {}'.format(scheduler_setting['lr_start'] * factor))
        return scheduler_setting['lr_start'] * factor

    def generate_optimizer(self, optimizer_setting, model):
        optimizer, _, _ = optimizer_build.make_optimizer(optimizer_setting, model)
        # make one var one group !!!
        # notice weight decay for bias item and for BN layer !!!!!!!
        return optimizer

    def generate_center_optimizer(self, optimizer_setting, model, center_criterion):
        optimizer, center_optimizer, _, _ = optimizer_build.make_optimizer_with_center(optimizer_setting, model,
                                                                                 center_criterion)
        # one var one group !!!
        return optimizer, center_optimizer

    def calculate_topn_mAP(self, query_feature, query_label, test_feature, test_label,
                           if_rerank = False, if_Euclidean = False, if_sqrt = False):
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

        def EuclideanDistanceMatrix(M1,M2,if_sqrt = False):
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
            similarity_matrix = EuclideanDistanceMatrix(query_feature_norm,test_feature_norm,if_sqrt=if_sqrt)
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

    def independent_evaluate(self,
                             image_size = [224,224],
                             if_rerank = False,
                             if_Euclidean = False,
                             batch_size=128,
                             target_model_path=None):
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

        augmentation_preprocess = {'image_size': image_size,
                                   'pixel_mean': [0.485, 0.456, 0.406],
                                   'pixel_stddev': [0.229, 0.224, 0.225],
                                   'flip_probability': 0.5,
                                   'padding_size': 10,
                                   'random_eras_probability': 0.5,
                                   's_ratio_min': 0.02,
                                   's_ratio_max': 0.4,
                                   'aspect_ratio_min': 0.3
                                   }
        dataloader_setting = {'if_triplet_loss': False,
                              'K_num': 4,
                              'if_center_loss': False,
                              'num_workers': 8
                              }

        query_dataloader, query_img_num, query_person_num = self.construct_dataset_dataloader(
                self.query_dir, augmentation_preprocess, dataloader_setting,
                if_reset_id=False, is_train=False, batch_size=batch_size)

        test_dataloader, test_img_num, test_person_num = self.construct_dataset_dataloader(
                self.test_dir, augmentation_preprocess, dataloader_setting,
                if_reset_id=False, is_train=False, batch_size=batch_size)

        model_setting = self.net_setting_load()
        my_model = self.construct_model(2, model_setting)  # random class_num
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
        my_model.load_param(target_model_path)

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

                pid_array = np.array(pid)[:,np.newaxis]
                cam_array = np.array(cam)[:,np.newaxis]
                tem_label = np.concatenate((pid_array,cam_array),axis=1).tolist() # [B,2]
                query_label.extend(tem_label)

            query_feature = t.cat(query_feature, dim=0)
            print('query_feature: {}\tquery_label: {}'.format(query_feature.shape, np.shape(query_label)))

            print('extract test set feature')
            for data in test_dataloader:
                image, pid, cam = data
                image = image.to(available_device)
                feature = my_model(image)
                test_feature.append(feature)

                pid_array = np.array(pid)[:,np.newaxis]
                cam_array = np.array(cam)[:,np.newaxis]
                tem_label = np.concatenate((pid_array,cam_array),axis=1).tolist() # [B,2]
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

    def independent_evaluate_all(self,
                                 image_size = [224,224],
                                 batch_size = 128,
                                 if_rerank = False, if_Euclidean = False):
        f = open(self.independent_eval_dir, 'w')
        f.close()
        fname_list = os.listdir(self.save_dir)
        for fname in fname_list:
            target_model_path = os.path.join(self.save_dir, fname)
            self.independent_evaluate(image_size=image_size, batch_size=batch_size,
                                      target_model_path=target_model_path,
                                      if_rerank=if_rerank, if_Euclidean=if_Euclidean)

        self.plot_SBP_loss_acc_lr(eval_dir=self.independent_eval_dir,
                                  SaveDir='./SBP/independent_SBP_loss_acc_lr.jpg')

    def train(self,
              Lr_Start = 0.00035,
              Epoch=120,
              train_batch_size=64,
              eval_batch_size=128,
              if_triplet_loss=True,
              if_center_loss=True,
              center_loss_weight=0.0005,
              eval_step=2,  # 4
              save_step=4,  # 4
              if_rerank = False,
              if_Euclidean = False,
              if_show=False,
              plot_dir = './SBP/loss_acc_lr.jpg'
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
        model_setting = {'last_stride': 1,
                         'neck': True,
                         'eval_neck': True,
                         'model_name': 'resnet50',
                         'ImageNet_Init': True,
                         'ImageNet_model_path': './resnet50-19c8e357.pth'
                         }
        optimizer_setting = {'lr_start': Lr_Start,
                             'weight_decay': 0.0005,
                             'bias_lr_factor': 1,
                             'weight_decay_bias': 0.0005,
                             'optimizer_name': 'Adam',  # 'SGD' 'Adam'
                             'SGD_momentum': 0.9,
                             'center_lr': 0.5,
                             'OP_lr':0.1
                             }
        scheduler_setting = {'lr_start': Lr_Start,
                             'decay_rate': 10,
                             'warmup_rate': 100
                             }
        loss_setting = {'eval_feature_dim': 2048,
                        'if_triplet_loss': if_triplet_loss,
                        'triplet_margin': 0.3,  # 0.1-0.3 Zheng suggest, LuoHao 0.3
                        'label_smooth_rate': 0.1  # 0. original softmax
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
        history_f.write('->->->->->-> Augmentation Preprocess:\n')
        history_f.write('Image Size:\t{}\n'.format(augmentation_preprocess['image_size']))
        history_f.write('Pixel Mean:\t{}\tPixel Stddev:\t{}\n'.format(augmentation_preprocess['pixel_mean'],
                                                                      augmentation_preprocess['pixel_stddev']))
        history_f.write('Random Flip Probability:\t{}\n'.format(augmentation_preprocess['flip_probability']))
        history_f.write('Random Pad Crop -- Pad:\t{}\tCrop:\t{}\n'.format(augmentation_preprocess['padding_size'],
                                                                          augmentation_preprocess['image_size']))
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
        history_f.write('DataLoader Worker Num:\t{}\n'.format(dataloader_setting['num_workers']))
        history_f.write('->->->->->-> Optimizer Setting:\n')
        history_f.write('Lr_Start:\t{}\n'.format(Lr_Start))
        history_f.write('Optimizer:\t{}\n'.format(optimizer_setting['optimizer_name']))
        history_f.write('Weight Decay:\t{}\tWeight Decay Bias:\t{}\tBias Lr Factor:\t{}\n'.format(
                optimizer_setting['weight_decay'],optimizer_setting['weight_decay_bias'],
                optimizer_setting['bias_lr_factor']))
        history_f.write('SGD Momentum:\t{}\n'.format(optimizer_setting['SGD_momentum']))
        history_f.write('Center Lr:\t{}\n'.format(optimizer_setting['center_lr']))
        history_f.write('->->->->->-> Scheduler Setting:\n')
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

        train_dataloader, train_img_num, train_person_num = self.construct_dataset_dataloader(
                self.train_dir, augmentation_preprocess, dataloader_setting,
                if_reset_id=True, is_train=True, batch_size=train_batch_size)

        query_dataloader, query_img_num, query_person_num = self.construct_dataset_dataloader(
                self.query_dir, augmentation_preprocess, dataloader_setting,
                if_reset_id=False, is_train=False, batch_size=eval_batch_size)

        test_dataloader, test_img_num, test_person_num = self.construct_dataset_dataloader(
                self.test_dir, augmentation_preprocess, dataloader_setting,
                if_reset_id=False, is_train=False, batch_size=eval_batch_size)

        print('train iteration num per epoch: {}'.format(len(train_dataloader)))
        print('train iter images num: {} dataset image: {}'.format(len(train_dataloader) * train_batch_size,
                                                                   train_img_num))

        # construct model and initialize weights
        my_model = self.construct_model(train_person_num, model_setting)
        # my_model.cuda()
        my_model.to(available_device)

        # construct optimizer
        if if_center_loss:
            loss_calculator, center_criterion = self.build_loss_structure(person_num=train_person_num,
                                                                          if_center_loss=True,
                                                                          loss_setting=loss_setting,
                                                                          use_GPU=use_GPU)
            # loss_cal param: fc_logits, GAP_feature, label, if_triplet_loss, if_center_loss, center_loss_weight
            my_optimizer, center_optimizer = self.generate_center_optimizer(optimizer_setting, my_model,
                                                                            center_criterion)
        else:
            my_optimizer = self.generate_optimizer(optimizer_setting, my_model)
            loss_calculator = self.build_loss_structure(person_num=train_person_num,
                                                        if_center_loss=False,
                                                        loss_setting=loss_setting,
                                                        use_GPU=use_GPU)
            # loss_cal param: fc_logits, GAP_feature, label, if_triplet_loss, if_center_loss, center_loss_weight

        # train and eval
        t0 = time.clock()
        for ep in range(Epoch):
            print('--------->  Start epoch {}'.format(ep))
            my_model.train()
            time_start = time.clock()
            avg_loss = []
            avg_acc = []
            current_basic_lr = self.lr_scheduler(my_optimizer, ep, scheduler_setting)
            for iter, data in enumerate(train_dataloader):
                batch_image, batch_label = data
                batch_image = batch_image.to(available_device)
                batch_label = batch_label.to(available_device)

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
                        param.grad.data *= (1. / center_loss_weight)  # recovery grad weighted by CenterLossWeight
                    center_optimizer.step()

                acc = (fc_logits.max(1)[1] == batch_label).float().mean().item()
                print('iter {}\tacc {}\tloss {}\ts_loss {}\tt_loss {}\tc_loss {}\tlr {}'.format(
                        iter, acc, loss[1], loss[2], loss[3], loss[4], current_basic_lr))
                avg_loss.append(loss[1:])
                avg_acc.append(acc)

            print('-------> Epoch {} End'.format(ep))
            avg_loss = np.mean(avg_loss, axis=0)
            avg_acc = np.mean(avg_acc, axis=0)
            print('Avg_acc: {}\tAvg_loss: {}\tAvg_s_loss: {}\tAvg_t_loss: {}\tAvg_c_loss: {}\tLr: {}'.format(
                    avg_acc, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], current_basic_lr))

            # write loss lr
            time_current = time.clock() - time_start
            time_elapsed_hour = int(time_current // 3600)
            time_elapsed_minute = int(time_current // 60 - 60 * time_elapsed_hour)
            time_elapsed_second = int(time_current - 3600 * time_elapsed_hour - 60 * time_elapsed_minute)

            loss_f = open(self.loss_lr_dir, 'a')
            loss_f.write('epoch {} train_loss {} softmax_loss {} triplet_loss {} center_loss {} '
                         'lr {} time {}:{}:{}\n'.format(ep, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3],
                                                        current_basic_lr, time_elapsed_hour, time_elapsed_minute,
                                                        time_elapsed_second))
            loss_f.close()

            if ep % save_step == 0 or ep + 1 == Epoch:
                print('-------> save model')
                t.save(my_model.state_dict(), os.path.join(self.save_dir, '{}_{}.pth'.format(self.save_name,ep)))

            if ep % eval_step == 0 or ep + 1 == Epoch:
                print('-------> start evaluate Epoch{} model'.format(ep))
                eval_time_start = time.clock()
                my_model.eval()
                query_feature = []
                query_label = []
                test_feature = []
                test_label = []
                with t.no_grad():
                    print('extract query set feature')
                    for data in query_dataloader:
                        image, pid, cam = data # image(4D tensor) pid(1D tuple) cam(1D tuple)
                        image = image.to(available_device)
                        feature = my_model(image)
                        query_feature.append(feature)

                        pid_array = np.array(pid)[:,np.newaxis]
                        cam_array = np.array(cam)[:,np.newaxis]
                        tem_label = np.concatenate((pid_array,cam_array),axis=1).tolist() # [B,2]
                        query_label.extend(tem_label)

                    query_feature = t.cat(query_feature, dim=0) # tensor
                    print('query_feature: {}\tquery_label: {}'.format(query_feature.shape, np.shape(query_label)))

                    print('extract test set feature')
                    for data in test_dataloader:
                        image, pid, cam = data
                        image = image.to(available_device)
                        feature = my_model(image)
                        test_feature.append(feature)

                        pid_array = np.array(pid)[:,np.newaxis]
                        cam_array = np.array(cam)[:,np.newaxis]
                        tem_label = np.concatenate((pid_array,cam_array),axis=1).tolist() # [B,2]
                        test_label.extend(tem_label)

                    test_feature = t.cat(test_feature, dim=0) # tensor
                    print('test_feature: {}\ttest_label: {}'.format(test_feature.shape, np.shape(test_label)))

                    # input: feature 4D tensor, label [Num,2] list
                    rank, mAP = self.calculate_topn_mAP(query_feature, query_label, test_feature, test_label,
                                                        if_rerank=if_rerank, if_Euclidean=if_Euclidean,
                                                        if_sqrt=False)

                    print('Epoch {} Dataset {}'.format(ep, self.target_dataset_name))
                    print('1: %f\t5: %f\t10: %f\t20: %f\tmAP: %f' % (rank[0], rank[1], rank[2], rank[3], mAP))

                    time_current = time.clock() - eval_time_start
                    time_elapsed_hour = int(time_current // 3600)
                    time_elapsed_minute = int(time_current // 60 - 60 * time_elapsed_hour)
                    time_elapsed_second = int(time_current - 3600 * time_elapsed_hour - 60 * time_elapsed_minute)
                    print('Time elapsed: {0}:{1}:{2}'.format(
                            time_elapsed_hour, time_elapsed_minute, time_elapsed_second))

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
                 'lr {}\n'.format(ep, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], current_basic_lr))
        history_f.write('Time Elapsed Overall {}:{}:{}\n'.format(time_elapsed_hour, time_elapsed_minute,
                                                        time_elapsed_second))
        history_f.close()

        self.plot_SBP_loss_acc_lr(SaveDir=plot_dir, if_show=if_show)


if __name__ == '__main__':
    # #####################################################################
    # Market
    # Train: 12936 image (751 identity)
    # Test: 19732 image (exist 0--distractor -1--junk; in evaluate--preserve distractor, discard junk)
    # Query: 3368 image (750 identity)

    # Duke
    # Train: 16522 image (702 identity)
    # Test: 17661 image
    # Query: 2228 image (702 identity)

    # CUHK03(new_protocol,detected)
    # Train: 7365 image (767 identity)
    # Test: 5332 image
    # Query: 1400 image (700 identity)
    # #####################################################################

    parser = argparse.ArgumentParser(description='MIE Train Setting')
    parser.add_argument('--target_dataset_name', type=str, default='MARKET', metavar='Name',
                        help='target dataset name (default: MARKET)')
    args = parser.parse_args()

    dataset_name = args.target_dataset_name.strip()
    save_dir = './SBP/model_save' + '_{}'.format(dataset_name)
    net_setting_dir = './SBP/structure_{}.txt'.format(dataset_name)
    loss_lr_dir = './SBP/loss_lr_{}.txt'.format(dataset_name)
    acc_dir = './SBP/acc_{}.txt'.format(dataset_name)
    plot_dir = './SBP/loss_acc_lr_{}.jpg'.format(dataset_name)

    myModel = pytorch_SBP(target_dataset_name=dataset_name,
                          save_dir=save_dir,
                          net_setting_dir=net_setting_dir,
                          loss_lr_dir=loss_lr_dir,
                          acc_dir=acc_dir)


    # myModel.compute_dataset_train_pixel_mean_stddev('../ViewInvariantNet/CUHK_part')
    # myModel.compute_dataset_train_pixel_mean_stddev(target_size=[256,128])
    # REA: mean (0.4914, 0.4822, 0.4465) ?????
    # #####################################################################
    # Market train 224x224 , 256x128
    # pixel mean: [0.41475891 0.38901752 0.38423163]
    # pixel stddev: [0.21395619 0.20609914 0.20543369]
    #
    # Duke train
    # pixel mean: [0.43995332 0.43091059 0.44581371]
    # pixel stddev: [0.23044895 0.23225201 0.22228246]
    #
    # CUHK03 train
    # pixel mean: [0.3644157  0.36088115 0.35251895]
    # pixel stddev: [0.24143442 0.24459431 0.24552803]
    # #####################################################################


    # myModel.forward_BP_logic_check()
    myModel.train(if_show=False,plot_dir=plot_dir)
    # myModel.plot_SBP_loss_acc_lr()

    # myModel.independent_evaluate([256,128],if_rerank=False,if_Euclidean=True,
    #                              target_model_path='./SBP/Eculidean_Success/model_save/pytorch_SBP_119.pth')

