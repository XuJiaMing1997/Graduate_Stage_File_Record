# This File is assistant, OP_annotation_name is Core
import os
import time
import sys
import math

import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image

import shutil
import argparse


def show_img(target_folder, label_list_dir, show_num_once = 8):
    # fname_sort
    fname_list = os.listdir(target_folder)
    if  'Thumbs.db' in fname_list:
        fname_list.pop(fname_list.index('Thumbs.db'))
        print('Pop Thumbs.db')
    fname_list.sort() # IMPORTANT !!!!!
    print('target folder: {}\ninclude images: {}'.format(target_folder, len(fname_list)))

    if os.path.isfile(label_list_dir):
        label_f = open(label_list_dir,'r')
        lines = label_f.readlines()
        label_f.close()
        start = len(lines)
        print('----> read: {}'.format(label_list_dir))
        print('Have annotated {}  Rest: {}'.format(len(lines), len(fname_list) - len(lines)))
        print('Last Annotation image: {}'.format(lines[-1].strip()))
    else:
        start = 0

    fname_dir = []
    count = 0
    for idx in range(start, len(fname_list)):
        if count != show_num_once:
            fname_dir.append(os.path.join(target_folder,fname_list[idx]))
            count += 1
        if count == show_num_once:
            fig = plt.figure()
            fig.suptitle('rest: {}\nFirst img: {}\nFront:1  Back:2  Side:3'.format(
                    len(fname_list) - idx - 1, fname_dir[0].split('\\')[-1]))
            for i,fd in enumerate(fname_dir):
                img = cv2.imread(fd)
                img = img[:,:,::-1]
                subfig = fig.add_subplot(2, int(math.ceil(show_num_once / 2.)),i+1)
                subfig.imshow(img)
                subfig.set_xticks([])
                subfig.set_yticks([])
            count = 0
            fname_dir = []
            plt.show()

    if count != 0:
        fig = plt.figure()
        fig.suptitle('rest: {}\nFirst img: {}\nFront:1  Back:2  Side:3'.format(
                len(fname_list) - idx - 1, fname_dir[0].split('\\')[-1]))
        for i,fd in enumerate(fname_dir):
            img = cv2.imread(fd)
            img = img[:,:,::-1]
            subfig = fig.add_subplot(2, int(math.ceil(show_num_once / 2.)),i+1)
            subfig.imshow(img)
            subfig.set_xticks([])
            subfig.set_yticks([])
        plt.show()




def RandomCheck(check_list_dir, check_folder_dir ,check_num = 10):
    OP_dict = {0:'Front',
               1:'Back',
               2:'Side'}
    f = open(check_list_dir,'r')
    lines = f.readlines()
    f.close()
    print('{} include label: {}'.format(check_list_dir, len(lines)))

    percentage_check(check_list_dir)

    for i in range(check_num):
        rand_idx = np.random.randint(0,len(lines))
        lines_list = lines[rand_idx].strip().split(' ')
        file_dir = os.path.join(check_folder_dir, lines_list[0])
        plt.figure()
        plt.title('Rest: {}\nName: {}\nLabel: {}'.format(check_num-i-1,file_dir,OP_dict[int(lines_list[1])]))
        img = cv2.imread(file_dir)
        img = img[:,:,::-1]
        plt.imshow(img)
        plt.show()


def percentage_check(check_list_dir):
    f = open(check_list_dir,'r')
    lines = f.readlines()
    f.close()

    fname_list = []
    orient_list = []
    for line in lines:
        line_list = line.strip().split(' ')
        fname_list.append(line_list[0])
        orient_list.append(int(line_list[1]))
    label = np.array(orient_list)  # or np.where will not work !!!!!
    count_front = len(np.where(label == 0)[0])
    count_back = len(np.where(label == 1)[0])
    count_side = len(np.where(label == 2)[0])
    print('File {}\ninclude labels: {}'.format(check_list_dir, len(lines)))
    print('percentage for each view:')
    print('Front: {0} Back: {1} Side: {2}'.format(count_front, count_back, count_side))
    print('Front: {0:.2%} Back: {1:.2%} Side: {2:.2%}'.format(
            count_front / float(len(label)), count_back / float(len(label)), count_side / float(len(label))
    ))
    return count_front, count_back, count_side, len(label)


def copy_image_with_OP_label(source_folder, label_list_dir, target_folder):
    folder_fname_list = os.listdir(source_folder)
    if  'Thumbs.db' in folder_fname_list:
        folder_fname_list.pop(folder_fname_list.index('Thumbs.db'))
        print('Pop Thumbs.db')
    folder_fname_list.sort() # IMPORTANT !!!!!
    print('Folder Num: {}'.format(len(folder_fname_list)))

    f = open(label_list_dir,'r')
    lines = f.readlines()
    f.close()
    fname_list = []
    orient_list = []
    for line in lines:
        line_list = line.strip().split(' ')
        fname_list.append(line_list[0])
        orient_list.append(int(line_list[1]))
    print('List fname num: {}'.format(len(fname_list)))
    assert len(fname_list) == len(folder_fname_list), 'List name num MUST == Folder image num'

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for idx, (fname, label) in enumerate(zip(fname_list, orient_list)):
        if idx % 1000 == 0:
            print(idx)
        fname_split_list = fname.split('.')
        fname_new = fname_split_list[0] + '_' + str(label) + '.' + fname_split_list[-1]
        shutil.copy(os.path.join(source_folder, fname),os.path.join(target_folder, fname_new))


if __name__ == '__main__':
    #
    # File ./Market_train_OP_label.list
    # include labels: 12936
    # percentage for each view:
    # Front: 4899 Back: 3987 Side: 4050
    # Front: 37.87% Back: 30.82% Side: 31.31%
    #

    parser = argparse.ArgumentParser(description='OP Annotation Setting')
    parser.add_argument('--target_folder', type=str, default='../Dataset_256x128/Market/bounding_box_train', metavar='Dir',
                        help='folder dir')
    parser.add_argument('--label_list_dir', type=str, default='./Market_train_OP_label.list', metavar='Dir',
                        help='label list dir')
    parser.add_argument('--show_num_once', type=int, default=10, metavar='N',
                        help='show num one (default: 10)')
    # parser.add_argument('--if_load_OP_model', action='store_true', default=False,
    #                     help='load OP model to initialize')  # action means no input var
    args = parser.parse_args()

    # show_img(target_folder=args.target_folder, label_list_dir=args.label_list_dir, show_num_once=args.show_num_once)
    RandomCheck(check_list_dir=args.label_list_dir, check_folder_dir=args.target_folder,check_num=20)
    # copy_image_with_OP_label(source_folder=args.target_folder, label_list_dir=args.label_list_dir,
    #                          target_folder='./Market_OP')









