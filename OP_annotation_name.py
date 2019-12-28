# This File is Core, OP_annotation_show is assistant
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


def name_fn(target_folder, label_list_dir, label_num_once = 8):
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

    fname_once = []
    count = 0
    for idx in range(start, len(fname_list)):
        if count != label_num_once:
            fname_once.append(fname_list[idx])
            count += 1
        if count == label_num_once:
            while(True):
                recv_str = input('rest: {}\nFirst fname: {}\nInput-{}(1,2,3): '.format(len(fname_list) - idx - 1,
                                                                                  fname_once[0], label_num_once))
                recv_str = recv_str.strip()

                # Correct Check
                if len(recv_str) != label_num_once:
                    print('\n\n======!!!======Input Length Wrong!  Try Again=========!!!=====\n')
                    continue
                elif False in [i in ['1','2','3'] for i in recv_str]:
                    print('\n\n======!!!======Input Not in (1,2,3)!  Try Again=========!!!=====\n')
                    continue
                else:
                    break
                # Correct Check End

            label_f = open(label_list_dir,'a')
            for fname,label in zip(fname_once, recv_str):
                label_f.write('{} {}\n'.format(fname, int(label) - 1))
            label_f.close()
            count = 0
            fname_once = []

    if count != 0:
        while(True):
            recv_str = input('rest: {}\nFirst fname: {}\nInput-{}(1,2,3): '.format(len(fname_list) - idx - 1,
                                                                              fname_once[0], label_num_once))
            recv_str = recv_str.strip()

            # Correct Check
            if len(recv_str) != count:
                print('\n\n======!!!======Input Length Wrong!  Try Again=========!!!=====\n')
                continue
            elif False in [i in ['1','2','3'] for i in recv_str]:
                print('\n\n======!!!======Input Not in (1,2,3)!  Try Again=========!!!=====\n')
                continue
            else:
                break
            # Correct Check End

        label_f = open(label_list_dir,'a')
        for fname,label in zip(fname_once, recv_str):
            label_f.write('{} {}\n'.format(fname, int(label) - 1))
        label_f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OP Annotation Setting')
    parser.add_argument('--target_folder', type=str, default='../Dataset_256x128/Market/bounding_box_train', metavar='Dir',
                        help='folder dir')
    parser.add_argument('--label_list_dir', type=str, default='./Market_train_OP_label.list', metavar='Dir',
                        help='label list dir')
    parser.add_argument('--label_num_once', type=int, default=10, metavar='N',
                        help='label num one (default: 10)')
    # parser.add_argument('--if_load_OP_model', action='store_true', default=False,
    #                     help='load OP model to initialize')  # action means no input var
    args = parser.parse_args()

    name_fn(target_folder=args.target_folder, label_list_dir=args.label_list_dir, label_num_once=args.label_num_once)

