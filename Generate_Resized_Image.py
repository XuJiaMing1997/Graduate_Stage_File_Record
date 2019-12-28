import os
import shutil
import PIL
import PIL.Image as Image
import math
import numpy as np
import matplotlib.pyplot as plt

# TODO: use terminal call, avoid Pycharm indexing
# TODO: change dir
DatasetRootDir = '../Unsupervised-Person-Re-identification-Clustering-and-Fine-tuning-master/dataset'

DatasetDir = {}
DatasetDir['MARKET'] = os.path.join(DatasetRootDir,'Market')
DatasetDir['CUHK03'] = os.path.join(DatasetRootDir,'CUHK03')
DatasetDir['DUKE'] = os.path.join(DatasetRootDir,'Duke')

TrainDir = {}
TrainDir['MARKET'] = os.path.join(DatasetDir['MARKET'],'bounding_box_train')
TrainDir['CUHK03'] = os.path.join(DatasetDir['CUHK03'],'bounding_box_train')
TrainDir['DUKE'] = os.path.join(DatasetDir['DUKE'],'bounding_box_train')

TestDir = {}
TestDir['MARKET'] = os.path.join(DatasetDir['MARKET'],'bounding_box_test')
TestDir['CUHK03'] = os.path.join(DatasetDir['CUHK03'],'bounding_box_test')
TestDir['DUKE'] = os.path.join(DatasetDir['DUKE'],'bounding_box_test')

QueryDir = {}
QueryDir['MARKET'] = os.path.join(DatasetDir['MARKET'],'query')
QueryDir['CUHK03'] = os.path.join(DatasetDir['CUHK03'],'query')
QueryDir['DUKE'] = os.path.join(DatasetDir['DUKE'],'query')


target_size = [[224,224],[256,128]]
# 224 x 224
for t_size in target_size:
    size_tag = '{}x{}'.format(t_size[0],t_size[1])
    print('current generate {}'.format(size_tag))
    NewRootDir = '../Dataset_Resized_' # TODO: change dir
    NewRootDir += size_tag
    NewDir = {}
    NewDir['MARKET'] = os.path.join(NewRootDir,'Market')
    NewDir['CUHK03'] = os.path.join(NewRootDir,'CUHK03')
    NewDir['DUKE'] = os.path.join(NewRootDir,'Duke')

    NewTrainDir = {}
    NewTrainDir['MARKET'] = os.path.join(NewDir['MARKET'],'bounding_box_train')
    NewTrainDir['CUHK03'] = os.path.join(NewDir['CUHK03'],'bounding_box_train')
    NewTrainDir['DUKE'] = os.path.join(NewDir['DUKE'],'bounding_box_train')

    NewTestDir = {}
    NewTestDir['MARKET'] = os.path.join(NewDir['MARKET'],'bounding_box_test')
    NewTestDir['CUHK03'] = os.path.join(NewDir['CUHK03'],'bounding_box_test')
    NewTestDir['DUKE'] = os.path.join(NewDir['DUKE'],'bounding_box_test')

    NewQueryDir = {}
    NewQueryDir['MARKET'] = os.path.join(NewDir['MARKET'],'query')
    NewQueryDir['CUHK03'] = os.path.join(NewDir['CUHK03'],'query')
    NewQueryDir['DUKE'] = os.path.join(NewDir['DUKE'],'query')


    if not os.path.exists(NewRootDir):
        os.makedirs(NewRootDir)

    TargetDatasetName = ['MARKET','CUHK03','DUKE']

    for dataset in TargetDatasetName:
        print('current dataset: {}'.format(dataset))
        if not os.path.exists(NewDir[dataset]):
            os.makedirs(NewDir[dataset])
            print('make folder {}'.format(NewDir[dataset]))
        if not os.path.exists(NewTrainDir[dataset]):
            os.makedirs(NewTrainDir[dataset])
            print('make folder {}'.format(NewTrainDir[dataset]))
        else:
            shutil.rmtree(NewTrainDir[dataset],ignore_errors=True)
            os.makedirs(NewTrainDir[dataset])
            print('clean & make folder {}'.format(NewTrainDir[dataset]))
        if not os.path.exists(NewTestDir[dataset]):
            os.makedirs(NewTestDir[dataset])
            print('make folder {}'.format(NewTestDir[dataset]))
        else:
            shutil.rmtree(NewTestDir[dataset],ignore_errors=True)
            os.makedirs(NewTestDir[dataset])
            print('clean & make folder {}'.format(NewTestDir[dataset]))
        if not os.path.exists(NewQueryDir[dataset]):
            os.makedirs(NewQueryDir[dataset])
            print('make folder {}'.format(NewQueryDir[dataset]))
        else:
            shutil.rmtree(NewQueryDir[dataset],ignore_errors=True)
            os.makedirs(NewQueryDir[dataset])
            print('clean & make folder {}'.format(NewQueryDir[dataset]))

        print(TrainDir[dataset])
        print('    ---> {}'.format(NewTrainDir[dataset]))
        fname_list = os.listdir(TrainDir[dataset])
        print('image num: {}'.format(len(fname_list)))
        for idx,fname in enumerate(fname_list):
            if idx % 1000 == 0:
                print(idx)
            if fname == 'Thumbs.db':
                continue
            img = Image.open(os.path.join(TrainDir[dataset], fname))
            img = img.resize((t_size[1],t_size[0]),resample= PIL.Image.BILINEAR)
            img.save(os.path.join(NewTrainDir[dataset],fname))

        print(TestDir[dataset])
        print('    ---> {}'.format(NewTestDir[dataset]))
        fname_list = os.listdir(TestDir[dataset])
        print('image num: {}'.format(len(fname_list)))
        for idx,fname in enumerate(fname_list):
            if idx % 1000 == 0:
                print(idx)
            if fname == 'Thumbs.db':
                continue
            img = Image.open(os.path.join(TestDir[dataset], fname))
            img = img.resize((t_size[1],t_size[0]),resample= PIL.Image.BILINEAR)
            img.save(os.path.join(NewTestDir[dataset],fname))

        print(QueryDir[dataset])
        print('    ---> {}'.format(NewQueryDir[dataset]))
        fname_list = os.listdir(QueryDir[dataset])
        print('image num: {}'.format(len(fname_list)))
        for idx,fname in enumerate(fname_list):
            if idx % 1000 == 0:
                print(idx)
            if fname == 'Thumbs.db':
                continue
            img = Image.open(os.path.join(QueryDir[dataset],fname))
            img = img.resize((t_size[1],t_size[0]),resample= PIL.Image.BILINEAR)
            img.save(os.path.join(NewQueryDir[dataset],fname))



















