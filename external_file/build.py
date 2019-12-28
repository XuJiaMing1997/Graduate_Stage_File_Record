# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing


def build_transforms(aug_setting, is_train=True):
    # augmentation_preprocess = {'image_size': [256, 128],
    #                            'pixel_mean': [0.485, 0.456, 0.406],
    #                            'pixel_stddev': [0.229, 0.224, 0.225],
    #                            'flip_probability': 0.5,
    #                            'padding_size': 10,
    #                            'random_eras_probability': 0.5,
    #                            's_ratio_min': 0.02,
    #                            's_ratio_max': 0.4,
    #                            'aspect_ratio_min': 0.3
    #                            }
    normalize_transform = T.Normalize(mean=aug_setting['pixel_mean'], std=aug_setting['pixel_stddev'])
    if is_train:
        transform = T.Compose([
            T.Resize(aug_setting['image_size']),
            T.RandomHorizontalFlip(p=aug_setting['flip_probability']),
            T.Pad(aug_setting['padding_size']),
            T.RandomCrop(aug_setting['image_size']),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=aug_setting['random_eras_probability'], sl=aug_setting['s_ratio_min'],
                          sh=aug_setting['s_ratio_max'],r1=aug_setting['aspect_ratio_min'])
        ])
    else:
        transform = T.Compose([
            T.Resize(aug_setting['image_size']),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
