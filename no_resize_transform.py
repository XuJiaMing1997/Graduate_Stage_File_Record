import torchvision.transforms as T
import PIL.Image

from external_file.transforms import RandomErasing
from zoom_out_transform import RandomZoomOutPad


def build_transforms(aug_setting, is_train=True):
    # augmentation_preprocess = {'image_size': image_size,
    #                            'pixel_mean': [0.485, 0.456, 0.406],
    #                            'pixel_stddev': [0.229, 0.224, 0.225],
    #                            'flip_probability': 0.5,
    #                            'padding_size': 10,
    #                            'random_eras_probability': 0.5,
    #                            's_ratio_min': 0.02,
    #                            's_ratio_max': 0.4,
    #                            'aspect_ratio_min': 0.3,
    #                            'if_REA':if_REA
    #                            }
    normalize_transform = T.Normalize(mean=aug_setting['pixel_mean'], std=aug_setting['pixel_stddev'])
    if is_train:
        if aug_setting['if_REA']:
            transform = T.Compose([
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
                T.RandomHorizontalFlip(p=aug_setting['flip_probability']),
                T.Pad(aug_setting['padding_size']),
                T.RandomCrop(aug_setting['image_size']),
                T.ToTensor(),
                normalize_transform
            ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            normalize_transform
        ])

    return transform


def build_transforms_OP(aug_setting, is_train = True):
    normalize_transform = T.Normalize(mean=aug_setting['pixel_mean'], std=aug_setting['pixel_stddev'])
    if is_train:
        transform = T.Compose([
            T.RandomHorizontalFlip(p=aug_setting['flip_probability']),
            T.Pad(aug_setting['padding_size']),
            T.RandomCrop(aug_setting['image_size']),
            RandomZoomOutPad(size=aug_setting['image_size'],Probability=aug_setting['zoom_out_pad_prob']),
            T.RandomRotation(10,resample=PIL.Image.BILINEAR),
            T.ToTensor(),
            normalize_transform
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            normalize_transform
        ])

    return transform