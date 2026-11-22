import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform

TARGET_SIZE = (512, 512)

def get_train_transform(cfg):

    seed = cfg["general"]["seed"]

    train_transform = A.Compose(
        [
            A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
            A.PadIfNeeded(min_height=TARGET_SIZE[0], min_width=TARGET_SIZE[1], border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=255),
            A.RandomCrop(height=TARGET_SIZE[0], width=TARGET_SIZE[1], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(std_range=(0.001, 0.01), p=0.25),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ],
        additional_targets={'ir_image': 'image'},
        seed=seed,
    )

    return train_transform


def get_val_transform(cfg):

    seed = cfg["general"]["seed"]

    val_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=255),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.ToTensorV2(),
        ],
        additional_targets={'ir_image': 'image'},
        seed=seed,
    )

    return val_transform
