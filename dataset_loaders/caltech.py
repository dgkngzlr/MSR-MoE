import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


class CaltechDataset(Dataset):

    NEGLECT_FOR_METRICS = [0, 1]

    NUM_CLASSES = 12

    CLASS_NAMES = [
        "unknown",
        "background",
        "bare ground",
        "rocky terrain",
        "human made",
        "road",
        "shrubs",
        "trees",
        "sky",
        "water",
        "vegetation",
        "person",
    ]

    COLORMAP = {
        0: (255, 36, 0),      # unknown
        1: (0, 0, 0),         # background
        2: (242, 216, 196),   # bare ground
        3: (89, 70, 54),      # rocky terrain
        4: (166, 166, 166),   # human made
        5: (82, 89, 90),      # road
        6: (155, 230, 0),     # shrubs
        7: (0, 138, 53),      # trees
        8: (0, 216, 245),     # sky
        9: (13, 127, 252),    # water
        10: (255, 249, 0),    # vegetation
        11: (254, 0, 170),    # person
    }

    # Class pixel frequencies on the test split (sum = 1).
    CLASS_WEIGHTS = [
        0.0000842107,  # unknown
        0.0000010578,  # background
        0.1968565188,  # bare ground
        0.1860852499,  # rocky terrain
        0.0306497723,  # human made
        0.0175004480,  # road
        0.1052430369,  # shrubs
        0.0499775737,  # trees
        0.0563772028,  # sky
        0.3567871553,  # water
        0.0003525301,  # vegetation
        0.0000852437,  # person
    ]

    def __init__(self, dataset_dir, split, transform=None):
        super().__init__()

        assert split in ["train", "test", "val"]

        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        self.rgb_images_path = []
        self.ir_images_path = []
        self.masks_path = []
        with open(os.path.join(self.dataset_dir, "rgb_" + split + ".txt"), "r") as f:
            for line in f:
                self.rgb_images_path.append(
                    os.path.join(self.dataset_dir, line.split(",")[0].rstrip())
                )
                self.masks_path.append(
                    os.path.join(self.dataset_dir, line.split(",")[1].rstrip())
                )
        with open(os.path.join(self.dataset_dir, "thermal8_" + split + ".txt"), "r") as f:
            for line in f:
                self.ir_images_path.append(
                    os.path.join(self.dataset_dir, line.split(",")[0].rstrip())
                )
                

        assert len(self.rgb_images_path) == len(self.masks_path)
        assert len(self.ir_images_path) == len(self.masks_path)
        self.length = len(self.rgb_images_path)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        rgb = np.array(Image.open(self.rgb_images_path[index]), dtype=np.uint8)
        ir = np.array(Image.open(self.ir_images_path[index]), dtype=np.uint8)
        mask = np.array(Image.open(self.masks_path[index]), dtype=np.int64)

        if self.transform is not None:
            
            augmented = self.transform(image=rgb, ir_image=ir, mask=mask)

            rgb = augmented["image"]
            ir = augmented["ir_image"]
            mask = augmented["mask"]

        return {"rgb": rgb, "ir": ir, "mask": mask}
