import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


class Pst900Dataset(Dataset):

    NEGLECT_FOR_METRICS = []

    NUM_CLASSES = 5

    CLASS_NAMES = [
        "background",
        "fire extinguisher",
        "backpack",
        "hand drill",
        "survivor",
    ]

    COLORMAP = {
        0: (0, 0, 0),          # Background
        1: (255, 0, 0),        # Fire Extinguisher (red)
        2: (0, 255, 0),        # Backpack (green)
        3: (0, 0, 255),        # Hand Drill (blue)
        4: (255, 255, 255),    # Survivor (white)
    }

    # Class pixel frequencies on the test split (sum = 1).
    CLASS_WEIGHTS = [
        0.9717858962,  # background
        0.0023568311,  # fire extinguisher
        0.0168604344,  # backpack
        0.0026646216,  # hand drill
        0.0063322166,  # survivor
    ]

    def __init__(self, dataset_dir, split, transform=None):
        super().__init__()

        assert split in ["train", "test", "val"]

        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        split_dir = "test" if split == "val" else split

        rgb_dir = os.path.join(self.dataset_dir, split_dir, "rgb")
        ir_dir = os.path.join(self.dataset_dir, split_dir, "thermal")
        mask_dir = os.path.join(self.dataset_dir, split_dir, "labels")

        self.rgb_images_path = []
        self.ir_images_path = []
        self.masks_path = []
        for name in sorted(os.listdir(rgb_dir)):
            self.rgb_images_path.append(os.path.join(rgb_dir, name))
            self.ir_images_path.append(os.path.join(ir_dir, name))
            self.masks_path.append(os.path.join(mask_dir, name))

        assert len(self.rgb_images_path) == len(self.masks_path)
        assert len(self.ir_images_path) == len(self.masks_path)
        self.length = len(self.rgb_images_path)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        rgb = np.array(Image.open(self.rgb_images_path[index]), dtype=np.uint8)
        ir = np.array(Image.open(self.ir_images_path[index]).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(self.masks_path[index]), dtype=np.int64)

        if self.transform is not None:
            
            augmented = self.transform(image=rgb, ir_image=ir, mask=mask)

            rgb = augmented["image"]
            ir = augmented["ir_image"]
            mask = augmented["mask"]

        return {"rgb": rgb, "ir": ir, "mask": mask}
