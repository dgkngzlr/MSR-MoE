import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


class SemanticRTDataset(Dataset):

    NEGLECT_FOR_METRICS = [0]

    NUM_CLASSES = 13

    CLASS_NAMES = [
        "background",
        "car stop",
        "bike",
        "bicyclist",
        "motorcycle",
        "motorcyclist",
        "car",
        "tricycle",
        "traffic light",
        "box",
        "pole",
        "curve",
        "person",
    ]

    COLORMAP = {
        0:  (0, 0, 0),         # background
        1:  (72, 61, 39),      # car stop
        2:  (0, 0, 255),       # bike
        3:  (148, 0, 211),     # bicyclist
        4:  (128, 128, 0),     # motorcycle
        5:  (64, 64, 128),     # motorcyclist
        6:  (0, 139, 139),     # car
        7:  (131, 139, 139),   # tricycle
        8:  (192, 64, 0),      # traffic light
        9:  (126, 192, 238),   # box
        10:  (244, 164, 96),   # pole
        11: (211, 211, 211),   # curve
        12: (205, 155, 155),   # person
    }

    # Class pixel frequencies on the test split (sum = 1).
    CLASS_WEIGHTS = [
        0.7856780672,  # background
        0.0038944143,  # car stop
        0.0052994324,  # bike
        0.0042485387,  # bicyclist
        0.0069797630,  # motorcycle
        0.0046607755,  # motorcyclist
        0.1345320952,  # car
        0.0031747433,  # tricycle
        0.0058481610,  # traffic light
        0.0083713857,  # box
        0.0075049185,  # pole
        0.0145374734,  # curve
        0.0152702318,  # person
    ]

    SPLITS = [
        "train",
        "val",
        "test",
        "test_day",
        "test_night",
        "test_hard",
        "test_mc",
        "test_mo",
    ]

    def __init__(self, dataset_dir, split, transform=None):
        super().__init__()

        assert split in self.SPLITS, f"split must be one of {self.SPLITS}, got {split}"

        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        self.rgb_path = []
        self.ir_path = []
        self.masks_path = []
        with open(os.path.join(self.dataset_dir, split + ".txt"), "r") as f:
            for line in f:
                stem = line.rstrip()
                self.rgb_path.append(
                    os.path.join(self.dataset_dir, "rgb", f"{stem}.jpg")
                )
                self.ir_path.append(
                    os.path.join(self.dataset_dir, "thermal", f"{stem}.jpg")
                )
                self.masks_path.append(
                    os.path.join(self.dataset_dir, "labels", f"{stem}.png")
                )

        assert len(self.rgb_path) == len(self.ir_path) == len(self.masks_path)
        self.length = len(self.rgb_path)

    def _load(self, index):
        rgb = np.array(Image.open(self.rgb_path[index]).convert("RGB"), dtype=np.uint8)
        ir = np.array(Image.open(self.ir_path[index]).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(self.masks_path[index]), dtype=np.uint8)
        # Concat along channel axis so downstream transforms see a single HxWx6 image.
        image = np.concatenate([rgb, ir], axis=2)
        return image, mask

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        image, mask = self._load(index)

        rgb = image[:, :, :3]
        ir = image[:, :, 3:6]

        if self.transform is not None:
            
            augmented = self.transform(image=rgb, ir_image=ir, mask=mask)

            rgb = augmented["image"]
            ir = augmented["ir_image"]
            mask = augmented["mask"]

        return {"rgb": rgb, "ir": ir, "mask": mask}
