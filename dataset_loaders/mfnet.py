import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


class MfnetDataset(Dataset):

    NEGLECT_FOR_METRICS = [0]

    NUM_CLASSES = 9

    CLASS_NAMES = [
        "unlabelled",
        "car",
        "person",
        "bike",
        "curve",
        "car stop",
        "guardrail",
        "color cone",
        "bump",
    ]

    COLORMAP = {
        0: (0, 0, 0),        # unlabelled
        1: (64, 0, 128),     # car
        2: (64, 64, 0),      # person
        3: (0, 128, 192),    # bike
        4: (0, 0, 192),      # curve
        5: (128, 128, 0),    # car stop
        6: (64, 64, 128),    # guardrail
        7: (192, 128, 128),  # color cone
        8: (192, 64, 0),     # bump
    }

    # Class pixel frequencies on the test split (sum = 1).
    CLASS_WEIGHTS = [
        0.9278879827,  # unlabelled
        0.0402097249,  # car
        0.0111486495,  # person
        0.0104495252,  # bike
        0.0049914851,  # curve
        0.0029430562,  # car stop
        0.0001473872,  # guardrail
        0.0007060903,  # color cone
        0.0015160988,  # bump
    ]

    def __init__(self, dataset_dir, split, transform=None):
        super().__init__()

        assert split in ["train", "val", "test", "test_day", "test_night"]

        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform


        self.images_path = []
        self.masks_path = []
        with open(os.path.join(self.dataset_dir, split + ".txt"), "r") as f:
            for line in f:
                self.images_path.append(
                    os.path.join(self.dataset_dir, "images", f"{line.rstrip()}.png")
                )
                self.masks_path.append(
                    os.path.join(self.dataset_dir, "labels", f"{line.rstrip()}.png")
                )

        assert len(self.images_path) == len(self.masks_path)
        self.length = len(self.images_path)
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):

        image = np.array(Image.open(self.images_path[index]), dtype=np.uint8)
        mask = np.array(Image.open(self.masks_path[index]), dtype=np.uint8)

        rgb = image[:, :, :3]
        ir = np.repeat(image[:, :, 3:4], 3, axis=-1)

        if self.transform is not None:
            
            augmented = self.transform(image=rgb, ir_image=ir, mask=mask)

            rgb = augmented["image"]
            ir = augmented["ir_image"]
            mask = augmented["mask"]

        return {"rgb": rgb, "ir": ir, "mask": mask}
