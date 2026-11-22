import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


class FmbDataset(Dataset):

    NEGLECT_FOR_METRICS = [0]

    NUM_CLASSES = 15

    CLASS_NAMES = [
        "unlabeled",
        "road",
        "sidewalk",
        "building",
        "traffic light",
        "traffic sign",
        "vegetation",
        "sky",
        "person",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "pole",
    ]


    COLORMAP = {
        0:  (0, 0, 0),          # Unlabeled
        1:  (179, 228, 228),    # Road
        2:  (181, 57, 133),     # Sidewalk
        3:  (67, 162, 177),     # Building
        4:  (200, 178, 50),     # Traffic Light
        5:  (132, 45, 199),     # Traffic Sign
        6:  (66, 172, 84),      # Vegetation
        7:  (179, 73, 79),      # Sky
        8:  (76, 99, 166),      # Person
        9:  (66, 121, 253),     # Car
        10:  (139, 165, 91),     # Truck
        11: (156, 99, 152),     # Bus
        12: (105, 153, 140),    # Motorcycle
        13: (222, 215, 158),    # Bicycle
        14: (135, 113, 90),     # Pole
    }

    # Class pixel frequencies on the test split (sum = 1).
    CLASS_WEIGHTS = [
        0.0141257068,  # unlabeled
        0.1133052604,  # road
        0.0159623289,  # sidewalk
        0.1962109226,  # building
        0.0018304762,  # traffic light
        0.0042387128,  # traffic sign
        0.2348506845,  # vegetation
        0.3252249405,  # sky
        0.0063527307,  # person
        0.0584234598,  # car
        0.0038221205,  # truck
        0.0085361905,  # bus
        0.0004065104,  # motorcycle
        0.0000000000,  # bicycle (no pixels in test split)
        0.0167099554,  # pole
    ]

    def __init__(self, dataset_dir, split, transform=None):
        super().__init__()

        assert split in ["train", "test", "val"]

        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform

        split_dir = "test" if split == "val" else split

        rgb_dir = os.path.join(self.dataset_dir, split_dir, "Visible")
        ir_dir = os.path.join(self.dataset_dir, split_dir, "Infrared")
        mask_dir = os.path.join(self.dataset_dir, split_dir, "Label")

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
