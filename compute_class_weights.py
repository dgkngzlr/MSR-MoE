import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset_loaders import (
    MfnetDataset,
    FmbDataset,
    Pst900Dataset,
    SemanticRTDataset,
    CaltechDataset,
)


DATASETS = {
    "mfnet": (MfnetDataset, "data/mfnet"),
    "fmb": (FmbDataset, "data/fmb"),
    "pst900": (Pst900Dataset, "data/pst900"),
    "semanticrt": (SemanticRTDataset, "data/semanticrt"),
    "caltech": (CaltechDataset, "data/caltech"),
}


def load_mask(path):
    return np.array(Image.open(path))


def compute_counts(mask_paths, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for path in tqdm(mask_paths, desc="masks"):
        mask = load_mask(path).astype(np.int64)
        valid = (mask >= 0) & (mask < num_classes)
        m = mask[valid]
        if m.size == 0:
            continue
        binc = np.bincount(m, minlength=num_classes)
        counts += binc
    return counts


def direct_freq_weights(counts):
    counts = counts.astype(np.float64)
    s = counts.sum()
    if s == 0:
        return np.zeros_like(counts)
    return counts / s


def main():
    for name, (cls, data_dir) in DATASETS.items():
        ds = cls(dataset_dir=data_dir, split="test", transform=None)
        mask_paths = ds.masks_path
        num_classes = cls.NUM_CLASSES
        counts = compute_counts(mask_paths, num_classes)
        weights = direct_freq_weights(counts)
        print(f"\n=== {name} (num_classes={num_classes}) ===")
        print(f"counts: {counts.tolist()}")
        print(f"weights (sum={weights.sum():.6f}):")
        print(f"CLASS_WEIGHTS = [")
        for i, w in enumerate(weights):
            print(f"    {w:.10f},  # {cls.CLASS_NAMES[i]}")
        print(f"]")


if __name__ == "__main__":
    main()
