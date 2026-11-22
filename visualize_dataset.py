import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from dataset_loaders import (
    MfnetDataset,
    FmbDataset,
    Pst900Dataset,
    SemanticRTDataset,
    CaltechDataset,
    get_train_transform,
    get_val_transform,
)


DATASETS = {
    "mfnet": (MfnetDataset, "data/mfnet"),
    "fmb": (FmbDataset, "data/fmb"),
    "pst900": (Pst900Dataset, "data/pst900"),
    "semanticrt": (SemanticRTDataset, "data/semanticrt"),
    "caltech": (CaltechDataset, "data/caltech")
}

IGNORE_INDEX = 255
IGNORE_COLOR = (255, 0, 255)  # magenta

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(image_tensor):
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def colorize_mask(mask, colormap):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in colormap.items():
        colored[mask == cls] = color
    colored[mask == IGNORE_INDEX] = IGNORE_COLOR
    return colored


def build_legend(class_names, colormap):
    handles = []
    for cls_idx, color in colormap.items():
        name = class_names[cls_idx] if cls_idx < len(class_names) else f"class {cls_idx}"
        handles.append(
            Patch(facecolor=np.array(color) / 255.0, edgecolor="black", label=f"{cls_idx}: {name}")
        )
    handles.append(
        Patch(
            facecolor=np.array(IGNORE_COLOR) / 255.0,
            edgecolor="black",
            label=f"{IGNORE_INDEX}: ignore",
        )
    )
    return handles


def parse_args():
    parser = argparse.ArgumentParser(description="Veriseti gorsellestirme aracligi.")
    parser.add_argument("dataset", choices=list(DATASETS.keys()), help="Hangi veriseti.")
    parser.add_argument("split", help="Hangi split (ornek: train, val, test).")
    parser.add_argument("-n", "--num-samples", type=int, default=40, help="Kac adet ornek.")
    parser.add_argument("--output-dir", default="./viz", help="Cikti kok dizini.")
    parser.add_argument("--data-root", default=None, help="Veriseti dizini (varsayilan: data/<dataset>).")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik seed degeri.")
    parser.add_argument(
        "--transform",
        choices=["auto", "train", "val"],
        default="auto",
        help="Hangi transform kullanilsin (auto: split=='train' ise train, aksi halde val).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_cls, default_dir = DATASETS[args.dataset]
    data_dir = args.data_root or default_dir

    cfg = {"general": {"seed": args.seed}}
    if args.transform == "train" or (args.transform == "auto" and args.split == "train"):
        transform = get_train_transform(cfg)
        transform_name = "train"
    else:
        transform = get_val_transform(cfg)
        transform_name = "val"

    dataset = dataset_cls(dataset_dir=data_dir, split=args.split, transform=transform)

    num = min(args.num_samples, len(dataset))
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=num, replace=False)

    out_dir = os.path.join(args.output_dir, args.dataset, args.split)
    os.makedirs(out_dir, exist_ok=True)

    colormap = dataset_cls.COLORMAP
    class_names = dataset_cls.CLASS_NAMES
    legend_handles = build_legend(class_names, colormap)

    print(
        f"[viz] dataset={args.dataset} split={args.split} transform={transform_name} "
        f"samples={num}/{len(dataset)} out={out_dir}"
    )

    for i, idx in enumerate(indices):
        sample = dataset[int(idx)]
        rgb = denormalize(sample["rgb"])
        ir = denormalize(sample["ir"])
        mask_colored = colorize_mask(sample["mask"], colormap)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[0].axis("off")
        axes[1].imshow(ir)
        axes[1].set_title("IR")
        axes[1].axis("off")
        axes[2].imshow(mask_colored)
        axes[2].set_title("GT Mask")
        axes[2].axis("off")

        ncol = min(5, len(legend_handles))
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=ncol,
            bbox_to_anchor=(0.5, 0.0),
            fontsize=9,
            frameon=False,
        )
        fig.suptitle(
            f"{args.dataset} / {args.split} / idx={int(idx)}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0.12, 1, 0.95])

        out_path = os.path.join(out_dir, f"{i:04d}_idx{int(idx):05d}.png")
        fig.savefig(out_path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"[{i + 1}/{num}] -> {out_path}")

    print(f"Bitti. {num} gorsel kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()
