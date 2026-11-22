import os
import yaml
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

import evaluate
from accelerate import Accelerator

from accelerate.utils import set_seed

from dataset_loaders import get_train_transform, get_val_transform
from utils import (
    build_accelerator,
    build_dataset,
    build_loss,
    build_model,
    build_scheduler,
    seed_worker
)

import logging

logger = logging.getLogger("train")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IGNORE_INDEX = 255
IGNORE_COLOR = (255, 0, 255)


def _denormalize(image_tensor):
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _colorize_mask(mask, colormap):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in colormap.items():
        colored[mask == cls] = color
    colored[mask == IGNORE_INDEX] = IGNORE_COLOR
    return colored


def _build_legend(class_names, colormap):
    handles = []
    for cls_idx, color in colormap.items():
        name = class_names[cls_idx] if cls_idx < len(class_names) else f"class {cls_idx}"
        handles.append(
            Patch(facecolor=np.array(color) / 255.0, edgecolor="black", label=f"{cls_idx}: {name}")
        )
    handles.append(
        Patch(facecolor=np.array(IGNORE_COLOR) / 255.0, edgecolor="black", label=f"{IGNORE_INDEX}: ignore")
    )
    return handles


class Tester:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        test_dataloader: DataLoader,
        accelerator: Accelerator,
        cfg: dict,
        save_dir: str = None,
    ):
        self.model = model
        self.criterion = criterion
        self.test_dataloader = test_dataloader
        self.accelerator = accelerator
        self.cfg = cfg
        self.device = accelerator.device

        # Hyperparameters
        self.class_weights = cfg["meta"]["class_weights"]
        self.class_names = cfg["meta"]["class_names"]
        self.num_classes = cfg["meta"]["num_classes"]
        self.neglect_for_metrics = cfg["meta"]["neglect_for_metrics"]
        self.class_colormap = cfg["meta"].get("class_colormap")

        # Where to dump per-sample visualizations (None disables saving).
        self.save_dir = save_dir
        if self.save_dir is not None and self._is_main():
            os.makedirs(self.save_dir, exist_ok=True)
            self._legend_handles = _build_legend(self.class_names, self.class_colormap)

        # Scratchpad
        self._test_loss = 0.0
        self._stats = None

    def _is_main(self) -> bool:
        return self.accelerator.is_main_process

    def _load_best(self):
        """Load best.pt weights into the (prepared) model for evaluation."""
        best_path = os.path.join(self.cfg["test"]["checkpoint"])
        if not os.path.isfile(best_path):
            if self._is_main():
                print(f"⚠️ best.pt not found at {best_path}, testing with current weights")
            return

        state_dict = torch.load(best_path, map_location=self.device)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.load_state_dict(state_dict)
        if self._is_main():
            print(f"▶️ Loaded best weights from {best_path}")

    @torch.no_grad()
    def _test_epoch(self):
        self.model.eval()

        metric = evaluate.load("mean_iou")

        total_loss = torch.zeros(1, device=self.device)
        num_batches = 0

        pbar = tqdm(
            self.test_dataloader,
            desc="Test",
            disable=not self.accelerator.is_local_main_process,
        )
        sample_idx = 0
        for batch in pbar:
            logits = self.model(batch["rgb"], batch["ir"])
            loss = self.criterion(logits, batch["mask"].long())

            preds = logits.argmax(dim=1)

            if self.save_dir is not None and self._is_main():
                self._save_batch_viz(batch, preds, sample_idx)
            sample_idx += batch["mask"].shape[0]

            preds_global, mask_global = self.accelerator.gather_for_metrics(
                (preds, batch["mask"])
            )
            metric.add_batch(predictions=preds_global, references=mask_global)

            batch_loss = loss.detach().float()
            total_loss += batch_loss
            num_batches += 1
            if self.accelerator.is_main_process:
                pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

        self._stats = metric.compute(
            num_labels=self.num_classes, ignore_index=255, reduce_labels=False
        )
        avg = self.accelerator.reduce(
            total_loss / max(num_batches, 1), reduction="mean"
        )
        self._test_loss = avg.item()

    def _compute_metrics(self):

        per_category_iou = {c: float(v) for c, v in zip(self.class_names, self._stats["per_category_iou"])}
        per_category_accuracy = {c: float(v) for c, v in zip(self.class_names, self._stats["per_category_accuracy"])}

        valid = np.ones(self.num_classes, dtype=bool)
        valid[self.neglect_for_metrics] = False

        iou_valid = self._stats["per_category_iou"][valid]
        acc_valid = self._stats["per_category_accuracy"][valid]

        mean_iou = float(np.nanmean(iou_valid))
        mean_accuracy = float(np.nanmean(acc_valid))
        overall_accuracy = float(self._stats["overall_accuracy"])

        weights_arr = np.array(self.class_weights, dtype=np.float32)[valid]
        mask = ~np.isnan(iou_valid)
        w = weights_arr[mask] / weights_arr[mask].sum()
        weighted_iou = float(np.sum(w * iou_valid[mask]))

        return {
            "mean_iou": mean_iou,
            "mean_accuracy": mean_accuracy,
            "overall_accuracy": overall_accuracy,
            "per_category_iou": per_category_iou,
            "per_category_accuracy": per_category_accuracy,
            "weighted_iou": weighted_iou,
        }

    def _log(self, metrics: dict):
        if not self._is_main():
            return

        bar_width = 30

        print(
            f"\n{'*' * 10} Test Results {'*' * 7}\n"
            f"test_loss        = {self._test_loss:.4f}\n"
            f"mean_iou         = {metrics['mean_iou']:.4f}\n"
            f"weighted_iou     = {metrics['weighted_iou']:.4f}\n"
            f"mean_accuracy    = {metrics['mean_accuracy']:.4f}\n"
            f"overall_accuracy = {metrics['overall_accuracy']:.4f}"
        )

        name_width = max(len(c) for c in metrics["per_category_iou"])

        print("  per_category_iou:")
        for c, v in metrics["per_category_iou"].items():
            print(f"    {c:<{name_width}}  {v:.4f}")

        print("  per_category_accuracy:")
        for c, v in metrics["per_category_accuracy"].items():
            print(f"    {c:<{name_width}}  {v:.4f}")

        print("*" * bar_width)

        if self.cfg["general"].get("logger") == "wandb":
            try:
                self.accelerator.log(
                    {
                        "test/loss": self._test_loss,
                        "test/mean_iou": metrics["mean_iou"],
                        "test/weighted_iou": metrics["weighted_iou"],
                        "test/mean_accuracy": metrics["mean_accuracy"],
                        "test/overall_accuracy": metrics["overall_accuracy"],
                        **{f"test/iou/{c}": v for c, v in metrics["per_category_iou"].items()},
                        **{f"test/acc/{c}": v for c, v in metrics["per_category_accuracy"].items()},
                    }
                )
            except Exception as e:
                print(f"⚠️ wandb logging failed: {e}")

    def _save_batch_viz(self, batch, preds, start_idx):
        rgb_t = batch["rgb"]
        ir_t = batch["ir"]
        mask_t = batch["mask"]
        for i in range(mask_t.shape[0]):
            rgb = _denormalize(rgb_t[i])
            ir = _denormalize(ir_t[i])
            gt = _colorize_mask(mask_t[i], self.class_colormap)
            pred = _colorize_mask(preds[i], self.class_colormap)

            fig, axes = plt.subplots(1, 4, figsize=(22, 6))
            axes[0].imshow(rgb)
            axes[0].set_title("RGB")
            axes[1].imshow(ir)
            axes[1].set_title("IR")
            axes[2].imshow(gt)
            axes[2].set_title("GT")
            axes[3].imshow(pred)
            axes[3].set_title("Pred")
            for ax in axes:
                ax.axis("off")

            ncol = min(6, len(self._legend_handles))
            fig.legend(
                handles=self._legend_handles,
                loc="lower center",
                ncol=ncol,
                bbox_to_anchor=(0.5, 0.0),
                fontsize=9,
                frameon=False,
            )
            fig.tight_layout(rect=[0, 0.12, 1, 0.97])

            idx = start_idx + i
            out_path = os.path.join(self.save_dir, f"{idx:05d}.png")
            fig.savefig(out_path, dpi=90, bbox_inches="tight")
            plt.close(fig)

    def test(self):
        self._load_best()
        self._test_epoch()
        metrics = self._compute_metrics()
        self._log(metrics)
        self.accelerator.wait_for_everyone()
        return metrics

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logger = logging.getLogger("test")

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to config file")

    args = parser.parse_args()

    cfg_path = os.path.abspath(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    accelerator = build_accelerator(cfg)
    
    # Seeds must come after Accelerator() so device_specific=True can read
    # accelerator.process_index — each rank gets a different augmentation stream.
    set_seed(cfg["general"]["seed"], device_specific=True)
    g = torch.Generator()
    g.manual_seed(cfg["general"]["seed"] + accelerator.process_index)
    
    # Setup dataset
    train_dataset, \
    val_dataset, \
    test_dataset, \
    num_classes, \
    class_names, \
    class_weights, \
    class_colormap, \
    neglect_for_metrics = build_dataset(cfg)

    # Add metadata
    cfg.setdefault("meta", {})
    cfg["meta"]["num_classes"] = num_classes
    cfg["meta"]["class_names"] = class_names
    cfg["meta"]["class_weights"] = class_weights
    cfg["meta"]["neglect_for_metrics"] = neglect_for_metrics
    cfg["meta"]["class_colormap"] = class_colormap

    # Set transormation for dataset
    train_transform = get_train_transform(cfg)
    val_transform = get_val_transform(cfg)
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform
    test_dataset.transform = val_transform

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1, # some images have diff. res.
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g)

    # Setup model and loss
    model = build_model(cfg)

    criterion = build_loss(cfg)

    # Prepare everything together
    test_dataloader, model = accelerator.prepare(
        test_dataloader, model
    )

    save_dir = os.path.join(
        "outputs",
        cfg["general"]["project"],
        cfg["general"]["experiment"],
    )

    tester = Tester(
        model=model,
        criterion=criterion,
        test_dataloader=test_dataloader,
        accelerator=accelerator,
        cfg=cfg,
        save_dir=save_dir,
    )

    tester.test()



