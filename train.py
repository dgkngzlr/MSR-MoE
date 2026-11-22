import os
import sys
import yaml
import random
import logging
import argparse
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import optuna
import numpy as np
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


from test import Tester

logger = logging.getLogger("train")


def setup_logging(log_dir: str, is_main: bool) -> None:
    """Send `logger` output to both stdout and {log_dir}/train.log on main rank;
    silence non-main ranks so DDP doesn't duplicate lines."""
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    if not is_main:
        logger.addHandler(logging.NullHandler())
        return

    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

@dataclass
class TrainingState:
    next_epoch: int = 0
    max_mean_iou: float = 0.0
    early_stop_cnt: int = 0

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, sd):
        self.next_epoch = sd["next_epoch"]
        self.max_mean_iou = sd["max_mean_iou"]
        self.early_stop_cnt = sd["early_stop_cnt"]

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        accelerator: Accelerator,
        cfg: dict,
        trial=None,
        optuna_enabled: bool = False,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.accelerator = accelerator
        self.cfg = cfg
        self.device = accelerator.device
        self.trial = trial
        self.optuna_enabled = optuna_enabled

        # Hyperparameters
        self.epochs = cfg["train"]["epochs"]
        self.early_stop_th = cfg["train"]["early_stop"]
        self.class_weights = cfg["meta"]["class_weights"]
        self.class_names = cfg["meta"]["class_names"]
        self.num_classes = cfg["meta"]["num_classes"]
        self.neglect_for_metrics = cfg["meta"]["neglect_for_metrics"]

        self.state = TrainingState()
        self.accelerator.register_for_checkpointing(self.state)

        # Per-epoch metric scratchpad
        self._train_loss = 0.0
        self._val_loss = 0.0
        self._stats = None

    def _is_main(self) -> bool:
        return self.accelerator.is_main_process

    def _train_epoch(self, epoch: int):

        self.model.train()

        total_loss = torch.zeros(1, device=self.device)
        num_batches = 0

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Train {epoch}/{self.epochs}",
            disable=not self.accelerator.is_local_main_process,
        )
        for batch in pbar:
            with self.accelerator.accumulate(self.model):
                logits = self.model(batch["rgb"], batch["ir"])
                loss = self.criterion(logits, batch["mask"].long())

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        

            batch_loss = loss.detach().float()
            total_loss += batch_loss
            num_batches += 1
            if self.accelerator.is_main_process:
                pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

        avg = self.accelerator.reduce(total_loss / max(num_batches, 1), reduction="mean")
        self._train_loss = avg.item()

    @torch.no_grad()
    def _val_epoch(self, epoch: int):

        self.model.eval()

        metric = evaluate.load("mean_iou")
         
        total_loss = torch.zeros(1, device=self.device)
        num_batches = 0

        pbar = tqdm(
            self.val_dataloader,
            desc=f"Val   {epoch}/{self.epochs}",
            disable=not self.accelerator.is_local_main_process,
        )
        for batch in pbar:
            logits = self.model(batch["rgb"], batch["ir"])
            loss = self.criterion(logits, batch["mask"].long())

            preds = logits.argmax(dim=1)

            # Gather across processes BEFORE computing stats so padded samples
            # added by the distributed sampler are dropped exactly once.
            preds_global, mask_global = self.accelerator.gather_for_metrics((preds, batch["mask"]))
            metric.add_batch(predictions=preds_global, references=mask_global)

            batch_loss = loss.detach().float()
            total_loss += batch_loss
            num_batches += 1
            if self.accelerator.is_main_process:
                pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

        # Stats are already global (gather_for_metrics handled it)
        # Loss still needs reducing because it was accumulated per-rank.
        self._stats = metric.compute(num_labels=self.num_classes, ignore_index=255, reduce_labels=False)
        avg = self.accelerator.reduce(total_loss / max(num_batches, 1), reduction="mean")
        self._val_loss = avg.item()

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

    def _log(self, epoch: int, metrics: dict):

        if not self._is_main():
            return

        lr = self.scheduler.get_last_lr()[0]
        bar_width = 30

        name_width = max(len(c) for c in metrics["per_category_iou"])
        iou_lines = "\n".join(
            f"    {c:<{name_width}}  {v:.4f}" for c, v in metrics["per_category_iou"].items()
        )
        acc_lines = "\n".join(
            f"    {c:<{name_width}}  {v:.4f}" for c, v in metrics["per_category_accuracy"].items()
        )

        logger.info(
            f"\n{'*' * 10} Logging {'*' * 10}\n"
            f"epoch            = {epoch}/{self.epochs}\n"
            f"lr               = {lr:.2e}\n"
            f"train_loss       = {self._train_loss:.4f}\n"
            f"val_loss         = {self._val_loss:.4f}\n"
            f"mean_iou         = {metrics['mean_iou']:.4f}\n"
            f"weighted_iou     = {metrics['weighted_iou']:.4f}\n"
            f"mean_accuracy    = {metrics['mean_accuracy']:.4f}\n"
            f"overall_accuracy = {metrics['overall_accuracy']:.4f}\n"
            f"  per_category_iou:\n{iou_lines}\n"
            f"  per_category_accuracy:\n{acc_lines}\n"
            f"{'*' * bar_width}"
        )

        if self.cfg["general"].get("logger") == "wandb":
            try:
                self.accelerator.log(
                    {
                        "lr": lr,
                        "train/loss": self._train_loss,
                        "val/loss": self._val_loss,
                        "val/mean_iou": metrics["mean_iou"],
                        "val/weighted_iou": metrics["weighted_iou"],
                        "val/mean_accuracy": metrics["mean_accuracy"],
                        "val/overall_accuracy": metrics["overall_accuracy"],
                        **{f"val/iou/{c}": v for c, v in metrics["per_category_iou"].items()},
                        **{f"val/acc/{c}": v for c, v in metrics["per_category_accuracy"].items()},
                    },
                    step=epoch,
                )
            except Exception as e:
                logger.warning(f"⚠️ wandb logging failed: {e}")


    def _maybe_resume(self):
        resume_from = self.cfg["train"].get("resume_from")
        if not resume_from:
            return
        if not os.path.isdir(resume_from):
            if self._is_main():
                logger.warning(f"⚠️ resume_from not found, starting fresh: {resume_from}")
            return
        self.accelerator.load_state(resume_from)
        if self._is_main():
            logger.info(
                f"▶️ Resumed from {resume_from} "
                f"(next_epoch={self.state.next_epoch}, max_mean_iou={self.state.max_mean_iou:.4f})"
            )

    def train(self):

        self._maybe_resume()

        last_dir = os.path.join(self.accelerator.project_dir, self.cfg["general"]["experiment"], "last")
        best_dir = os.path.join(self.accelerator.project_dir, self.cfg["general"]["experiment"], "best")

        for epoch in range(self.state.next_epoch, self.epochs):
            self._train_epoch(epoch)
            self._val_epoch(epoch)

            metrics = self._compute_metrics()
            self._log(epoch, metrics)

            # Update early-stop bookkeeping
            improved = metrics["mean_iou"] > self.state.max_mean_iou
            if improved:
                self.state.max_mean_iou = metrics["mean_iou"]
                self.state.early_stop_cnt = 0
            else:
                self.state.early_stop_cnt += 1
            self.state.next_epoch = epoch + 1

            # "last" is overwritten every epoch so resume_from points at the
            # most recent state. "best" is only refreshed when miou improves.
            self.accelerator.save_state(last_dir)
            if improved:
                self.accelerator.save_state(best_dir)

                # Also save just the model weights for easy inference loading.
                self.accelerator.wait_for_everyone()
                unwrapped = self.accelerator.unwrap_model(self.model)
                self.accelerator.save(
                    unwrapped.state_dict(),
                    os.path.join(self.accelerator.project_dir, self.cfg["general"]["experiment"], "best.pt"),
                )
            
            # Optuna pruning: report mIoU and check if Optuna wants to kill this trial.
            # All ranks must enter this block so the collective `reduce` doesn't deadlock;
            # only rank 0 holds the real trial object.
            if self.optuna_enabled:
                if self._is_main():
                    self.trial.report(metrics["mean_iou"], epoch)
                    should_prune = 1 if self.trial.should_prune() else 0
                else:
                    should_prune = 0

                # Broadcast rank 0's decision to all ranks
                prune_tensor = torch.tensor(should_prune, device=self.device)
                prune_tensor = self.accelerator.reduce(prune_tensor, reduction="sum")

                if prune_tensor.item() > 0:
                    if self._is_main():
                        logger.info(f"⏹️  Pruned at epoch {epoch} (mIoU={metrics['mean_iou']:.4f})")
                    raise optuna.TrialPruned()

            if self.state.early_stop_cnt >= self.early_stop_th:
                if self._is_main():
                    logger.info(
                        f"⏹️  Early stop at epoch {epoch}. "
                        f"Best miou: {self.state.max_mean_iou:.4f}"
                    )
                break

        self.accelerator.wait_for_everyone()


def main(cfg, accelerator=None, trial=None, optuna_enabled=False):

    if accelerator is None:
        accelerator = build_accelerator(cfg)

    setup_logging(os.path.join(accelerator.project_dir, cfg["general"]["experiment"]), accelerator.is_main_process)

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

    # Set transormation for dataset
    train_transform = get_train_transform(cfg)
    val_transform = get_val_transform(cfg)
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform
    test_dataset.transform = val_transform

    # Setup dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg["train"]["batch_per_device"], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g)
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=1, # some images have diff. res.
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g)
    
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

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"])
    )

    num_training_steps = (
        math.ceil(len(train_dataloader) / cfg["train"]["gradient_accumulation_steps"])
        * cfg["train"]["epochs"]
    )
    num_warmup_steps = (
        cfg["scheduler"].get("num_warmup_steps", 0) * accelerator.num_processes
    )
    scheduler = build_scheduler(optimizer, cfg, num_training_steps, num_warmup_steps)

    # Prepare everything together
    train_dataloader, val_dataloader, test_dataloader, model, optimizer, scheduler = accelerator.prepare(
        train_dataloader, val_dataloader, test_dataloader, model, optimizer, scheduler
    )

    try:
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            accelerator=accelerator,
            cfg=cfg,
            trial=trial,
            optuna_enabled=optuna_enabled,
        )

        trainer.train()

        tester = Tester(
            model=model,
            criterion=criterion,
            test_dataloader=test_dataloader,
            accelerator=accelerator,
            cfg=cfg,
        )
        tester.test()
    finally:
        accelerator.end_training()

    return trainer.state.max_mean_iou

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path to config file")

    args = parser.parse_args()

    cfg_path = os.path.abspath(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    main(cfg)