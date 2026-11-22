from accelerate import DistributedDataParallelKwargs
from transformers import (
    get_cosine_with_min_lr_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from dataset_loaders import (
    CaltechDataset,
    MfnetDataset,
    FmbDataset,
    Pst900Dataset,
    SemanticRTDataset
)
from models.oursnet import OursNet
from losses import DiceCELoss

import torch
import torch.nn as nn
import random
import numpy as np
import os

from accelerate import Accelerator

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_model(cfg):

    name = cfg["model"]["name"]
    if name == "OursNet":
        return OursNet(
            rgb_ch=cfg["model"]["rgb_ch"],
            ir_ch=cfg["model"]["ir_ch"],
            num_classes=cfg["meta"]["num_classes"],
            freeze_layer=cfg["model"]["freeze_layer"],
        )
    raise ValueError(f"Unknown model name: {name}")

def build_loss(cfg):

    name = cfg["loss"]["name"]
    if name == "dice_ce_loss":
        return DiceCELoss(smooth_factor=0.1, ignore_index=255)
    raise ValueError(f"Unknown loss name: {name}")


def build_scheduler(optimizer, cfg, num_training_steps, num_warmup_steps):

    sched_cfg = cfg["scheduler"]
    name = sched_cfg["name"]

    if name == "cosine_with_min_lr_schedule_with_warmup":
        return get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
            min_lr_rate=sched_cfg.get("min_lr_rate", 0.1),
        )

    if name == "constant_schedule_with_warmup":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )

    if name == "polynomial_decay_schedule_with_warmup":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            lr_end=sched_cfg.get("lr_end", 1e-7),
            power=sched_cfg.get("power", 2.0),
        )

    raise ValueError(f"Unknown scheduler name: {name}")
    
def build_dataset(cfg):

    registry = {
        "caltech":    (CaltechDataset,    os.path.join("./data", "caltech")),
        "mfnet":      (MfnetDataset,      os.path.join("./data", "mfnet")),
        "fmb":        (FmbDataset,        os.path.join("./data", "fmb")),
        "semanticrt": (SemanticRTDataset, os.path.join("./data", "semanticrt")),
        "pst900":     (Pst900Dataset,     os.path.join("./data", "pst900")),
    }
    name = cfg["general"]["dataset"]
    if name not in registry:
        raise ValueError(f"Unknown dataset name: {name}")
    dataset_cls, dataset_dir = registry[name]

    train_dataset = dataset_cls(dataset_dir, "train")
    val_dataset = dataset_cls(dataset_dir, "val")
    test_dataset = dataset_cls(dataset_dir, "test")
    num_classes = dataset_cls.NUM_CLASSES
    class_names = dataset_cls.CLASS_NAMES
    class_weights = dataset_cls.CLASS_WEIGHTS
    class_colormap = dataset_cls.COLORMAP
    neglect_for_metrics = dataset_cls.NEGLECT_FOR_METRICS

    return train_dataset, val_dataset, test_dataset, num_classes, class_names, class_weights, class_colormap, neglect_for_metrics

def build_accelerator(cfg) -> Accelerator:
    """Construct an Accelerator from a cfg. Used both by the standalone `train.py`
    entrypoint and by `sweep.py`, which builds one Accelerator and reuses it
    across trials."""
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    project_dir = f"checkpoints/{cfg['general']['project']}"
    common = dict(
        project_dir=project_dir,
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs],
    )
    if cfg["general"]["logger"] == "wandb":
        accelerator = Accelerator(log_with="wandb", **common)
        accelerator.init_trackers(
            project_name=cfg["general"]["project"],
            config=cfg,
            init_kwargs={"wandb": {"name": cfg["general"]["experiment"]}},
        )
    else:
        accelerator = Accelerator(**common)
    return accelerator