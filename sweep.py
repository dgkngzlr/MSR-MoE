# sweep.py
#
# Launch under accelerate so all ranks share one process group for the whole sweep:
#
#   accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 \
#       sweep.py --config configs/fmb.yaml --num_trials 20
#
# Rank 0 owns the Optuna study and samples hyperparameters; other ranks sit in a
# broadcast loop and run `train.main` whenever rank 0 dispatches a cfg.

import gc
import copy
import argparse

import yaml
import torch
import optuna
from accelerate.utils import broadcast_object_list

import train
from utils import build_accelerator
import os
import signal
import torch.distributed as dist

CMD_RUN = "run"
CMD_EXIT = "exit"


def _hard_exit(signum, frame):
    if dist.is_initialized():
        dist.destroy_process_group()
    os._exit(130)
             

def build_trial_cfg(base_cfg, trial: optuna.Trial) -> dict:
    cfg = copy.deepcopy(base_cfg)

    cfg["train"]["lr"] = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    cfg["train"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-5, 1e-2, log=True
    )
    cfg["train"]["batch_per_device"] = trial.suggest_categorical(
        "batch_per_device", [1, 2]
    )
    cfg["model"]["freeze_layer"] = trial.suggest_int("freeze_layer", 0, 4)

    cfg["general"]["experiment"] = (
        f"{base_cfg['general']['experiment']}-trial-{trial.number:04d}"
    )
    # wandb during a sweep would require per-trial init/finish bookkeeping; skip.
    cfg["general"]["logger"] = None
    cfg["train"]["resume_from"] = None

    return cfg


def run_trial(cfg, accelerator, trial):
    """Called on all ranks. `trial` is the real Optuna trial on rank 0, None elsewhere."""
    return train.main(
        cfg,
        accelerator=accelerator,
        trial=trial,
        optuna_enabled=True,
    )


def cleanup_between_trials(accelerator):
    accelerator._models.clear()
    accelerator._optimizers.clear()
    accelerator._schedulers.clear()
    accelerator._dataloaders.clear()
    accelerator._custom_objects.clear()
    gc.collect()
    torch.cuda.empty_cache()


def main_rank0(accelerator, base_cfg, args):
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=5,
        ),
    )

    try:
        for _ in range(args.num_trials):
            trial = study.ask()
            cfg = build_trial_cfg(base_cfg, trial)

            print("\n================================================")
            print(f"Starting trial {trial.number}")
            print(f"  lr               = {cfg['train']['lr']}")
            print(f"  weight_decay     = {cfg['train']['weight_decay']}")
            print(f"  batch_per_device = {cfg['train']['batch_per_device']}")
            print(f"  freeze_layer     = {cfg['model']['freeze_layer']}")
            print("================================================\n")

            payload = [cfg, CMD_RUN]
            broadcast_object_list(payload, from_process=0)
            accelerator.wait_for_everyone()

            try:
                val_best_miou = run_trial(cfg, accelerator, trial)
                study.tell(trial, val_best_miou)
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)

            cleanup_between_trials(accelerator)
    finally:
        # Unblock the other ranks regardless of how we leave the loop.
        broadcast_object_list([None, CMD_EXIT], from_process=0)
        accelerator.wait_for_everyone()

    print("\n====================================")
    print("Best trial")
    print("====================================")
    print(f"Value: {study.best_value:.6f}")
    print("\nParams:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


def main_other_rank(accelerator):
    while True:
        payload = [None, None]
        broadcast_object_list(payload, from_process=0)
        cfg, cmd = payload

        if cmd == CMD_EXIT:
            accelerator.wait_for_everyone()
            return

        accelerator.wait_for_everyone()
        try:
            run_trial(cfg, accelerator, trial=None)
        except optuna.TrialPruned:
            # Pruning is coordinated via collective reduce in Trainer.train;
            # non-rank-0 ranks raise alongside rank 0 and just continue.
            pass

        cleanup_between_trials(accelerator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="segmentation_sweep")
    parser.add_argument("--storage", type=str, default="sqlite:///sweep.db")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _hard_exit)
    signal.signal(signal.SIGTERM, _hard_exit)

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    # Force logger off before the accelerator is built so init_trackers is skipped.
    base_cfg["general"]["logger"] = None

    accelerator = build_accelerator(base_cfg)

    if accelerator.is_main_process:
        main_rank0(accelerator, base_cfg, args)
    else:
        main_other_rank(accelerator)


if __name__ == "__main__":
    main()
