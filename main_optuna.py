#!/usr/bin/env python3

# MIT License
# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from shutil import copytree, rmtree
import pickle
import os
import re
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from functools import partial

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except Exception:
    optuna = None

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import Dcm, class2one_hot, probs2one_hot, probs2class, tqdm_, dice_coef, save_images
from losses import CrossEntropy
from data_augmentation import HFlip, VFlip, Rotate, RandomAffine

OPTIMIZER_CHOICES = ["adamw"]

# ----------------------------
# Repro & DataLoader workers
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)

# ----------------------------
# Dataset params (adjust here)
# ----------------------------
datasets_params: dict[str, dict[str, Any]] = {}
datasets_params["SEGTHOR_CLEAN"] = {"K": 5, "net": ENet, "B": 8, "kernels": 8, "factor": 2}

# ----------------------------
# Transforms & Augmentations
# ----------------------------
def img_transform(img):
    img = img.convert("L")
    img = np.array(img)[np.newaxis, ...] / 255
    return torch.tensor(img, dtype=torch.float32)

def gt_transform(K, img):
    img = np.array(img)
    img = img / 63 if K == 5 else img / (255 / (K - 1))
    img = torch.tensor(img, dtype=torch.int64)[None, ...]
    return class2one_hot(img, K=K)[0]

def build_augmentations(args):
    augs = []
    for t in [a.lower() for a in args.augmentations]:
        if t == "hflip": augs.append(HFlip())
        elif t == "vflip": augs.append(VFlip())
        elif t == "rotate": augs.append(Rotate())
        elif t == "affine": augs.append(RandomAffine())
    return augs

# ----------------------------
# Folds utilities
# ----------------------------
def create_or_load_folds_by_patient(img_dir, num_folds=5, seed=42, path=None):
    path = Path(path) if path else Path(f"data/{num_folds}_folds.pkl")
    if path.is_dir():
        path = path / f"{num_folds}_folds.pkl"
    patient_ids = sorted({f.name.split('_')[1] for f in img_dir.glob('Patient_*_*.png')})
    if path.exists():
        with open(path, 'rb') as f:
            folds = pickle.load(f)
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        indices = np.arange(len(patient_ids))
        folds = [([patient_ids[i] for i in tr], [patient_ids[i] for i in va]) for tr, va in kf.split(indices)]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(folds, f)
    return folds

def build_fold_dataset(fold_root: Path, train_ids, val_ids, base_img_dir: Path, base_gt_dir: Path):
    for subset, ids in [("train", train_ids), ("val", val_ids)]:
        for sub in ["img", "gt"]:
            dest_dir = fold_root / subset / sub
            dest_dir.mkdir(parents=True, exist_ok=True)
            src_dir = base_img_dir if sub == "img" else base_gt_dir
            for pid in ids:
                pattern = re.compile(f"Patient_{pid}_\\d{{4}}\\.png")
                for file in src_dir.iterdir():
                    if pattern.match(file.name):
                        target = dest_dir / file.name
                        if not target.exists():
                            os.symlink(file.resolve(), target)

# ----------------------------
# Optimizer
# ----------------------------
def build_optimizer(net: nn.Module, args) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )

# ----------------------------
# Setup loaders & model
# ----------------------------
def setup(args):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    K = datasets_params[args.dataset]['K']
    net_cls = datasets_params[args.dataset]['net']
    net = net_cls(1, K, kernels=datasets_params[args.dataset]['kernels'], factor=datasets_params[args.dataset]['factor'])
    net.init_weights()
    net.to(device)

    optimizer = build_optimizer(net, args)

    train_set = SliceDataset(
        'train', args.dest,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        augmentations=build_augmentations(args),
        debug=args.debug
    )
    val_set = SliceDataset(
        'val', args.dest,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        debug=args.debug
    )

    train_loader = DataLoader(
        train_set,
        batch_size=datasets_params[args.dataset]['B'],
        num_workers=5,
        worker_init_fn=worker_init_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=datasets_params[args.dataset]['B'],
        num_workers=5,
        worker_init_fn=worker_init_fn,
        shuffle=False
    )
    return net, optimizer, device, train_loader, val_loader, K

# ----------------------------
# Train one run (with real val metric + pruning)
# ----------------------------
def train_one_run(args, fold_dest, train_ids, val_ids, base_img_dir, base_gt_dir, trial=None):
    build_fold_dataset(fold_dest, train_ids, val_ids, base_img_dir, base_gt_dir)

    local_args = argparse.Namespace(**{**vars(args), "dest": fold_dest})
    net, optimizer, device, train_loader, val_loader, K = setup(local_args)

    # Optional schedulers chosen by Optuna or CLI
    scheduler = None
    if getattr(args, "scheduler", None) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.tune_epochs if getattr(args, "tune_epochs", None) else args.epochs))
    elif getattr(args, "scheduler", None) == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    loss_fn = CrossEntropy(idk=list(range(K)))
    best_dice = 0.0
    best_epoch = -1

    max_epochs = args.tune_epochs if getattr(args, "tune_epochs", None) else args.epochs

    for e in range(max_epochs):
        # ---- Train ----
        net.train()
        for i, data in tqdm_(enumerate(train_loader), total=len(train_loader), desc=f">> Train (Epoch {e})"):
            img, gt = data['images'].to(device), data['gts'].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net(img)
            pred_probs = torch.softmax(logits, dim=1)
            loss = loss_fn(pred_probs, gt)
            loss.backward()
            optimizer.step()

        # ---- Validate ----
        net.eval()
        dices = []
        with torch.no_grad():
            for i, data in tqdm_(enumerate(val_loader), total=len(val_loader), desc=f">> Val   (Epoch {e})"):
                img, gt = data['images'].to(device), data['gts'].to(device)
                logits = net(img)
                probs = torch.softmax(logits, dim=1)

                hard = probs2one_hot(probs)          # (B, K, H, W)
                d = dice_coef(hard, gt)              # (B, K) or (K,)
                if d.dim() == 2:
                    d = d.mean(dim=0)                # mean over batch -> (K,)
                # foreground-only (ignore bg=0) if requested
                if getattr(args, "ignore_bg_in_val", False) and d.numel() > 1:
                    val_dice = d[1:].mean().item()
                else:
                    val_dice = d.mean().item()
                dices.append(val_dice)

        current_dice = float(np.mean(dices)) if dices else 0.0

        # Step schedulers
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_dice)
            else:
                scheduler.step()

        # Track best
        if current_dice > best_dice:
            best_dice, best_epoch = current_dice, e

        # Report to Optuna and allow pruning per-epoch
        if trial is not None:
            trial.report(current_dice, step=e)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_dice, best_epoch

# ----------------------------
# Optuna tuning
# ----------------------------
def tune_hyperparams(args):
    if optuna is None:
        raise RuntimeError("Optuna is not installed or failed to import. Install with `pip install optuna`.")

    base_img_dir = Path('data') / args.dataset / 'img'
    base_gt_dir  = Path('data') / args.dataset / 'gt'
    folds = create_or_load_folds_by_patient(base_img_dir, num_folds=args.folds, path=args.fold_path)

    sampler = TPESampler(seed=args.seed)
    pruner  = MedianPruner(n_startup_trials=3) if args.use_pruner else None
    study   = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    def objective(trial):
        # --- explore *different learning rates* (log scale) ---
        lr = trial.suggest_float('lr', args.lr_min, args.lr_max, log=True)
        # other helpful hyperparams
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        beta1 = trial.suggest_float('beta1', 0.85, 0.99)
        beta2 = trial.suggest_float('beta2', 0.95, 0.9999)
        eps   = trial.suggest_float('eps', 1e-9, 1e-7, log=True)
        scheduler = trial.suggest_categorical('scheduler', [None, 'cosine', 'plateau'])

        trial_args = argparse.Namespace(**{
            **vars(args),
            'lr': lr,
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps,
            'scheduler': scheduler,
            'epochs': args.tune_epochs,  # keep short for tuning
        })

        fold_scores = []
        for fold_idx, (tr, va) in enumerate(folds):
            if args.max_folds_in_tune and fold_idx >= args.max_folds_in_tune:
                break
            opt_dest = args.dest / f"trial_{trial.number:04d}" / f"fold_{fold_idx+1}"
            opt_dest.mkdir(parents=True, exist_ok=True)
            best_dice, best_epoch = train_one_run(trial_args, opt_dest, tr, va, base_img_dir, base_gt_dir, trial=trial)
            fold_scores.append(best_dice)

        return float(np.mean(fold_scores)) if fold_scores else 0.0

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    home_path = Path.home() / 'MIProject' / 'results'
    home_path.mkdir(parents=True, exist_ok=True)
    best_params_path = home_path / 'best_params_optuna_adamw.pkl'
    best_epoch_path  = home_path / 'best_epoch_info.txt'

    with open(best_params_path, 'wb') as f:
        pickle.dump(study.best_params, f)
    with open(best_epoch_path, 'w') as f:
        f.write(f"Best value: {study.best_value}\nBest params: {study.best_params}\n")

    print(f"Saved best params to {best_params_path} and epoch info to {best_epoch_path}")

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SEGTHOR_CLEAN', choices=datasets_params.keys())
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--fold_path', type=str, default='data')
    parser.add_argument('--augmentations', nargs='*', default=[])

    parser.add_argument('--seed', type=int, default=42)

    # Base optimizer hyperparams (used for non-tuning runs or as defaults)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)

    # Training lengths
    parser.add_argument('--epochs', type=int, default=20)

    # Tuning toggles
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--n_trials', type=int, default=25)
    parser.add_argument('--tune_epochs', type=int, default=8)
    parser.add_argument('--max_folds_in_tune', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--use_pruner', action='store_true')

    # LR search range
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--lr_max', type=float, default=5e-3)

    # Validation behavior
    parser.add_argument('--ignore_bg_in_val', action='store_true', help='Average Dice over classes 1..K-1 on validation')

    # Optional schedulers: None | cosine | plateau
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'cosine', 'plateau'])

    args = parser.parse_args()

    set_seed(args.seed)
    pprint(vars(args))

    if args.tune:
        tune_hyperparams(args)
    else:
        # Single run without tuning (uses --epochs and base optimizer hparams)
        base_img_dir = Path('data') / args.dataset / 'img'
        base_gt_dir  = Path('data') / args.dataset / 'gt'
        folds = create_or_load_folds_by_patient(base_img_dir, num_folds=args.folds, path=args.fold_path)
        # Just run the first fold as a quick baseline:
        tr, va = folds[0]
        fold_dest = args.dest / "single_run_fold1"
        fold_dest.mkdir(parents=True, exist_ok=True)
        best_dice, best_epoch = train_one_run(args, fold_dest, tr, va, base_img_dir, base_gt_dir, trial=None)
        print(f"[Single run] Best Dice: {best_dice:.4f} at epoch {best_epoch}")

if __name__ == '__main__':
    main()