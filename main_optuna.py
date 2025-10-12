#!/usr/bin/env python3

# MIT License
# Copyright (c) 2025 Hoel Kervadec, Caroline Magg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

# Optuna for HPO
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except Exception as e:
    optuna = None

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (
    Dcm,
    class2one_hot,
    probs2one_hot,
    probs2class,
    tqdm_,
    dice_coef,
    save_images,
)
from losses import CrossEntropy

from data_augmentation import HFlip, VFlip, Rotate, RandomAffine

# Optimizer: AdamW 
OPTIMIZER_CHOICES = ["adamw"]


def build_optimizer(net: nn.Module, args) -> torch.optim.Optimizer:
    lr = getattr(args, "lr", 5e-4)
    wd = getattr(args, "weight_decay", 0.0)
    beta1 = getattr(args, "beta1", 0.9)
    beta2 = getattr(args, "beta2", 0.999)
    eps = getattr(args, "eps", 1e-8)

    return torch.optim.AdamW(
        net.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd, eps=eps
    )


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



datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {"K": 2, "net": shallowCNN, "B": 2, "kernels": 8, "factor": 2}
datasets_params["SEGTHOR"] = {"K": 5, "net": ENet, "B": 8, "kernels": 8, "factor": 2}
datasets_params["SEGTHOR_CLEAN"] = {"K": 5, "net": ENet, "B": 8, "kernels": 8, "factor": 2}


def img_transform(img):
    img = img.convert("L")
    img = np.array(img)[np.newaxis, ...]
    img = img / 255  # max <= 1
    img = torch.tensor(img, dtype=torch.float32)
    return img


def gt_transform(K, img):
    img = np.array(img)[...]
    # The idea is that the classes are mapped to {0, 255} for binary cases
    # {0, 85, 170, 255} for 4 classes
    # {0, 51, 102, 153, 204, 255} for 6 classes
    # Very sketchy but that works here and that simplifies visualization
    img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
    img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
    img = class2one_hot(img, K=K)
    return img[0]


# -------------------- Data Augmentation --------------------

def build_augmentations(args):
    tokens = [s.lower() for s in args.augmentations]
    augs = []
    for t in tokens:
        if t == "hflip":
            augs.append(HFlip())
        elif t == "vflip":
            augs.append(VFlip())
        elif t == "rotate":
            augs.append(Rotate())
        elif t == "affine":
            augs.append(RandomAffine())
        else:
            raise ValueError(f"Unknown augmentation: {t}")
    return augs


# -------------------- K-Fold utilities --------------------

def create_or_load_folds_by_patient(img_dir, num_folds=5, seed=42, path=None):
    """Load or create K-fold splits by patient ID."""
    path = Path(path) if path is not None else None
    if path is None:
        path = Path(f"data/{num_folds}_folds.pkl")
    elif path.is_dir():
        path = path / f"{num_folds}_folds.pkl"

    patient_ids = sorted({f.name.split("_")[1] for f in img_dir.glob("Patient_*_*.png")})
    print(f">> Found {len(patient_ids)} unique patients")

    if path.exists():
        print(f">> Loading existing folds from {path}")
        with open(path, "rb") as f:
            folds = pickle.load(f)
    else:
        print(f">> Creating new {num_folds}-fold splits and saving to {path}")
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        indices = np.arange(len(patient_ids))
        folds = [
            ([patient_ids[i] for i in train_idx], [patient_ids[i] for i in val_idx])
            for train_idx, val_idx in kf.split(indices)
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(folds, f)
    return folds


def build_fold_dataset(fold_root: Path, train_ids, val_ids, base_img_dir: Path, base_gt_dir: Path):
    """Create train/val folders for one fold with symlinks to PNG slices."""
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


# -------------------- Setup & Training --------------------

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int, Path]:
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]["K"]
    kernels: int = datasets_params[args.dataset].get("kernels", 8)
    factor: int = datasets_params[args.dataset].get("factor", 2)
    net = datasets_params[args.dataset]["net"](1, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    optimizer = build_optimizer(net, args)

    B: int = datasets_params[args.dataset]["B"]

    train_set = SliceDataset(
        "train",
        args.dest,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        augmentations=build_augmentations(args),
        debug=args.debug,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=B,
        num_workers=5,
        worker_init_fn=worker_init_fn,
        shuffle=True,
    )

    val_set = SliceDataset(
        "val",
        args.dest,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        debug=args.debug,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=B,
        num_workers=5,
        worker_init_fn=worker_init_fn,
        shuffle=False,
    )

    args.dest.mkdir(parents=True, exist_ok=True)

    return net, optimizer, device, train_loader, val_loader, K, args.dest


def train_one_run(args, fold_dest: Path, train_ids, val_ids, base_img_dir: Path, base_gt_dir: Path):
    build_fold_dataset(fold_dest, train_ids, val_ids, base_img_dir, base_gt_dir)

    local_args = argparse.Namespace(**{**vars(args), "dest": fold_dest})
    net, optimizer, device, train_loader, val_loader, K, _ = setup(local_args)

    # Loss
    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))
    elif args.mode == "partial" and args.dataset == "SEGTHOR":
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])
    else:
        raise ValueError(args.mode, args.dataset)

    log_loss_tra = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0.0
    best_epoch: int = -1

    for e in range(args.epochs):
        for mode in ["train", "val"]:
            if mode == "train":
                net.train()
                cm = Dcm
                opt = optimizer
                desc = f">> Training   (Epoch {e})"
                loader = train_loader
                log_loss, log_dice = log_loss_tra, log_dice_tra
            else:
                net.eval()
                cm = torch.no_grad
                opt = None
                desc = f">> Validation (Epoch {e})"
                loader = val_loader
                log_loss, log_dice = log_loss_val, log_dice_val

            with cm():
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data["images"].to(device)
                    gt = data["gts"].to(device)

                    if opt:
                        opt.zero_grad()

                    assert 0 <= img.min() and img.max() <= 1
                    pred_logits = net(img)
                    pred_probs = F.softmax(pred_logits, dim=1)

                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j : j + img.shape[0], :] = dice_coef(pred_seg, gt)

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()

                    if opt:
                        loss.backward()
                        opt.step()

                    if mode == "val" and not args.skip_save_images:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(
                                predicted_class * mult, data["stems"], fold_dest / f"iter{e:03d}" / mode
                            )

                    j += img.shape[0]
                    tq_iter.set_postfix(
                        {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}", "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    )

        # Save logs
        np.save(fold_dest / "loss_tra.npy", log_loss_tra)
        np.save(fold_dest / "dice_tra.npy", log_dice_tra)
        np.save(fold_dest / "loss_val.npy", log_loss_val)
        np.save(fold_dest / "dice_val.npy", log_dice_val)
        current_dice = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved Dice {best_dice:05.3f} → {current_dice:05.3f} (Epoch {e})"
            print(message)
            best_dice = current_dice
            best_epoch = e

            with open(fold_dest / "best_epoch.txt", "w") as f:
                f.write(message)

            # When tuning, images may be disabled (skip_save_images=True), so the src may not exist.
            src = fold_dest / f"iter{e:03d}"
            best_folder = fold_dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            if (not args.skip_save_images) and src.exists():
                copytree(src, best_folder)
            else:
                # Create an empty marker folder so downstream scripts don't fail
                best_folder.mkdir(parents=True, exist_ok=True)

            torch.save(net, fold_dest / "bestmodel.pkl")
            torch.save(net.state_dict(), fold_dest / "bestweights.pt")

            print(f">> Saved best predictions to {best_folder}")

    return best_dice, best_epoch


def runTraining(args):
    print(f">>> Setting up {args.folds}-fold training on {args.dataset} (AdamW)")

    base_img_dir = Path("data") / args.dataset / "img"
    base_gt_dir = Path("data") / args.dataset / "gt"
    folds = create_or_load_folds_by_patient(base_img_dir, num_folds=args.folds, path=args.fold_path)

    fold_results = []

    # loop through folds
    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        fold_num = fold_idx + 1
        if args.fold_index is not None and fold_num != args.fold_index:
            continue

        print(f"\n--------------------- Fold {fold_num}/{args.folds} ---------------------")
        fold_dest = args.dest / f"fold_{fold_num}"
        fold_dest.mkdir(parents=True, exist_ok=True)

        best_dice, best_epoch = train_one_run(args, fold_dest, train_ids, val_ids, base_img_dir, base_gt_dir)
        fold_results.append(best_dice)

    if fold_results:
        avg = float(np.mean(fold_results))
        std = float(np.std(fold_results))
        print(f"\n>> AdamW: Average Dice across folds: {avg:.4f} ± {std:.4f}")


# -------------------- Optuna Hyperparameter Tuning (AdamW) --------------------

def tune_hyperparams(args):
    if optuna is None:
        raise RuntimeError(
            "Optuna is not installed. Please install it with `pip install optuna` to use --tune."
        )

    base_img_dir = Path("data") / args.dataset / "img"
    base_gt_dir = Path("data") / args.dataset / "gt"
    folds = create_or_load_folds_by_patient(base_img_dir, num_folds=args.folds, path=args.fold_path)

    storage = None
    if args.study_storage:
        args.dest.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{(args.dest / args.study_storage).resolve()}"

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=min(5, max(1, args.n_trials // 4))) if args.use_pruner else None

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(storage),
    )

    def objective(trial: optuna.Trial) -> float:
        # AdamW hyperparameters to tune
        lr = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
        beta1 = trial.suggest_float("beta1", 0.85, 0.99)
        beta2 = trial.suggest_float("beta2", 0.95, 0.9999)
        eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)

        trial_args_dict = {**vars(args)}
        trial_args_dict.update(
            dict(
                lr=lr,
                weight_decay=weight_decay,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                epochs=args.tune_epochs,
                skip_save_images=True,
            )
        )
        trial_args = argparse.Namespace(**trial_args_dict)

        fold_scores = []
        for fold_idx, (train_ids, val_ids) in enumerate(folds):
            fold_num = fold_idx + 1
            if args.fold_index is not None and fold_num != args.fold_index:
                continue
            if args.max_folds_in_tune and fold_num > args.max_folds_in_tune:
                break

            opt_dest = args.dest / f"tune_adamw" / f"trial_{trial.number:04d}" / f"fold_{fold_num}"
            opt_dest.mkdir(parents=True, exist_ok=True)

            best_dice, _ = train_one_run(trial_args, opt_dest, train_ids, val_ids, base_img_dir, base_gt_dir)
            fold_scores.append(best_dice)

            trial.report(float(best_dice), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores)) if fold_scores else 0.0

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    print("\n===================== Optuna Results (AdamW) =====================")
    print(f"Best value (mean Dice): {study.best_value:.5f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")

    best_params_path = args.dest / "best_params_optuna_adamw.pkl"
    with open(best_params_path, "wb") as f:
        pickle.dump(study.best_params, f)
    print(f"Saved best params to: {best_params_path}")

    if args.final_train_after_tune:
        best = study.best_params
        final_args_dict = {**vars(args)}
        final_args_dict.update(
            dict(
                lr=best["lr"],
                weight_decay=best["weight_decay"],
                beta1=best["beta1"],
                beta2=best["beta2"],
                eps=best["eps"],
                epochs=args.epochs,
                skip_save_images=False,
            )
        )
        final_args = argparse.Namespace(**final_args_dict)
        runTraining(final_args)


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument(
        "--dataset",
        default="SEGTHOR_CLEAN",
        choices=datasets_params.keys(),
        help=(
            "The dataset folder should contain *all* patient images together under 'data/<dataset>/img' and 'data/<dataset>/gt', "
            "not pre-divided into train/val. K-fold splitting will automatically handle train/val separation."
        ),
    )

    parser.add_argument("--mode", default="full", choices=["partial", "full"])
    parser.add_argument(
        "--dest", type=Path, required=True, help="Destination directory to save the results (predictions and weights)."
    )

    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Keep only a fraction (10 samples) of the datasets, to test the logics around epochs and logging easily.",
    )

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--fold_path",
        type=str,
        default="data",
        help=(
            "Path or directory to load/save the folds file. If a directory is given, the file will be named {num_folds}_folds.pkl."
        ),
    )
    parser.add_argument(
        "--fold_index",
        type=int,
        default=None,
        help="If set (1-based), train only this fold instead of all folds.",
    )
    parser.add_argument(
        "--augmentations",
        nargs="*",
        default=[],
        help="List of data augmentation to use during training. Available: HFlip, VFlip, Rotate, Affine.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # AdamW hyperparameters
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for AdamW.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for AdamW.")

    parser.add_argument("--skip_save_images", action="store_true")

    # --- Optuna HPO flags (AdamW) ---
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning for AdamW.")
    parser.add_argument("--n_trials", type=int, default=25, help="Number of Optuna trials.")
    parser.add_argument("--tune_epochs", type=int, default=8, help="Epochs per trial during tuning.")
    parser.add_argument(
        "--max_folds_in_tune", type=int, default=1, help="Max number of folds to evaluate per trial."
    )
    parser.add_argument("--timeout", type=int, default=None, help="Timeout (seconds) for tuning.")
    parser.add_argument(
        "--study_storage",
        type=str,
        default="optuna_study.sqlite3",
        help="SQLite filename to persist the study under dest/. Set empty to disable.",
    )
    parser.add_argument("--study_name", type=str, default="seg_hpo_adamw", help="Name of the Optuna study.")
    parser.add_argument("--use_pruner", action="store_true", help="Enable Optuna MedianPruner.")
    parser.add_argument("--final_train_after_tune", action="store_true", help="Run full training with best params.")

    # LR search bounds
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Min LR for tuning (log scale).")
    parser.add_argument("--lr_max", type=float, default=5e-3, help="Max LR for tuning (log scale).")

    args = parser.parse_args()

    set_seed(args.seed)

    pprint(args)

    if args.tune:
        tune_hyperparams(args)
    else:
        runTraining(args)


if __name__ == "__main__":
    main()
