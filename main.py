#!/usr/bin/env python3

# MIT License
# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

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
from operator import itemgetter
from shutil import copytree, rmtree
import pickle
import os
import shutil
import re

import random
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
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
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)
from losses import CrossEntropy

from data_augmentation import HFlip, VFlip, Rotate, RandomAffine, Elastic2D

def _link_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink if possible, else hardlink, else copy (Windows-safe)."""
    if dst.exists():
        return
    try:
        os.symlink(src, dst)         # works on Linux/macOS; on Win only with Dev Mode/admin
        return
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(src, dst)            # hardlink (NTFS), no admin needed
        return
    except OSError:
        pass
    shutil.copy2(src, dst)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def img_transform(img):
    # If the dataset hands us a (C,H,W) float tensor in [0,1] (2.5D path), keep it.
    if isinstance(img, torch.Tensor):
        return img
    # Fallback for PIL inputs (old 2D behavior)
    img = img.convert('L')
    img = np.array(img, dtype=np.float32)[np.newaxis, ...] / 255.0
    return torch.tensor(img, dtype=torch.float32)

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
    norm = lambda s: s.lower()
    tokens = [norm(s) for s in args.augmentations]

    augs = []
    for t in tokens:
        if t == 'hflip':
            augs.append(HFlip())
        elif t == 'vflip':
            augs.append(VFlip())
        elif t == 'rotate':
            augs.append(Rotate())
        elif t == 'affine':
            augs.append(RandomAffine())
        elif t == 'elastic':
           augs.append(Elastic2D())
        else:
            raise ValueError(f"Unknown augmentation: {t}")
    return augs

# -------------------- add K-Fold splits --------------------
def create_or_load_folds_by_patient(img_dir, num_folds=5, seed=42, path=None):
    """
    This fumnction loads in pre-sorted K-fold splits based on patient IDs, or
    creates the folds if not yet created. Each fold contains unique patients.
    """
    # construct file name automatically if directory is passed
    path = Path(path) if path is not None else None
    if path is not None and path.is_dir():
        path = path / f"{num_folds}_folds.pkl"

    # if no path is given create path name
    if path is None:
        path = Path(f"data/{num_folds}_folds.pkl")

    patient_ids = sorted({f.name.split('_')[1] for f in img_dir.glob("Patient_*_*.png")})
    print(f">> Found {len(patient_ids)} unique patients")

    if path.exists():
        print(f">> Loading existing folds from {path}")
        with open(path, "rb") as f:
            folds = pickle.load(f)
    else:
        print(f">> Creating new {num_folds}-fold splits and saving to {path}")
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        indices = np.arange(len(patient_ids))
        folds = [( [patient_ids[i] for i in train_idx],
                   [patient_ids[i] for i in val_idx]) for train_idx, val_idx in kf.split(indices)]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(folds, f)
    return folds

# --- build per-fold train/val folders to remain consistent with expected input to dataset.py ---
def build_fold_dataset(fold_root: Path, train_ids, val_ids, base_img_dir: Path, base_gt_dir: Path):
    """
    Create train/val folders for one fold with symlinks to PNG slices.

    Args:
        fold_root: destination (e.g., results/kfold_baseline/fold_1)
        train_ids: list of patient IDs for training (e.g., ['01', '02', ...])
        val_ids: list of patient IDs for validation
        base_img_dir, base_gt_dir: source directories with all img/gt PNGs
    """

    for subset, ids in [('train', train_ids), ('val', val_ids)]:
        for sub in ['img', 'gt']:
            dest_dir = fold_root / subset / sub
            dest_dir.mkdir(parents=True, exist_ok=True)

            src_dir = base_img_dir if sub == 'img' else base_gt_dir

            # Each patient ID is like '01', '02', ...
            # Files are named 'Patient_XX_####.png'
            for pid in ids:
                pattern = f"Patient_{pid}_*.png"
                for file in src_dir.glob(pattern):
                    target = dest_dir / file.name
                    _link_or_copy(file.resolve(), target)

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2

    in_ch = 2 * args.half_ctx + 1
    NetClass = datasets_params[args.dataset]['net']
    try:
        net = NetClass(in_ch, K, kernels=kernels, factor=factor)
    except TypeError:
        net = NetClass(in_ch, K)

    net.init_weights()
    net.to(device)

    # AdamW 
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = args.dest

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             augmentations=build_augmentations(args),
                             debug=args.debug,
                             half_ctx=args.half_ctx)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              worker_init_fn=worker_init_fn,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug,
                           half_ctx=args.half_ctx)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            worker_init_fn=worker_init_fn,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return net, optimizer, device, train_loader, val_loader, K

def runTraining(args):
    print(f">>> Setting up {args.folds}-fold training on {args.dataset}")

    # load consistent folds
    root_dir = Path("data") / args.dataset

    # set base directories and get folds
    base_img_dir = Path("data") / args.dataset / "img"
    base_gt_dir  = Path("data") / args.dataset / "gt"
    folds = create_or_load_folds_by_patient(base_img_dir, num_folds=args.folds, path=args.fold_path)

    fold_results = []

    # loop through folds
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_num = fold_idx + 1
        if args.fold_index is not None and fold_num != args.fold_index:
            continue  # skip other folds

        print(f"\n===================== Fold {fold_idx + 1}/{args.folds} =====================")

        # define directories for current fold and build dataset structure
        fold_dest = args.dest / f"fold_{fold_idx + 1}"
        fold_dest.mkdir(parents=True, exist_ok=True)
        train_ids, val_ids = train_idx, val_idx
        build_fold_dataset(fold_dest, train_ids, val_ids, base_img_dir, base_gt_dir)

        # initialise for current fold
        net, optimizer, device, train_loader, val_loader, K = setup(argparse.Namespace(**{**vars(args), "dest": fold_dest}))

        # Choose loss
        if args.mode == "full":
            loss_fn = CrossEntropy(idk=list(range(K)))
        elif args.mode == "partial" and args.dataset == 'SEGTHOR':
            loss_fn = CrossEntropy(idk=[0, 1, 3, 4])
        else:
            raise ValueError(args.mode, args.dataset)

        # Notice one has the length of the _loader_, and the other one of the _dataset_
        log_loss_tra = torch.zeros((args.epochs, len(train_loader)))
        log_dice_tra = torch.zeros((args.epochs, len(train_loader.dataset), K))
        log_loss_val = torch.zeros((args.epochs, len(val_loader)))
        log_dice_val = torch.zeros((args.epochs, len(val_loader.dataset), K))

        best_dice: float = 0

        # Epoch loop
        for e in range(args.epochs):
            for mode in ['train', 'val']:
                if mode == 'train':
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

                with cm(): # Either dummy context manager, or the torch.no_grad for validation
                    j = 0
                    tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                    for i, data in tq_iter:
                        img = data['images'].to(device)
                        gt = data['gts'].to(device)

                        if opt:  # So only for training
                            opt.zero_grad()

                        # Sanity tests to see we loaded and encoded the data correctly
                        assert 0 <= img.min() and img.max() <= 1
                        B, _, W, H = img.shape

                        pred_logits = net(img)
                        pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                        # Metrics computation, not used for training
                        pred_seg = probs2one_hot(pred_probs)
                        log_dice[e, j:j + img.shape[0], :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                        loss = loss_fn(pred_probs, gt)
                        log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                        if opt:  # Only for training
                            loss.backward()
                            opt.step()

                        if mode == 'val':
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=UserWarning)
                                predicted_class: Tensor = probs2class(pred_probs)
                                mult: int = 63 if K == 5 else (255 / (K - 1))
                                save_images(predicted_class * mult,
                                            data['stems'],
                                            fold_dest / f"iter{e:03d}" / mode)

                        # changed B to img.shape[0] here and above to be safe if last batch if smaller than B
                        j += img.shape[0]
                        # Removed the printing of each dice score
                        # For the DSC average: do not take the background class (0) into account:
                        tq_iter.set_postfix({
                            "Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                            "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"
                        })

            # I save it at each epochs, in case the code crashes or I decide to stop it early
            np.save(fold_dest / "loss_tra.npy", log_loss_tra)
            np.save(fold_dest / "dice_tra.npy", log_dice_tra)
            np.save(fold_dest / "loss_val.npy", log_loss_val)
            np.save(fold_dest / "dice_val.npy", log_dice_val)

            current_dice = log_dice_val[e, :, 1:].mean().item()
            if current_dice > best_dice:
                message = f">>> Fold {fold_idx + 1}: Improved Dice {best_dice:05.3f} -> {current_dice:05.3f} (Epoch {e})"
                print(message)
                best_dice = current_dice

                with open(fold_dest / "best_epoch.txt", 'w') as f:
                    f.write(message)

                best_folder = fold_dest / "best_epoch"
                if best_folder.exists():
                    rmtree(best_folder)
                copytree(fold_dest / f"iter{e:03d}", best_folder)

                torch.save(net, fold_dest / "bestmodel.pkl")
                torch.save(net.state_dict(), fold_dest / "bestweights.pt")

                print(f">> Saved best predictions to {best_folder}")

        fold_results.append(best_dice)

    print(f"\n>> Average Dice across folds: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

# optuna tuning 
def _optuna_objective(args, folds, base_img_dir: Path, base_gt_dir: Path):
    def objective(trial):
        lr = trial.suggest_float('lr', args.lr_min, args.lr_max, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        beta1 = trial.suggest_float('beta1', 0.85, 0.99)
        beta2 = trial.suggest_float('beta2', 0.95, 0.9999)
        eps   = trial.suggest_float('eps', 1e-9, 1e-7, log=True)
        scheduler = trial.suggest_categorical('scheduler', [None, 'cosine', 'plateau'])

        fold_scores = []
        for fold_idx, (train_ids, val_ids) in enumerate(folds):
            if args.max_folds_in_tune and fold_idx >= args.max_folds_in_tune:
                break

            fold_dest = args.dest / f"trial_{trial.number:04d}" / f"fold_{fold_idx+1}"
            fold_dest.mkdir(parents=True, exist_ok=True)
            build_fold_dataset(fold_dest, train_ids, val_ids, base_img_dir, base_gt_dir)

            local = argparse.Namespace(**{
                **vars(args),
                "dest": fold_dest,
                "lr": lr,
                "weight_decay": weight_decay,
                "beta1": beta1,
                "beta2": beta2,
                "eps": eps
            })

            net, optimizer, device, train_loader, val_loader, K = setup(local)
            loss_fn = CrossEntropy(idk=list(range(K)))

            if scheduler == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.tune_epochs))
            elif scheduler == "plateau":
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
            else:
                sched = None

            best = 0.0
            for e in range(args.tune_epochs):
                # train
                net.train()
                for i, data in tqdm_(enumerate(train_loader), total=len(train_loader), desc=f">> Train (E{e})"):
                    img, gt = data['images'].to(device), data['gts'].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = net(img)
                    probs = torch.softmax(logits, dim=1)
                    loss = loss_fn(probs, gt)
                    loss.backward()
                    optimizer.step()

                # validate
                net.eval()
                vals = []
                with torch.no_grad():
                    for i, data in tqdm_(enumerate(val_loader), total=len(val_loader), desc=f">> Val   (E{e})"):
                        img, gt = data['images'].to(device), data['gts'].to(device)
                        logits = net(img)
                        probs = torch.softmax(logits, dim=1)
                        hard = probs2one_hot(probs)
                        d = dice_coef(hard, gt)
                        if d.dim() == 2:
                            d = d.mean(dim=0)
                        if args.ignore_bg_in_val and d.numel() > 1:
                            vals.append(d[1:].mean().item())
                        else:
                            vals.append(d.mean().item())

                score = float(np.mean(vals)) if vals else 0.0
                if sched is not None:
                    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        sched.step(score)
                    else:
                        sched.step()

                trial.report(score, step=e)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                best = max(best, score)

            fold_scores.append(best)

        return float(np.mean(fold_scores)) if fold_scores else 0.0
    return objective

def tune_hyperparams(args):
    if optuna is None:
        raise RuntimeError("Optuna is not installed or failed to import. Install with `pip install optuna`.")

    base_img_dir = Path("data") / args.dataset / "img"
    base_gt_dir  = Path("data") / args.dataset / "gt"
    folds = create_or_load_folds_by_patient(base_img_dir, num_folds=args.folds, path=args.fold_path)

    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=3) if args.use_pruner else None
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    study.optimize(_optuna_objective(args, folds, base_img_dir, base_gt_dir),
                   n_trials=args.n_trials, timeout=args.timeout)

    results_dir = Path.home() / "MIProject" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_params_path = results_dir / "best_params_optuna_adamw.pkl"
    with open(best_params_path, "wb") as f:
        pickle.dump(study.best_params, f)

    best_txt_path = results_dir / "best_trial_info_adamw.txt"
    with open(best_txt_path, "w") as f:
        f.write(f"Best value (mean Dice): {study.best_value:.6f}\nBest parameters:\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTrials finished: {len(study.trials)}\n")

    trials_csv_path = results_dir / "optuna_all_trials_adamw.csv"
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(trials_csv_path, index=False)

    trials_json_path = results_dir / "optuna_all_trials_adamw.json"
    import json
    with open(trials_json_path, "w") as f:
        json.dump([t.params | {"value": t.value, "state": t.state.name} for t in study.trials], f, indent=2)

    print("\n=== Optuna Results Saved ===")
    print(f"Best params (pickle): {best_params_path}")
    print(f"Summary text:          {best_txt_path}")
    print(f"All trials CSV:        {trials_csv_path}")
    print(f"All trials JSON:       {trials_json_path}")
    print("============================\n")

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='SEGTHOR_CLEAN', choices=datasets_params.keys(),
                        help="The dataset folder should contain *all* patient images together "
                        "under 'data/<dataset>/img' and 'data/<dataset>/gt', "
                        "not pre-divided into train/val. "
                        "K-fold splitting will automatically handle train/val separation.")

    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")

    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--fold_path', type=str, default='data',
                        help="Path or directory to load/save the folds file. "
                             "If a directory is given, the file will be named {num_folds}_folds.pkl.")
    parser.add_argument('--fold_index', type=int, default=None,
                        help="If set (1-based), train only this fold instead of all folds.")
    parser.add_argument('--augmentations', nargs='*', default=[],
                        help="List of data augmentation to use during training. "
                             "Available: HFlip, VFlip, Rotate, Affine, Elastic.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument('--half_ctx', type=int, default=0,
                        help='Neighbors per side for 2.5D (0=2D, 1→3ch, 2→5ch, …)')

    # AdamW params 
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)

    # Validation behavior
    parser.add_argument('--ignore_bg_in_val', action='store_true',
                        help='Average Dice over classes 1..K-1 on validation')

    # Optuna tuning switches
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--n_trials', type=int, default=25)
    parser.add_argument('--tune_epochs', type=int, default=8)
    parser.add_argument('--max_folds_in_tune', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--use_pruner', action='store_true')
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--lr_max', type=float, default=5e-3)
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'cosine', 'plateau'])

    args = parser.parse_args()

    set_seed(args.seed)

    pprint(args)

    if args.tune:
        tune_hyperparams(args)
    else:
        runTraining(args)

if __name__ == '__main__':
    main()