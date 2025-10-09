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


import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from functools import partial

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

from data_augmentation import HFlip, VFlip, Rotate, RandomAffine

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def img_transform(img):
    img = img.convert('L')
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
    path = Path(path)
    if path.is_dir():
        path = path / f"{num_folds}_folds.pkl"

    # if no path is given create path name
    elif path is None:
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
    # loop through train and val ids and create destination folders
    for subset, ids in [('train', train_ids), ('val', val_ids)]:
        for sub in ['img', 'gt']:
            dest_dir = fold_root / subset / sub
            dest_dir.mkdir(parents=True, exist_ok=True)

            # get actual source directory for current subset
            src_dir = base_img_dir if sub == 'img' else base_gt_dir

            # loop through pateint IDs and create symlinks for all their slices
            for pid in ids:
                pattern = re.compile(f"Patient_{pid}_\\d{{4}}\\.png")
                for file in src_dir.iterdir():
                    if pattern.match(file.name):
                        target = dest_dir / file.name
                        if not target.exists():
                            os.symlink(file.resolve(), target)


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = args.dest

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             augmentations=build_augmentations(args),
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
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
                message = f">>> Fold {fold_idx + 1}: Improved Dice {best_dice:05.3f} → {current_dice:05.3f} (Epoch {e})"
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
                             "Available: HFlip, VFlip, Rotate, Affine.")

    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
