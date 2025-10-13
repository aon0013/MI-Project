#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

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

#!/usr/bin/env python3
# MIT License
# (header unchanged)

from pathlib import Path
from typing import Callable, Union, List, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

# NEW: for consistent affine on 2.5D
from torchvision.transforms import InterpolationMode, RandomAffine
from torchvision.transforms import functional as TF


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


def _parse_pid_sid(p: Path) -> Tuple[str, int]:
    """'Patient_01_0034.png' -> ('Patient_01', 34) (splits on the LAST underscore)"""
    stem = p.stem
    if "_" not in stem:
        return stem, 0
    pid, sid = stem.rsplit("_", 1)
    try:
        sid_i = int(sid)
    except ValueError:
        sid_i = 0
    return pid, sid_i


# NEW: one affine applied identically to all context planes + center GT
def apply_consistent_affine_2p5d(
    img_tensor: Tensor,        # (C,H,W) float in [0,1]
    gt_pil: Image.Image,       # PIL (center mask)
    *,
    degrees: float = 10.0,
    translate: Tuple[float, float] = (0.05, 0.05),   # fraction of size
    scale_ranges: Tuple[float, float] = (0.95, 1.05),
    shear: float = 5.0,
) -> tuple[Tensor, Image.Image]:
    H, W = img_tensor.shape[-2], img_tensor.shape[-1]
    angle, translations, scale, shear_vals = RandomAffine.get_params(
        degrees=(-degrees, degrees),
        translate=translate,
        scale_ranges=scale_ranges,
        shears=(-shear, shear, -shear, shear),
        img_size=[H, W],
    )
    # Apply identical params to all channels (bilinear) and to GT (nearest)
    img_aug = TF.affine(
        img_tensor, angle=angle, translate=translations, scale=scale, shear=shear_vals,
        interpolation=InterpolationMode.BILINEAR, fill=0.0
    )
    gt_aug = TF.affine(
        gt_pil, angle=angle, translate=translations, scale=scale, shear=shear_vals,
        interpolation=InterpolationMode.NEAREST, fill=0
    )
    return img_aug, gt_aug


class SliceDataset(Dataset):
    def __init__(self,
                 subset: str,
                 root_dir,
                 img_transform: Callable | None = None,
                 gt_transform: Callable | None = None,
                 augmentations: List[Callable] | None = None,
                 equalize: bool = False,
                 debug: bool = False,
                 half_ctx: int = 0,
                 # NEW: 2.5D affine controls
                 affine_2p5d: bool = True,
                 affine_degrees: float = 10.0,
                 affine_translate: Tuple[float, float] = (0.05, 0.05),
                 affine_scale: Tuple[float, float] = (0.95, 1.05),
                 affine_shear: float = 5.0):
        """
        half_ctx = 0 -> standard 2D (C=1)
        half_ctx = 1 -> 2.5D with 3 channels (center ±1)
        half_ctx = 2 -> 2.5D with 5 channels (center ±2), etc.

        When half_ctx > 0 and affine_2p5d=True, a single RandomAffine is sampled
        and applied consistently to all context channels and to the center GT.
        """
        self.root_dir: str = root_dir
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.augmentations: List[Callable] = augmentations if augmentations is not None else []
        self.equalize: bool = equalize
        self.half_ctx: int = int(half_ctx)

        # 2.5D affine knobs
        self.affine_2p5d: bool = bool(affine_2p5d)
        self.affine_degrees: float = float(affine_degrees)
        self.affine_translate: Tuple[float, float] = tuple(affine_translate)
        self.affine_scale: Tuple[float, float] = tuple(affine_scale)
        self.affine_shear: float = float(affine_shear)

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        # Variant logic (keep your original behavior for 2D)
        self.variants_per = 1 if self.test_mode else (1 + len(self.augmentations))

        # --- 2.5D indexing setup ---
        self.img_paths: List[Path] = [ip for ip, _ in self.files]
        self.gt_paths: List[Path | None] = [gp for _, gp in self.files]
        self.index_pid_sid: List[Tuple[str, int]] = [_parse_pid_sid(p) for p in self.img_paths]
        self.lookup: Dict[Tuple[str, int], Path] = {k: v for k, v in zip(self.index_pid_sid, self.img_paths)}

        from collections import defaultdict
        bounds = defaultdict(lambda: [10**9, -10**9])
        for (pid, sid) in self.index_pid_sid:
            lo, hi = bounds[pid]
            bounds[pid] = [min(lo, sid), max(hi, sid)]
        self.bounds = dict(bounds)

        # To keep stacks consistent, we disable arbitrary 2D augs when half_ctx>0
        # (You can still enable consistent affine via affine_2p5d=True.)
        if self.half_ctx > 0 and len(self.augmentations) > 0 and not self.test_mode:
            print(">> [SliceDataset] half_ctx>0: disabling generic PIL augs "
                  "(2.5D stacks must remain slice-aligned). Use affine_2p5d=True if needed.")
            self.augmentations = []
            self.variants_per = 1

        print(f">> Created {subset} dataset with {len(self)} images... "
              f"(half_ctx={self.half_ctx}, variants_per={self.variants_per}, affine_2p5d={self.affine_2p5d})")

    def __len__(self):
        return len(self.files) * self.variants_per

    def _map_index(self, index):
        base_idx = index // self.variants_per
        variant_id = index % self.variants_per
        return base_idx, variant_id

    def _load_png_u8(self, p: Path) -> np.ndarray:
        return np.array(Image.open(p).convert("L"), dtype=np.uint8)  # (H,W)

    def _get_ctx_stack(self, pid: str, sid: int) -> np.ndarray:
        """Return (C,H,W) uint8 stack with context slices; clamp at patient edges."""
        lo, hi = self.bounds[pid]
        planes = []
        for off in range(-self.half_ctx, self.half_ctx + 1):
            s = sid + off
            if s < lo: s = lo
            if s > hi: s = hi
            planes.append(self._load_png_u8(self.lookup[(pid, s)]))
        return np.stack(planes, axis=0)  # (C,H,W)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        base_idx, variant_id = self._map_index(index)
        img_path, gt_path = self.files[base_idx]
        pid, sid = self.index_pid_sid[base_idx]

        # --- Build image tensor (C,H,W) float in [0,1] ---
        if self.half_ctx > 0:
            arr = self._get_ctx_stack(pid, sid)        # (C,H,W) uint8
            img = torch.tensor(arr, dtype=torch.float32) / 255.0

            # If training with masks, load center GT for consistent affine (if requested)
            gt_pil = None
            if not self.test_mode and gt_path is not None:
                gt_pil = Image.open(gt_path)

                if self.affine_2p5d:
                    img, gt_pil = apply_consistent_affine_2p5d(
                        img, gt_pil,
                        degrees=self.affine_degrees,
                        translate=self.affine_translate,
                        scale_ranges=self.affine_scale,
                        shear=self.affine_shear,
                    )

            data: dict[str, Union[Tensor, int, str]] = {"stems": img_path.stem, "images": img}

            if gt_pil is not None:
                gt: Tensor = self.gt_transform(gt_pil)
                _, W, H = img.shape
                K, _, _ = gt.shape
                assert gt.shape == (K, W, H)
                data["gts"] = gt

            return data

        else:
            # 2D path (preserve your augmentation flow on PIL)
            img_pil = Image.open(img_path)
            data: dict[str, Union[Tensor, int, str]] = {"stems": img_path.stem}

            if self.test_mode:
                data["images"] = self.img_transform(img_pil)
                return data

            gt_pil = Image.open(gt_path)

            if variant_id == 0:
                img_aug_pil, gt_aug_pil = img_pil, gt_pil
            else:
                offset = 1
                aug_idx = variant_id - offset
                img_aug_pil, gt_aug_pil = self.augmentations[aug_idx](img_pil.copy(), gt_pil.copy())

            img: Tensor = self.img_transform(img_aug_pil)
            gt: Tensor = self.gt_transform(gt_aug_pil)

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data["images"] = img
            data["gts"] = gt
            return data
