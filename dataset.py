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

from pathlib import Path
from typing import Callable, Union

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset


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


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augmentations=None, equalize=False, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentations: List[Callable] = augmentations if augmentations is not None else []
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        self.variants_per = 1 + (0 if self.test_mode else len(self.augmentations))
        if self.test_mode:  # no augs in test mode
            self.variants_per = 1

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files) * self.variants_per

    def _map_index(self, index):
        base_idx = index // self.variants_per
        variant_id = index % self.variants_per
        return base_idx, variant_id

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        base_idx, variant_id = self._map_index(index)
        img_path, gt_path = self.files[base_idx]

        img_pil = Image.open(img_path)

        data_dict = {"stems": img_path.stem}

        if self.test_mode:
            data_dict["images"] = self.img_transform(img_pil)
            return data_dict

        gt_pil = Image.open(gt_path)

        if variant_id == 0:
            img, gt = img_pil, gt_pil
        else:
            offset = 1
            aug_idx = variant_id - offset
            img, gt = self.augmentations[aug_idx](img_pil.copy(), gt_pil.copy())

        img: Tensor = self.img_transform(img)
        gt: Tensor = self.gt_transform(gt)

        _, W, H = img.shape
        K, _, _ = gt.shape
        assert gt.shape == (K, W, H)

        data_dict["gts"] = gt
        data_dict["images"] = img

        return data_dict
