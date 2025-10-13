import argparse
from pathlib import Path
from functools import partial
from typing import Callable
from utils import tqdm_

import nibabel as nib
import numpy as np
import re
import skimage.io
from skimage.transform import resize

from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
from skimage import measure

resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)

# Post-processing functions
def keep_largest_volume(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    return (labels == largest_label).astype(np.uint8)

def smooth_mask(mask: np.ndarray, structure_size: int=3) -> np.ndarray:
    structure = np.ones((structure_size,) * mask.ndim, dtype=bool)
    mask = binary_opening(mask, structure=structure)
    mask = binary_closing(mask, structure=structure)
    return mask.astype(np.uint8)

def post_process(volume: np.ndarray) -> np.ndarray:
    processed = np.zeros_like(volume)
    esophagus, heart, trachea, aorta = 1, 2, 3, 4

    for organ in [esophagus, heart, trachea, aorta]:
        organ_mask = (volume == organ).astype(np.uint8)

        if organ in [heart]:
            organ_mask = keep_largest_volume(organ_mask)

        if organ in [heart, trachea, aorta]:
            organ_mask = smooth_mask(organ_mask, structure_size=2)

        if organ in [heart]:
            organ_mask = binary_fill_holes(organ_mask)

        processed[(organ_mask > 0) & (processed == 0)] = organ

    processed[volume == esophagus] = esophagus

    return processed

def store_nifti(volume, dest_path, header=None, affine=None):
    nifti_img = nib.Nifti1Image(volume, affine=np.diag([1, 1, 1, 1]) if affine is None else affine,
                                header=header)

    nib.save(nifti_img, dest_path)


def stitch_slices(data_folder, dest_folder, num_classes, grp_regex, source_scan_pattern, postprocess=True):
    patient_regex = re.compile(grp_regex)

    id_set = set()
    source_shapes = dict()
    stitched_volumes = dict()
    headers = dict()
    affines = dict()

    px_multiplier = int(255 / (num_classes - 1))

    for pred_slice in tqdm_(data_folder.iterdir(), desc="Processing slices"):
        match = patient_regex.fullmatch(pred_slice.stem)

        if pred_slice.is_file():
            # Extract patient ID from the match
            patient_id = match.group(1)
            image_id = int(match.group(0).split('_')[-1])

            if patient_id not in id_set:
                id_set.add(patient_id)
                pattern_path = source_scan_pattern.format(id_=patient_id)
                source_volume = nib.load(pattern_path)
                source_shapes[patient_id] = source_volume.shape
                headers[patient_id] = source_volume.header
                affines[patient_id] = source_volume.affine
                stitched_volumes[patient_id] = np.zeros(source_shapes[patient_id], dtype=np.uint8)

            # Load the png slice image as a numpy array
            slice_img = skimage.io.imread(pred_slice)
            slice_img = (slice_img / px_multiplier).astype(np.uint8)

            stitched_volumes[patient_id][:, :, image_id] = resize_(slice_img, source_shapes[patient_id][:2], 
                                                                   order=0).astype(np.uint8)

    dest_folder.mkdir(parents=True, exist_ok=True)

    for patient_id, volume in stitched_volumes.items():
        if postprocess:
            volume = post_process(volume)
        dest_path = dest_folder / f"{patient_id}.nii.gz"
        store_nifti(volume, dest_path, header=headers[patient_id], affine=affines[patient_id])

    print(f"Stitched and post-processed for patients: {id_set}")
    print(f"Stitched volumes saved to {dest_folder}")


def main(args):
    data_folder = Path(args.data_folder)
    dest_folder = Path(args.dest_folder)
    num_classes = args.num_classes
    grp_regex = args.grp_regex
    source_scan_pattern = args.source_scan_pattern

    assert data_folder.exists()

    stitch_slices(data_folder, dest_folder, num_classes, grp_regex, source_scan_pattern, postprocess=True)


def get_args():
    parser = argparse.ArgumentParser(description='Stitching and post-processing parameters')

    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--dest_folder', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--grp_regex', type=str, required=True, default=r"(Patient_\d\d)_\d\d\d\d")
    parser.add_argument('--source_scan_pattern', type=str, required=True)

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)

