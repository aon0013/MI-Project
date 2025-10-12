import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
from skimage import measure
import os

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
    heart, aorta, trachea, esophagus = 1, 2, 3, 4

    for organ in [heart, aorta, trachea, esophagus]:
        organ_mask = (volume == organ).astype(np.uint8)

        if organ in [aorta]:
            organ_mask = keep_largest_volume(organ_mask)

        if organ in [aorta, trachea, esophagus]:
            organ_mask = smooth_mask(organ_mask, structure_size=2)

        if organ in [aorta]:
            organ_mask = binary_fill_holes(organ_mask)

        processed[(organ_mask > 0) & (processed == 0)] = organ

    processed[volume == heart] = heart

    return processed

# Process predicted volumes
def process_predicted_volumes(input_folder: str, output_folder: str):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    nii_files = list(input_folder.glob("*.nii*"))

    for nii_path in nii_files:
        # Load predicted volume
        img = nib.load(str(nii_path))
        volume = img.get_fdata().astype(np.uint8)
        affine = img.affine

        # Apply post-processing
        volume_processed = post_process(volume)

        # Save processed volume
        out_path = output_folder / nii_path.name
        nib.save(nib.Nifti1Image(volume_processed, affine), str(out_path))
        print(f"Processed {nii_path.name} → {out_path.name}")


if __name__ == "__main__":
    # input_dir = "pred_volumes/volumes_1"
    # output_dir = "testing/pred/scen5/volumes_1"
    # process_predicted_volumes(input_dir, output_dir)

    base_input_dir = "pred_volumes"
    base_output_dir = "testing/pred/scen5"

    # Create output base directory
    os.makedirs(base_output_dir, exist_ok=True)

    # Loop through all subfolders in pred_volumes
    for folder_name in os.listdir(base_input_dir):
        input_dir = os.path.join(base_input_dir, folder_name)
        output_dir = os.path.join(base_output_dir, folder_name)

        # Process only if it's a directory
        if os.path.isdir(input_dir):
            print(f"Processing {input_dir} → {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            process_predicted_volumes(input_dir, output_dir)