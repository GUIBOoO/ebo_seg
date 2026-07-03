from pathlib import Path
import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import json

# =========================
# CONFIG
# =========================

SCRATCH = os.environ.get("SCRATCH", ".")

DATASET_3D = Path(SCRATCH) / "nnUNet_raw/Dataset003_CHAOS3D"
DATASET_2D = Path(SCRATCH) / "nnUNet_raw/Dataset003_CHAOS2D"

IMAGES_TR = DATASET_2D / "imagesTr"
LABELS_TR = DATASET_2D / "labelsTr"

IMAGES_TR.mkdir(parents=True, exist_ok=True)
LABELS_TR.mkdir(parents=True, exist_ok=True)


# =========================
# SLICE EXTRACTION
# =========================

def save_slice(img_slice, seg_slice, case_id, z):

    img_out = IMAGES_TR / f"{case_id}_{z:04d}_0000.nii.gz"
    seg_out = LABELS_TR / f"{case_id}_{z:04d}.nii.gz"

    img_slice = img_slice.astype(np.float32)[None, ...]
    seg_slice = seg_slice.astype(np.uint8)[None, ...]

    nib.save(nib.Nifti1Image(img_slice, np.eye(4)), img_out)
    nib.save(nib.Nifti1Image(seg_slice, np.eye(4)), seg_out)


def process_case(img_path, seg_path):

    img = nib.load(str(img_path)).get_fdata()
    seg = nib.load(str(seg_path)).get_fdata()

    assert img.shape == seg.shape, f"Shape mismatch {img.shape} vs {seg.shape}"

    n_slices = img.shape[-1]
    case_id = img_path.name.replace("_0000.nii.gz", "")

    count = 0

    for z in range(n_slices):

        img_slice = img[:, :, z]
        seg_slice = seg[:, :, z]

        # skip empty slices (optionnel mais recommandé)
        if np.sum(seg_slice) == 0:
            continue

        save_slice(img_slice, seg_slice, case_id, z)
        count += 1

    return count


# =========================
# MAIN
# =========================

def main():

    img_dir = DATASET_3D / "imagesTr"
    seg_dir = DATASET_3D / "labelsTr"

    images = sorted(img_dir.glob("*_0000.nii.gz"))

    total = 0

    for img_path in tqdm(images):

        case_id = img_path.name.replace("_0000.nii.gz", "")
        seg_path = seg_dir / f"{case_id}.nii.gz"

        if not seg_path.exists():
            print(f"Missing label for {case_id}")
            continue

        total += process_case(img_path, seg_path)

    print(f"\nDone. Total 2D slices: {total}")


if __name__ == "__main__":
    main()