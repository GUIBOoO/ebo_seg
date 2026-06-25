from pathlib import Path
import json
import os

import nibabel as nib
import numpy as np
from tqdm import tqdm

SCRATCH = os.environ["SCRATCH"]

TRAIN_DIR = Path(SCRATCH) / "datasets/Brats/training_data1_v2"

DATASET_ID = 1
DATASET_NAME = f"Dataset{DATASET_ID:03d}_BraTS2D"

NNUNET_ROOT = Path(SCRATCH) / "nnUNet_raw" / DATASET_NAME
IMAGES_TR = NNUNET_ROOT / "imagesTr"
LABELS_TR = NNUNET_ROOT / "labelsTr"

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

SKIP_EMPTY_SLICES = True


def find_patient_files(case_dir: Path):
    imgs = {}

    for modality in MODALITIES:
        files = (
            list(case_dir.glob(f"*{modality}.nii.gz"))
            + list(case_dir.glob(f"*_{modality}.nii.gz"))
        )

        if len(files) == 0:
            return None

        imgs[modality] = files[0]

    seg = (
        list(case_dir.glob("*seg.nii.gz"))
        + list(case_dir.glob("*_seg.nii.gz"))
    )

    if len(seg) == 0:
        return None

    return imgs, seg[0]


def remap_brats_labels(seg):
    """
    BraTS:
        0 = background
        1 = NCR/NET
        2 = edema
        4 = enhancing tumor

    nnU-Net préfère:
        0,1,2,3
    """
    seg = seg.astype(np.uint8)

    seg[seg == 4] = 3

    return seg


def save_slice_case(
    img_vol,
    seg_slice,
    case_id,
):
    for channel in range(len(MODALITIES)):
        img_slice = img_vol[channel].astype(np.float32)

        # nnU-Net attend du NIfTI.
        # On ajoute une dimension z=1.
        img_slice = img_slice[..., None]

        nib.save(
            nib.Nifti1Image(img_slice, affine=np.eye(4)),
            IMAGES_TR / f"{case_id}_{channel:04d}.nii.gz",
        )

    seg_slice = seg_slice[..., None].astype(np.uint8)

    nib.save(
        nib.Nifti1Image(seg_slice, affine=np.eye(4)),
        LABELS_TR / f"{case_id}.nii.gz",
    )


def process_patient(case_dir: Path, patient_idx: int):
    result = find_patient_files(case_dir)

    if result is None:
        print(f"Skipping {case_dir.name}")
        return 0

    imgs_paths, seg_path = result

    volumes = []

    for modality in MODALITIES:
        vol = nib.load(str(imgs_paths[modality])).get_fdata()
        volumes.append(vol)

    img_vol = np.stack(volumes, axis=0)  # (4,H,W,D)

    seg_vol = nib.load(str(seg_path)).get_fdata()
    seg_vol = remap_brats_labels(seg_vol)

    n_slices = img_vol.shape[-1]

    written = 0

    for slice_idx in range(n_slices):

        seg_slice = seg_vol[:, :, slice_idx]

        if SKIP_EMPTY_SLICES and np.sum(seg_slice > 0) == 0:
            continue

        case_id = f"BraTS_{patient_idx:04d}_{slice_idx:03d}"

        img_slice_vol = img_vol[:, :, :, slice_idx]

        save_slice_case(
            img_slice_vol,
            seg_slice,
            case_id,
        )

        written += 1

    return written


def write_dataset_json(num_cases: int):
    dataset_json = {
        "channel_names": {
            "0": "T1",
            "1": "T1ce",
            "2": "T2",
            "3": "FLAIR",
        },
        "labels": {
            "background": 0,
            "NCR_NET": 1,
            "Edema": 2,
            "EnhancingTumor": 3,
        },
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
    }

    with open(NNUNET_ROOT / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}"
        )

    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)

    patients = sorted(
        [p for p in TRAIN_DIR.iterdir() if p.is_dir()]
    )

    print(f"Found {len(patients)} patients")

    total_cases = 0

    for patient_idx, patient_dir in enumerate(tqdm(patients)):
        total_cases += process_patient(
            patient_dir,
            patient_idx,
        )

    write_dataset_json(total_cases)

    print(f"Done.")
    print(f"Dataset location: {NNUNET_ROOT}")
    print(f"Number of 2D cases: {total_cases}")


if __name__ == "__main__":
    main()