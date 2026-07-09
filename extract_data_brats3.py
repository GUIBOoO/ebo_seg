"""Prepare a 3-channel BraTS slice dataset (T1ce, T2, FLAIR) for TransUNet.

The vendored R50-ViT-B_16 TransUNet stem is a hard-coded `StdConv2d(3, ...)`
(see transunet/vit_seg_modeling_resnet_skip.py), so it only accepts 1- or
3-channel input. BraTS ships 4 modalities; dropping T1 (the least informative
for tumour sub-regions) leaves exactly the 3 channels the ImageNet-pretrained
stem expects.

Labels are remapped 4 -> 3 so classes are contiguous (0..3), unlike
extract_data.py which keeps the raw BraTS 0/1/2/4 encoding and therefore needs
num_classes=5 with a dead class.

Output layout matches extract_data.py so datasets.py picks it up unchanged:
    <OUTPUT_DIR>/<patient>/slice_<idx>_img.npy   float32 (3, H, W)
    <OUTPUT_DIR>/<patient>/slice_<idx>_seg.npy   int64   (H, W)
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

SCRATCH = os.environ["SCRATCH"]

TRAIN_DIR = Path(os.environ.get("BRATS_TRAIN_DIR", Path(SCRATCH) / "datasets/Brats/training_data1_v2"))
OUTPUT_DIR = Path(os.environ.get("BRATS3_OUTPUT_DIR", Path(SCRATCH) / "datasets/Brats/data_slices_3ch"))

# Order matters: this is the channel order the model sees.
MODALITIES = ["t1c", "t2w", "t2f"]  # T1ce, T2, FLAIR
CHANNEL_NAMES = ["T1ce", "T2", "FLAIR"]

SKIP_EMPTY_SLICES = os.environ.get("BRATS3_SKIP_EMPTY_SLICES", "1") == "1"


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

    seg = list(case_dir.glob("*seg.nii.gz")) + list(case_dir.glob("*_seg.nii.gz"))
    if len(seg) == 0:
        return None

    return imgs, seg[0]


def remap_brats_labels(seg: np.ndarray) -> np.ndarray:
    """BraTS encodes enhancing tumour as 4; make labels contiguous 0..3.

    0 = background, 1 = NCR/NET, 2 = edema, 3 = enhancing tumour.
    """
    seg = seg.astype(np.int64)
    seg[seg == 4] = 3
    return seg


def process_patient(case_dir: Path, out_dir: Path) -> int:
    result = find_patient_files(case_dir)
    if result is None:
        return 0

    imgs_paths, seg_path = result

    volumes = [nib.load(str(imgs_paths[modality])).get_fdata() for modality in MODALITIES]
    img_vol = np.stack(volumes, axis=0)  # (3, H, W, D)

    seg_vol = remap_brats_labels(nib.load(str(seg_path)).get_fdata())

    patient_out = out_dir / case_dir.name
    patient_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for slice_idx in range(img_vol.shape[-1]):
        seg_slice = seg_vol[:, :, slice_idx]

        if SKIP_EMPTY_SLICES and not np.any(seg_slice > 0):
            continue

        img_slice = img_vol[:, :, :, slice_idx].astype(np.float32)  # (3, H, W)

        np.save(patient_out / f"slice_{slice_idx:03d}_img.npy", img_slice)
        np.save(patient_out / f"slice_{slice_idx:03d}_seg.npy", seg_slice)
        written += 1

    if written == 0:
        patient_out.rmdir()

    return written


def main() -> None:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"BraTS training directory not found: {TRAIN_DIR}")

    patients = sorted(p for p in TRAIN_DIR.iterdir() if p.is_dir())
    print(f"Found {len(patients)} patients in {TRAIN_DIR}")
    print(f"Channels        : {CHANNEL_NAMES} (from {MODALITIES})")
    print(f"Skip empty      : {SKIP_EMPTY_SLICES}")
    print(f"Output dir      : {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_slices = 0
    kept_patients = 0
    for patient_dir in tqdm(patients):
        written = process_patient(patient_dir, OUTPUT_DIR)
        total_slices += written
        kept_patients += int(written > 0)

    print("Done preprocessing.")
    print(f"Patients written: {kept_patients} / {len(patients)}")
    print(f"Slices written  : {total_slices}")
    print("Train with: --dataset brats --in-channels 3 --num-classes 4")


if __name__ == "__main__":
    main()
