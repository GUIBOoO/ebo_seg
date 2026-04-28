import zipfile
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm
import os

SCRATCH = os.environ.get("SCRATCH")

ZIP_PATH = Path(SCRATCH) / "datasets/Brats/brats_train.zip"
EXTRACT_DIR = Path(SCRATCH) / "datasets/Brats"
OUTPUT_DIR = Path(SCRATCH) / "datasets/Brats/data_slices"

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def find_patient_files(case_dir: Path):
    imgs = {}

    for m in MODALITIES:
        files = list(case_dir.glob(f"*{m}.nii.gz")) + list(case_dir.glob(f"*_{m}.nii.gz"))
        if len(files) == 0:
            return None
        imgs[m] = files[0]

    seg = list(case_dir.glob("*seg.nii.gz")) + list(case_dir.glob("*_seg.nii.gz"))
    if len(seg) == 0:
        return None

    return imgs, seg[0]


def process_patient(case_dir: Path, out_dir: Path):
    res = find_patient_files(case_dir)
    if res is None:
        return

    imgs_paths, seg_path = res

    vols = []
    for m in MODALITIES:
        vol = nib.load(str(imgs_paths[m])).get_fdata()
        vols.append(vol)

    img_vol = np.stack(vols, axis=0)  # (4, H, W, D)
    seg_vol = nib.load(str(seg_path)).get_fdata()

    n_slices = img_vol.shape[-1]

    patient_out = out_dir / case_dir.name
    patient_out.mkdir(parents=True, exist_ok=True)

    for i in range(n_slices):
        img = img_vol[:, :, :, i]
        seg = seg_vol[:, :, i]

        seg = seg.astype(np.int64)   

        img = img.astype(np.float32)

        np.save(patient_out / f"slice_{i:03d}_img.npy", img)
        np.save(patient_out / f"slice_{i:03d}_seg.npy", seg)


def main():
    train_dir = EXTRACT_DIR / "training_data1_v2"
    if not train_dir.exists():
        raise FileNotFoundError(f"{train_dir} not found")

    patients = sorted([p for p in train_dir.iterdir() if p.is_dir()])

    print(f"Found {len(patients)} patients")

    for p in tqdm(patients):
        process_patient(p, OUTPUT_DIR)

    print("Done preprocessing.")


if __name__ == "__main__":
    main()