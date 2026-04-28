from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm
import os

SCRATCH = os.environ.get("SCRATCH")

EXTRACT_DIR = Path(SCRATCH) / "datasets/Brats"
TRAIN_DIR = EXTRACT_DIR / "training_data1_v2"


def inspect_raw_brats(train_dir: Path):

    all_labels = set()
    patient_stats = {}

    patients = sorted([p for p in train_dir.iterdir() if p.is_dir()])

    print(f"Found {len(patients)} patients\n")

    for patient in tqdm(patients):

        seg_files = list(patient.glob("*seg.nii.gz")) + list(patient.glob("*_seg.nii.gz"))
        if len(seg_files) == 0:
            continue

        seg_path = seg_files[0]

        seg = nib.load(str(seg_path)).get_fdata()
        uniq = np.unique(seg)

        patient_stats[patient.name] = uniq
        all_labels.update(uniq)

    # -------- GLOBAL --------
    print("\n================ GLOBAL REPORT ================\n")
    print("All labels found:", sorted(all_labels))

    invalid = [x for x in all_labels if x not in [0, 1, 2, 3, 4]]
    if invalid:
        print("\n⚠️ Unexpected labels:", invalid)
    else:
        print("\nLabels are in BraTS raw format [0,1,2,4] (expected)")


    # -------- PER PATIENT --------
    print("\n================ PER PATIENT ================\n")
    for k, v in patient_stats.items():
        print(f"{k}: {sorted(v)}")


if __name__ == "__main__":
    inspect_raw_brats(TRAIN_DIR)