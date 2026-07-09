from pathlib import Path
import shutil
import json
import os

from nnunet_split_utils import (
    split_ids_train_val,
    write_splits_final_json,
    write_test_identifiers,
)

# ==========================
# PARAMETRES
# ==========================

SCRATCH = os.environ["SCRATCH"]

EXTRACT_DIR = Path(SCRATCH) / "datasets/ACDC_nnUnet/ACDC/database"

DATASET_ID = 1
DATASET_NAME = "ACDC"

NNUNET_RAW = Path(SCRATCH) / "nnUNet_raw"

NNUNET_SPLIT_VAL_SIZE = float(os.environ.get("NNUNET_SPLIT_VAL_SIZE", "0.2"))
NNUNET_SPLIT_SEED = int(os.environ.get("NNUNET_SPLIT_SEED", "42"))


# ==========================
# RECHERCHE TRAIN / TEST
# ==========================

training_root = next(EXTRACT_DIR.rglob("training"))
testing_root = next(EXTRACT_DIR.rglob("testing"))

print("Training trouvé :", training_root)
print("Testing trouvé :", testing_root)


# ==========================
# DATASET NNUNET
# ==========================

dataset_folder = (
    NNUNET_RAW /
    f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
)

imagesTr = dataset_folder / "imagesTr"
labelsTr = dataset_folder / "labelsTr"


for folder in [imagesTr, labelsTr]:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================
# EXTRACTION D'UN GROUPE DE PATIENTS
# ==========================

def extract_patients(patient_dirs):
    """Copie les frames + labels d'un groupe de patients vers imagesTr/labelsTr.
    Retourne la liste des case_id effectivement écrits."""

    case_ids = []

    for patient_dir in patient_dirs:

        if not patient_dir.is_dir():
            continue

        for image_file in sorted(patient_dir.glob("*frame*.nii.gz")):

            if "_gt" in image_file.name:
                continue

            gt_file = patient_dir / image_file.name.replace(
                ".nii.gz",
                "_gt.nii.gz"
            )

            if not gt_file.exists():
                print("Label absent :", gt_file)
                continue

            case_id = image_file.stem.replace(".nii", "")

            shutil.copy2(
                image_file,
                imagesTr / f"{case_id}_0000.nii.gz"
            )

            shutil.copy2(
                gt_file,
                labelsTr / f"{case_id}.nii.gz"
            )

            case_ids.append(case_id)

    return case_ids


# ==========================
# SPLIT TRAIN / VAL (dossier "training") + TEST (dossier "testing")
# ==========================

training_patient_dirs = sorted(
    [p for p in training_root.iterdir() if p.is_dir()]
)
testing_patient_dirs = sorted(
    [p for p in testing_root.iterdir() if p.is_dir()]
)

train_patient_dirs, val_patient_dirs = split_ids_train_val(
    training_patient_dirs,
    val_size=NNUNET_SPLIT_VAL_SIZE,
    seed=NNUNET_SPLIT_SEED,
)

train_case_ids = extract_patients(train_patient_dirs)
val_case_ids = extract_patients(val_patient_dirs)
# Le dossier "testing" natif d'ACDC est poolé lui aussi (mêmes imagesTr/labelsTr,
# même preprocessing nnU-Net), mais ses case ids sont gardés à part et exclus de
# splits_final.json : nnU-Net ne les voit jamais pendant l'entraînement/la CV.
test_case_ids = extract_patients(testing_patient_dirs)

n_training_cases = len(train_case_ids) + len(val_case_ids) + len(test_case_ids)


# ==========================
# SPLITS_FINAL.JSON / TEST_IDENTIFIERS.JSON
# ==========================

write_splits_final_json(
    dataset_folder / "splits_final.json",
    train_case_ids=train_case_ids,
    val_case_ids=val_case_ids,
)
write_test_identifiers(
    dataset_folder / "test_identifiers.json",
    test_case_ids=test_case_ids,
)


# ==========================
# DATASET.JSON
# ==========================

dataset_json = {
    "channel_names": {
        "0": "MRI"
    },
    "labels": {
        "background": 0,
        "RV": 1,
        "MYO": 2,
        "LV": 3
    },
    "numTraining": n_training_cases,
    "file_ending": ".nii.gz"
}


with open(dataset_folder / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)



print()
print("================================")
print("Conversion terminée")
print("================================")
print("Dataset :", dataset_folder)
print(f"Patients -> train: {len(train_patient_dirs)}, val: {len(val_patient_dirs)}, "
      f"test: {len(testing_patient_dirs)}")
print(f"Cases    -> train: {len(train_case_ids)}, val: {len(val_case_ids)}, "
      f"test: {len(test_case_ids)}")
print("imagesTr :", imagesTr)
print("labelsTr :", labelsTr)
print("================================")
