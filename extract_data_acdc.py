from pathlib import Path
import shutil
import json
import os

# ==========================
# PARAMETRES
# ==========================

SCRATCH = os.environ["SCRATCH"]

EXTRACT_DIR = Path(SCRATCH) / "datasets/ACDC_nnUnet/ACDC/database"

DATASET_ID = 1
DATASET_NAME = "ACDC"

NNUNET_RAW = Path(SCRATCH) / "nnUNet_raw"


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
imagesTs = dataset_folder / "imagesTs"
labelsTs = dataset_folder / "labelsTs"   # <-- ajouté


for folder in [imagesTr, labelsTr, imagesTs, labelsTs]:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================
# TRAINING
# ==========================

n_training_cases = 0

for patient_dir in sorted(training_root.iterdir()):

    if not patient_dir.is_dir():
        continue

    for image_file in patient_dir.glob("*frame*.nii.gz"):

        if "_gt" in image_file.name:
            continue

        gt_file = patient_dir / image_file.name.replace(
            ".nii.gz",
            "_gt.nii.gz"
        )

        if not gt_file.exists():
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


        n_training_cases += 1



# ==========================
# TESTING + LABELS TEST
# ==========================

n_test_cases = 0

for patient_dir in sorted(testing_root.iterdir()):

    if not patient_dir.is_dir():
        continue


    for image_file in patient_dir.glob("*frame*.nii.gz"):

        if "_gt" in image_file.name:
            continue


        case_id = image_file.stem.replace(".nii", "")


        # image test
        shutil.copy2(
            image_file,
            imagesTs / f"{case_id}_0000.nii.gz"
        )


        # label test si présent
        gt_file = patient_dir / image_file.name.replace(
            ".nii.gz",
            "_gt.nii.gz"
        )


        if gt_file.exists():

            shutil.copy2(
                gt_file,
                labelsTs / f"{case_id}.nii.gz"
            )

        else:
            print("Label test absent :", gt_file)


        n_test_cases += 1



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
print("Train :", n_training_cases)
print("Test :", n_test_cases)
print("imagesTr :", imagesTr)
print("labelsTr :", labelsTr)
print("imagesTs :", imagesTs)
print("labelsTs :", labelsTs)
print("================================")