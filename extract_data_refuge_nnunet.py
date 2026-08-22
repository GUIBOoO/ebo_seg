"""Build the nnU-Net raw dataset for REFUGE2 (optic disc / cup).

Companion of extract_data_refuge.py, which builds the .npy slice dataset for the
U-Net pipeline: both import the same label remapping, so a nnU-Net run and a
U-Net run are trained on the same targets and can be compared.

REFUGE images are natural RGB images, so the dataset uses nnU-Net's
NaturalImage2DIO: one RGB .png per case (3 channels in a single _0000 file) and
a .png label map. Images are resized to a fixed size for the same reason as on
the U-Net side: the three official splits ship three different resolutions.

Output:
    $nnUNet_raw/Dataset004_REFUGE/imagesTr/<case>_0000.png
    $nnUNet_raw/Dataset004_REFUGE/labelsTr/<case>.png
    $nnUNet_raw/Dataset004_REFUGE/dataset.json
    $nnUNet_raw/Dataset004_REFUGE/splits_final.json     (official train / val)
    $nnUNet_raw/Dataset004_REFUGE/test_identifiers.json (official test, held out)

As in extract_data_acdc.py the test images are written to imagesTr too, so that
nnU-Net preprocesses them, but their ids are kept out of splits_final.json and
nnU-Net never sees them during training or CV.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from extract_data_refuge import RAW_DIR, find_mask, remap_refuge_mask
from nnunet_split_utils import write_splits_final_json, write_test_identifiers
from slice_dataset_utils import resize_image, resize_mask

SCRATCH = os.environ.get("SCRATCH", ".")

DATASET_ID = int(os.environ.get("REFUGE_NNUNET_DATASET_ID", 4))
DATASET_NAME = "REFUGE"

NNUNET_RAW = Path(os.environ.get("nnUNet_raw", Path(SCRATCH) / "nnUNet_raw"))
dataset_folder = NNUNET_RAW / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
imagesTr = dataset_folder / "imagesTr"
labelsTr = dataset_folder / "labelsTr"

IMAGE_SIZE = int(os.environ.get("REFUGE_IMAGE_SIZE", 512))
SIZE = (IMAGE_SIZE, IMAGE_SIZE)

SPLITS = ("train", "val", "test")


def convert_split(split: str) -> list[str]:
    images_dir = RAW_DIR / split / "images"
    mask_dir = RAW_DIR / split / "mask"
    if not images_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected {images_dir} and {mask_dir}. Unzip refuge.zip first.")

    case_ids: list[str] = []
    for image_path in tqdm(sorted(images_dir.iterdir()), desc=f"REFUGE {split}"):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        mask_path = find_mask(mask_dir, image_path.stem)
        if mask_path is None:
            print(f"Label absent : {image_path.name}")
            continue

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if image.shape[:2] != mask.shape[:2]:
            print(f"Shape mismatch {image_path.name}: {image.shape[:2]} vs {mask.shape[:2]}")
            continue

        image = resize_image(image, SIZE)
        mask = resize_mask(remap_refuge_mask(mask), SIZE)

        case_id = f"{split}_{image_path.stem}"
        Image.fromarray(image).save(imagesTr / f"{case_id}_0000.png")
        Image.fromarray(mask.astype(np.uint8)).save(labelsTr / f"{case_id}.png")
        case_ids.append(case_id)

    return case_ids


def main() -> None:
    for folder in (imagesTr, labelsTr):
        folder.mkdir(parents=True, exist_ok=True)

    split_cases = {split: convert_split(split) for split in SPLITS}
    num_cases = sum(len(cases) for cases in split_cases.values())

    write_splits_final_json(
        dataset_folder / "splits_final.json",
        train_case_ids=split_cases["train"],
        val_case_ids=split_cases["val"],
    )
    write_test_identifiers(
        dataset_folder / "test_identifiers.json",
        test_case_ids=split_cases["test"],
    )

    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "optic disc": 1, "optic cup": 2},
        "numTraining": num_cases,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (dataset_folder / "dataset.json").write_text(json.dumps(dataset_json, indent=4), encoding="utf-8")

    print()
    print("================================")
    print("Conversion terminée")
    print("================================")
    print("Dataset :", dataset_folder)
    for split, cases in split_cases.items():
        print(f"{split:<6} -> {len(cases)} images")
    print("Next    : nnUNetv2_plan_and_preprocess -d", DATASET_ID, "--verify_dataset_integrity")


if __name__ == "__main__":
    main()
