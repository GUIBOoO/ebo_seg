"""Build the nnU-Net raw dataset for CATARACTS 2020 / CaDISv2.

Companion of extract_data_cataracts.py (U-Net .npy slices): the class remapping,
the frame size and the video-level split are imported from it, so both pipelines
train on the same targets.

Frames are natural RGB images, hence nnU-Net's NaturalImage2DIO: one RGB .png
per case (3 channels in a single _0000 file) plus a .png label map.

Output:
    $nnUNet_raw/Dataset005_CaDIS/imagesTr/<Video>_<frame>_0000.png
    $nnUNet_raw/Dataset005_CaDIS/labelsTr/<Video>_<frame>.png
    $nnUNet_raw/Dataset005_CaDIS/dataset.json
    $nnUNet_raw/Dataset005_CaDIS/splits_final.json     (official train / val videos)
    $nnUNet_raw/Dataset005_CaDIS/test_identifiers.json (official test videos)

A case is one frame, but the split stays at the video level: frames of one
surgery are near duplicates, so splitting frames would leak across sets. The
test videos are written to imagesTr as well (nnU-Net has to preprocess them) but
their ids never appear in splits_final.json.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from extract_data_cataracts import (
    CLASS_NAMES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RAW_DIR,
    build_remap_array,
    read_splits,
)
from nnunet_split_utils import write_splits_final_json, write_test_identifiers
from slice_dataset_utils import resize_image, resize_mask

SCRATCH = os.environ.get("SCRATCH", ".")

DATASET_ID = int(os.environ.get("CADIS_NNUNET_DATASET_ID", 5))
DATASET_NAME = "CaDIS"

NNUNET_RAW = Path(os.environ.get("nnUNet_raw", Path(SCRATCH) / "nnUNet_raw"))
dataset_folder = NNUNET_RAW / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
imagesTr = dataset_folder / "imagesTr"
labelsTr = dataset_folder / "labelsTr"


def convert_video(video_dir: Path, remap: np.ndarray, size: tuple[int, int]) -> list[str]:
    images_dir = video_dir / "Images"
    labels_dir = video_dir / "Labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"Skipping {video_dir.name}: missing Images/ or Labels/")
        return []

    case_ids: list[str] = []
    for image_path in tqdm(sorted(images_dir.glob("*.png")), desc=video_dir.name):
        label_path = labels_dir / image_path.name
        if not label_path.is_file():
            print(f"Label absent : {image_path.name}")
            continue

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(label_path).convert("L"))
        if mask.max() >= remap.size:
            raise ValueError(f"Unexpected CaDIS class id {mask.max()} in {label_path}")

        image = resize_image(image, size)
        mask = resize_mask(remap[mask] + 1, size)

        # Video01/Video1_frame000090.png -> Video01_frame000090
        case_id = f"{video_dir.name}_{image_path.stem.split('_', 1)[1]}"
        Image.fromarray(image).save(imagesTr / f"{case_id}_0000.png")
        Image.fromarray(mask.astype(np.uint8)).save(labelsTr / f"{case_id}.png")
        case_ids.append(case_id)

    return case_ids


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3],
                        default=int(os.environ.get("CADIS_EXPERIMENT", 1)),
                        help="CaDIS class remapping: 1 (8 classes, default), 2 (18), 3 (26).")
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    size = (args.height, args.width)
    remap, num_classes = build_remap_array(args.experiment)

    splits = read_splits(RAW_DIR / "splits.txt")
    video_dirs = sorted(
        path for path in RAW_DIR.iterdir() if path.is_dir() and path.name.startswith("Video")
    )
    if not video_dirs:
        raise FileNotFoundError(f"No VideoXX directories under {RAW_DIR}. Unzip cataracts.zip first.")

    for folder in (imagesTr, labelsTr):
        folder.mkdir(parents=True, exist_ok=True)

    cases_by_video = {video.name: convert_video(video, remap, size) for video in video_dirs}
    expand = lambda videos: [case for video in videos for case in cases_by_video.get(video, [])]

    train_cases = expand(splits["train"])
    val_cases = expand(splits["val"])
    test_cases = expand(splits["test"])
    num_cases = sum(len(cases) for cases in cases_by_video.values())

    write_splits_final_json(
        dataset_folder / "splits_final.json",
        train_case_ids=train_cases,
        val_case_ids=val_cases,
    )
    write_test_identifiers(dataset_folder / "test_identifiers.json", test_case_ids=test_cases)

    # CaDIS class 0 is "Pupil", a real class: every pixel of a frame belongs to
    # some class and there is no background. nnU-Net refuses a label map without
    # a background at 0, so every class is shifted by +1 and label 0 stays empty.
    # Its mean_fg_dice then covers exactly the CaDIS classes (label 0 is excluded
    # from the foreground metrics), which is what the U-Net pipeline averages
    # over too - only the class indices differ by one between the two datasets.
    labels = {"background": 0}
    labels.update({name: index + 1 for index, name in CLASS_NAMES[args.experiment].items()})
    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": labels,
        "numTraining": num_cases,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (dataset_folder / "dataset.json").write_text(json.dumps(dataset_json, indent=4), encoding="utf-8")

    print()
    print("================================")
    print("Conversion terminée")
    print("================================")
    print("Dataset    :", dataset_folder)
    print("Experiment :", args.experiment,
          f"({num_classes} classes CaDIS -> labels 1..{num_classes}, 0 = background vide)")
    print(f"Videos     : {len(cases_by_video)} "
          f"({len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test)")
    print(f"Frames     : {num_cases} ({len(train_cases)} train / {len(val_cases)} val / "
          f"{len(test_cases)} test)")
    print("Next       : nnUNetv2_plan_and_preprocess -d", DATASET_ID, "--verify_dataset_integrity")


if __name__ == "__main__":
    main()
