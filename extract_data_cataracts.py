"""Prepare the CATARACTS 2020 / CaDISv2 dataset for the U-Net pipeline.

Input (unzipped cataracts.zip):
    <CADIS_RAW>/VideoXX/Images/VideoN_frameNNNNNN.png
    <CADIS_RAW>/VideoXX/Labels/VideoN_frameNNNNNN.png
    <CADIS_RAW>/splits.txt

The raw labels use the 36 CaDIS class ids. The benchmark defines three class
remappings (Experiment I/II/III, see cadis_visualization.py shipped with the
dataset); they are reproduced here.

Experiment I (8 classes) is the default because it is the only one without an
ignore label: Experiments II and III map the rarest classes to 255, and neither
CEDiceLoss nor the EBO losses have an ignore_index. With --experiment 2 or 3
those pixels are therefore folded into one extra trailing "ignore/other" class
(index 17 resp. 25) so every label stays a valid class index. Use experiment 1
to reproduce the published protocol exactly.

A case is a video, so the official video-level split in splits.txt is kept:
frames of one surgery are near duplicates, and splitting the flat frame list
would put the same surgery on both sides.

Output layout (see slice_dataset_utils.py):
    <OUTPUT_DIR>/VideoXX/slice_<idx>_img.npy   uint8 (3, H, W)
    <OUTPUT_DIR>/VideoXX/slice_<idx>_seg.npy   uint8 (H, W)
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from slice_dataset_utils import (
    resize_image,
    resize_mask,
    save_slice,
    summarize,
    write_dataset_info,
    write_sources,
    write_splits,
)

SCRATCH = os.environ.get("SCRATCH", ".")

RAW_DIR = Path(
    os.environ.get("CADIS_RAW_DIR", Path(SCRATCH) / "datasets/cataracts/raw/CATARACTS_2020_CaDISv2")
)
OUTPUT_DIR = Path(
    os.environ.get("CADIS_OUTPUT_DIR", Path(SCRATCH) / "datasets/cataracts/data_slices")
)

# CaDIS frames are 960x540; 512x288 keeps that 16:9 aspect and stays divisible
# by 16, which the U-Net's four poolings need.
IMAGE_HEIGHT = int(os.environ.get("CADIS_IMAGE_HEIGHT", 288))
IMAGE_WIDTH = int(os.environ.get("CADIS_IMAGE_WIDTH", 512))

IGNORE_LABEL = 255

# From cadis_visualization.py: {new class id: [raw CaDIS class ids]}.
EXPERIMENTS: dict[int, dict[int, list[int]]] = {
    1: {
        0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6],
        7: list(range(7, 36)),
    },
    2: {
        0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6],
        7: [7, 8, 10, 27, 20, 32],
        8: [9, 22],
        9: [11, 33],
        10: [12, 28],
        11: [13, 21],
        12: [14, 24],
        13: [15, 18],
        14: [16, 23],
        15: [17],
        16: [19],
        IGNORE_LABEL: [25, 26, 29, 30, 31, 34, 35],
    },
    3: {
        **{index: [index] for index in range(25)},
        IGNORE_LABEL: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    },
}

CLASS_NAMES: dict[int, dict[int, str]] = {
    1: {
        0: "Pupil", 1: "Surgical Tape", 2: "Hand", 3: "Eye Retractors",
        4: "Iris", 5: "Skin", 6: "Cornea", 7: "Instrument",
    },
    2: {
        0: "Pupil", 1: "Surgical Tape", 2: "Hand", 3: "Eye Retractors",
        4: "Iris", 5: "Skin", 6: "Cornea", 7: "Cannula", 8: "Cap. Cystotome",
        9: "Tissue Forceps", 10: "Primary Knife", 11: "Ph. Handpiece",
        12: "Lens Injector", 13: "I/A Handpiece", 14: "Secondary Knife",
        15: "Micromanipulator", 16: "Cap. Forceps", 17: "Ignore/Other",
    },
    3: {
        0: "Pupil", 1: "Surgical Tape", 2: "Hand", 3: "Eye Retractors",
        4: "Iris", 5: "Skin", 6: "Cornea", 7: "Hydro. Cannula",
        8: "Visc. Cannula", 9: "Cap. Cystotome", 10: "Rycroft Cannula",
        11: "Bonn Forceps", 12: "Primary Knife", 13: "Ph. Handpiece",
        14: "Lens Injector", 15: "I/A Handpiece", 16: "Secondary Knife",
        17: "Micromanipulator", 18: "I/A Handpiece Handle", 19: "Cap. Forceps",
        20: "R. Cannula Handle", 21: "Ph. Handpiece Handle",
        22: "Cap. Cystotome Handle", 23: "Sec. Knife Handle",
        24: "Lens Injector Handle", 25: "Ignore/Other",
    },
}


def build_remap_array(experiment: int) -> tuple[np.ndarray, int]:
    """Lookup table raw CaDIS id -> class id, plus the resulting class count.

    The ignore group of Experiments II and III becomes the last class index
    instead of 255, so the label map stays inside [0, num_classes).
    """
    remapping = EXPERIMENTS[experiment]
    kept = sorted(key for key in remapping if key != IGNORE_LABEL)
    num_classes = len(kept) + (1 if IGNORE_LABEL in remapping else 0)
    other_class = len(kept)

    table = np.zeros(36, dtype=np.uint8)
    for new_id, raw_ids in remapping.items():
        target = other_class if new_id == IGNORE_LABEL else new_id
        for raw_id in raw_ids:
            table[raw_id] = target

    return table, num_classes


def read_splits(splits_path: Path) -> dict[str, list[str]]:
    """Parse the '# Training:' / '# Validation:' / '# Test:' sections."""
    if not splits_path.is_file():
        raise FileNotFoundError(f"splits.txt not found: {splits_path}")

    sections = {"train": [], "val": [], "test": []}
    current = None
    for raw_line in splits_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            header = line.lstrip("#").strip().lower()
            if header.startswith("training"):
                current = "train"
            elif header.startswith("validation"):
                current = "val"
            elif header.startswith("test"):
                current = "test"
            else:
                current = None
            continue
        if current is not None:
            sections[current].append(line)

    if not all(sections.values()):
        raise ValueError(f"Incomplete split sections in {splits_path}: "
                         f"{ {name: len(videos) for name, videos in sections.items()} }")
    return sections


def process_video(video_dir: Path, remap: np.ndarray, size: tuple[int, int]) -> int:
    images_dir = video_dir / "Images"
    labels_dir = video_dir / "Labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"Skipping {video_dir.name}: missing Images/ or Labels/")
        return 0

    case_dir = OUTPUT_DIR / video_dir.name
    sources = []

    for index, image_path in enumerate(tqdm(sorted(images_dir.glob("*.png")), desc=video_dir.name)):
        label_path = labels_dir / image_path.name
        if not label_path.is_file():
            print(f"Label absent : {image_path.name}")
            continue

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(label_path).convert("L"))

        if mask.max() >= remap.size:
            raise ValueError(f"Unexpected CaDIS class id {mask.max()} in {label_path}")

        image = resize_image(image, size)
        mask = resize_mask(remap[mask], size)

        save_slice(case_dir, len(sources), image, mask)
        sources.append(image_path)

    if sources:
        write_sources(case_dir, sources)
    return len(sources)


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
    video_dirs = sorted(path for path in RAW_DIR.iterdir() if path.is_dir() and path.name.startswith("Video"))
    if not video_dirs:
        raise FileNotFoundError(f"No VideoXX directories under {RAW_DIR}. Unzip cataracts.zip first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    kept_videos = []
    for video_dir in video_dirs:
        frames = process_video(video_dir, remap, size)
        total_frames += frames
        if frames:
            kept_videos.append(video_dir.name)

    # splits.txt names videos as Video05; keep only what was actually converted.
    known = set(kept_videos)
    missing = {name for videos in splits.values() for name in videos} - known
    if missing:
        print(f"Videos listed in splits.txt but not converted: {sorted(missing)}")

    write_splits(
        OUTPUT_DIR,
        train=[name for name in splits["train"] if name in known],
        val=[name for name in splits["val"] if name in known],
        test=[name for name in splits["test"] if name in known],
        rule="official CaDISv2 splits.txt (video level)",
    )
    write_dataset_info(
        OUTPUT_DIR,
        dataset="cataracts",
        in_channels=3,
        num_classes=num_classes,
        labels=CLASS_NAMES[args.experiment],
        image_size=size,
        image_dtype="uint8",
        experiment=args.experiment,
        ignore_folded_into_last_class=args.experiment in (2, 3),
        source=str(RAW_DIR),
        **summarize(OUTPUT_DIR),
    )

    print()
    print("================================")
    print("Conversion terminée")
    print("================================")
    print("Output     :", OUTPUT_DIR)
    print("Experiment :", args.experiment, f"({num_classes} classes)")
    print("Videos     :", len(kept_videos), f"({len(splits['train'])} train / "
          f"{len(splits['val'])} val / {len(splits['test'])} test)")
    print("Frames     :", total_frames)


if __name__ == "__main__":
    main()
