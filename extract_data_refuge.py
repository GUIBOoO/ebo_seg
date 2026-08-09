"""Prepare the REFUGE2 fundus dataset (optic disc / cup) for the U-Net pipeline.

Input (unzipped refuge.zip):
    <REFUGE_RAW>/REFUGE2/{train,val,test}/images/*.jpg
    <REFUGE_RAW>/REFUGE2/{train,val,test}/mask/*.{bmp,png}

REFUGE encodes its masks the other way round from what the pipeline expects:
0 = optic cup, 128 = optic disc (rim), 255 = background. They are remapped to
contiguous 0 = background, 1 = disc, 2 = cup.

Each image is one case with a single slice: unlike ACDC or BraTS there is no
volume to split, and the official train/val/test folders are kept as-is in
splits.json (400 images each, disjoint patients).

Output layout (see slice_dataset_utils.py):
    <OUTPUT_DIR>/<split>_<image_id>/slice_000_img.npy   uint8 (3, H, W)
    <OUTPUT_DIR>/<split>_<image_id>/slice_000_seg.npy   uint8 (H, W) in {0,1,2}
"""

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

RAW_DIR = Path(os.environ.get("REFUGE_RAW_DIR", Path(SCRATCH) / "datasets/refuge/raw/REFUGE2"))
OUTPUT_DIR = Path(os.environ.get("REFUGE_OUTPUT_DIR", Path(SCRATCH) / "datasets/refuge/data_slices"))

# The three official splits come with different native resolutions (2124x2056,
# 1940x1940, 1634x1634), so a fixed size is required for the batches to stack.
IMAGE_SIZE = int(os.environ.get("REFUGE_IMAGE_SIZE", 512))
SIZE = (IMAGE_SIZE, IMAGE_SIZE)

SPLITS = ("train", "val", "test")
MASK_SUFFIXES = (".bmp", ".png", ".gif", ".tif")

LABELS = {0: "background", 1: "optic disc", 2: "optic cup"}
NUM_CLASSES = 3
IN_CHANNELS = 3


def remap_refuge_mask(mask: np.ndarray) -> np.ndarray:
    """0 = cup, 128 = disc, 255 = background  ->  0 = background, 1 = disc, 2 = cup."""
    remapped = np.zeros(mask.shape, dtype=np.uint8)
    remapped[mask <= 64] = 2
    remapped[(mask > 64) & (mask < 192)] = 1
    return remapped


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    for suffix in MASK_SUFFIXES:
        candidate = mask_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def process_split(split: str) -> list[str]:
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
        case_dir = OUTPUT_DIR / case_id
        save_slice(case_dir, 0, image, mask)
        write_sources(case_dir, [image_path])
        case_ids.append(case_id)

    return case_ids


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_cases = {split: process_split(split) for split in SPLITS}

    write_splits(
        OUTPUT_DIR,
        train=split_cases["train"],
        val=split_cases["val"],
        test=split_cases["test"],
        rule="official REFUGE2 train/val/test folders",
    )
    write_dataset_info(
        OUTPUT_DIR,
        dataset="refuge",
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        labels=LABELS,
        image_size=SIZE,
        image_dtype="uint8",
        source=str(RAW_DIR),
        **summarize(OUTPUT_DIR),
    )

    print()
    print("================================")
    print("Conversion terminée")
    print("================================")
    print("Output   :", OUTPUT_DIR)
    for split, cases in split_cases.items():
        print(f"{split:<6} -> {len(cases)} images")
    print("Classes  :", LABELS)


if __name__ == "__main__":
    main()
