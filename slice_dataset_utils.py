"""Helpers shared by the extract_data_*.py scripts that build 2D slice datasets.

Every converter writes the layout datasets.py already reads for BraTS, so a new
dataset needs no new loader:

    <root>/<case_id>/slice_<idx>_img.npy   (C, H, W) uint8 or float32
    <root>/<case_id>/slice_<idx>_seg.npy   (H, W)    uint8, contiguous 0..K-1
    <root>/<case_id>/sources.json          slice index -> source file
    <root>/dataset_info.json               in_channels / num_classes / labels
    <root>/splits.json                     train/val/test case ids

`splits.json` is what makes the difference with the BraTS layout: REFUGE and
CaDIS ship an official split, and CHAOS has several series per patient, so the
random case-level split datasets.py falls back to would either ignore the
benchmark protocol or leak a patient across splits.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from PIL import Image

DATASET_INFO_NAME = "dataset_info.json"
SPLITS_NAME = "splits.json"


def resize_image(image: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """Bilinear resize of a (H, W) or (H, W, C) array to `size` = (H, W)."""
    if size is None or image.shape[:2] == tuple(size):
        return image

    height, width = size
    if image.ndim == 2:
        if image.dtype == np.uint8:
            resized = Image.fromarray(image).resize((width, height), Image.BILINEAR)
        else:
            resized = Image.fromarray(image.astype(np.float32), mode="F").resize(
                (width, height), Image.BILINEAR
            )
        return np.asarray(resized, dtype=image.dtype)

    channels = [resize_image(image[..., c], size) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def resize_mask(mask: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    """Nearest-neighbour resize: a label map must not be interpolated."""
    if size is None or mask.shape[:2] == tuple(size):
        return mask
    height, width = size
    resized = Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.NEAREST)
    return np.asarray(resized, dtype=mask.dtype)


def to_channels_first(image: np.ndarray) -> np.ndarray:
    """(H, W) or (H, W, C) -> (C, H, W), the layout datasets.py feeds the model."""
    if image.ndim == 2:
        return image[None, ...]
    return np.transpose(image, (2, 0, 1))


def save_slice(case_dir: Path, index: int, image: np.ndarray, mask: np.ndarray) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / f"slice_{index:03d}_img.npy", to_channels_first(image))
    np.save(case_dir / f"slice_{index:03d}_seg.npy", mask.astype(np.uint8))


def write_sources(case_dir: Path, sources: Sequence[Any]) -> None:
    (case_dir / "sources.json").write_text(
        json.dumps({str(i): str(src) for i, src in enumerate(sources)}, indent=2),
        encoding="utf-8",
    )


def write_dataset_info(
    root: Path,
    dataset: str,
    in_channels: int,
    num_classes: int,
    labels: Dict[int, str],
    image_size: tuple[int, int] | None,
    **extra: Any,
) -> None:
    """train_unet.py reads this to default --in-channels / --num-classes."""
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "labels": {str(key): value for key, value in labels.items()},
        "image_size": list(image_size) if image_size else None,
        **extra,
    }
    (root / DATASET_INFO_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_splits(
    root: Path,
    train: Iterable[str],
    val: Iterable[str],
    test: Iterable[str],
    rule: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "rule": rule,
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }
    (root / SPLITS_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(root: Path) -> Dict[str, int]:
    case_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    num_slices = sum(len(list(case.glob("slice_*_img.npy"))) for case in case_dirs)
    return {"cases": len(case_dirs), "slices": num_slices}
