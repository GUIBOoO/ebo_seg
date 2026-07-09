import json
import os
import shutil
from pathlib import Path
from typing import Sequence, Tuple

from sklearn.model_selection import train_test_split


def split_ids_train_val_test(
    ids: Sequence,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[list, list, list]:
    """Split ids into train/val/test by a two-stage sklearn split (patient-level, no leakage)."""
    ids = sorted(ids, key=str)

    train_ids, temp_ids = train_test_split(
        ids,
        test_size=(val_size + test_size),
        random_state=seed,
        shuffle=True,
    )

    val_ratio = val_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1 - val_ratio),
        random_state=seed,
        shuffle=True,
    )

    return list(train_ids), list(val_ids), list(test_ids)


def split_ids_train_val(
    ids: Sequence,
    val_size: float,
    seed: int,
) -> Tuple[list, list]:
    """Split ids into train/val only (used when test membership comes from elsewhere)."""
    ids = sorted(ids, key=str)

    train_ids, val_ids = train_test_split(
        ids,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
    )

    return list(train_ids), list(val_ids)


def write_splits_final_json(
    path: Path,
    train_case_ids: Sequence[str],
    val_case_ids: Sequence[str],
) -> None:
    """Write a single-fold nnU-Net splits_final.json (fold 0 only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "train": sorted(str(case_id) for case_id in train_case_ids),
            "val": sorted(str(case_id) for case_id in val_case_ids),
        }
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_test_identifiers(path: Path, test_case_ids: Sequence[str]) -> None:
    """Write the held-out test case ids, kept out of nnU-Net's splits_final.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = sorted(str(case_id) for case_id in test_case_ids)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_dataset_name(dataset_id: int, raw_root: Path) -> str:
    matches = sorted(Path(raw_root).glob(f"Dataset{int(dataset_id):03d}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No nnUNet_raw dataset directory found for Dataset ID {dataset_id} in {raw_root}"
        )
    return matches[0].name


def sync_split_files(dataset_name: str, raw_root: Path, preprocessed_root: Path) -> None:
    """Copy splits_final.json / test_identifiers.json from the raw dataset dir into the
    preprocessed dataset dir, so nnU-Net's CV and our test-set inference both see our
    custom train/val/test split instead of nnU-Net's auto-generated 5-fold CV."""
    raw_dataset_dir = Path(raw_root) / dataset_name
    preprocessed_dataset_dir = Path(preprocessed_root) / dataset_name
    preprocessed_dataset_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("splits_final.json", "test_identifiers.json"):
        src = raw_dataset_dir / filename
        if src.is_file():
            shutil.copy2(src, preprocessed_dataset_dir / filename)


def sync_split_files_by_id(dataset_id: int) -> None:
    """Convenience wrapper for shell scripts: reads nnUNet_raw/nnUNet_preprocessed from env."""
    raw_root = Path(os.environ["nnUNet_raw"])
    preprocessed_root = Path(os.environ["nnUNet_preprocessed"])
    dataset_name = resolve_dataset_name(dataset_id, raw_root)
    sync_split_files(dataset_name, raw_root, preprocessed_root)
    print(f"Synced splits_final.json / test_identifiers.json for {dataset_name}")
