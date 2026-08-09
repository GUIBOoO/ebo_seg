import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict

import nibabel as nib
import numpy as np
from monai.data import DataLoader, Dataset, pad_list_data_collate
from monai.transforms import Compose, MapTransform, Resized, ScaleIntensityd, ToTensord
from sklearn.model_selection import train_test_split

DEFAULT_BASE_DIR = os.environ.get("PYTHON_DATA_DIR")
DEFAULT_DATASET = os.environ.get("PYTHON_DATASET")

# Datasets stored as per-case .npy slices by the extract_data_*.py scripts. They
# all share one loader; only ACDC reads NIfTI volumes lazily.
SLICE_DATASETS = ("brats", "refuge", "chaos", "cataracts")
SUPPORTED_DATASETS = ("acdc",) + SLICE_DATASETS

DATASET_INFO_NAME = "dataset_info.json"
SPLITS_NAME = "splits.json"

print("nibabel version:", nib.__version__)
print(f"Path used by Python: {DEFAULT_BASE_DIR}")


def _resolve_base_dir(base_dir: str | os.PathLike | None) -> Path:
    resolved = Path(base_dir) if base_dir is not None else (Path(DEFAULT_BASE_DIR) if DEFAULT_BASE_DIR else None)
    if resolved is None:
        raise ValueError("A dataset root is required. Pass `base_dir` or set `PYTHON_DATA_DIR`.")
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset root not found: {resolved}")
    return resolved


def _load_slice(path: str | os.PathLike, slice_index: int) -> np.ndarray:
    return nib.load(str(path)).get_fdata()[:, :, slice_index]


def _load_array(path: str | os.PathLike) -> np.ndarray:
    return np.load(str(path))


class LoadSlicedACDC(MapTransform):
    """Load a single 2D slice from an ACDC 3D volume."""

    def __call__(self, data):
        img = _load_slice(data["image"], data["slice_index"])[None, ...]
        lbl = _load_slice(data["label"], data["slice_index"])[None, ...]

        return {
            "image": img.astype(np.float32),
            "label": lbl.astype(np.float32),
        }


class LoadSlicedNpy(MapTransform):
    """Load one prepared 2D slice: (C, H, W) image + (H, W) label, both .npy.

    Written by extract_data.py / extract_data_brats3.py (BraTS) and by
    extract_data_refuge.py, extract_data_cataracts.py and
    extract_data_chaos_unet.py. RGB datasets are stored as uint8 to keep the
    dataset four times smaller; the cast to float32 here plus the
    ScaleIntensityd below give the same [0, 1] input either way.

    The label gets a leading channel axis so downstream channel-first
    transforms (e.g. Resized) see (1, H, W) like the ACDC loader; train_unet.py
    squeezes it back out.
    """

    def __call__(self, data):
        img = _load_array(data["image"])
        lbl = _load_array(data["label"])

        return {
            "image": img.astype(np.float32),
            "label": lbl[None, ...].astype(np.float32),
        }


# Kept so existing imports of the BraTS-specific name keep working.
LoadSlicedBraTS = LoadSlicedNpy


def load_dataset_info(base_dir: str | os.PathLike | None) -> Dict:
    """Read dataset_info.json (in_channels, num_classes, labels) if present.

    train_unet.py uses it to default --in-channels / --num-classes so a new
    dataset does not need those flags spelled out at every call site.
    """
    root = Path(base_dir) if base_dir is not None else None
    if root is None or not (root / DATASET_INFO_NAME).is_file():
        return {}
    try:
        return json.loads((root / DATASET_INFO_NAME).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_splits(root: Path) -> Optional[Dict[str, List[str]]]:
    """Official / patient-level split written by the extract_data_*.py scripts."""
    splits_path = root / SPLITS_NAME
    if not splits_path.is_file():
        return None
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if not all(key in splits for key in ("train", "val", "test")):
        raise ValueError(f"{splits_path} must define train, val and test.")
    return splits


def infer_dataset_type(base_dir: str | os.PathLike | None = None, dataset: str | None = None) -> str:
    if dataset is not None:
        normalized = dataset.lower()
    elif DEFAULT_DATASET is not None:
        normalized = DEFAULT_DATASET.lower()
    else:
        root = _resolve_base_dir(base_dir)
        root_name = root.name.lower()
        info_name = str(load_dataset_info(root).get("dataset", "")).lower()
        # dataset_info.json first: it is the only source that survives a
        # dataset directory being renamed or copied somewhere else.
        if info_name in SUPPORTED_DATASETS:
            normalized = info_name
        elif "acdc" in root_name:
            normalized = "acdc"
        elif any(name in str(root).lower() for name in SLICE_DATASETS):
            normalized = next(name for name in SLICE_DATASETS if name in str(root).lower())
        elif list(root.glob("patient*")):
            normalized = "acdc"
        elif list(root.rglob("slice_*_img.npy")):
            normalized = "brats"
        else:
            raise ValueError(
                f"Unable to infer dataset type from {root}. "
                f"Pass `dataset=` one of {SUPPORTED_DATASETS}."
            )

    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset `{dataset}`. Expected one of {SUPPORTED_DATASETS}.")
    return normalized


def _get_acdc_patient_dirs(data_dir: str | os.PathLike) -> List[Path]:
    return sorted(Path(data_dir).glob("patient*"))


def get_acdc_slices_from_patients(patient_dirs: List[Path]) -> List[dict]:
    data: List[dict] = []

    for patient in patient_dirs:
        for img_path in sorted(patient.glob("*frame*.nii.gz")):
            if "_gt" in img_path.name:
                continue

            label_path = img_path.with_name(img_path.name.replace(".nii.gz", "_gt.nii.gz"))
            if not label_path.exists():
                continue

            num_slices = nib.load(str(img_path)).shape[2]
            for slice_index in range(num_slices):
                data.append(
                    {
                        "image": str(img_path),
                        "label": str(label_path),
                        "slice_index": slice_index,
                    }
                )

    return data


def get_acdc_slices(data_dir: str | os.PathLike) -> List[dict]:
    return get_acdc_slices_from_patients(_get_acdc_patient_dirs(data_dir))


def _get_slice_case_dirs(data_dir: Path) -> List[Path]:
    return sorted(
        path for path in data_dir.iterdir()
        if path.is_dir() and list(path.glob("slice_*_img.npy"))
    )


# Kept so existing imports of the BraTS-specific names keep working.
_get_brats_case_dirs = _get_slice_case_dirs


def get_slices_from_cases(
    patient_dirs: List[Path],
    foreground_only: bool = True,
) -> List[dict]:

    data: List[dict] = []

    for case_dir in patient_dirs:
        for img_path in sorted(case_dir.glob("slice_*_img.npy")):
            seg_path = case_dir / img_path.name.replace("_img.npy", "_seg.npy")
            if not seg_path.exists():
                continue

            if foreground_only and np.sum(_load_array(seg_path) > 0) == 0:
                continue

            data.append(
                {
                    "image": str(img_path),
                    "label": str(seg_path),
                }
            )

    return data


get_brats_slices_from_patients = get_slices_from_cases


def _resolve_acdc_split_dirs(base_dir: Path) -> Tuple[Path, Path | None]:
    training_dir = base_dir / "training"
    testing_dir = base_dir / "testing"

    if training_dir.exists():
        return training_dir, testing_dir if testing_dir.exists() else None
    return base_dir, None


def _resolve_named_split_dirs(base_dir: Path) -> Tuple[Path, None]:
    # next() instead of list(): one hit is enough, and the RGB datasets have
    # thousands of slice files to walk.
    if next(base_dir.rglob("slice_*_img.npy"), None) is None:
        raise FileNotFoundError(
            f"Expected prepared .npy slices under {base_dir}. Build them with the matching "
            "extract_data_*.py (see README)."
        )
    return base_dir, None

def _split_randomly(
    items: Sequence[Path],
    val_size: float,
    test_size: float,
    random_state: int,
) -> Tuple[List[Path], List[Path], List[Path]]:

    items = list(items)

    train_patients, temp_patients = train_test_split(
        items,
        test_size=(val_size + test_size),
        random_state=random_state,
        shuffle=True,
    )

    val_ratio = val_size / (val_size + test_size)

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=(1 - val_ratio),
        random_state=random_state,
        shuffle=True,
    )

    return train_patients, val_patients, test_patients

def get_transforms(dataset: str, image_size: int | None = None) -> Tuple[Compose, Compose]:
    dataset = dataset.lower()
    if dataset == "acdc":
        loader = LoadSlicedACDC(keys=["image", "label"])
    elif dataset in SLICE_DATASETS:
        loader = LoadSlicedNpy(keys=["image", "label"])
    else:
        raise ValueError(f"Unsupported dataset `{dataset}`. Expected one of {SUPPORTED_DATASETS}.")

    steps = [loader, ScaleIntensityd(keys=["image"])]
    if image_size is not None:
        steps.append(
            Resized(
                keys=["image", "label"],
                spatial_size=(image_size, image_size),
                mode=("bilinear", "nearest"),
            )
        )
    steps.append(ToTensord(keys=["image", "label"]))

    transforms = Compose(steps)
    return transforms, transforms


def get_acdc_dataloaders(
    base_dir: str | os.PathLike | None = None,
    batch_size: int = 4,
    val_size: float = 0.2,
    random_state: int = 42,
    image_size: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root = _resolve_base_dir(base_dir)
    train_root, test_root = _resolve_acdc_split_dirs(root)

    # Patient-level split: slices of one volume are near duplicates of their
    # neighbours, so splitting the flat slice list puts a patient on both sides
    # and makes val optimistic (energy separation especially). test comes from
    # the official testing/ patients, which are unseen either way.
    train_patients, val_patients = train_test_split(
        _get_acdc_patient_dirs(train_root), test_size=val_size, random_state=random_state
    )

    train_data = get_acdc_slices_from_patients(train_patients)
    val_data = get_acdc_slices_from_patients(val_patients)
    test_data = get_acdc_slices(test_root)

    train_transforms, val_transforms = get_transforms("acdc", image_size=image_size)
    train_ds = Dataset(train_data, transform=train_transforms)
    val_ds = Dataset(val_data, transform=val_transforms)
    test_ds = Dataset(test_data, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=pad_list_data_collate)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=pad_list_data_collate)

    print("Patients train:", len(train_patients))
    print("Patients val:", len(val_patients))

    print("Slices train:", len(train_data))
    print("Slices val:", len(val_data))
    print("Slices test:", len(test_data))
    return train_loader, val_loader, test_loader

def get_slice_dataloaders(
    dataset: str = "brats",
    base_dir: str | os.PathLike | None = None,
    batch_size: int = 4,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    foreground_only: bool = True,
    image_size: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Loaders for any dataset stored as per-case .npy slices.

    When the dataset root carries a splits.json (REFUGE's and CaDIS's official
    protocol, CHAOS's patient-level split), it wins over the random case-level
    split: those datasets either define the benchmark split or have several
    series per patient, which a random split would spread across train and val.
    """

    root = _resolve_base_dir(base_dir)
    train_root, _ = _resolve_named_split_dirs(root)

    patient_dirs = _get_slice_case_dirs(train_root)
    splits = _load_splits(train_root)

    if splits is not None:
        by_name = {path.name: path for path in patient_dirs}
        missing = [
            name
            for key in ("train", "val", "test")
            for name in splits[key]
            if name not in by_name
        ]
        if missing:
            raise FileNotFoundError(
                f"{train_root / SPLITS_NAME} lists cases with no slices on disk: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        train_patients = [by_name[name] for name in splits["train"]]
        val_patients = [by_name[name] for name in splits["val"]]
        test_patients = [by_name[name] for name in splits["test"]]
        print(f"Split: {train_root / SPLITS_NAME} ({splits.get('rule', 'no rule recorded')})")
    else:
        train_patients, val_patients, test_patients = _split_randomly(
            patient_dirs,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
        )
        print(f"Split: random case level (val={val_size}, test={test_size}, seed={random_state})")

    train_data = get_slices_from_cases(train_patients, foreground_only)
    val_data   = get_slices_from_cases(val_patients, foreground_only)
    test_data  = get_slices_from_cases(test_patients, foreground_only)

    train_transforms, val_transforms = get_transforms(dataset, image_size=image_size)

    train_ds = Dataset(train_data, transform=train_transforms)
    val_ds   = Dataset(val_data, transform=val_transforms)
    test_ds  = Dataset(test_data, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_list_data_collate,num_workers=4,pin_memory=True,
    persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=pad_list_data_collate,num_workers=4,pin_memory=True,
    persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=pad_list_data_collate,num_workers=4,pin_memory=True,
    persistent_workers=True)
    print("Patients train:", len(train_patients))
    print("Patients val:", len(val_patients))
    print("Patients test:", len(test_patients))

    print("Slices train:", len(train_data))
    print("Slices val:", len(val_data))
    print("Slices test:", len(test_data))
    return train_loader, val_loader, test_loader


def get_brats_dataloaders(
    base_dir: str | os.PathLike | None = None,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return get_slice_dataloaders(dataset="brats", base_dir=base_dir, **kwargs)


def get_dataloaders(
    dataset: str | None = None,
    base_dir: str | os.PathLike | None = None,
    batch_size: int = 4,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    foreground_only: bool = True,
    image_size: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_name = infer_dataset_type(base_dir=base_dir, dataset=dataset)
    root = _resolve_base_dir(base_dir)

    print(f"Dataset detected: {dataset_name}")
    print(f"Dataset root: {root}")

    if dataset_name == "acdc":
        return get_acdc_dataloaders(
            base_dir=root,
            batch_size=batch_size,
            val_size=val_size,
            random_state=random_state,
            image_size=image_size,
        )

    return get_slice_dataloaders(
        dataset=dataset_name,
        base_dir=root,
        batch_size=batch_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        foreground_only=foreground_only,
        image_size=image_size,
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    batch = next(iter(train_loader))
    print("Batch image shape:", batch["image"].shape)
    print("Batch label shape:", batch["label"].shape)
