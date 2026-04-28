import os
from pathlib import Path
from typing import List, Sequence, Tuple, Dict

import nibabel as nib
import numpy as np
from monai.data import DataLoader, Dataset, pad_list_data_collate
from monai.transforms import Compose, MapTransform, ScaleIntensityd, ToTensord
from sklearn.model_selection import train_test_split

DEFAULT_BASE_DIR = os.environ.get("PYTHON_DATA_DIR")
DEFAULT_DATASET = os.environ.get("PYTHON_DATASET")

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


class LoadSlicedBraTS(MapTransform):
    """Load a single 2D BraTS slice prepared by extract_data.py."""

    def __call__(self, data):
        img = _load_array(data["image"])
        lbl = _load_array(data["label"])
        #lbl = (lbl > 0).astype(np.float32)[None, ...]

        return {
            "image": img.astype(np.float32),
            "label": lbl,
        }


def infer_dataset_type(base_dir: str | os.PathLike | None = None, dataset: str | None = None) -> str:
    if dataset is not None:
        normalized = dataset.lower()
    elif DEFAULT_DATASET is not None:
        normalized = DEFAULT_DATASET.lower()
    else:
        root = _resolve_base_dir(base_dir)
        root_name = root.name.lower()
        if "acdc" in root_name:
            normalized = "acdc"
        elif "brats" in root_name:
            normalized = "brats"
        elif list(root.glob("patient*")):
            normalized = "acdc"
        elif list(root.rglob("slice_*_img.npy")):
            normalized = "brats"
        else:
            raise ValueError(
                f"Unable to infer dataset type from {root}. Pass `dataset='acdc'` or `dataset='brats'`."
            )

    if normalized not in {"acdc", "brats"}:
        raise ValueError(f"Unsupported dataset `{dataset}`. Expected `acdc` or `brats`.")
    return normalized


def get_acdc_slices(data_dir: str | os.PathLike) -> List[dict]:
    patient_dirs = sorted(Path(data_dir).glob("patient*"))
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


def _get_brats_case_dirs(data_dir: Path) -> List[Path]:
    return sorted(
        path for path in data_dir.iterdir()
        if path.is_dir() and list(path.glob("slice_*_img.npy"))
    )


def get_brats_slices_from_patients(
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


def _resolve_acdc_split_dirs(base_dir: Path) -> Tuple[Path, Path | None]:
    training_dir = base_dir / "training"
    testing_dir = base_dir / "testing"

    if training_dir.exists():
        return training_dir, testing_dir if testing_dir.exists() else None
    return base_dir, None


def _resolve_named_split_dirs(base_dir: Path) -> Tuple[Path, None]:
    if not list(base_dir.rglob("slice_*_img.npy")):
        raise FileNotFoundError(f"Expected prepared BraTS slices under {base_dir}")
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

def get_transforms(dataset: str) -> Tuple[Compose, Compose]:
    dataset = dataset.lower()
    if dataset == "acdc":
        loader = LoadSlicedACDC(keys=["image", "label"])
    elif dataset == "brats":
        loader = LoadSlicedBraTS(keys=["image", "label"])
    else:
        raise ValueError(f"Unsupported dataset `{dataset}`. Expected `acdc` or `brats`.")

    transforms = Compose(
        [
            loader,
            ScaleIntensityd(keys=["image"]),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return transforms, transforms


def get_acdc_dataloaders(
    base_dir: str | os.PathLike | None = None,
    batch_size: int = 4,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root = _resolve_base_dir(base_dir)
    train_root, test_root = _resolve_acdc_split_dirs(root)

    train_data = get_acdc_slices(train_root)
    test_data = get_acdc_slices(test_root)
    train_data_split, val_data_split = train_test_split(train_data, test_size=val_size, random_state=random_state)

    train_transforms, val_transforms = get_transforms("acdc")
    train_ds = Dataset(train_data_split, transform=train_transforms)
    val_ds = Dataset(val_data_split, transform=val_transforms)
    test_ds = Dataset(test_data, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=pad_list_data_collate)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=pad_list_data_collate)
    return train_loader, val_loader, test_loader

def get_brats_dataloaders(
    base_dir: str | os.PathLike | None = None,
    batch_size: int = 4,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    foreground_only: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    root = _resolve_base_dir(base_dir)
    train_root, _ = _resolve_named_split_dirs(root)

    patient_dirs = _get_brats_case_dirs(train_root)

    train_patients, val_patients, test_patients = _split_randomly(
        patient_dirs,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    train_data = get_brats_slices_from_patients(train_patients, foreground_only)
    val_data   = get_brats_slices_from_patients(val_patients, foreground_only)
    test_data  = get_brats_slices_from_patients(test_patients, foreground_only)

    train_transforms, val_transforms = get_transforms("brats")

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


def get_dataloaders(
    dataset: str | None = None,
    base_dir: str | os.PathLike | None = None,
    batch_size: int = 4,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    foreground_only: bool = True,
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
        )

    return get_brats_dataloaders(
        base_dir=root,
        batch_size=batch_size,
        val_size=val_size,
        random_state=random_state,
        foreground_only=foreground_only,
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    batch = next(iter(train_loader))
    print("Batch image shape:", batch["image"].shape)
    print("Batch label shape:", batch["label"].shape)
