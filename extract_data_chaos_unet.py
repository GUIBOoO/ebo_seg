"""Prepare the CHAOS abdominal dataset for the U-Net pipeline.

Not to be confused with extract_data_chaos.py, which slices an existing 3D
nnU-Net dataset into a 2D nnU-Net dataset. This one reads the released archive
directly and writes the .npy slice layout datasets.py consumes.

Input (unzipped chaos.zip, then Train_Sets.zip):
    <CHAOS_RAW>/Train_Sets/CT/<case>/DICOM_anon/*.dcm
    <CHAOS_RAW>/Train_Sets/CT/<case>/Ground/liver_GT_*.png
    <CHAOS_RAW>/Train_Sets/MR/<case>/T1DUAL/DICOM_anon/{InPhase,OutPhase}/*.dcm
    <CHAOS_RAW>/Train_Sets/MR/<case>/T1DUAL/Ground/*.png
    <CHAOS_RAW>/Train_Sets/MR/<case>/T2SPIR/DICOM_anon/*.dcm
    <CHAOS_RAW>/Train_Sets/MR/<case>/T2SPIR/Ground/*.png

Test_Sets ships no ground truth, so only Train_Sets can be used here; the
train/val/test split is drawn from it, at the patient level.

CT and MR are written as two separate datasets: CT is liver-only (2 classes)
while MR labels four organs (5 classes), and one model cannot be trained on
both label sets at once.

    <OUTPUT_ROOT>_ct/CT<case>/slice_<idx>_*.npy          1 channel, 2 classes
    <OUTPUT_ROOT>_mr/MR<case>_T2SPIR/slice_<idx>_*.npy   1 channel, 5 classes

MR labels come as grey levels (63 liver, 126 right kidney, 189 left kidney,
252 spleen) and are remapped to 1..4. CT ground truth is binary (liver = 255).
DICOM pixels are converted to their real intensities with RescaleSlope /
RescaleIntercept and stored as float32; datasets.py's ScaleIntensityd then
min-max normalises each slice.

Image and label are paired by sorted filename order within a series, which is
the acquisition order in the CHAOS release; the counts are checked to match.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

from nnunet_split_utils import split_ids_train_val_test
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
    os.environ.get("CHAOS_RAW_DIR", Path(SCRATCH) / "datasets/chaos/raw/CHAOS_Original")
)
OUTPUT_ROOT = Path(os.environ.get("CHAOS_OUTPUT_DIR", Path(SCRATCH) / "datasets/chaos/data_slices"))

IMAGE_SIZE = int(os.environ.get("CHAOS_IMAGE_SIZE", 256))
SIZE = (IMAGE_SIZE, IMAGE_SIZE)

VAL_SIZE = float(os.environ.get("CHAOS_VAL_SIZE", 0.2))
TEST_SIZE = float(os.environ.get("CHAOS_TEST_SIZE", 0.1))
SPLIT_SEED = int(os.environ.get("CHAOS_SPLIT_SEED", 42))

CT_LABELS = {0: "background", 1: "liver"}
MR_LABELS = {0: "background", 1: "liver", 2: "right kidney", 3: "left kidney", 4: "spleen"}

# Grey level -> class, with the tolerance the CHAOS documentation gives.
MR_LABEL_RANGES = ((55, 70, 1), (110, 135, 2), (175, 200, 3), (240, 255, 4))


def read_dicom(path: Path) -> np.ndarray:
    dataset = pydicom.dcmread(str(path))
    pixels = dataset.pixel_array.astype(np.float32)
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    return pixels * slope + intercept


def remap_ct_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def remap_mr_mask(mask: np.ndarray) -> np.ndarray:
    remapped = np.zeros(mask.shape, dtype=np.uint8)
    for low, high, class_id in MR_LABEL_RANGES:
        remapped[(mask >= low) & (mask <= high)] = class_id
    return remapped


def convert_series(
    case_dir: Path,
    dicom_paths: list[Path],
    mask_paths: list[Path],
    remap,
    skip_empty: bool,
) -> int:
    if len(dicom_paths) != len(mask_paths):
        print(f"Skipping {case_dir.name}: {len(dicom_paths)} DICOM vs {len(mask_paths)} labels")
        return 0

    sources = []
    for dicom_path, mask_path in zip(dicom_paths, mask_paths):
        image = read_dicom(dicom_path)
        mask = remap(np.asarray(Image.open(mask_path).convert("L")))

        if image.shape[:2] != mask.shape[:2]:
            print(f"Shape mismatch {dicom_path.name}: {image.shape} vs {mask.shape}")
            continue
        if skip_empty and not mask.any():
            continue

        save_slice(case_dir, len(sources), resize_image(image, SIZE), resize_mask(mask, SIZE))
        sources.append(dicom_path)

    if sources:
        write_sources(case_dir, sources)
    return len(sources)


def convert_ct(output_dir: Path, skip_empty: bool) -> dict[str, list[str]]:
    ct_root = RAW_DIR / "Train_Sets" / "CT"
    if not ct_root.is_dir():
        raise FileNotFoundError(f"CHAOS CT folder not found: {ct_root}")

    patient_cases: dict[str, list[str]] = {}
    for patient_dir in tqdm(sorted(ct_root.iterdir(), key=lambda p: p.name), desc="CHAOS CT"):
        if not patient_dir.is_dir():
            continue
        dicoms = sorted((patient_dir / "DICOM_anon").glob("*.dcm"))
        masks = sorted((patient_dir / "Ground").glob("*.png"))
        if not dicoms or not masks:
            continue

        case_id = f"CT{patient_dir.name}"
        if convert_series(output_dir / case_id, dicoms, masks, remap_ct_mask, skip_empty):
            patient_cases.setdefault(patient_dir.name, []).append(case_id)

    return patient_cases


def convert_mr(output_dir: Path, skip_empty: bool) -> dict[str, list[str]]:
    mr_root = RAW_DIR / "Train_Sets" / "MR"
    if not mr_root.is_dir():
        raise FileNotFoundError(f"CHAOS MR folder not found: {mr_root}")

    patient_cases: dict[str, list[str]] = {}
    for patient_dir in tqdm(sorted(mr_root.iterdir(), key=lambda p: p.name), desc="CHAOS MR"):
        if not patient_dir.is_dir():
            continue

        # T1DUAL: two phases sharing one Ground; T2SPIR: a single series.
        # The dual echo interleaves instances, so OutPhase files are numbered
        # 00001, 00003, ... while the Ground is named after the InPhase ones
        # (00002, 00004, ...). Same slice locations, same count: pairing by
        # sorted position is right for both phases, pairing by name is not.
        series: list[tuple[str, list[Path], list[Path]]] = []
        t1_ground = sorted((patient_dir / "T1DUAL" / "Ground").glob("*.png"))
        for phase in ("InPhase", "OutPhase"):
            phase_dir = patient_dir / "T1DUAL" / "DICOM_anon" / phase
            if phase_dir.is_dir() and t1_ground:
                series.append((f"T1DUAL_{phase}", sorted(phase_dir.glob("*.dcm")), t1_ground))

        t2_dicoms = sorted((patient_dir / "T2SPIR" / "DICOM_anon").glob("*.dcm"))
        t2_ground = sorted((patient_dir / "T2SPIR" / "Ground").glob("*.png"))
        if t2_dicoms and t2_ground:
            series.append(("T2SPIR", t2_dicoms, t2_ground))

        for name, dicoms, masks in series:
            case_id = f"MR{patient_dir.name}_{name}"
            if convert_series(output_dir / case_id, dicoms, masks, remap_mr_mask, skip_empty):
                patient_cases.setdefault(patient_dir.name, []).append(case_id)

    return patient_cases


def split_by_patient(patient_cases: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]:
    """Split patients, then expand to cases: one patient has up to 3 MR series,
    and they must not straddle two splits."""
    patients = sorted(patient_cases)
    if len(patients) < 4:
        raise ValueError(
            f"Only {len(patients)} patient(s) converted: a train/val/test split needs at least 4. "
            "Check CHAOS_RAW_DIR points at the unzipped Train_Sets."
        )
    train_ids, val_ids, test_ids = split_ids_train_val_test(
        patients, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=SPLIT_SEED
    )
    expand = lambda ids: [case for patient in ids for case in patient_cases[patient]]
    return expand(train_ids), expand(val_ids), expand(test_ids)


def finalize(
    output_dir: Path,
    modality: str,
    patient_cases: dict[str, list[str]],
    labels: dict[int, str],
    skip_empty: bool,
) -> None:
    if not patient_cases:
        print(f"No {modality} case converted, skipping {output_dir}")
        return

    train_cases, val_cases, test_cases = split_by_patient(patient_cases)
    write_splits(
        output_dir,
        train=train_cases,
        val=val_cases,
        test=test_cases,
        rule=f"patient-level random split (val={VAL_SIZE}, test={TEST_SIZE}, seed={SPLIT_SEED})",
    )
    write_dataset_info(
        output_dir,
        dataset="chaos",
        in_channels=1,
        num_classes=len(labels),
        labels=labels,
        image_size=SIZE,
        image_dtype="float32",
        modality=modality,
        patients=len(patient_cases),
        skip_empty_slices=skip_empty,
        source=str(RAW_DIR),
        **summarize(output_dir),
    )
    stats = summarize(output_dir)
    print(f"{modality}: {len(patient_cases)} patients, {stats['cases']} series, "
          f"{stats['slices']} slices -> {output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modality", choices=["ct", "mr", "both"], default="both")
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        default=os.environ.get("CHAOS_SKIP_EMPTY_SLICES", "0") == "1",
        help="Drop slices without any labelled organ (datasets.py also filters them at load time).",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if args.modality in ("ct", "both"):
        ct_dir = Path(f"{OUTPUT_ROOT}_ct")
        ct_dir.mkdir(parents=True, exist_ok=True)
        finalize(ct_dir, "CT", convert_ct(ct_dir, args.skip_empty), CT_LABELS, args.skip_empty)

    if args.modality in ("mr", "both"):
        mr_dir = Path(f"{OUTPUT_ROOT}_mr")
        mr_dir.mkdir(parents=True, exist_ok=True)
        finalize(mr_dir, "MR", convert_mr(mr_dir, args.skip_empty), MR_LABELS, args.skip_empty)

    print()
    print("================================")
    print("Conversion terminée")
    print("================================")


if __name__ == "__main__":
    main()
