"""Build the nnU-Net raw dataset for CHAOS MR (liver, kidneys, spleen).

Companion of extract_data_chaos_unet.py (U-Net .npy slices): the DICOM reading
and the label remapping are imported from it, so both pipelines see the same
targets. Unlike the U-Net version, the slices are stacked back into 3D NIfTI
volumes with their real voxel spacing, which is what nnU-Net expects; training
in CONFIGURATION=2d then slices them again itself, exactly like ACDC.

Only Train_Sets is usable: the released Test_Sets has no ground truth.

One case = one series (T1DUAL InPhase, T1DUAL OutPhase, T2SPIR), so a patient
contributes up to three cases; the split is drawn at the patient level to keep
them together.

Output:
    $nnUNet_raw/Dataset006_CHAOSMR/imagesTr/MR<case>_<series>_0000.nii.gz
    $nnUNet_raw/Dataset006_CHAOSMR/labelsTr/MR<case>_<series>.nii.gz
    $nnUNet_raw/Dataset006_CHAOSMR/dataset.json
    $nnUNet_raw/Dataset006_CHAOSMR/splits_final.json
    $nnUNet_raw/Dataset006_CHAOSMR/test_identifiers.json
"""

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

from extract_data_chaos_unet import MR_LABELS, RAW_DIR, read_dicom, remap_mr_mask
from nnunet_split_utils import split_ids_train_val_test, write_splits_final_json, write_test_identifiers

SCRATCH = os.environ.get("SCRATCH", ".")

DATASET_ID = int(os.environ.get("CHAOS_MR_NNUNET_DATASET_ID", 6))
DATASET_NAME = "CHAOSMR"

NNUNET_RAW = Path(os.environ.get("nnUNet_raw", Path(SCRATCH) / "nnUNet_raw"))
dataset_folder = NNUNET_RAW / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
imagesTr = dataset_folder / "imagesTr"
labelsTr = dataset_folder / "labelsTr"

VAL_SIZE = float(os.environ.get("CHAOS_VAL_SIZE", 0.2))
TEST_SIZE = float(os.environ.get("CHAOS_TEST_SIZE", 0.1))
SPLIT_SEED = int(os.environ.get("CHAOS_SPLIT_SEED", 42))


def slice_geometry(dicom_paths: list[Path]) -> tuple[list[int], tuple[float, float, float]]:
    """Return the slice order (by z position) and the (row, col, z) spacing.

    The Ground PNGs are index-aligned with the sorted DICOM list, so the order
    computed here is applied to both, never to the images alone.
    """
    headers = [pydicom.dcmread(str(path), stop_before_pixels=True) for path in dicom_paths]

    positions = []
    for index, header in enumerate(headers):
        position = getattr(header, "ImagePositionPatient", None)
        instance = getattr(header, "InstanceNumber", index)
        positions.append(float(position[2]) if position is not None else float(instance))
    order = sorted(range(len(headers)), key=lambda i: positions[i])

    pixel_spacing = getattr(headers[0], "PixelSpacing", [1.0, 1.0])
    ordered = [positions[i] for i in order]
    gaps = np.abs(np.diff(ordered)) if len(ordered) > 1 else np.array([])
    if gaps.size and np.median(gaps) > 0:
        spacing_z = float(np.median(gaps))
    else:
        spacing_z = float(
            getattr(headers[0], "SpacingBetweenSlices", 0) or getattr(headers[0], "SliceThickness", 1) or 1
        )

    return order, (float(pixel_spacing[0]), float(pixel_spacing[1]), spacing_z)


def convert_series(case_id: str, dicom_paths: list[Path], mask_paths: list[Path]) -> bool:
    if len(dicom_paths) != len(mask_paths):
        print(f"Skipping {case_id}: {len(dicom_paths)} DICOM vs {len(mask_paths)} labels")
        return False

    order, spacing = slice_geometry(dicom_paths)

    volume, segmentation = [], []
    for index in order:
        image = read_dicom(dicom_paths[index])
        mask = remap_mr_mask(np.asarray(Image.open(mask_paths[index]).convert("L")))
        if image.shape[:2] != mask.shape[:2]:
            print(f"Shape mismatch {dicom_paths[index].name}: {image.shape} vs {mask.shape}")
            return False
        volume.append(image.astype(np.float32))
        segmentation.append(mask.astype(np.uint8))

    # (row, col, z), matching the diagonal affine below.
    volume_array = np.stack(volume, axis=-1)
    segmentation_array = np.stack(segmentation, axis=-1)
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])

    nib.save(nib.Nifti1Image(volume_array, affine), imagesTr / f"{case_id}_0000.nii.gz")
    nib.save(nib.Nifti1Image(segmentation_array, affine), labelsTr / f"{case_id}.nii.gz")
    return True


def convert_all() -> dict[str, list[str]]:
    mr_root = RAW_DIR / "Train_Sets" / "MR"
    if not mr_root.is_dir():
        raise FileNotFoundError(f"CHAOS MR folder not found: {mr_root}")

    patient_cases: dict[str, list[str]] = {}
    for patient_dir in tqdm(sorted(mr_root.iterdir(), key=lambda p: p.name), desc="CHAOS MR"):
        if not patient_dir.is_dir():
            continue

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
            if convert_series(case_id, dicoms, masks):
                patient_cases.setdefault(patient_dir.name, []).append(case_id)

    return patient_cases


def main() -> None:
    for folder in (imagesTr, labelsTr):
        folder.mkdir(parents=True, exist_ok=True)

    patient_cases = convert_all()
    patients = sorted(patient_cases)
    if len(patients) < 4:
        raise RuntimeError(
            f"Only {len(patients)} patient(s) converted: a train/val/test split needs at least 4. "
            "Check CHAOS_RAW_DIR points at the unzipped Train_Sets."
        )

    train_ids, val_ids, test_ids = split_ids_train_val_test(
        patients, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=SPLIT_SEED
    )
    expand = lambda ids: [case for patient in ids for case in patient_cases[patient]]
    train_cases, val_cases, test_cases = expand(train_ids), expand(val_ids), expand(test_ids)
    num_cases = sum(len(cases) for cases in patient_cases.values())

    write_splits_final_json(
        dataset_folder / "splits_final.json",
        train_case_ids=train_cases,
        val_case_ids=val_cases,
    )
    write_test_identifiers(dataset_folder / "test_identifiers.json", test_case_ids=test_cases)

    dataset_json = {
        "channel_names": {"0": "MR"},
        "labels": {name: index for index, name in MR_LABELS.items()},
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
    }
    (dataset_folder / "dataset.json").write_text(json.dumps(dataset_json, indent=4), encoding="utf-8")

    print()
    print("================================")
    print("Conversion terminée")
    print("================================")
    print("Dataset  :", dataset_folder)
    print(f"Patients : {len(patients)} ({len(train_ids)} train / {len(val_ids)} val / "
          f"{len(test_ids)} test)")
    print(f"Series   : {num_cases} ({len(train_cases)} train / {len(val_cases)} val / "
          f"{len(test_cases)} test)")
    print("Next     : nnUNetv2_plan_and_preprocess -d", DATASET_ID, "--verify_dataset_integrity")


if __name__ == "__main__":
    main()
