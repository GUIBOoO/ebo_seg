"""Compute the RELU D matrix for an nnU-Net model (ACDC or BraTS).

Mirrors d_matrix.py, but iterates nnU-Net's preprocessed cases through a
trained nnUNetPredictor instead of the custom UNet/TransUNet dataloaders.
Everything model-folder related is reused from inference_nnunet.py so the two
scripts can never drift apart.

The D matrix should be estimated on a held-out split that is NOT the one you
report on: default --split val, then evaluate with inference_nnunet.py
--split test --d-matrix <this output>.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm

from d_matrix_core import DMatrixAccumulator, save_d_matrix
from inference_nnunet import (
    load_best_trial,
    load_case_arrays,
    load_cases_for_split,
    patch_size_of,
    predict_padded,
    resolve_model_folder_and_fold,
    resolve_preprocessed_dir,
    shape_must_be_divisible_by,
)

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute the RELU D matrix for a trained nnU-Net model.")
    parser.add_argument("--best-json", type=Path, default=None, help="nnunet_grid_search_best.json from select_nnunet_grid_best.py.")
    parser.add_argument("--model-folder", type=Path, default=None, help="Folder containing fold_* directories.")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_best.pth")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--preprocessed-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where D will be saved.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tile-step-size", type=float, default=0.5)
    parser.add_argument("--disable-mirroring", action="store_true")
    parser.add_argument("--perform-everything-on-device", action="store_true")
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="val",
        help="Split to estimate D on. Use 'val' so the test set stays untouched for reporting.",
    )
    parser.add_argument("--lambda-weight", type=float, default=0.5, help="RELU paper lambda weight for X_-.")
    return parser


def accumulate_2d_case(
    network: torch.nn.Module,
    data: np.ndarray,
    seg: np.ndarray,
    device: torch.device,
    divisor: tuple[int, ...],
    accumulator: DMatrixAccumulator,
    patch_size: tuple[int, ...] | None = None,
) -> None:
    if data.ndim == 3:
        data = data[:, None]
    if seg.ndim == 4 and seg.shape[0] == 1:
        seg = seg[0]
    if seg.ndim == 2:
        seg = seg[None]
    if data.ndim != 4 or seg.ndim != 3:
        raise ValueError(f"Expected 2d data (C, Z, H, W) and seg (Z, H, W), got {data.shape} and {seg.shape}")

    for slice_idx in range(data.shape[1]):
        image = torch.from_numpy(data[:, slice_idx]).unsqueeze(0).to(device)
        target = torch.from_numpy(seg[slice_idx]).to(device).unsqueeze(0)
        logits = predict_padded(network, image, divisor, patch_size)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1)
        # nnU-Net marks ignored voxels with a negative label.
        accumulator.update(probs, preds == target, valid_mask=target >= 0)


def accumulate_nd_case(
    network: torch.nn.Module,
    data: np.ndarray,
    seg: np.ndarray,
    device: torch.device,
    divisor: tuple[int, ...],
    accumulator: DMatrixAccumulator,
    patch_size: tuple[int, ...] | None = None,
) -> None:
    if seg.ndim == data.ndim and seg.shape[0] == 1:
        seg = seg[0]
    image = torch.from_numpy(data).unsqueeze(0).to(device)
    target = torch.from_numpy(seg).to(device).unsqueeze(0)
    logits = predict_padded(network, image, divisor, patch_size)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(1)
    accumulator.update(probs, preds == target, valid_mask=target >= 0)


def main() -> None:
    args = build_argparser().parse_args()
    best_trial = load_best_trial(args.best_json)
    model_folder, fold, checkpoint_name = resolve_model_folder_and_fold(args, best_trial)
    dataset_preprocessed, plans_name, configuration, dataset_name = resolve_preprocessed_dir(args, model_folder)
    case_dir = dataset_preprocessed / f"{plans_name}_{configuration}"
    cases = load_cases_for_split(dataset_preprocessed, dataset_name, fold, args.split)
    device = torch.device(args.device)

    predictor = nnUNetPredictor(
        tile_step_size=args.tile_step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_mirroring,
        perform_everything_on_device=args.perform_everything_on_device,
        device=device,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(fold,),
        checkpoint_name=checkpoint_name,
    )
    predictor.network = predictor.network.to(device)
    predictor.network.eval()

    num_classes = predictor.label_manager.num_segmentation_heads
    divisor = shape_must_be_divisible_by(predictor.configuration_manager)
    patch_size = patch_size_of(predictor.configuration_manager)
    accumulator = DMatrixAccumulator(num_classes, device)

    print(f"Model folder : {model_folder}")
    print(f"Checkpoint   : fold_{fold}/{checkpoint_name}")
    print(f"Dataset      : {dataset_name}")
    print(f"Preprocessed : {dataset_preprocessed}")
    print(f"Split        : {args.split}")
    print(f"Cases        : {len(cases)}")
    print(f"Classes      : {num_classes}")
    print(f"Patch size   : {patch_size} (min padded shape)")

    with torch.no_grad():
        for case in tqdm.tqdm(cases, desc="Computing D"):
            data, seg = load_case_arrays(case_dir, case)
            if configuration == "2d":
                accumulate_2d_case(predictor.network, data, seg, device, divisor, accumulator, patch_size)
            else:
                accumulate_nd_case(predictor.network, data, seg, device, divisor, accumulator, patch_size)

    d_matrix, stats = accumulator.finalize(num_classes=num_classes, lambda_weight=args.lambda_weight)

    stem = f"d_matrix_nnunet_{dataset_name}_{args.split}_lambda_{args.lambda_weight:g}"
    metadata = {
        **stats,
        "dataset": dataset_name,
        "split": args.split,
        "fold": fold,
        "model_folder": str(model_folder),
        "checkpoint": str(model_folder / f"fold_{fold}" / checkpoint_name),
        "configuration": configuration,
        "plans": plans_name,
        "num_cases": len(cases),
    }
    torch_path, npy_path, json_path = save_d_matrix(args.output_dir, stem, d_matrix, metadata)

    print("\nD matrix:")
    print(d_matrix)
    print(f"Trace(D D^T)     : {stats['trace_ddt']:.6f}")
    print(f"Correct pixels   : {stats['num_correct_pixels']}")
    print(f"Incorrect pixels : {stats['num_incorrect_pixels']}")
    print(f"Saved: {torch_path}")
    print(f"Saved: {npy_path}")
    print(f"Saved: {json_path}")
    print(f"\nNow run: inference_nnunet.py --split test --d-matrix {npy_path}")


if __name__ == "__main__":
    main()
