import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import blosc2
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from metrics import compute_binary_metrics, compute_multiclass_metrics
from nnunet_split_utils import sync_split_files
from utils import SIRC, doctor, energy, load_d_matrix, relu_score


BASE_SCORE_NAMES = ("energy", "msp", "alpha", "beta", "sirc")


def resolve_score_names(d_matrix: torch.Tensor | None) -> tuple[str, ...]:
    """RELU needs a precomputed D matrix, so it is only scored when one is given."""
    return BASE_SCORE_NAMES + (("relu",) if d_matrix is not None else ())


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the best nnU-Net grid-search checkpoint on a held-out split.")
    parser.add_argument("--best-json", type=Path, default=None, help="nnunet_grid_search_best.json from select_nnunet_grid_best.py.")
    parser.add_argument("--model-folder", type=Path, default=None, help="Folder containing fold_* directories.")
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_best.pth")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--preprocessed-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tile-step-size", type=float, default=0.5)
    parser.add_argument("--disable-mirroring", action="store_true")
    parser.add_argument("--perform-everything-on-device", action="store_true")
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val"],
        default="test",
        help="Which held-out split to run inference on. 'test' (default) is the true "
        "held-out test set (never used for training/CV); 'val' is the CV validation fold.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the energy score.")
    parser.add_argument(
        "--d-matrix",
        type=Path,
        default=None,
        help="D matrix (.npy/.pt) from d_matrix_nnunet.py. Enables the RELU score. "
        "Estimate it on --split val, then report on --split test.",
    )
    return parser


def load_best_trial(best_json: Path | None) -> dict[str, Any]:
    if best_json is None:
        return {}
    payload = json.loads(best_json.read_text(encoding="utf-8"))
    return payload.get("best_trial", payload)


def resolve_model_folder_and_fold(args: argparse.Namespace, best_trial: dict[str, Any]) -> tuple[Path, int, str]:
    checkpoint_path = Path(best_trial["checkpoint"]) if best_trial.get("checkpoint") else None
    model_folder = args.model_folder
    fold = args.fold
    checkpoint_name = args.checkpoint_name

    if checkpoint_path is not None:
        checkpoint_name = checkpoint_path.name
        fold_folder = checkpoint_path.parent
        if model_folder is None:
            model_folder = fold_folder.parent
        if fold is None:
            match = re.fullmatch(r"fold_(\d+)", fold_folder.name)
            if match:
                fold = int(match.group(1))

    if model_folder is None:
        raise ValueError("Provide --model-folder or --best-json containing best_trial.checkpoint.")
    if fold is None:
        fold = 0

    checkpoint = model_folder / f"fold_{fold}" / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    return model_folder, fold, checkpoint_name


def parse_model_folder(model_folder: Path) -> tuple[str, str, str]:
    dataset_name = model_folder.parent.name
    parts = model_folder.name.split("__")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse trainer/plans/configuration from model folder: {model_folder}")
    plans_name = parts[-2]
    configuration = parts[-1]
    return dataset_name, plans_name, configuration


def resolve_preprocessed_dir(args: argparse.Namespace, model_folder: Path) -> tuple[Path, str, str, str]:
    dataset_name, plans_name, configuration = parse_model_folder(model_folder)
    preprocessed_root = args.preprocessed_dir or Path(os.environ.get("nnUNet_preprocessed", ""))
    if not str(preprocessed_root):
        raise ValueError("Provide --preprocessed-dir or export nnUNet_preprocessed.")

    dataset_preprocessed = preprocessed_root / dataset_name
    if not dataset_preprocessed.is_dir():
        raise FileNotFoundError(f"Preprocessed dataset folder not found: {dataset_preprocessed}")
    return dataset_preprocessed, plans_name, configuration, dataset_name


def load_validation_cases(dataset_preprocessed: Path, fold: int) -> list[str]:
    splits_path = dataset_preprocessed / "splits_final.json"
    if not splits_path.is_file():
        raise FileNotFoundError(f"nnU-Net splits file not found: {splits_path}")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} is not present in {splits_path}")

    validation = splits[fold].get("val")
    if not validation:
        raise ValueError(f"No validation cases found for fold {fold} in {splits_path}")
    return list(validation)


def load_test_cases(dataset_preprocessed: Path) -> list[str]:
    test_path = dataset_preprocessed / "test_identifiers.json"
    if not test_path.is_file():
        raise FileNotFoundError(
            f"test_identifiers.json not found: {test_path}. Run the extract_data_*.py "
            "script (writes it to nnUNet_raw) and then train_nnunet.sh / "
            "nnunet_grid_search.sh at least once (they sync it into nnUNet_preprocessed)."
        )
    test_cases = json.loads(test_path.read_text(encoding="utf-8"))
    if not test_cases:
        raise ValueError(f"No test cases found in {test_path}")
    return list(test_cases)


def load_cases_for_split(dataset_preprocessed: Path, dataset_name: str, fold: int, split: str) -> list[str]:
    raw_root = os.environ.get("nnUNet_raw", "")
    if raw_root:
        try:
            sync_split_files(dataset_name, Path(raw_root), dataset_preprocessed.parent)
        except OSError:
            pass  # best-effort: fall back to whatever is already in the preprocessed dir

    if split == "test":
        return load_test_cases(dataset_preprocessed)
    return load_validation_cases(dataset_preprocessed, fold)


def load_case_arrays(case_dir: Path, case: str):
    data_path = case_dir / f"{case}.b2nd"
    seg_path = case_dir / f"{case}_seg.b2nd"

    if not data_path.is_file():
        raise FileNotFoundError(f"Missing data: {data_path}")

    data = blosc2.open(urlpath=str(data_path), mode="r", dparams={"nthreads": 1})[:]
    seg = blosc2.open(urlpath=str(seg_path), mode="r", dparams={"nthreads": 1})[:]

    return data.astype(np.float32), seg.astype(np.int64)


def shape_must_be_divisible_by(configuration_manager) -> tuple[int, ...]:
    strides = np.vstack(configuration_manager.pool_op_kernel_sizes)
    return tuple(int(s) for s in np.prod(strides, axis=0))


def patch_size_of(configuration_manager) -> tuple[int, ...]:
    return tuple(int(s) for s in configuration_manager.patch_size)


def predict_padded(
    network: torch.nn.Module,
    image: torch.Tensor,
    divisor: tuple[int, ...],
    patch_size: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Run the network on a whole (already preprocessed) image, then crop the padding away.

    `patch_size` acts as a minimum shape. nnU-Net crops each case to its nonzero
    region, so 2D BraTS slices near the top of the head can be as small as 6x6.
    Padding only to a multiple of `divisor` would shrink those to 1x1 at the
    bottleneck, and InstanceNorm (track_running_stats=False) rejects a single
    spatial element. nnU-Net's own predictor pads to at least the patch size.
    """
    padded, slicer = pad_nd_image(
        image,
        new_shape=patch_size,
        shape_must_be_divisible_by=divisor,
        return_slicer=True,
    )
    logits = network(padded)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    return logits[(slice(None), slice(None)) + tuple(slicer[2:])]


def fpr95(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return float(fpr[idx])


def compute_scores(
    logits: torch.Tensor,
    temperature: float = 1.0,
    d_matrix: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    softmax = torch.softmax(logits, dim=1)
    msp = torch.max(softmax, dim=1).values
    alpha, beta = doctor(softmax)
    energy_map = energy(logits, temperature)
    sirc = SIRC(msp, 1, energy_map)
    scores = {
        "energy": energy_map,
        "msp": -msp,
        "alpha": alpha,
        "beta": beta,
        "sirc": -sirc,
    }
    if d_matrix is not None:
        scores["relu"] = relu_score(softmax, d_matrix)
    return scores


def _append_segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    num_classes: int,
    seg_metrics: dict[str, list[float]],
) -> None:
    target_batched = target.unsqueeze(0)
    valid_mask = valid.unsqueeze(0)

    if num_classes == 1:
        dice, iou, pixel_acc = compute_binary_metrics(
            logits, target_batched.unsqueeze(1).float(), valid_mask=valid_mask
        )
    else:
        dice, iou, pixel_acc = compute_multiclass_metrics(
            logits, target_batched, num_classes, valid_mask=valid_mask
        )

    if not np.isnan(dice):
        seg_metrics["dice"].append(dice)
        seg_metrics["iou"].append(iou)
        seg_metrics["pixel_acc"].append(pixel_acc)


def append_2d_case_scores(
    network: torch.nn.Module,
    data: np.ndarray,
    seg: np.ndarray,
    device: torch.device,
    num_classes: int,
    temperature: float,
    divisor: tuple[int, ...],
    all_scores: dict[str, list[np.ndarray]],
    all_errors: list[np.ndarray],
    seg_metrics: dict[str, list[float]],
    d_matrix: torch.Tensor | None = None,
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
        target = torch.from_numpy(seg[slice_idx]).to(device)
        logits = predict_padded(network, image, divisor, patch_size)
        pred = logits.argmax(1)[0]
        valid = target >= 0
        error = (pred[valid] != target[valid]).detach().cpu().numpy()

        scores = compute_scores(logits, temperature, d_matrix)
        for name, score in scores.items():
            all_scores[name].append(score[0][valid].detach().cpu().numpy().reshape(-1))
        all_errors.append(error.reshape(-1))

        _append_segmentation_metrics(logits, target, valid, num_classes, seg_metrics)


def append_nd_case_scores(
    network: torch.nn.Module,
    data: np.ndarray,
    seg: np.ndarray,
    device: torch.device,
    num_classes: int,
    temperature: float,
    divisor: tuple[int, ...],
    all_scores: dict[str, list[np.ndarray]],
    all_errors: list[np.ndarray],
    seg_metrics: dict[str, list[float]],
    d_matrix: torch.Tensor | None = None,
    patch_size: tuple[int, ...] | None = None,
) -> None:
    if seg.ndim == data.ndim and seg.shape[0] == 1:
        seg = seg[0]
    image = torch.from_numpy(data).unsqueeze(0).to(device)
    target = torch.from_numpy(seg).to(device)
    logits = predict_padded(network, image, divisor, patch_size)
    pred = logits.argmax(1)[0]
    valid = target >= 0
    error = (pred[valid] != target[valid]).detach().cpu().numpy()

    scores = compute_scores(logits, temperature, d_matrix)
    for name, score in scores.items():
        all_scores[name].append(score[0][valid].detach().cpu().numpy().reshape(-1))
    all_errors.append(error.reshape(-1))

    _append_segmentation_metrics(logits, target, valid, num_classes, seg_metrics)


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

    d_matrix = load_d_matrix(args.d_matrix, num_classes=num_classes) if args.d_matrix else None
    if d_matrix is not None:
        d_matrix = d_matrix.to(device)
    resolved_score_names = resolve_score_names(d_matrix)

    all_scores = {name: [] for name in resolved_score_names}
    all_errors = []
    seg_metrics = {"dice": [], "iou": [], "pixel_acc": []}

    print(f"Model folder : {model_folder}")
    print(f"Checkpoint   : fold_{fold}/{checkpoint_name}")
    print(f"Dataset      : {dataset_name}")
    print(f"Preprocessed : {dataset_preprocessed}")
    print(f"Split        : {args.split}")
    print(f"Cases        : {len(cases)}")
    print(f"D matrix     : {args.d_matrix or 'none (RELU score disabled)'}")
    print(f"Scores       : {', '.join(resolved_score_names)}")

    with torch.no_grad():
        for case in cases:
            data, seg = load_case_arrays(case_dir, case)
            print(f"Evaluating {case}: data={data.shape} seg={seg.shape}")
            if configuration == "2d":
                append_2d_case_scores(
                    predictor.network, data, seg, device, num_classes, args.temperature, divisor,
                    all_scores, all_errors, seg_metrics, d_matrix, patch_size,
                )
            else:
                append_nd_case_scores(
                    predictor.network, data, seg, device, num_classes, args.temperature, divisor,
                    all_scores, all_errors, seg_metrics, d_matrix, patch_size,
                )

    labels = np.concatenate(all_errors).astype(np.uint8)
    detection_metrics = {}
    for name in resolved_score_names:
        scores = np.concatenate(all_scores[name])
        if np.unique(labels).size < 2:
            auroc = aupr = fpr = float("nan")
        else:
            auroc = float(roc_auc_score(labels, scores))
            aupr = float(average_precision_score(labels, scores))
            fpr = fpr95(scores, labels)
        detection_metrics[name] = {"auroc": auroc, "aupr": aupr, "fpr95": fpr}

    segmentation_metrics = {
        name: (float(np.mean(values)) if values else float("nan"))
        for name, values in seg_metrics.items()
    }

    payload = {
        "best_json": str(args.best_json) if args.best_json else None,
        "model_folder": str(model_folder),
        "checkpoint": str(model_folder / f"fold_{fold}" / checkpoint_name),
        "fold": fold,
        "dataset": dataset_name,
        "split": args.split,
        "preprocessed_dir": str(dataset_preprocessed),
        "d_matrix": str(args.d_matrix) if args.d_matrix else None,
        "num_cases": len(cases),
        "num_pixels": int(labels.size),
        "num_error_pixels": int(labels.sum()),
        "segmentation": segmentation_metrics,
        "uncertainty": detection_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    print("\n========= RESULTS =========")
    print("Segmentation metrics:")
    print(json.dumps(segmentation_metrics, indent=2, allow_nan=True))
    print("Uncertainty metrics:")
    print(json.dumps(detection_metrics, indent=2, allow_nan=True))
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
