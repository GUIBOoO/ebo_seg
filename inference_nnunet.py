import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from utils import SIRC, doctor, energy


SCORE_NAMES = ("energy", "msp", "alpha", "beta", "sirc")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the best nnU-Net grid-search checkpoint on its validation fold.")
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

def load_case_arrays(case_dir: Path, case: str):
    data_path = case_dir / f"{case}.b2nd"
    seg_path = case_dir / f"{case}_seg.b2nd"

    if not data_path.is_file():
        raise FileNotFoundError(f"Missing data: {data_path}")

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    seg = torch.load(seg_path, map_location="cpu", weights_only=False)

    return data.numpy().astype(np.float32), seg.numpy().astype(np.int64)


def fpr95(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return float(fpr[idx])


def compute_scores(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    softmax = torch.softmax(logits, dim=1)
    msp = torch.max(softmax, dim=1).values
    alpha, beta = doctor(softmax)
    energy_map = energy(logits)
    sirc = SIRC(msp, 1, energy_map)
    return {
        "energy": energy_map,
        "msp": -msp,
        "alpha": alpha,
        "beta": beta,
        "sirc": -sirc,
    }


def append_2d_case_scores(
    network: torch.nn.Module,
    data: np.ndarray,
    seg: np.ndarray,
    device: torch.device,
    all_scores: dict[str, list[np.ndarray]],
    all_errors: list[np.ndarray],
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
        logits = network(image)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        pred = logits.argmax(1)[0]
        valid = target >= 0
        error = (pred[valid] != target[valid]).detach().cpu().numpy()

        scores = compute_scores(logits)
        for name, score in scores.items():
            all_scores[name].append(score[0][valid].detach().cpu().numpy().reshape(-1))
        all_errors.append(error.reshape(-1))


def append_nd_case_scores(
    network: torch.nn.Module,
    data: np.ndarray,
    seg: np.ndarray,
    device: torch.device,
    all_scores: dict[str, list[np.ndarray]],
    all_errors: list[np.ndarray],
) -> None:
    if seg.ndim == data.ndim and seg.shape[0] == 1:
        seg = seg[0]
    image = torch.from_numpy(data).unsqueeze(0).to(device)
    target = torch.from_numpy(seg).to(device)
    logits = network(image)
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    pred = logits.argmax(1)[0]
    valid = target >= 0
    error = (pred[valid] != target[valid]).detach().cpu().numpy()

    scores = compute_scores(logits)
    for name, score in scores.items():
        all_scores[name].append(score[0][valid].detach().cpu().numpy().reshape(-1))
    all_errors.append(error.reshape(-1))


def main() -> None:
    args = build_argparser().parse_args()
    best_trial = load_best_trial(args.best_json)
    model_folder, fold, checkpoint_name = resolve_model_folder_and_fold(args, best_trial)
    dataset_preprocessed, plans_name, configuration, dataset_name = resolve_preprocessed_dir(args, model_folder)
    case_dir = dataset_preprocessed / f"{plans_name}_{configuration}"
    validation_cases = load_validation_cases(dataset_preprocessed, fold)
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
    predictor.network.eval()

    all_scores = {name: [] for name in SCORE_NAMES}
    all_errors = []

    print(f"Model folder : {model_folder}")
    print(f"Checkpoint   : fold_{fold}/{checkpoint_name}")
    print(f"Dataset      : {dataset_name}")
    print(f"Preprocessed : {dataset_preprocessed}")
    print(f"Cases        : {len(validation_cases)}")

    with torch.no_grad():
        for case in validation_cases:
            data, seg = load_case_arrays(case_dir, case)
            print(f"Evaluating {case}: data={data.shape} seg={seg.shape}")
            if configuration == "2d":
                append_2d_case_scores(predictor.network, data, seg, device, all_scores, all_errors)
            else:
                append_nd_case_scores(predictor.network, data, seg, device, all_scores, all_errors)

    labels = np.concatenate(all_errors).astype(np.uint8)
    metrics = {}
    for name in SCORE_NAMES:
        scores = np.concatenate(all_scores[name])
        if np.unique(labels).size < 2:
            auroc = aupr = fpr = float("nan")
        else:
            auroc = float(roc_auc_score(labels, scores))
            aupr = float(average_precision_score(labels, scores))
            fpr = fpr95(scores, labels)
        metrics[name] = {"auroc": auroc, "aupr": aupr, "fpr95": fpr}

    payload = {
        "best_json": str(args.best_json) if args.best_json else None,
        "model_folder": str(model_folder),
        "checkpoint": str(model_folder / f"fold_{fold}" / checkpoint_name),
        "fold": fold,
        "dataset": dataset_name,
        "preprocessed_dir": str(dataset_preprocessed),
        "num_cases": len(validation_cases),
        "num_pixels": int(labels.size),
        "metrics": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")

    print("\n========= RESULTS =========")
    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
