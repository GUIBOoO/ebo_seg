import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm

from datasets import get_dataloaders, infer_dataset_type
from models import build_model_acdc, build_model_brats


def _normalize_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    return masks.long()


def _prediction_probabilities(logits: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    if num_classes == 1:
        foreground = torch.sigmoid(logits[:, 0])
        probs = torch.stack((1.0 - foreground, foreground), dim=1)
        preds = (foreground > 0.5).long()
        return probs, preds

    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    return probs, preds


def _load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    dataset_name: str,
    model_name: str | None = None,
    num_classes: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    resolved_model_name = model_name or train_args.get("model", "unet")
    resolved_num_classes = int(num_classes if num_classes is not None else train_args.get("num_classes", 1))

    if dataset_name == "acdc":
        model = build_model_acdc(resolved_model_name, resolved_num_classes)
    else:
        model = build_model_brats(resolved_model_name, resolved_num_classes)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    train_args = dict(train_args)
    train_args.setdefault("model", resolved_model_name)
    train_args.setdefault("num_classes", resolved_num_classes)
    return model, train_args


def compute_d_matrix(
    loader,
    model: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    lambda_weight: float = 0.5,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError(f"lambda_weight must be in [0, 1], got {lambda_weight}")

    matrix_classes = 2 if num_classes == 1 else num_classes
    incorrect_sum = torch.zeros((matrix_classes, matrix_classes), dtype=torch.float64, device=device)
    correct_sum = torch.zeros_like(incorrect_sum)
    incorrect_count = 0
    correct_count = 0

    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Computing D"):
            images = batch["image"].to(device, non_blocking=True)
            targets = _normalize_masks(batch["label"].to(device, non_blocking=True))

            logits = model(images)
            probs, preds = _prediction_probabilities(logits, num_classes)

            correct_mask = preds == targets
            incorrect_mask = ~correct_mask

            probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, matrix_classes).double()
            correct_flat = correct_mask.reshape(-1)
            incorrect_flat = incorrect_mask.reshape(-1)

            if incorrect_flat.any():
                p_miss = probs_flat[incorrect_flat]
                incorrect_sum += p_miss.T @ p_miss
                incorrect_count += int(p_miss.shape[0])

            if correct_flat.any():
                p_corr = probs_flat[correct_flat]
                correct_sum += p_corr.T @ p_corr
                correct_count += int(p_corr.shape[0])

    if model_was_training:
        model.train()

    incorrect_mean = incorrect_sum / max(incorrect_count, 1)
    correct_mean = correct_sum / max(correct_count, 1)

    d_star = torch.relu(lambda_weight * incorrect_mean - (1.0 - lambda_weight) * correct_mean)
    d_star.fill_diagonal_(0.0)

    frobenius_sq = torch.sum(d_star * d_star)
    if frobenius_sq > eps:
        d_matrix = d_star * math.sqrt(matrix_classes / float(frobenius_sq.item()))
    else:
        d_matrix = d_star.clone()

    stats = {
        "lambda_weight": float(lambda_weight),
        "num_classes": int(num_classes),
        "matrix_classes": int(matrix_classes),
        "num_correct_pixels": int(correct_count),
        "num_incorrect_pixels": int(incorrect_count),
        "trace_ddt": float(torch.sum(d_matrix * d_matrix).detach().cpu().item()),
        "unnormalized_trace_ddt": float(frobenius_sq.detach().cpu().item()),
        "has_incorrect_pixels": bool(incorrect_count > 0),
        "has_correct_pixels": bool(correct_count > 0),
    }
    return d_matrix.float().cpu(), stats


def _select_split(loaders: tuple, split: str):
    split_to_index = {"train": 0, "val": 1, "test": 2}
    return loaders[split_to_index[split]]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute the RELU D matrix for a pretrained segmentation model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint saved by train_unet.py.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root.")
    parser.add_argument("--dataset", type=str, choices=["acdc", "brats"], default=None)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where D will be saved.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lambda-weight", type=float, default=0.5, help="RELU paper lambda weight for X_-.")
    parser.add_argument("--model", type=str, default=None, help="Override model name from checkpoint args.")
    parser.add_argument("--num-classes", type=int, default=None, help="Override num_classes from checkpoint args.")
    parser.add_argument("--seed", type=int, default=42, help="Random state for dataloader splits.")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--foreground-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def main() -> None:
    args = build_argparser().parse_args()
    device = resolve_device(args.device)

    checkpoint_for_args = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint_for_args.get("args", {})
    dataset_name = infer_dataset_type(
        base_dir=args.dataset_root,
        dataset=args.dataset or checkpoint_args.get("dataset"),
    )
    num_classes = int(args.num_classes if args.num_classes is not None else checkpoint_args.get("num_classes", 1))

    loaders = get_dataloaders(
        dataset=dataset_name,
        base_dir=args.dataset_root,
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        foreground_only=args.foreground_only,
    )
    loader = _select_split(loaders, args.split)

    model, train_args = _load_checkpoint_model(
        checkpoint_path=args.checkpoint,
        device=device,
        dataset_name=dataset_name,
        model_name=args.model,
        num_classes=num_classes,
    )

    d_matrix, stats = compute_d_matrix(
        loader=loader,
        model=model,
        device=device,
        num_classes=num_classes,
        lambda_weight=args.lambda_weight,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"d_matrix_{dataset_name}_{args.split}_lambda_{args.lambda_weight:g}"
    torch_path = args.output_dir / f"{stem}.pt"
    npy_path = args.output_dir / f"{stem}.npy"
    json_path = args.output_dir / f"{stem}.json"

    torch.save(d_matrix, torch_path)
    np.save(npy_path, d_matrix.numpy())

    metadata = {
        **stats,
        "dataset": dataset_name,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "model": train_args.get("model", args.model or "unet"),
        "output_pt": str(torch_path),
        "output_npy": str(npy_path),
    }
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    print("D matrix:")
    print(d_matrix)
    print(f"Trace(D D^T): {stats['trace_ddt']:.6f}")
    print(f"Correct pixels: {stats['num_correct_pixels']}")
    print(f"Incorrect pixels: {stats['num_incorrect_pixels']}")
    print(f"Saved: {torch_path}")
    print(f"Saved: {npy_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
