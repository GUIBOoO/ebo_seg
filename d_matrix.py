import argparse
from pathlib import Path
from typing import Any

import torch
import tqdm

from d_matrix_core import DMatrixAccumulator, save_d_matrix
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
    # img_size fixes TransUNet's position-embedding grid, so it must match training.
    img_size = int(train_args.get("image_size", 256))
    in_channels = int(train_args.get("in_channels", 4))

    if dataset_name == "acdc":
        model = build_model_acdc(resolved_model_name, resolved_num_classes, img_size=img_size)
    else:
        model = build_model_brats(
            resolved_model_name, resolved_num_classes, img_size=img_size, in_channels=in_channels
        )

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
    matrix_classes = 2 if num_classes == 1 else num_classes
    accumulator = DMatrixAccumulator(matrix_classes, device)

    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Computing D"):
            images = batch["image"].to(device, non_blocking=True)
            targets = _normalize_masks(batch["label"].to(device, non_blocking=True))

            logits = model(images)
            probs, preds = _prediction_probabilities(logits, num_classes)
            accumulator.update(probs, preds == targets)

    if model_was_training:
        model.train()

    return accumulator.finalize(num_classes=num_classes, lambda_weight=lambda_weight, eps=eps)


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

    # TransUNet was trained on resized slices and its position-embedding grid is
    # fixed by img_size; feed it the same geometry it saw during training.
    model_name = str(args.model or checkpoint_args.get("model", "unet")).lower()
    resize_to = int(checkpoint_args.get("image_size", 256)) if model_name == "transunet" else None

    loaders = get_dataloaders(
        dataset=dataset_name,
        base_dir=args.dataset_root,
        batch_size=args.batch_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        foreground_only=args.foreground_only,
        image_size=resize_to,
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

    stem = f"d_matrix_{dataset_name}_{args.split}_lambda_{args.lambda_weight:g}"
    metadata = {
        **stats,
        "dataset": dataset_name,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "model": train_args.get("model", args.model or "unet"),
    }
    torch_path, npy_path, json_path = save_d_matrix(args.output_dir, stem, d_matrix, metadata)

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
