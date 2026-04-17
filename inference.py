import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from datasets import get_dataloaders
from metrics import compute_binary_metrics, compute_multiclass_metrics
from models import build_model
from utils import SIRC, doctor, energy


SCORE_NAMES = ("msp", "alpha", "beta", "energy", "sirc")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference for ebo_seg checkpoints.")
    parser.add_argument(
        "modes",
        nargs="+",
        choices=["metrics", "distrib", "visu", "all"],
        help="Which inference outputs to generate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/guibo/links/scratch/models/ebo_seg_unet/best_ce_dice.pt"),
        help="Path to a training checkpoint saved by train_unet.py.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root. Defaults to PYTHON_DATA_DIR if set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where metrics, plots and visualizations will be saved.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-pixels-kde", type=int, default=200000)
    parser.add_argument("--energy-threshold", type=int, default= -25)
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)



def _normalize_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    return masks.long()



def _to_numpy_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().float()
    if image.ndim == 3 and image.shape[0] == 1:
        return image[0].numpy()
    if image.ndim == 3:
        return image.permute(1, 2, 0).numpy()
    return image.numpy()



def _compute_score_maps(logits: torch.Tensor, temperature: float) -> Dict[str, torch.Tensor]:
    softmax = torch.softmax(logits, dim=1)
    msp = torch.max(softmax, dim=1).values
    alpha, beta = doctor(softmax)
    energy_map = energy(logits, temperature)
    sirc = SIRC(msp, 1, energy_map, b=1, a=1)

    return {
        "msp": -msp,
        "alpha": alpha,
        "beta": beta,
        "energy": energy_map,
        "sirc": -sirc,
    }

def _compute_detection_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float | None]:
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": None, "aupr": None, "fpr95": None}

    auroc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[valid[0]]) if valid.size > 0 else 1.0
    return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}



def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    model_name = train_args.get("model", "unet")
    num_classes = int(train_args.get("num_classes", 1))

    model = build_model(model_name, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, train_args



def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    temperature: float,
) -> Tuple[dict, Dict[str, np.ndarray], List[dict]]:
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    num_batches = 0

    error_labels: List[np.ndarray] = []
    score_values = {name: [] for name in SCORE_NAMES}
    per_sample: List[dict] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc="Inference")):
            images = batch["image"].to(device, non_blocking=True)
            masks = _normalize_masks(batch["label"].to(device, non_blocking=True))

            logits = model(images)
            preds = torch.argmax(logits, dim=1) if num_classes > 1 else (torch.sigmoid(logits[:, 0]) > 0.5).long()

            if num_classes == 1:
                dice, iou, pixel_acc = compute_binary_metrics(logits, masks.unsqueeze(1).float())
            else:
                dice, iou, pixel_acc = compute_multiclass_metrics(logits, masks, num_classes)

            scores = _compute_score_maps(logits, temperature)
            error_map = (preds != masks).long()

            total_dice += dice
            total_iou += iou
            total_acc += pixel_acc
            num_batches += 1

            error_labels.append(error_map.reshape(-1).cpu().numpy())
            for name, tensor in scores.items():
                score_values[name].append(tensor.reshape(-1).detach().cpu().numpy())

            batch_size = images.shape[0]
            for sample_offset in range(batch_size):
                sample_id = batch_idx * batch_size + sample_offset
                per_sample.append(
                    {
                        "sample_id": sample_id,
                        "image": images[sample_offset].detach().cpu(),
                        "mask": masks[sample_offset].detach().cpu(),
                        "pred": preds[sample_offset].detach().cpu(),
                        "error": error_map[sample_offset].detach().cpu(),
                        "scores": {name: tensor[sample_offset].detach().cpu() for name, tensor in scores.items()},
                    }
                )

    segmentation_metrics = {
        "dice": total_dice / max(num_batches, 1),
        "iou": total_iou / max(num_batches, 1),
        "pixel_acc": total_acc / max(num_batches, 1),
    }
    flattened_labels = np.concatenate(error_labels) if error_labels else np.array([], dtype=np.int64)
    flattened_scores = {
        name: np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        for name, chunks in score_values.items()
    }
    return segmentation_metrics, {"labels": flattened_labels, **flattened_scores}, per_sample



def save_metrics(segmentation_metrics: dict, score_arrays: Dict[str, np.ndarray], output_dir: Path) -> None:
    detection_metrics = {
        name: _compute_detection_metrics(score_arrays["labels"], score_arrays[name])
        for name in SCORE_NAMES
    }
    payload = {
        "segmentation": segmentation_metrics,
        "uncertainty": detection_metrics,
        "num_pixels": int(score_arrays["labels"].size),
        "num_error_pixels": int(score_arrays["labels"].sum()),
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)

    print("Segmentation metrics:")
    print(json.dumps(segmentation_metrics, indent=2))
    print("Uncertainty metrics:")
    print(json.dumps(detection_metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")



def save_distributions(score_arrays: Dict[str, np.ndarray], output_dir: Path, max_pixels_kde: int) -> None:
    labels = score_arrays["labels"]
    if labels.size == 0:
        print("No pixels available for distributions.")
        return

    rng = np.random.default_rng(42)
    if labels.size > max_pixels_kde:
        indices = rng.choice(labels.size, size=max_pixels_kde, replace=False)
    else:
        indices = np.arange(labels.size)

    distribution_dir = output_dir / "distributions"
    distribution_dir.mkdir(parents=True, exist_ok=True)

    label_names = np.where(labels[indices] == 1, "error", "correct")
    for name in SCORE_NAMES:
        df = pd.DataFrame(
            {
                "score": score_arrays[name][indices],
                "label": label_names,
            }
        )
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=df, x="score", hue="label", common_norm=False)
        plt.title(f"Distribution - {name.upper()}")
        plt.xlabel("Score")
        plt.ylabel("Density")
        save_path = distribution_dir / f"distribution_{name}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved distribution to {save_path}")



def save_visualizations(per_sample: Iterable[dict], output_dir: Path, num_samples: int, energy_threshold: float=0.0) -> None:
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    score_order = ["msp", "alpha", "beta", "energy", "sirc"]
    for sample in list(per_sample)[:num_samples]:
        image = _to_numpy_image(sample["image"])
        mask = sample["mask"].numpy()
        pred = sample["pred"].numpy()
        error = sample["error"].numpy()

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        axes[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axes[0].set_title("Image")

        axes[1].imshow(mask, cmap="tab20")
        axes[1].set_title("Mask")

        axes[2].imshow(pred, cmap="tab20")
        axes[2].set_title("Prediction")

        axes[3].imshow(error, cmap="gray")
        axes[3].set_title("Error")

        for axis, score_name in zip(axes[4:8], score_order):
            axis.imshow(sample["scores"][score_name].numpy(), cmap="viridis")
            axis.set_title(score_name.upper())

        energy = sample["scores"]["energy"].numpy()
        high_energy = (energy > energy_threshold).astype(np.uint8)

        axes[8].imshow(high_energy, cmap="gray")
        axes[8].set_title(f"Energy > {energy_threshold}")

        axes[9].axis("off")

        plt.tight_layout()
        save_path = vis_dir / f"sample_{sample['sample_id']:04d}.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved visualization to {save_path}")



def main() -> None:
    args = build_argparser().parse_args()
    modes = set(args.modes)
    if "all" in modes:
        modes = {"metrics", "distrib", "visu"}

    device = resolve_device(args.device)
    model, train_args = load_checkpoint_model(args.checkpoint, device)
    num_classes = int(train_args.get("num_classes", 1))

    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root_env = os.environ.get("PYTHON_DATA_DIR")
        if not dataset_root_env:
            raise ValueError("Dataset root is required. Pass --dataset-root or set PYTHON_DATA_DIR.")
        dataset_root = Path(dataset_root_env)

    default_output_dir = args.checkpoint.parent / f"inference_{args.checkpoint.stem}"
    output_dir = args.output_dir or default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_dataloaders(base_dir=str(dataset_root), batch_size=args.batch_size)

    segmentation_metrics, score_arrays, per_sample = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=num_classes,
        temperature=args.temperature,
    )

    run_metadata = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(dataset_root),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_classes": num_classes,
        "modes": sorted(modes),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(run_metadata, file_obj, indent=2)

    if "metrics" in modes:
        save_metrics(segmentation_metrics, score_arrays, output_dir)
    if "distrib" in modes:
        save_distributions(score_arrays, output_dir, args.max_pixels_kde)
    if "visu" in modes:
        save_visualizations(per_sample, output_dir, args.num_samples, args.energy_threshold)

    print(f"Artifacts saved in {output_dir}")


if __name__ == "__main__":
    main()
