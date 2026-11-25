import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from datasets import SUPPORTED_DATASETS, get_dataloaders, infer_dataset_type
from metrics import (
    compute_binary_metrics,
    compute_binary_metrics_from_preds,
    compute_multiclass_metrics,
    compute_multiclass_metrics_from_preds,
    hausdorff_distance_95_from_preds,
)
from models import build_model_2d, build_model_acdc
from utils import SIRC, doctor, energy, hybrid_energy, load_d_matrix, relu_score


BASE_SCORE_NAMES = ("msp", "alpha", "beta", "energy", "hybrid_energy", "sirc")

# Scores plotted on a fixed x-range so distributions stay comparable across
# runs, datasets and models instead of being auto-scaled to each run's spread.
FIXED_SCORE_XLIM: Dict[str, Tuple[float, float]] = {"energy": (-80.0, 0.0)}


def energy_threshold_arg(raw: str) -> float | None:
    """--energy-threshold accepts a number or 'auto' (resolved after the pass)."""
    if raw.strip().lower() == "auto":
        return None
    try:
        return float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or 'auto', got '{raw}'") from None


def parse_score_thresholds(raw_values: Iterable[str]) -> Dict[str, float]:
    """Turn the repeatable --score-threshold NAME=VALUE flags into a dict."""
    known = set(BASE_SCORE_NAMES) | {"relu"}
    thresholds: Dict[str, float] = {}
    for raw in raw_values:
        name, separator, value = raw.partition("=")
        name = name.strip().lower()
        if not separator:
            raise ValueError(f"--score-threshold expects NAME=VALUE, got '{raw}'")
        if name not in known:
            raise ValueError(
                f"--score-threshold: unknown score '{name}'. Known scores: {', '.join(sorted(known))}"
            )
        thresholds[name] = float(value)
    return thresholds


def resolve_score_names(d_matrix: torch.Tensor | None) -> Tuple[str, ...]:
    """RELU needs a precomputed D matrix, so it is only scored when one is given."""
    return BASE_SCORE_NAMES + (("relu",) if d_matrix is not None else ())


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
        "--dataset",
        type=str,
        choices=list(SUPPORTED_DATASETS),
        default=None,
        help="Dataset name. If omitted, inferred from the checkpoint args or dataset root.",
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
    parser.add_argument(
        "--energy-threshold",
        type=energy_threshold_arg,
        default=-25.0,
        help="Energy cut behind dice_identified_correct and energy_below_threshold. "
        "A number is used as is; 'auto' puts energy on the same operating point as "
        "every other score, i.e. the energy value that flags --score-tpr of the errors.",
    )
    parser.add_argument("--msp-threshold", type=float, default=0.9)
    parser.add_argument(
        "--score-threshold",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Absolute threshold for one score in the 'segmentation.score_thresholds' "
        "block, written in the convention shared by every score map: the higher the "
        "score, the more suspicious the pixel, so the kept pixels are score < VALUE. "
        "Repeatable, e.g. --score-threshold sirc=2.5 --score-threshold alpha=-1. "
        "Scores left out fall back to their --score-tpr operating point.",
    )
    parser.add_argument(
        "--score-tpr",
        type=float,
        default=0.95,
        help="Default operating point for the score_thresholds block: each score is "
        "cut where it flags this fraction of the error pixels, which is the same "
        "point the reported FPR95 describes.",
    )
    parser.add_argument(
        "--score-threshold-max-pixels",
        type=int,
        default=2_000_000,
        help="Pixels sampled to locate the --score-tpr thresholds. The metrics "
        "themselves always use every pixel. 0 disables the subsampling.",
    )
    parser.add_argument(
        "--d-matrix",
        type=Path,
        default=None,
        help="D matrix (.npy/.pt) from d_matrix.py. Enables the RELU score. "
        "Estimate it on the val split, then report on the test split.",
    )
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
    if image.ndim == 3 and image.shape[0] == 3:
        return image.permute(1, 2, 0).numpy()
    if image.ndim == 3:
        return image[0].numpy()
    return image.numpy()



def _compute_score_maps(
    logits: torch.Tensor,
    images: torch.Tensor,
    temperature: float,
    d_matrix: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    softmax = torch.softmax(logits, dim=1)
    msp = torch.max(softmax, dim=1).values
    alpha, beta = doctor(softmax)
    energy_map = energy(logits, temperature)
    hybrid_energy_map = hybrid_energy(logits, images, T=temperature)
    sirc = SIRC(msp, 1, energy_map, b=1, a=1)

    scores = {
        "msp": -msp,
        "alpha": alpha,
        "beta": beta,
        "energy": energy_map,
        "hybrid_energy": hybrid_energy_map,
        "sirc": -sirc,
    }
    if d_matrix is not None:
        # d_matrix.py builds a 2x2 matrix for binary models (stacking 1-p, p),
        # while softmax over a single logit channel is degenerate (all ones).
        if logits.shape[1] == 1:
            foreground = torch.sigmoid(logits[:, 0])
            relu_probs = torch.stack((1.0 - foreground, foreground), dim=1)
        else:
            relu_probs = softmax
        scores["relu"] = relu_score(relu_probs, d_matrix)
    return scores

def _threshold_at_tpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_tpr: float,
    max_pixels: int,
) -> Tuple[float | None, float | None]:
    """Score value at which `target_tpr` of the error pixels are flagged.

    Every score map follows the same convention: the higher, the more suspicious.
    A pixel is therefore *rejected* when `score >= threshold`, which is exactly
    how roc_curve defines its thresholds, and the FPR95 already reported is the
    false positive rate at this very point. Estimating it on a subsample keeps
    the extra sort cheap; the metrics themselves use every pixel.
    """
    if labels.size == 0 or np.unique(labels).size < 2:
        return None, None

    if 0 < max_pixels < labels.size:
        rng = np.random.default_rng(42)
        indices = rng.choice(labels.size, size=max_pixels, replace=False)
        labels = labels[indices]
        scores = scores[indices]

    _, tpr, thresholds = roc_curve(labels, scores)
    reached = np.where(tpr >= target_tpr)[0]
    if reached.size == 0:
        return None, None
    idx = int(reached[0])
    return float(thresholds[idx]), float(tpr[idx])


def _segmentation_metrics_from_arrays(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> Tuple[float, float, float]:
    if preds.size == 0:
        return float("nan"), float("nan"), float("nan")
    pred_tensor = torch.from_numpy(preds.astype(np.int64, copy=False))
    target_tensor = torch.from_numpy(targets.astype(np.int64, copy=False))
    if num_classes == 1:
        return compute_binary_metrics_from_preds(pred_tensor, target_tensor)
    return compute_multiclass_metrics_from_preds(pred_tensor, target_tensor, num_classes)


def _macro_dice_over_batches(
    preds: np.ndarray,
    targets: np.ndarray,
    kept: np.ndarray,
    batch_sizes: List[int],
    num_classes: int,
) -> float:
    """Dice of the kept pixels averaged per batch, batches keeping none skipped.

    Matches what the evaluation loop used to accumulate on the fly; it is only
    computed afterwards now, so the threshold no longer has to be known upfront.
    """
    total = 0.0
    counted = 0
    start = 0
    for size in batch_sizes:
        stop = start + size
        batch_kept = kept[start:stop]
        if batch_kept.any():
            dice, _, _ = _segmentation_metrics_from_arrays(
                preds[start:stop][batch_kept],
                targets[start:stop][batch_kept],
                num_classes,
            )
            total += dice
            counted += 1
        start = stop
    return total / counted if counted else float("nan")


def _score_threshold_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    scores: Dict[str, np.ndarray],
    num_classes: int,
    explicit_thresholds: Dict[str, float],
    target_tpr: float,
    max_pixels_threshold: int,
) -> Dict[str, dict]:
    """Segmentation quality of the pixels each score keeps, one entry per score.

    `energy_below_threshold` and `msp_above_threshold` answer the same question
    for two hand-picked absolute thresholds. Those two numbers are not
    comparable across scores, so the default here is the operating point where
    each score flags `target_tpr` of the errors: every score then rejects the
    same share of the errors and only the pixels it sacrifices differ.
    """
    blocks: Dict[str, dict] = {}
    for name in _present_score_names(scores):
        score = scores[name]
        explicit = explicit_thresholds.get(name)
        if explicit is not None:
            threshold: float | None = float(explicit)
            achieved_tpr: float | None = None
            source = "explicit"
        else:
            threshold, achieved_tpr = _threshold_at_tpr(labels, score, target_tpr, max_pixels_threshold)
            source = f"tpr>={target_tpr:g}"

        block: dict = {
            "threshold": threshold,
            "threshold_source": source,
            "rule": "kept pixels are score < threshold",
        }
        if explicit is None:
            block["target_tpr"] = target_tpr
            block["achieved_tpr"] = achieved_tpr

        if threshold is None:
            block.update(
                {
                    "num_pixels": 0,
                    "frac_pixels": float("nan"),
                    "dice": float("nan"),
                    "iou": float("nan"),
                    "pixel_acc": float("nan"),
                }
            )
            blocks[name] = block
            continue

        kept = score < threshold
        num_kept = int(kept.sum())
        dice, iou, pixel_acc = _segmentation_metrics_from_arrays(preds[kept], targets[kept], num_classes)
        block.update(
            {
                "num_pixels": num_kept,
                "frac_pixels": float(num_kept / score.size) if score.size else float("nan"),
                "dice": dice,
                "iou": iou,
                "pixel_acc": pixel_acc,
            }
        )
        blocks[name] = block
    return blocks


def _compute_detection_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float | None]:
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": None, "aupr": None, "fpr95": None}
    print("labels shape:", labels.shape)
    print("scores shape:", scores.shape)

    print("NaN in scores:", np.isnan(scores).sum())
    print("Inf in scores:", np.isinf(scores).sum())

    print("scores min:", np.nanmin(scores))
    print("scores max:", np.nanmax(scores))
    auroc = float(roc_auc_score(labels, scores))
    aupr = float(average_precision_score(labels, scores))
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.where(tpr >= 0.95)[0]
    if valid.size > 0:
        print("valid.size >0")
        fpr95 = float(fpr[valid[0]]) 
    else:
        fpr95= 1.0
    return {"auroc": auroc, "aupr": aupr, "fpr95": fpr95}



def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    dataset_name: str,
) -> Tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    model_name = train_args.get("model", "unet")
    num_classes = int(train_args.get("num_classes", 1))
    # img_size must match training: it fixes TransUNet's position-embedding grid.
    img_size = int(train_args.get("image_size", 256))
    in_channels = int(train_args.get("in_channels", 4))

    if dataset_name == "acdc":
        model = build_model_acdc(model_name, num_classes, img_size=img_size)
    else:
        model = build_model_2d(model_name, num_classes, img_size=img_size, in_channels=in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, train_args



def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    temperature: float,
    energy_threshold: float | None,
    msp_threshold: float,
    d_matrix: torch.Tensor | None = None,
    score_thresholds: Dict[str, float] | None = None,
    score_tpr: float = 0.95,
    score_threshold_max_pixels: int = 2_000_000,
) -> Tuple[dict, Dict[str, np.ndarray], List[dict]]:
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    num_batches = 0

    total_hd95 = 0.0
    num_hd95_batches = 0
    num_hd95_defined = 0
    num_hd95_undefined = 0

    error_labels: List[np.ndarray] = []
    # Predictions and targets of the whole test set, kept so any score can be
    # thresholded after the pass. uint8 is enough for every dataset here and
    # costs far less than the per-score lists of selected pixels.
    label_dtype = np.uint8 if num_classes <= 256 else np.int16
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    score_values = {name: [] for name in resolve_score_names(d_matrix)}
    per_sample: List[dict] = []
    # Length of each batch in the flattened arrays, so the per-batch averages can
    # be rebuilt after the pass, once the thresholds are known.
    batch_sizes: List[int] = []

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

            batch_hd95, hd95_defined, hd95_undefined = hausdorff_distance_95_from_preds(
                preds, masks, num_classes
            )
            num_hd95_defined += hd95_defined
            num_hd95_undefined += hd95_undefined
            if not np.isnan(batch_hd95):
                total_hd95 += batch_hd95
                num_hd95_batches += 1

            scores = _compute_score_maps(logits, images, temperature, d_matrix)
            for name, tensor in scores.items():
                arr = tensor.detach().cpu().numpy()
                if not np.isfinite(arr).all():
                    print(f"[NaN DETECTED] score: {name}")
                    print("NaN:", np.isnan(arr).sum(), "Inf:", np.isinf(arr).sum())
                    print("Min:", np.nanmin(arr), "Max:", np.nanmax(arr))
            error_map = (preds != masks).long()

            total_dice += dice
            total_iou += iou
            total_acc += pixel_acc
            num_batches += 1

            error_labels.append(error_map.reshape(-1).cpu().numpy())
            all_preds.append(preds.reshape(-1).cpu().numpy().astype(label_dtype, copy=False))
            all_targets.append(masks.reshape(-1).cpu().numpy().astype(label_dtype, copy=False))
            batch_sizes.append(int(error_map.numel()))
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
        # Pixels, foreground classes only, (slice, class) pairs where the class
        # is absent from the prediction or the ground truth are skipped.
        "hd95": total_hd95 / num_hd95_batches if num_hd95_batches > 0 else float("nan"),
        "hd95_num_defined_pairs": num_hd95_defined,
        "hd95_num_undefined_pairs": num_hd95_undefined,
    }

    flattened_labels = np.concatenate(error_labels) if error_labels else np.array([], dtype=np.int64)
    flattened_scores = {
        name: np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        for name, chunks in score_values.items()
    }

    flattened_preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=label_dtype)
    flattened_targets = np.concatenate(all_targets) if all_targets else np.array([], dtype=label_dtype)

    # energy_threshold=None asks for the same operating point the other scores
    # use by default: the energy value that flags `score_tpr` of the errors.
    energy_scores = flattened_scores.get("energy", np.array([], dtype=np.float32))
    if energy_threshold is None:
        resolved_energy_threshold, energy_achieved_tpr = _threshold_at_tpr(
            flattened_labels, energy_scores, score_tpr, score_threshold_max_pixels
        )
        energy_threshold_source = f"tpr>={score_tpr:g}"
    else:
        resolved_energy_threshold = float(energy_threshold)
        energy_achieved_tpr = None
        energy_threshold_source = "explicit"

    if resolved_energy_threshold is None:
        energy_kept = np.zeros(energy_scores.size, dtype=bool)
    else:
        energy_kept = energy_scores < resolved_energy_threshold
    energy_dice, energy_iou, energy_acc = _segmentation_metrics_from_arrays(
        flattened_preds[energy_kept], flattened_targets[energy_kept], num_classes
    )

    # Same pixels as energy_below_threshold, averaged per batch instead of
    # pooled, so it stays comparable to `dice` above.
    segmentation_metrics["dice_identified_correct"] = _macro_dice_over_batches(
        flattened_preds, flattened_targets, energy_kept, batch_sizes, num_classes
    )
    segmentation_metrics["dice_identified_correct_threshold"] = resolved_energy_threshold

    segmentation_metrics["energy_below_threshold"] = {
        "threshold": resolved_energy_threshold,
        "threshold_source": energy_threshold_source,
        "num_pixels": int(energy_kept.sum()),
        "dice": energy_dice,
        "iou": energy_iou,
        "pixel_acc": energy_acc,
    }
    if energy_threshold is None:
        segmentation_metrics["energy_below_threshold"]["target_tpr"] = score_tpr
        segmentation_metrics["energy_below_threshold"]["achieved_tpr"] = energy_achieved_tpr

    # scores["msp"] holds -max_softmax, so "max_softmax > t" is "score < -t".
    msp_scores = flattened_scores.get("msp", np.array([], dtype=np.float32))
    msp_kept = msp_scores < -msp_threshold
    msp_dice, msp_iou, msp_acc = _segmentation_metrics_from_arrays(
        flattened_preds[msp_kept], flattened_targets[msp_kept], num_classes
    )
    segmentation_metrics["msp_above_threshold"] = {
        "threshold": msp_threshold,
        "num_pixels": int(msp_kept.sum()),
        "dice": msp_dice,
        "iou": msp_iou,
        "pixel_acc": msp_acc,
    }

    segmentation_metrics["score_thresholds"] = _score_threshold_metrics(
        preds=flattened_preds,
        targets=flattened_targets,
        labels=flattened_labels,
        scores=flattened_scores,
        num_classes=num_classes,
        explicit_thresholds=score_thresholds or {},
        target_tpr=score_tpr,
        max_pixels_threshold=score_threshold_max_pixels,
    )
    del flattened_preds, flattened_targets, all_preds, all_targets

    return segmentation_metrics, {"labels": flattened_labels, **flattened_scores}, per_sample



def _present_score_names(score_arrays: Dict[str, np.ndarray]) -> Tuple[str, ...]:
    return tuple(name for name in (*BASE_SCORE_NAMES, "relu") if name in score_arrays)


def save_metrics(segmentation_metrics: dict, score_arrays: Dict[str, np.ndarray], output_dir: Path) -> None:
    detection_metrics = {
        name: _compute_detection_metrics(score_arrays["labels"], score_arrays[name])
        for name in _present_score_names(score_arrays)
    }
    payload = {
        "segmentation": segmentation_metrics,
        "uncertainty": detection_metrics,
        "num_pixels": int(score_arrays["labels"].size),
        "num_error_pixels": int(score_arrays["labels"].sum()),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
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
    for name in _present_score_names(score_arrays):
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
        xlim = FIXED_SCORE_XLIM.get(name)
        if xlim is not None:
            plt.xlim(*xlim)
        save_path = distribution_dir / f"distribution_{name}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved distribution to {save_path}")



def save_visualizations(
    per_sample: Iterable[dict],
    output_dir: Path,
    num_samples: int,
    energy_threshold: float = 0.0,
    msp_threshold: float = 0.9,
) -> None:
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for sample in list(per_sample)[:num_samples]:
        image = _to_numpy_image(sample["image"])
        pred = sample["pred"].numpy()
        error = sample["error"].numpy()

        energy = sample["scores"]["energy"].numpy()

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
        axes[0].set_title("Image")

        axes[1].imshow(pred, cmap="tab20")
        axes[1].set_title("Prediction")

        axes[2].imshow(error, cmap="gray_r")
        axes[2].set_title("Error")

        axes[3].imshow(energy, cmap="viridis")
        axes[3].set_title("Energy")

        for axis in axes:
            axis.axis("off")

        msp = -sample["scores"]["msp"].numpy()
        error_pixels = error.astype(bool)
        correct_pixels = ~error_pixels
        summary_energy_detected = energy > -5.0
        summary_msp_detected = msp < 0.999

        rejection_summary = np.zeros((*error.shape, 3), dtype=np.float32)
        rejection_summary[correct_pixels & summary_energy_detected] = (0.95, 0.55, 0.15)
        rejection_summary[correct_pixels & summary_msp_detected] = (0.55, 0.30, 0.85)
        rejection_summary[correct_pixels & summary_energy_detected & summary_msp_detected] = (0.10, 0.75, 0.35)
        rejection_summary[error_pixels & summary_energy_detected] = (0.90, 0.10, 0.10)
        rejection_summary[error_pixels & summary_msp_detected] = (0.10, 0.35, 0.95)
        rejection_summary[error_pixels & summary_energy_detected & summary_msp_detected] = (1.00, 0.90, 0.05)

        legend_handles = [
            Patch(color=(0.90, 0.10, 0.10), label="Error: energy"),
            Patch(color=(0.10, 0.35, 0.95), label="Error: MSP"),
            Patch(color=(1.00, 0.90, 0.05), label="Error: both"),
            Patch(color=(0.95, 0.55, 0.15), label="Correct: energy"),
            Patch(color=(0.55, 0.30, 0.85), label="Correct: MSP"),
            Patch(color=(0.10, 0.75, 0.35), label="Correct: both"),
        ]

        plt.tight_layout()
        save_path = vis_dir / f"sample_{sample['sample_id']:04d}.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved visualization to {save_path}")

        summary_fig, summary_axis = plt.subplots(1, 1, figsize=(8, 8))
        summary_axis.imshow(rejection_summary)
        summary_axis.set_title("Detected pixels summary")
        summary_axis.axis("off")
        summary_fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)
        summary_fig.tight_layout(rect=(0, 0.08, 1, 1))
        summary_path = vis_dir / f"sample_{sample['sample_id']:04d}_detected_pixels_summary.png"
        summary_fig.savefig(summary_path)
        plt.close(summary_fig)
        print(f"Saved detected pixels summary to {summary_path}")



def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    modes = set(args.modes)
    if "all" in modes:
        modes = {"metrics", "distrib", "visu"}

    device = resolve_device(args.device)
    try:
        score_thresholds = parse_score_thresholds(args.score_threshold)
    except ValueError as error:
        parser.error(str(error))

    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root_env = os.environ.get("PYTHON_DATA_DIR")
        if not dataset_root_env:
            raise ValueError("Dataset root is required. Pass --dataset-root or set PYTHON_DATA_DIR.")
        dataset_root = Path(dataset_root_env)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})
    dataset_name = infer_dataset_type(
        base_dir=dataset_root,
        dataset=args.dataset or train_args.get("dataset"),
    )
    num_classes = int(train_args.get("num_classes", 1))
    model, train_args = load_checkpoint_model(args.checkpoint, device, dataset_name)

    # Binary models get a 2x2 D (background/foreground), see d_matrix.py.
    matrix_classes = 2 if num_classes == 1 else num_classes
    d_matrix = load_d_matrix(args.d_matrix, num_classes=matrix_classes).to(device) if args.d_matrix else None

    default_output_dir = args.checkpoint.parent / f"inference_{args.checkpoint.stem}"
    output_dir = args.output_dir or default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # TransUNet was trained on resized slices; feed it the same geometry.
    resize_to = int(train_args.get("image_size", 256)) if str(train_args.get("model", "unet")).lower() == "transunet" else None
    _, _, test_loader = get_dataloaders(
        dataset=dataset_name,
        base_dir=str(dataset_root),
        batch_size=args.batch_size,
        image_size=resize_to,
    )

    segmentation_metrics, score_arrays, per_sample = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=num_classes,
        temperature=args.temperature,
        energy_threshold=args.energy_threshold,
        msp_threshold=args.msp_threshold,
        d_matrix=d_matrix,
        score_thresholds=score_thresholds,
        score_tpr=args.score_tpr,
        score_threshold_max_pixels=args.score_threshold_max_pixels,
    )

    run_metadata = {
        "checkpoint": str(args.checkpoint),
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_classes": num_classes,
        "modes": sorted(modes),
        "energy_threshold": "auto" if args.energy_threshold is None else args.energy_threshold,
        "msp_threshold": args.msp_threshold,
        "score_thresholds": score_thresholds,
        "score_tpr": args.score_tpr,
    }
    # Inference can take minutes; recreate the directory in case it was removed
    # in the meantime, rather than losing the whole pass on a write error.
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(run_metadata, file_obj, indent=2)

    if "metrics" in modes:
        save_metrics(segmentation_metrics, score_arrays, output_dir)
    if "distrib" in modes:
        save_distributions(score_arrays, output_dir, args.max_pixels_kde)
    if "visu" in modes:
        save_visualizations(
            per_sample,
            output_dir,
            args.num_samples,
            args.energy_threshold,
            args.msp_threshold,
        )

    print(f"Artifacts saved in {output_dir}")


if __name__ == "__main__":
    main()
