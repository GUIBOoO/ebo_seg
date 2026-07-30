from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


@dataclass
class EpochStats:
    loss: float
    dice: float
    iou: float
    pixel_acc: float
    fpr95: Optional[float]=None


def compute_binary_metrics_from_preds(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
    preds = preds.float()
    targets = targets.float()

    intersection = (preds * targets).sum().item()
    pred_sum = preds.sum().item()
    target_sum = targets.sum().item()
    union = pred_sum + target_sum - intersection
    total = targets.numel()
    correct = (preds == targets).sum().item()

    dice = (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    pixel_acc = correct / total
    return dice, iou, pixel_acc


def compute_binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:
    preds = (torch.sigmoid(logits) > 0.5).float()
    targets = targets.float()

    if valid_mask is not None:
        if valid_mask.ndim == preds.ndim - 1:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.bool()
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    if preds.numel() == 0:
        return float("nan"), float("nan"), float("nan")

    return compute_binary_metrics_from_preds(preds, targets)


def compute_multiclass_metrics_from_preds(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Tuple[float, float, float]:
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds={preds.shape}, targets={targets.shape}")

    total = targets.numel()
    correct = (preds == targets).sum().item()
    pixel_acc = correct / total

    dices = []
    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().item()
        pred_sum = pred_cls.sum().item()
        target_sum = target_cls.sum().item()
        union = pred_sum + target_sum - intersection
        dices.append((2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6))
        ious.append((intersection + 1e-6) / (union + 1e-6))

    return float(np.mean(dices)), float(np.mean(ious)), pixel_acc


def _surface_distances(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from every surface pixel of `mask` to the closest surface pixel of `reference`."""
    footprint = generate_binary_structure(mask.ndim, connectivity=1)
    # border_value=0 keeps the part of an object touching the image edge in its
    # surface; otherwise a structure cropped by the frame looks closed.
    mask_surface = mask ^ binary_erosion(mask, structure=footprint, border_value=0)
    reference_surface = reference ^ binary_erosion(reference, structure=footprint, border_value=0)
    distance_to_reference = distance_transform_edt(~reference_surface)
    return distance_to_reference[mask_surface]


def hausdorff_distance_95_binary(pred: np.ndarray, target: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff distance, in pixels.

    Undefined (NaN) when either mask is empty: there is no surface to measure
    against.
    """
    if not pred.any() or not target.any():
        return float("nan")
    distances = np.concatenate([
        _surface_distances(pred, target),
        _surface_distances(target, pred),
    ])
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def hausdorff_distance_95_from_preds(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
) -> Tuple[float, int, int]:
    """HD95 averaged over the (slice, class) pairs where it is defined.

    Background is excluded by default: its surface is dominated by the image
    frame, which says nothing about segmentation quality. A class missing from
    the prediction or from the ground truth is skipped, so the returned count of
    undefined pairs is what tells you a class was missed entirely.

    Returns (mean HD95, number of defined pairs, number of undefined pairs).
    """
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    if preds_np.ndim == 2:
        preds_np = preds_np[None]
        targets_np = targets_np[None]

    if num_classes == 1:
        class_ids: Iterable[int] = (1,)
    else:
        class_ids = range(0 if include_background else 1, num_classes)

    values = []
    num_undefined = 0
    for index in range(preds_np.shape[0]):
        for class_id in class_ids:
            value = hausdorff_distance_95_binary(
                preds_np[index] == class_id,
                targets_np[index] == class_id,
            )
            if np.isnan(value):
                num_undefined += 1
            else:
                values.append(value)

    mean_value = float(np.mean(values)) if values else float("nan")
    return mean_value, len(values), num_undefined


def hausdorff_distance_95(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
) -> Tuple[float, int, int]:
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    if num_classes == 1:
        preds = (torch.sigmoid(logits) > 0.5).long()
        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)
    else:
        preds = torch.argmax(logits, dim=1)

    return hausdorff_distance_95_from_preds(preds, targets.long(), num_classes, include_background)


def compute_multiclass_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float]:

    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    preds = torch.argmax(logits, dim=1)

    if valid_mask is not None:
        if valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask.squeeze(1)
        valid_mask = valid_mask.bool()
        preds = preds[valid_mask]
        targets = targets[valid_mask]

    if preds.numel() == 0:
        return float("nan"), float("nan"), float("nan")

    return compute_multiclass_metrics_from_preds(preds, targets, num_classes)
