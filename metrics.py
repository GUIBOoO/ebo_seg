from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch


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
