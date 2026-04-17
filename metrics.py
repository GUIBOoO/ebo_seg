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


def compute_binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
    preds = (torch.sigmoid(logits) > 0.5).float()
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


def compute_multiclass_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
    preds = torch.argmax(logits, dim=1)
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
