"""Framework-agnostic core of the RELU D matrix.

Deliberately depends on nothing but torch/numpy so that both pipelines can use
it: the custom UNet/TransUNet one (d_matrix.py, needs monai) and the nnU-Net
one (d_matrix_nnunet.py, runs in a venv without monai).
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


class DMatrixAccumulator:
    """Streams the RELU second-moment matrices E[p p^T] over correct/incorrect pixels.

    Works for any spatial rank (2D slices or 3D volumes) and tolerates an
    ignore label via `valid_mask`.
    """

    def __init__(self, matrix_classes: int, device: torch.device) -> None:
        self.matrix_classes = matrix_classes
        self.incorrect_sum = torch.zeros((matrix_classes, matrix_classes), dtype=torch.float64, device=device)
        self.correct_sum = torch.zeros_like(self.incorrect_sum)
        self.incorrect_count = 0
        self.correct_count = 0

    def update(
        self,
        probs: torch.Tensor,
        correct_mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        """probs: (B, C, *spatial); correct_mask/valid_mask: (B, *spatial)."""
        probs_flat = probs.movedim(1, -1).reshape(-1, self.matrix_classes).double()
        correct_flat = correct_mask.reshape(-1)

        if valid_mask is not None:
            valid_flat = valid_mask.reshape(-1)
            probs_flat = probs_flat[valid_flat]
            correct_flat = correct_flat[valid_flat]

        incorrect_flat = ~correct_flat

        if incorrect_flat.any():
            p_miss = probs_flat[incorrect_flat]
            self.incorrect_sum += p_miss.T @ p_miss
            self.incorrect_count += int(p_miss.shape[0])

        if correct_flat.any():
            p_corr = probs_flat[correct_flat]
            self.correct_sum += p_corr.T @ p_corr
            self.correct_count += int(p_corr.shape[0])

    def finalize(
        self,
        num_classes: int,
        lambda_weight: float = 0.5,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not 0.0 <= lambda_weight <= 1.0:
            raise ValueError(f"lambda_weight must be in [0, 1], got {lambda_weight}")

        incorrect_mean = self.incorrect_sum / max(self.incorrect_count, 1)
        correct_mean = self.correct_sum / max(self.correct_count, 1)

        d_star = torch.relu(lambda_weight * incorrect_mean - (1.0 - lambda_weight) * correct_mean)
        d_star.fill_diagonal_(0.0)

        frobenius_sq = torch.sum(d_star * d_star)
        if frobenius_sq > eps:
            d_matrix = d_star * math.sqrt(self.matrix_classes / float(frobenius_sq.item()))
        else:
            d_matrix = d_star.clone()

        stats = {
            "lambda_weight": float(lambda_weight),
            "num_classes": int(num_classes),
            "matrix_classes": int(self.matrix_classes),
            "num_correct_pixels": int(self.correct_count),
            "num_incorrect_pixels": int(self.incorrect_count),
            "trace_ddt": float(torch.sum(d_matrix * d_matrix).detach().cpu().item()),
            "unnormalized_trace_ddt": float(frobenius_sq.detach().cpu().item()),
            "has_incorrect_pixels": bool(self.incorrect_count > 0),
            "has_correct_pixels": bool(self.correct_count > 0),
        }
        return d_matrix.float().cpu(), stats


def save_d_matrix(
    output_dir: Path,
    stem: str,
    d_matrix: torch.Tensor,
    metadata: dict[str, Any],
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_path = output_dir / f"{stem}.pt"
    npy_path = output_dir / f"{stem}.npy"
    json_path = output_dir / f"{stem}.json"

    torch.save(d_matrix, torch_path)
    np.save(npy_path, d_matrix.numpy())

    metadata = {**metadata, "output_pt": str(torch_path), "output_npy": str(npy_path)}
    with json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2)

    return torch_path, npy_path, json_path
