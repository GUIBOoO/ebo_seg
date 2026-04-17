import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

def fpr95_energy(energies: np.ndarray, labels: np.ndarray) -> float:
    energies = np.asarray(energies).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    if energies.size == 0 or np.unique(labels).size < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(labels, energies)

    idx = np.argmin(np.abs(tpr - 0.95))
    return float(fpr[idx])

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    args,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        path,
    )

def save_energy_distributions(
    energies: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    epoch,
    max_pixels_kde: int = 200_000,
) -> None:

    if energies.size == 0:
        print("No energies available.")
        return

    rng = np.random.default_rng(42)

    energies = energies.reshape(-1)
    labels = labels.reshape(-1)

    if energies.size > max_pixels_kde:
        indices = rng.choice(energies.size, size=max_pixels_kde, replace=False)
    else:
        indices = np.arange(energies.size)

    distribution_dir = output_dir / "distributions"
    distribution_dir.mkdir(parents=True, exist_ok=True)

    label_names = np.where(labels[indices] == 1, "error", "correct")

    df = pd.DataFrame(
        {
            "energy": energies[indices],
            "label": label_names,
        }
    )

    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x="energy", hue="label", common_norm=False)

    plt.title("Energy distribution (correct vs error)")
    plt.xlabel("Energy")
    plt.ylabel("Density")

    save_path = distribution_dir / f"energy_distribution_{epoch}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved distribution to {save_path}")

def energy(logits, T=1):
    energy =  -T * torch.logsumexp(logits / T, dim=1)
    return energy

# def doctor(softmax):
#     d_beta = torch.max(softmax, dim=1).values/(1-torch.max(softmax,dim=1).values)
#     d_alpha =torch.sum(softmax**2, dim=1)/(1-torch.sum(softmax**2, dim=1))
#     return d_alpha, d_beta

def doctor(softmax, eps=1e-6):
    max_p = torch.max(softmax, dim=1).values
    sum_sq = torch.sum(softmax**2, dim=1)

    d_beta = max_p / (1 - max_p + eps)
    d_alpha = sum_sq / (1 - sum_sq + eps)

    alpha_visu = -torch.clamp(d_alpha, 0, d_alpha.quantile(0.90))
    beta_visu  = -torch.clamp(d_beta, 0, d_beta.quantile(0.90))
    return alpha_visu, beta_visu

def SIRC(S1, S1max, S2, b=1, a=1):
    sirc = -(S1max - S1) * (1 + torch.exp(-b * (S2 - a)))
    return sirc