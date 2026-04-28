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
import torch.nn.functional as F

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

def save_gradients(
    grad_records,
    output_dir,
    epoch: int | None = None,
):
    if not grad_records:
        return

    save_dir = Path(output_dir) / "gradients"
    save_dir.mkdir(parents=True, exist_ok=True)

    grad_value_keys = sorted(
        {
            key for record in grad_records for key in record
            if key.endswith("_values")
        }
    )
    grad_names = [key.removesuffix("_values") for key in grad_value_keys]

    if grad_names:
        plt.figure(figsize=(8, 5))
        for grad_name in grad_names:
            grad_tensors = [record[f"{grad_name}_values"] for record in grad_records if f"{grad_name}_values" in record]
            if not grad_tensors:
                continue
            grad_all = torch.cat(grad_tensors).numpy()
            plt.hist(grad_all, bins=100, alpha=0.4, label=grad_name)
        plt.legend()
        plt.title("Gradient distribution")
        plt.xlabel("Gradient value")
        plt.ylabel("Count")
        filename = f"gradients_epoch_{epoch}.png" if epoch is not None else "gradients.png"
        plt.tight_layout()
        plt.savefig(save_dir / filename, dpi=200)
        plt.close()

    rows = []
    for record in grad_records:
        row = {key: value for key, value in record.items() if not key.endswith("_values")}
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = save_dir / "gradient_history.csv"
    if csv_path.exists():
        previous_df = pd.read_csv(csv_path)
        df = pd.concat([previous_df, df], ignore_index=True)
        df = df.drop_duplicates(subset=["epoch", "batch"], keep="last")

    df = df.sort_values(["epoch", "batch"], na_position="last")
    df["global_step"] = np.arange(1, len(df) + 1)
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5))
    plotted_any = False
    for grad_name in grad_names:
        norm_col = f"{grad_name}_norm"
        if norm_col in df:
            plt.plot(df["global_step"], df[norm_col], label=f"||{grad_name}||", alpha=0.6)
            plotted_any = True
    plt.title("Gradient norms during training")
    plt.xlabel("Global step")
    plt.ylabel("Gradient norm")
    if plotted_any:
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "gradient_norms_over_training.png", dpi=200)
    plt.close()

    if epoch is not None:
        epoch_df = df[df["epoch"] == epoch].copy()
        if not epoch_df.empty:
            plt.figure(figsize=(9, 5))
            plotted_any = False
            for grad_name in grad_names:
                norm_col = f"{grad_name}_norm"
                if norm_col in epoch_df:
                    plt.plot(epoch_df["batch"], epoch_df[norm_col], label=f"||{grad_name}||")
                    plotted_any = True
            plt.title(f"Gradient norms during epoch {epoch}")
            plt.xlabel("Batch")
            plt.ylabel("Gradient norm")
            if plotted_any:
                plt.legend()
                plt.tight_layout()
                plt.savefig(save_dir / f"gradient_norms_epoch_{epoch}.png", dpi=200)
            plt.close()

    last_row = df.iloc[-1]
    summary_parts = []
    for grad_name in grad_names:
        grad_norm = last_row.get(f"{grad_name}_norm", float("nan"))
        summary_parts.append(f"{grad_name}_norm={grad_norm:.4f}")
    if summary_parts:
        print(f"[Gradients] {' | '.join(summary_parts)}")

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
    q_alpha = d_alpha.quantile(0.90)
    q_beta = d_beta.quantile(0.90)

    alpha_visu = -torch.clamp(d_alpha, 0, q_alpha)
    beta_visu  = -torch.clamp(d_beta, 0, q_beta)

    return alpha_visu, beta_visu

def SIRC(S1, S1max, S2, b=1, a=1):
    x = -b * (S2 - a)

    x = torch.clamp(x, min=-50, max=50)

    sirc = -(S1max - S1) * (1 + torch.exp(x))
    return sirc

def bilateral_smoothing_local(E, image, radius=3, sigma_spatial=3, sigma_intensity=0.1):
    B, C, H, W = image.shape
    k = 2 * radius + 1

    # Padding pour gérer les bords
    pad = radius
    image_padded = F.pad(image, (pad, pad, pad, pad), mode='reflect')
    E_padded = F.pad(E.unsqueeze(1), (pad, pad, pad, pad), mode='reflect')

    # Extraire les patches locaux
    patches = image_padded.unfold(2, k, 1).unfold(3, k, 1)
    # shape: (B, C, H, W, k, k)

    E_patches = E_padded.unfold(2, k, 1).unfold(3, k, 1)
    # shape: (B, 1, H, W, k, k)

    # Pixel central
    center = image.unsqueeze(-1).unsqueeze(-1)  # (B, C, H, W, 1, 1)

    # --- Spatial kernel (pré-calculé une seule fois) ---
    coords = torch.stack(torch.meshgrid(
        torch.arange(k), torch.arange(k), indexing='ij'
    ), dim=-1).float().to(image.device)

    center_coord = torch.tensor([radius, radius], device=image.device)
    spatial = coords - center_coord
    spatial = torch.sum(spatial ** 2, dim=-1)
    spatial = torch.exp(-spatial / (2 * sigma_spatial ** 2))  # (k, k)

    spatial = spatial.view(1, 1, 1, 1, k, k)

    # --- Intensity kernel ---
    diff = (patches - center) ** 2
    diff = diff.sum(dim=1, keepdim=True)  # somme sur les canaux
    intensity = torch.exp(-diff / (2 * sigma_intensity ** 2))

    # --- Poids bilatéraux ---
    weights = spatial * intensity

    # Normalisation
    weights = weights / (weights.sum(dim=(-1, -2), keepdim=True) + 1e-8)

    # Application sur E
    E_smooth = (weights * E_patches).sum(dim=(-1, -2))

    return E_smooth.squeeze(1)


def hybrid_energy(logits, image, T=1, alpha=0.5, radius=3):
    E = -T * torch.logsumexp(logits / T, dim=1)
    E_spatial = bilateral_smoothing_local(E, image, radius=radius)
    return alpha * E + (1 - alpha) * E_spatial