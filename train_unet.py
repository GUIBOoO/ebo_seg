import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from datasets import get_dataloaders, infer_dataset_type
from losses import (
    BoundEBOLogBarrierLoss,
    EBOLossLogBarrier,
    HybridEBOLoss,
    build_loss,
    normalize_loss_name,
)
from metrics import EpochStats, compute_binary_metrics, compute_multiclass_metrics
from models import build_model_acdc, build_model_brats
from utils import (
    save_checkpoint,
    save_gradients,
    set_seed,
    energy as energy_fn,
    hybrid_energy,
    save_energy_distributions,
    fpr95_energy,
)


def _flatten_grads(grads: list[torch.Tensor | None]) -> torch.Tensor | None:
    valid_grads = [grad.reshape(-1) for grad in grads if grad is not None]
    if not valid_grads:
        return None
    return torch.cat(valid_grads)


def _get_base_loss_module(criterion: nn.Module) -> nn.Module:
    if hasattr(criterion, 'base_loss'):
        return criterion.base_loss
    return criterion


def _compute_tracked_losses(
    criterion: nn.Module,
    logits: torch.Tensor,
    loss_targets: torch.Tensor,
    energy: torch.Tensor,
    correct_mask: torch.Tensor,
    incorrect_mask: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    tracked_losses = {
        'loss_corr': None,
        'loss_miss': None,
        'ce_loss': None,
        'dice_loss': None,
    }

    if hasattr(criterion, 'margin_correct') and hasattr(criterion, 'margin_miss'):
        if hasattr(criterion, 'barrier'):
            corr_terms = criterion.barrier(energy - criterion.margin_correct, criterion.t)
            miss_terms = criterion.barrier(criterion.margin_miss - energy, criterion.t)
        else:
            corr_terms = F.relu(energy - criterion.margin_correct) ** 2
            miss_terms = F.relu(criterion.margin_miss - energy) ** 2

        corr_terms = corr_terms[correct_mask]
        if corr_terms.numel() > 0:
            tracked_losses['loss_corr'] = corr_terms.mean()

        miss_terms = miss_terms[incorrect_mask]
        if miss_terms.numel() > 0:
            tracked_losses['loss_miss'] = miss_terms.mean()

    base_loss_module = _get_base_loss_module(criterion)
    if hasattr(base_loss_module, 'ce'):
        ce_loss = base_loss_module.ce(logits, loss_targets)
        tracked_losses['ce_loss'] = getattr(base_loss_module, 'w_ce', 1.0) * ce_loss

    if hasattr(base_loss_module, 'dice'):
        dice_loss = base_loss_module.dice(logits, loss_targets)
        tracked_losses['dice_loss'] = getattr(base_loss_module, 'w_dice', 1.0) * dice_loss

    return tracked_losses


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    train: bool,
    save_energy: bool = False,
    energy_save_path: str | Path = None,
    epoch: int | None=None,
    track_loss_gradients: bool = False,
    barrier_t_growth: float = 1.0,
) -> EpochStats:
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    batches = 0

    all_energies=[]
    all_errors = []
    grad_records = []

    fpr95 = None

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(tqdm.tqdm(loader), start=1):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['label'].to(device, non_blocking=True)

            logits = model(images)
            loss_targets = masks if num_classes == 1 else masks.squeeze(1).long()
            uses_hybrid_ebo = isinstance(criterion, HybridEBOLoss)
            uses_logbar = isinstance(criterion, (EBOLossLogBarrier, BoundEBOLogBarrierLoss))
            if uses_hybrid_ebo:
                loss = criterion(logits, loss_targets, images)
                energy = hybrid_energy(logits, images)
            elif uses_logbar:
                if epoch is not None:
                    criterion.t = criterion.initial_t * (barrier_t_growth ** max(epoch - 1, 0))
                loss = criterion(logits, loss_targets)
                energy = energy_fn(logits)
            else:
                loss = criterion(logits, loss_targets)
                energy = energy_fn(logits)
            if num_classes == 1:
                preds = (torch.sigmoid(logits[:, 0]) > 0.5).long()
            else:
                preds = torch.argmax(logits, dim=1)
            targets = masks.squeeze(1) if masks.ndim == 4 and masks.shape[1] == 1 else masks
            correct_mask = preds == targets
            incorrect_mask = ~correct_mask

            tracked_losses = None
            if train and track_loss_gradients:
                tracked_losses = _compute_tracked_losses(
                    criterion=criterion,
                    logits=logits,
                    loss_targets=loss_targets,
                    energy=energy,
                    correct_mask=correct_mask,
                    incorrect_mask=incorrect_mask,
                )

            if train and track_loss_gradients and tracked_losses is not None:
                tracked_parameters = [p for p in model.parameters() if p.requires_grad]
                grad_record = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'num_correct_pixels': int(correct_mask.sum().item()),
                    'num_incorrect_pixels': int(incorrect_mask.sum().item()),
                }

                for loss_name, tracked_loss in tracked_losses.items():
                    grad_record[loss_name] = (
                        float(tracked_loss.detach().item()) if tracked_loss is not None else float('nan')
                    )
                    if tracked_loss is None:
                        continue

                    tracked_grads = torch.autograd.grad(
                        tracked_loss,
                        tracked_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    flat_grads = _flatten_grads(tracked_grads)
                    if flat_grads is None:
                        continue

                    grad_prefix = loss_name.replace('loss_', 'grad_').replace('_loss', '')
                    grad_record[f'{grad_prefix}_norm'] = float(flat_grads.norm().detach().cpu().item())
                    grad_record[f'{grad_prefix}_mean_abs'] = float(flat_grads.abs().mean().detach().cpu().item())
                    grad_record[f'{grad_prefix}_max_abs'] = float(flat_grads.abs().max().detach().cpu().item())
                    grad_record[f'{grad_prefix}_values'] = flat_grads.detach().cpu()

                grad_records.append(grad_record)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if num_classes == 1:
                dice, iou, pixel_acc = compute_binary_metrics(logits, masks)
            else:
                dice, iou, pixel_acc = compute_multiclass_metrics(
                    logits,
                    targets,
                    num_classes,
                )

            errors = (preds != targets).float()
            correct_mask = preds == targets
            incorrect_mask = ~correct_mask

            if uses_hybrid_ebo:
                energies = hybrid_energy(logits, images)
            else:
                energies = energy_fn(logits)

            all_energies.append(energies.detach().cpu().reshape(-1))
            all_errors.append(errors.detach().cpu().reshape(-1))
            
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_acc += pixel_acc
            batches += 1

    if train:
        if track_loss_gradients and grad_records:
            save_gradients(
                grad_records=grad_records,
                output_dir=energy_save_path,
                epoch=epoch,
            )
        if save_energy:
            energies = torch.cat(all_energies).numpy()
            errors = torch.cat(all_errors).numpy()

            save_energy_distributions(
                energies=energies,
                labels=errors,
                output_dir=energy_save_path,
                epoch=epoch
            )

    else:
        energies = torch.cat(all_energies).numpy()
        errors = torch.cat(all_errors).numpy()

        fpr95 = fpr95_energy(energies, errors)

        if save_energy:
            save_energy_distributions(
                energies=energies,
                labels=errors,
                output_dir=energy_save_path,
                epoch=epoch
            )

    return EpochStats(
        loss=total_loss / max(batches, 1),
        dice=total_dice / max(batches, 1),
        iou=total_iou / max(batches, 1),
        pixel_acc=total_acc / max(batches, 1),
        fpr95=fpr95
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a modular U-Net for segmentation.')
    parser.add_argument('--dataset', type=str, choices=['acdc', 'brats'], default=None)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--loss', type=str, default='ce_dice')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--num-classes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lambda-ebo-in', type=float, default=0.1)
    parser.add_argument('--lambda-ebo-corr', type=float, default=0.1)
    parser.add_argument('--lambda-ebo-cen-in', type=float, default=None)
    parser.add_argument('--lambda-ebo-out-in', type=float, default=None)
    parser.add_argument('--lambda-ebo-cen-corr', type=float, default=None)
    parser.add_argument('--lambda-ebo-out-corr', type=float, default=None)
    parser.add_argument('--boundary-k', type=int, default=1)
    parser.add_argument('--margin-correct', type=float, default=-17)
    parser.add_argument('--margin-miss', type=float, default=-5.0)
    parser.add_argument('--barrier-t', type=float, default=1.0)
    parser.add_argument('--barrier-t-growth', type=float, default=1.1)
    parser.add_argument(
        '--track-loss-gradients',
        action='store_true',
        help='Compute and save per-loss gradient diagnostics during training. Disabled by default to avoid OOM.',
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.loss = normalize_loss_name(args.loss)
    set_seed(args.seed)

    energy_save_path_val = args.output_dir / "val_distribs"
    energy_save_path_train= args.output_dir / "train_distribs"
    energy_save_path_val.mkdir(parents=True, exist_ok=True)
    energy_save_path_train.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = infer_dataset_type(base_dir=args.dataset_root, dataset=args.dataset)

    train_loader, val_loader, _ = get_dataloaders(
        dataset=dataset_name,
        base_dir=str(args.dataset_root),
        batch_size=args.batch_size,
    )

    if dataset_name == 'acdc':
        model = build_model_acdc(args.model, args.num_classes).to(device)
    else:
        model = build_model_brats(args.model, args.num_classes).to(device)
    criterion = build_loss(
        args.loss,
        args.num_classes,
        lambda_ebo_in=args.lambda_ebo_in,
        lambda_ebo_corr=args.lambda_ebo_corr,
        lambda_ebo_cen_in=args.lambda_ebo_cen_in,
        lambda_ebo_out_in=args.lambda_ebo_out_in,
        lambda_ebo_cen_corr=args.lambda_ebo_cen_corr,
        lambda_ebo_out_corr=args.lambda_ebo_out_corr,
        boundary_k=args.boundary_k,
        margin_correct=args.margin_correct,
        margin_miss=args.margin_miss,
        barrier_t=args.barrier_t,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with (args.output_dir / f'config_{args.loss}.json').open('w', encoding='utf-8') as file_obj:
        json.dump(vars(args), file_obj, indent=2, default=str)

    best_val_loss = math.inf
    history = []

    print(f'Device: {device}')
    print(f'Dataset: {dataset_name}')
    print(f'Model: {args.model}')
    print(f'Loss: {args.loss}')
    print(f'Track loss gradients: {args.track_loss_gradients}')
    print(
        'EBO params: '
        f'lambda_ebo_in={args.lambda_ebo_in}, '
        f'lambda_ebo_corr={args.lambda_ebo_corr}, '
        f'lambda_ebo_cen_in={args.lambda_ebo_cen_in}, '
        f'lambda_ebo_out_in={args.lambda_ebo_out_in}, '
        f'lambda_ebo_cen_corr={args.lambda_ebo_cen_corr}, '
        f'lambda_ebo_out_corr={args.lambda_ebo_out_corr}, '
        f'boundary_k={args.boundary_k}, '
        f'margin_correct={args.margin_correct}, '
        f'margin_miss={args.margin_miss}, '
        f'barrier_t={args.barrier_t}, '
        f'barrier_t_growth={args.barrier_t_growth}'
    )

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.num_classes,
            train=True,
            save_energy=True,
            energy_save_path=energy_save_path_train,
            epoch=epoch,
            track_loss_gradients=args.track_loss_gradients,
            barrier_t_growth=args.barrier_t_growth,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            args.num_classes,
            train=False,
            save_energy=True,
            energy_save_path=energy_save_path_val,
            epoch=epoch,
            track_loss_gradients=False,
            barrier_t_growth=args.barrier_t_growth,
        )

        history.append(
            {
                'epoch': epoch,
                'train': asdict(train_stats),
                'val': asdict(val_stats),
            }
        )

        print(
            f'[{epoch:03d}/{args.epochs:03d}] '
            f'train_loss={train_stats.loss:.4f} val_loss={val_stats.loss:.4f} '
            f'val_dice={val_stats.dice:.4f} val_iou={val_stats.iou:.4f} val_acc={val_stats.pixel_acc:.4f} '
            f'val_fpr95={val_stats.fpr95:.4f}'
        )

        save_checkpoint(args.output_dir / f'last_{args.loss}.pt', model, optimizer, epoch, best_val_loss, args)
        if val_stats.loss < best_val_loss:
            best_val_loss = val_stats.loss
            save_checkpoint(args.output_dir / f'best_{args.loss}.pt', model, optimizer, epoch, best_val_loss, args)

        with (args.output_dir / f'history_{args.loss}.json').open('w', encoding='utf-8') as file_obj:
            json.dump(history, file_obj, indent=2)

    print(f'Training complete. Best val loss: {best_val_loss:.4f}')
    print(f'Artifacts saved in: {args.output_dir}')


if __name__ == '__main__':
    print('launching training')
    main()
