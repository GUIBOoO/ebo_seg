import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from datasets import get_dataloaders
from losses import build_loss
from metrics import EpochStats, compute_binary_metrics, compute_multiclass_metrics
from models import build_model
from utils import save_checkpoint, set_seed, energy as energy_fn, save_energy_distributions, fpr95_energy


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
    epoch: int | None=None
) -> EpochStats:
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    batches = 0

    all_energies=[]
    all_errors = []

    fpr95 = None

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm.tqdm(loader):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['label'].to(device, non_blocking=True)

            logits = model(images)
            loss_targets = masks if num_classes == 1 else masks.squeeze(1).long()

            loss = criterion(logits, loss_targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if num_classes == 1:
                dice, iou, pixel_acc = compute_binary_metrics(logits, masks)
            else:
                dice, iou, pixel_acc = compute_multiclass_metrics(
                    logits,
                    masks,
                    num_classes,
                )

            energies = energy_fn(logits) 
            preds = logits.argmax(dim=1)

            errors = (preds != masks.squeeze(1)).float()

            all_energies.append(energies.detach().cpu().reshape(-1))
            all_errors.append(errors.detach().cpu().reshape(-1))
            
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_acc += pixel_acc
            batches += 1

    if train:
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
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--loss', type=str, default='auto')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--num-classes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lambda-ebo-in', type=float, default=2)
    parser.add_argument('--lambda-ebo-corr', type=float, default=1)
    parser.add_argument('--margin-correct', type=float, default=-25.0)
    parser.add_argument('--margin-miss', type=float, default=-5.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    energy_save_path_val = args.output_dir / "val_distribs"
    energy_save_path_train= args.output_dir / "train_distribs"
    energy_save_path_val.mkdir(parents=True, exist_ok=True)
    energy_save_path_train.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders()

    model = build_model(args.model, args.num_classes).to(device)
    criterion = build_loss(
        args.loss,
        args.num_classes,
        lambda_ebo_in=args.lambda_ebo_in,
        lambda_ebo_corr=args.lambda_ebo_corr,
        margin_correct=args.margin_correct,
        margin_miss=args.margin_miss,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with (args.output_dir / f'config_{args.loss}.json').open('w', encoding='utf-8') as file_obj:
        json.dump(vars(args), file_obj, indent=2, default=str)

    best_val_loss = math.inf
    history = []

    print(f'Device: {device}')
    print(f'Model: {args.model}')
    print(f'Loss: {args.loss}')
    print(
        'EBO params: '
        f'lambda_ebo_in={args.lambda_ebo_in}, '
        f'lambda_ebo_corr={args.lambda_ebo_corr}, '
        f'margin_correct={args.margin_correct}, '
        f'margin_miss={args.margin_miss}'
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
            epoch=epoch
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
            epoch=epoch
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
    main()
