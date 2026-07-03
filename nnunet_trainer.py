import os
import json
import math
from time import time

import numpy as np
import torch
from torch import autocast
from torch import distributed as dist
from batchgenerators.utilities.file_and_folder_operations import join, isfile

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from losses import build_loss, normalize_loss_name


def _energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def _fpr95_energy(energies: np.ndarray, labels: np.ndarray) -> float:
    energies = np.asarray(energies).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)

    if energies.size == 0 or np.unique(labels).size < 2:
        return float("nan")

    order = np.argsort(energies)[::-1]
    labels = labels[order]
    positives = labels.sum()
    negatives = labels.size - positives
    tpr = np.concatenate([[0.0], np.cumsum(labels) / positives])
    fpr = np.concatenate([[0.0], np.cumsum(~labels) / negatives])
    idx = np.argmin(np.abs(tpr - 0.95))
    return float(fpr[idx])


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _sample_flat_tensors(
    values: torch.Tensor,
    labels: torch.Tensor,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.reshape(-1)
    labels = labels.reshape(-1)
    if max_samples > 0 and values.numel() > max_samples:
        indices = torch.randperm(values.numel(), device=values.device)[:max_samples]
        values = values[indices]
        labels = labels[indices]
    return values, labels


def _set_loss_attr(module: torch.nn.Module, attr: str, value: float) -> None:
    if hasattr(module, attr):
        setattr(module, attr, value)
    if isinstance(module, DeepSupervisionWrapper):
        _set_loss_attr(module.loss, attr, value)


class _EBOTrainerBase(nnUNetTrainer):
    loss_name = "ebo_ce"

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        # self.num_epochs = 100
        self._best_val_loss = None
        self._best_loss_fpr95 = None
        self._last_val_energy_fpr95 = float("nan")
        self._ebo_validation_history = []

    def _build_loss(self):
        num_classes = self.label_manager.num_segmentation_heads
        loss_name = normalize_loss_name(os.environ.get("NNUNET_EBO_LOSS", self.loss_name))

        loss = build_loss(
            loss_name=loss_name,
            num_classes=num_classes,
            lambda_ebo_in=_env_float("LAMBDA_EBO_IN", 0.1),
            lambda_ebo_corr=_env_float("LAMBDA_EBO_CORR", 0.1),
            lambda_ebo_cen_in=_env_float("LAMBDA_EBO_CEN_IN", 0.1),
            lambda_ebo_out_in=_env_float("LAMBDA_EBO_OUT_IN", 0.2),
            lambda_ebo_cen_corr=_env_float("LAMBDA_EBO_CEN_CORR", 0.05),
            lambda_ebo_out_corr=_env_float("LAMBDA_EBO_OUT_CORR", 0.1),
            boundary_k=_env_int("BOUNDARY_K", 1),
            margin_correct=_env_float("MARGIN_CORRECT", -25.0),
            margin_miss=_env_float("MARGIN_MISS", -5.0),
            barrier_t=_env_float("BARRIER_T", 1.0),
            rho=_env_float("RHO", 1.0),
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        print(
            f"Using {loss_name} with "
            f"lambda_ebo_in={os.environ.get('LAMBDA_EBO_IN', '0.1')}, "
            f"lambda_ebo_corr={os.environ.get('LAMBDA_EBO_CORR', '0.1')}, "
            f"lambda_ebo_cen_in={os.environ.get('LAMBDA_EBO_CEN_IN', '0.1')}, "
            f"lambda_ebo_out_in={os.environ.get('LAMBDA_EBO_OUT_IN', '0.2')}, "
            f"lambda_ebo_cen_corr={os.environ.get('LAMBDA_EBO_CEN_CORR', '0.05')}, "
            f"lambda_ebo_out_corr={os.environ.get('LAMBDA_EBO_OUT_CORR', '0.1')}, "
            f"boundary_k={os.environ.get('BOUNDARY_K', '1')}, "
            f"margin_correct={os.environ.get('MARGIN_CORRECT', '-25.0')}, "
            f"margin_miss={os.environ.get('MARGIN_MISS', '-5.0')}, "
            f"barrier_t={os.environ.get('BARRIER_T', '1.0')}, "
            f"barrier_t_growth={os.environ.get('BARRIER_T_GROWTH', '1.0')}, "
            f"rho={os.environ.get('RHO', '1.0')}"
        )
        return loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        barrier_t_growth = _env_float("BARRIER_T_GROWTH", 1.0)
        barrier_t = _env_float("BARRIER_T", 1.0) * (barrier_t_growth ** max(self.current_epoch, 0))
        _set_loss_attr(self.loss, "t", barrier_t)

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            del data
            loss = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            error_target = target.bool()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            error_target = target

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                valid_mask = target != self.label_manager.ignore_label
                mask = valid_mask.float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    valid_mask = ~target[:, -1:]
                else:
                    valid_mask = 1 - target[:, -1:]
                mask = valid_mask.float()
                target = target[:, :-1]
                error_target = error_target[:, :-1]
        else:
            valid_mask = None
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        if self.label_manager.has_regions:
            error_map = (predicted_segmentation_onehot.bool() != error_target).any(dim=1)
        else:
            error_map = output_seg[:, 0] != error_target[:, 0]
            del output_seg

        if valid_mask is not None:
            error_map = error_map[valid_mask[:, 0].bool()]
            energy_values = _energy(output)[valid_mask[:, 0].bool()]
        else:
            energy_values = _energy(output).reshape(-1)
            error_map = error_map.reshape(-1)

        energy_values, error_map = _sample_flat_tensors(
            energy_values,
            error_map,
            _env_int("NNUNET_FPR95_MAX_PIXELS_PER_BATCH", 50000),
        )

        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
            "energy_values": energy_values.detach().float().cpu().numpy().reshape(-1),
            "energy_errors": error_map.detach().cpu().numpy().astype(np.uint8).reshape(-1),
        }

    def on_validation_epoch_end(self, val_outputs):
        tp = np.sum([i["tp_hard"] for i in val_outputs], 0)
        fp = np.sum([i["fp_hard"] for i in val_outputs], 0)
        fn = np.sum([i["fn_hard"] for i in val_outputs], 0)
        losses = np.asarray([i["loss"] for i in val_outputs])
        energies = None
        errors = None

        if self.is_ddp:
            world_size = dist.get_world_size()
            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, losses)
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(losses)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        should_compute_fpr95 = (
            self._best_val_loss is None
            or loss_here < self._best_val_loss
            or math.isclose(float(loss_here), float(self._best_val_loss), rel_tol=0.0, abs_tol=1e-12)
        )
        if should_compute_fpr95:
            energies = np.concatenate([i["energy_values"] for i in val_outputs])
            errors = np.concatenate([i["energy_errors"] for i in val_outputs])
            if self.is_ddp:
                energy_values = [None for _ in range(world_size)]
                dist.all_gather_object(energy_values, energies)
                energies = np.concatenate(energy_values)

                error_values = [None for _ in range(world_size)]
                dist.all_gather_object(error_values, errors)
                errors = np.concatenate(error_values)

            max_pixels = _env_int("NNUNET_FPR95_MAX_PIXELS", 200000)
            if max_pixels > 0 and energies.size > max_pixels:
                rng = np.random.default_rng(self.current_epoch)
                indices = rng.choice(energies.size, size=max_pixels, replace=False)
                energies = energies[indices]
                errors = errors[indices]
            val_energy_fpr95 = _fpr95_energy(energies, errors)
        else:
            val_energy_fpr95 = float("nan")

        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
        self.logger.log("dice_per_class_or_region", global_dc_per_class, self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)
        self._last_val_energy_fpr95 = float(val_energy_fpr95)

    @staticmethod
    def _is_better_loss_fpr95(loss_value: float, fpr95_value: float, best_loss: float | None, best_fpr95: float | None) -> bool:
        if best_loss is None:
            return True
        if loss_value < best_loss:
            return True
        if math.isclose(loss_value, best_loss, rel_tol=0.0, abs_tol=1e-12):
            if best_fpr95 is None:
                return True
            if math.isnan(best_fpr95):
                return not math.isnan(fpr95_value)
            if math.isnan(fpr95_value):
                return False
            return fpr95_value < best_fpr95
        return False

    def _write_ebo_selection_files(self, selected: bool) -> None:
        if self.local_rank != 0:
            return

        val_loss = float(self.logger.get_value("val_losses", step=-1))
        val_fpr95 = float(self._last_val_energy_fpr95)
        record = {
            "epoch": int(self.current_epoch),
            "val_loss": val_loss,
            "val_energy_fpr95": val_fpr95,
            "selected_checkpoint": bool(selected),
            "checkpoint_name": "checkpoint_best.pth" if selected else None,
        }
        self._ebo_validation_history.append(record)

        history_path = join(self.output_folder, "ebo_validation_history.json")
        with open(history_path, "w", encoding="utf-8") as file_obj:
            json.dump(self._ebo_validation_history, file_obj, indent=2, allow_nan=True)

        summary_path = join(self.output_folder, "ebo_selection_summary.json")
        summary = {
            "selection_rule": "min_val_loss_then_min_val_energy_fpr95",
            "best_val_loss": self._best_val_loss,
            "best_val_energy_fpr95": self._best_loss_fpr95,
            "best_checkpoint": "checkpoint_best.pth",
            "output_folder": self.output_folder,
            "current_epoch": int(self.current_epoch),
        }
        with open(summary_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, indent=2, allow_nan=True)

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        self.print_to_log_file("train_loss", np.round(self.logger.get_value("train_losses", step=-1), decimals=4))
        self.print_to_log_file("val_loss", np.round(self.logger.get_value("val_losses", step=-1), decimals=4))
        self.print_to_log_file("val_energy_fpr95", np.round(self._last_val_energy_fpr95, decimals=4))
        self.print_to_log_file("Pseudo dice", [np.round(i, decimals=4) for i in self.logger.get_value("dice_per_class_or_region", step=-1)])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.get_value('epoch_end_timestamps', step=-1) - self.logger.get_value('epoch_start_timestamps', step=-1), decimals=2)} s"
        )

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        val_loss = float(self.logger.get_value("val_losses", step=-1))
        val_fpr95 = float(self._last_val_energy_fpr95)
        selected = self._is_better_loss_fpr95(val_loss, val_fpr95, self._best_val_loss, self._best_loss_fpr95)
        if selected:
            self._best_val_loss = val_loss
            self._best_loss_fpr95 = val_fpr95
            self.print_to_log_file(
                f"New best validation loss/FPR95 checkpoint: val_loss={np.round(val_loss, decimals=4)}, "
                f"val_energy_fpr95={np.round(val_fpr95, decimals=4)}"
            )
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        self._write_ebo_selection_files(selected)

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)
        if current_epoch == (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        summary_path = join(self.output_folder, "ebo_selection_summary.json")
        history_path = join(self.output_folder, "ebo_validation_history.json")
        if isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as file_obj:
                summary = json.load(file_obj)
            self._best_val_loss = summary.get("best_val_loss")
            self._best_loss_fpr95 = summary.get("best_val_energy_fpr95")
        if isfile(history_path):
            with open(history_path, "r", encoding="utf-8") as file_obj:
                self._ebo_validation_history = json.load(file_obj)


class EBOTrainer(_EBOTrainerBase):
    loss_name = "ebo_ce"


class EBOLossLogBarrierTrainer(_EBOTrainerBase):
    loss_name = "log_ebo"


class BoundEBOTrainer(_EBOTrainerBase):
    loss_name = "bound_ebo_ce"


class BoundEBOLogBarrierTrainer(_EBOTrainerBase):
    loss_name = "bound_log_ebo"
