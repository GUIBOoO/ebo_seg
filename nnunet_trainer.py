import os

import numpy as np
import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from losses import build_loss, normalize_loss_name


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _set_loss_attr(module: torch.nn.Module, attr: str, value: float) -> None:
    if hasattr(module, attr):
        setattr(module, attr, value)
    if isinstance(module, DeepSupervisionWrapper):
        _set_loss_attr(module.loss, attr, value)


class _EBOTrainerBase(nnUNetTrainer):
    loss_name = "ebo_ce"

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


class EBOTrainer(_EBOTrainerBase):
    loss_name = "ebo_ce"


class EBOLossLogBarrierTrainer(_EBOTrainerBase):
    loss_name = "log_ebo"


class BoundEBOTrainer(_EBOTrainerBase):
    loss_name = "bound_ebo_ce"


class BoundEBOLogBarrierTrainer(_EBOTrainerBase):
    loss_name = "bound_log_ebo"
