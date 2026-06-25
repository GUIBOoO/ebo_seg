import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnunetv2.training.loss.dice import DC_and_CE_loss

from losses import (
    CEDiceLoss,
    EBOLoss,
    BoundEBOLoss,
    BoundEBOLogBarrierLoss
)



class EBOTrainer(nnUNetTrainer):

    def _build_loss(self):

        num_classes = self.label_manager.num_classes


        base_loss = CEDiceLoss(
            num_classes=num_classes,
            weight_ce=1.0,
            weight_dice=1.0
        )


        self.loss = EBOLoss(
            base_loss,
            lambda_ebo_in=0.1,
            lambda_ebo_corr=0.1,
            margin_correct=-25,
            margin_miss=-5,
            temperature=1.0
        )


        print("🔥 Using EBOLoss")

class BoundEBOTrainer(nnUNetTrainer):

    def _build_loss(self):

        base_loss = CEDiceLoss(
            self.label_manager.num_classes
        )


        self.loss = BoundEBOLoss(
            base_loss,
            lambda_ebo_cen_in=0.1,
            lambda_ebo_out_in=0.2,
            lambda_ebo_cen_corr=0.05,
            lambda_ebo_out_corr=0.1,
            boundary_k=1
        )


        print("🔥 Using BoundEBOLoss")

class BoundEBOLogBarrierTrainer(nnUNetTrainer):

    def _build_loss(self):

        base_loss = CEDiceLoss(
            num_classes=self.label_manager.num_classes,
            weight_ce=1.0,
            weight_dice=1.0
        )

        self.loss = BoundEBOLogBarrierLoss(
            base_loss=base_loss,

            lambda_ebo_cen_in=0.1,
            lambda_ebo_out_in=0.2,

            lambda_ebo_cen_corr=0.05,
            lambda_ebo_out_corr=0.1,

            boundary_k=1,

            margin_correct=-25.0,
            margin_miss=-5.0,

            temperature=1.0,

            # paramètre barrière
            t=1.0
        )

        print("🔥 Using BoundEBOLogBarrierLoss")