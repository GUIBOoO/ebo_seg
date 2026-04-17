import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import energy as energy_fn


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)

        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets_onehot, dims)
        union = torch.sum(probs + targets_onehot, dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes, weight_ce=1.0, weight_dice=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.w_ce * ce_loss + self.w_dice * dice_loss


class EBOLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        lambda_ebo_in: float = 1000,
        lambda_ebo_corr: float = 0.1,
        margin_correct: float = -25.0,
        margin_miss: float = -5.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_ebo_in = lambda_ebo_in
        self.lambda_ebo_corr = lambda_ebo_corr
        self.margin_correct = margin_correct
        self.margin_miss = margin_miss
        self.temperature = temperature

    def forward(self, logits, targets):
        base = self.base_loss(logits, targets)

        energy = energy_fn(logits, self.temperature)
        pred = torch.argmax(logits, dim=1)
        correct_mask = pred == targets

        ebo_correct = F.relu(energy - self.margin_correct) ** 2
        ebo_in = F.relu(self.margin_miss - energy) ** 2

        ebo = torch.where(
            correct_mask,
            self.lambda_ebo_corr * ebo_correct,
            self.lambda_ebo_in * ebo_in,
        )

        return base + ebo.mean()


def build_loss(
    loss_name: str,
    num_classes: int,
    lambda_ebo_in: float = 0.1,
    lambda_ebo_corr: float = 0.1,
    margin_correct: float = -35.0,
    margin_miss: float = -5.0,
) -> nn.Module:
    loss_name = loss_name.lower()

    if loss_name in {'ce', 'cross_entropy'}:
        if num_classes < 2:
            raise ValueError('CrossEntropyLoss requires num_classes >= 2')
        return nn.CrossEntropyLoss()

    if loss_name in {'ce_dice'}:
        if num_classes < 2:
            raise ValueError('CE requires num_classes >=2')
        return CEDiceLoss(num_classes)

    if loss_name in {'ebo_ce', 'ebo_cross_entropy'}:
        if num_classes < 2:
            raise ValueError('ebo_cross_entropy requires num_classes >= 2')
        return EBOLoss(
            CEDiceLoss(num_classes),
            lambda_ebo_in=lambda_ebo_in,
            lambda_ebo_corr=lambda_ebo_corr,
            margin_correct=margin_correct,
            margin_miss=margin_miss,
        )

    raise ValueError(f'Unsupported loss: {loss_name}')
