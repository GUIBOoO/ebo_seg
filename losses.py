import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_boundary_mask, energy as energy_fn, hybrid_energy

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
        lambda_ebo_in: float = 0.1,
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

    def forward(self, logits, targets, image=None):
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

class BoundEBOLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        lambda_ebo_cen_in: float = 0.1,
        lambda_ebo_out_in: float = 0.1,
        lambda_ebo_cen_corr: float = 0.1,
        lambda_ebo_out_corr: float = 0.1,
        boundary_k: int = 1,
        margin_correct: float = -25.0,
        margin_miss: float = -5.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_ebo_cen_in = lambda_ebo_cen_in
        self.lambda_ebo_out_in = lambda_ebo_out_in
        self.lambda_ebo_cen_corr = lambda_ebo_cen_corr
        self.lambda_ebo_out_corr = lambda_ebo_out_corr
        self.boundary_k = boundary_k
        self.margin_correct = margin_correct
        self.margin_miss = margin_miss
        self.temperature = temperature

    def forward(self, logits, targets, image=None):
        base = self.base_loss(logits, targets)

        energy = energy_fn(logits, self.temperature)
        pred = torch.argmax(logits, dim=1)
        correct_mask = pred == targets
        boundary_mask = compute_boundary_mask(
            targets,
            k=self.boundary_k,
            num_classes=logits.shape[1],
        )

        ebo_correct = F.relu(energy - self.margin_correct) ** 2
        ebo_in = F.relu(self.margin_miss - energy) ** 2

        lambda_corr = torch.where(
            boundary_mask,
            torch.full_like(energy, self.lambda_ebo_out_corr),
            torch.full_like(energy, self.lambda_ebo_cen_corr),
        )
        lambda_in = torch.where(
            boundary_mask,
            torch.full_like(energy, self.lambda_ebo_out_in),
            torch.full_like(energy, self.lambda_ebo_cen_in),
        )

        ebo = torch.where(
            correct_mask,
            lambda_corr * ebo_correct,
            lambda_in * ebo_in,
        )

        return base + ebo.mean()


class HybridEBOLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        lambda_ebo_in: float = 0.1,
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

    def forward(self, logits, targets, image):
        base = self.base_loss(logits, targets)

        energy = hybrid_energy(logits, image, T=self.temperature)
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

class LogBarrierExtended(nn.Module):
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, z, t):
        """
        z : contrainte (doit être <= 0)
        t : temperature / paramètre de durcissement
        """

        threshold = -1.0 / (t ** 2)

        log_part = -(1.0 / t) * torch.log(torch.clamp(-z, min=self.eps))

        linear_part = (
            t * z
            - (1.0 / t) * torch.log(torch.tensor(1.0 / (t ** 2), device=z.device))
            + (1.0 / t)
        )

        return torch.where(z <= threshold, log_part, linear_part)

class EBOLossLogBarrier(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,
        lambda_ebo_in: float = 0.1,
        lambda_ebo_corr: float = 0.1,
        margin_correct: float = -25.0,
        margin_miss: float = -5.0,
        temperature: float = 1.0,
        t = 1
    ):
        super().__init__()

        self.base_loss = base_loss
        self.lambda_ebo_in = lambda_ebo_in
        self.lambda_ebo_corr = lambda_ebo_corr
        self.margin_correct = margin_correct
        self.margin_miss = margin_miss
        self.temperature = temperature
        self.t = t
        self.initial_t = t

        self.barrier = LogBarrierExtended()

    def forward(self, logits, targets):

        base = self.base_loss(logits, targets)

        energy = energy_fn(logits, self.temperature)

        pred = torch.argmax(logits, dim=1)
        correct_mask = pred == targets

        f_corr = energy - self.margin_correct  
        f_miss = self.margin_miss - energy       

        loss_corr = self.barrier(f_corr, self.t)
        loss_miss = self.barrier(f_miss, self.t)

        ebo = torch.where(
            correct_mask,
            self.lambda_ebo_corr * loss_corr,
            self.lambda_ebo_in * loss_miss,
        )

        return base + ebo.mean()


class BoundEBOLogBarrierLoss(nn.Module):
    def __init__(
        self,
        base_loss: nn.Module,

        lambda_ebo_cen_in: float = 0.1,
        lambda_ebo_out_in: float = 0.2,

        lambda_ebo_cen_corr: float = 0.05,
        lambda_ebo_out_corr: float = 0.1,

        boundary_k: int = 1,

        margin_correct: float = -25.0,
        margin_miss: float = -5.0,

        temperature: float = 1.0,
        t: float = 1.0,
    ):
        super().__init__()

        self.base_loss = base_loss

        self.lambda_ebo_cen_in = lambda_ebo_cen_in
        self.lambda_ebo_out_in = lambda_ebo_out_in

        self.lambda_ebo_cen_corr = lambda_ebo_cen_corr
        self.lambda_ebo_out_corr = lambda_ebo_out_corr

        self.boundary_k = boundary_k

        self.margin_correct = margin_correct
        self.margin_miss = margin_miss

        self.temperature = temperature
        self.t = t
        self.initial_t = t

        self.barrier = LogBarrierExtended()

    def forward(self, logits, targets, image=None):

        base = self.base_loss(logits, targets)

        energy = energy_fn(logits, self.temperature)

        pred = torch.argmax(logits, dim=1)
        correct_mask = pred == targets

        boundary_mask = compute_boundary_mask(
            targets,
            k=self.boundary_k,
            num_classes=logits.shape[1],
        )

        f_corr = energy - self.margin_correct

        f_miss = self.margin_miss - energy

        loss_corr = self.barrier(f_corr, self.t)
        loss_miss = self.barrier(f_miss, self.t)

        lambda_corr = torch.where(
            boundary_mask,
            torch.full_like(energy, self.lambda_ebo_out_corr),
            torch.full_like(energy, self.lambda_ebo_cen_corr),
        )

        lambda_in = torch.where(
            boundary_mask,
            torch.full_like(energy, self.lambda_ebo_out_in),
            torch.full_like(energy, self.lambda_ebo_cen_in),
        )

        ebo = torch.where(
            correct_mask,
            lambda_corr * loss_corr,
            lambda_in * loss_miss,
        )

        return base + ebo.mean()

class BoundEBOAugLagLoss(nn.Module):
    """
    Version pure Augmented Lagrangian, sans Log-Barrier.

    Contraintes :
        correct sample  -> energy <= margin_correct   (g_corr = energy - margin_correct <= 0)
        wrong sample    -> energy >= margin_miss      (g_miss = margin_miss - energy   <= 0)

    Les multiplicateurs de Lagrange sont mis a jour explicitement apres chaque batch.
    """

    def __init__(
        self,
        base_loss: nn.Module,

        boundary_k: int = 1,

        margin_correct: float = -25.0,
        margin_miss: float = -5.0,

        temperature: float = 1.0,

        rho: float = 1.0,

        lambda_init_corr_center: float = 0.01,
        lambda_init_corr_boundary: float = 0.01,

        lambda_init_miss_center: float = 0.01,
        lambda_init_miss_boundary: float = 0.01,
    ):
        super().__init__()

        self.base_loss = base_loss

        self.boundary_k = boundary_k

        self.margin_correct = margin_correct
        self.margin_miss = margin_miss

        self.temperature = temperature

        self.rho = rho

        self.lambda_corr_center_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_corr_center)),
            requires_grad=False,
        )
        self.lambda_corr_boundary_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_corr_boundary)),
            requires_grad=False,
        )
        self.lambda_miss_center_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_miss_center)),
            requires_grad=False,
        )
        self.lambda_miss_boundary_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_miss_boundary)),
            requires_grad=False,
        )

    @property
    def lambda_corr_center(self):
        return F.softplus(self.lambda_corr_center_raw)

    @property
    def lambda_corr_boundary(self):
        return F.softplus(self.lambda_corr_boundary_raw)

    @property
    def lambda_miss_center(self):
        return F.softplus(self.lambda_miss_center_raw)

    @property
    def lambda_miss_boundary(self):
        return F.softplus(self.lambda_miss_boundary_raw)

    @staticmethod
    def _inverse_softplus(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        value = torch.clamp(value, min=eps)
        return value + torch.log(-torch.expm1(-value))

    def forward(self, logits, targets, image=None):

        base = self.base_loss(logits, targets)

        energy = energy_fn(logits, self.temperature)

        pred = torch.argmax(logits, dim=1)
        correct_mask = pred == targets

        boundary_mask = compute_boundary_mask(
            targets,
            k=self.boundary_k,
            num_classes=logits.shape[1],
        )

        g_corr = energy - self.margin_correct
        g_miss = self.margin_miss - energy

        lambda_corr = torch.where(
            boundary_mask,
            self.lambda_corr_boundary.expand_as(energy),
            self.lambda_corr_center.expand_as(energy),
        )

        lambda_miss = torch.where(
            boundary_mask,
            self.lambda_miss_boundary.expand_as(energy),
            self.lambda_miss_center.expand_as(energy),
        )

        aug_corr = 0.5 / self.rho * (
            torch.relu(lambda_corr + self.rho * g_corr) ** 2 - lambda_corr ** 2
        )

        aug_miss = 0.5 / self.rho * (
            torch.relu(lambda_miss + self.rho * g_miss) ** 2 - lambda_miss ** 2
        )

        alm_term = torch.where(correct_mask, aug_corr, aug_miss)

        return base + alm_term.mean()

    @torch.no_grad()
    def update_lambdas(self, logits, targets):
        """
        Mise a jour duale classique :
            lambda <- max(0, lambda + rho * g(x))
        """

        energy = energy_fn(logits, self.temperature)

        pred = torch.argmax(logits, dim=1)
        correct_mask = pred == targets

        boundary_mask = compute_boundary_mask(
            targets,
            k=self.boundary_k,
            num_classes=logits.shape[1],
        )

        g_corr = energy - self.margin_correct
        g_miss = self.margin_miss - energy

        def masked_mean(x, mask):
            if mask.sum() == 0:
                return torch.tensor(0.0, device=x.device)
            return x[mask].mean()

        corr_boundary = correct_mask & boundary_mask
        corr_center = correct_mask & (~boundary_mask)

        miss_boundary = (~correct_mask) & boundary_mask
        miss_center = (~correct_mask) & (~boundary_mask)

        g_corr_boundary = masked_mean(g_corr, corr_boundary)
        g_corr_center = masked_mean(g_corr, corr_center)

        g_miss_boundary = masked_mean(g_miss, miss_boundary)
        g_miss_center = masked_mean(g_miss, miss_center)

        new_corr_boundary = torch.clamp(
            self.lambda_corr_boundary + self.rho * g_corr_boundary,
            min=0.0,
        )

        new_corr_center = torch.clamp(
            self.lambda_corr_center + self.rho * g_corr_center,
            min=0.0,
        )

        new_miss_boundary = torch.clamp(
            self.lambda_miss_boundary + self.rho * g_miss_boundary,
            min=0.0,
        )

        new_miss_center = torch.clamp(
            self.lambda_miss_center + self.rho * g_miss_center,
            min=0.0,
        )

        self.lambda_corr_boundary_raw.copy_(self._inverse_softplus(new_corr_boundary))

        self.lambda_corr_center_raw.copy_(self._inverse_softplus(new_corr_center))

        self.lambda_miss_boundary_raw.copy_(self._inverse_softplus(new_miss_boundary))

        self.lambda_miss_center_raw.copy_(self._inverse_softplus(new_miss_center))

class BoundEBOAugLogLoss(nn.Module):
    """
    Version Augmented Lagrangian avec Log-Barrier.

    Contraintes :
        correct sample  -> energy <= margin_correct   (g_corr = energy - margin_correct <= 0)
        wrong sample    -> energy >= margin_miss      (g_miss = margin_miss - energy   <= 0)

    Les contraintes sont transformees par la Log-Barrier, puis les multiplicateurs
    de Lagrange sont mis a jour explicitement apres chaque batch.
    """

    def __init__(
        self,
        base_loss: nn.Module,

        boundary_k: int = 1,

        margin_correct: float = -25.0,
        margin_miss: float = -5.0,

        temperature: float = 1.0,

        rho: float = 1.0,

        lambda_init_corr_center: float = 0.01,
        lambda_init_corr_boundary: float = 0.01,

        lambda_init_miss_center: float = 0.01,
        lambda_init_miss_boundary: float = 0.01,

        t: float = 1.0,
    ):
        super().__init__()

        self.base_loss = base_loss

        self.boundary_k = boundary_k

        self.margin_correct = margin_correct
        self.margin_miss = margin_miss

        self.temperature = temperature

        self.rho = rho

        self.lambda_corr_center_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_corr_center)),
            requires_grad=False,
        )
        self.lambda_corr_boundary_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_corr_boundary)),
            requires_grad=False,
        )
        self.lambda_miss_center_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_miss_center)),
            requires_grad=False,
        )
        self.lambda_miss_boundary_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_miss_boundary)),
            requires_grad=False,
        )
        self.t = t
        self.initial_t = t

        self.barrier = LogBarrierExtended()

    @property
    def lambda_corr_center(self):
        return F.softplus(self.lambda_corr_center_raw)

    @property
    def lambda_corr_boundary(self):
        return F.softplus(self.lambda_corr_boundary_raw)

    @property
    def lambda_miss_center(self):
        return F.softplus(self.lambda_miss_center_raw)

    @property
    def lambda_miss_boundary(self):
        return F.softplus(self.lambda_miss_boundary_raw)

    @staticmethod
    def _inverse_softplus(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        value = torch.clamp(value, min=eps)
        return value + torch.log(-torch.expm1(-value))

    def forward(self, logits, targets, image=None):

        base = self.base_loss(logits, targets)

        energy = energy_fn(logits, self.temperature)

        pred = torch.argmax(logits, dim=1)
        correct_mask = pred == targets

        boundary_mask = compute_boundary_mask(
            targets,
            k=self.boundary_k,
            num_classes=logits.shape[1],
        )

        g_miss = self.barrier(self.margin_miss - energy, self.t)

        g_corr = self.barrier(energy - self.margin_correct, self.t)

        lambda_corr = torch.where(
            boundary_mask,
            self.lambda_corr_boundary.expand_as(energy),
            self.lambda_corr_center.expand_as(energy),
        )

        lambda_miss = torch.where(
            boundary_mask,
            self.lambda_miss_boundary.expand_as(energy),
            self.lambda_miss_center.expand_as(energy),
        )

        aug_corr = 0.5 / self.rho * (
            torch.relu(lambda_corr + self.rho * g_corr) ** 2 - lambda_corr ** 2
        )

        aug_miss = 0.5 / self.rho * (
            torch.relu(lambda_miss + self.rho * g_miss) ** 2 - lambda_miss ** 2
        )

        alm_term = torch.where(correct_mask, aug_corr, aug_miss)

        return base + alm_term.mean()

    @torch.no_grad()
    def update_lambdas(self, logits, targets):
        """
        Mise a jour duale classique :
            lambda <- max(0, lambda + rho * g(x))
        """

        energy = energy_fn(logits, self.temperature)

        pred = torch.argmax(logits, dim=1)
        correct_mask = pred == targets

        boundary_mask = compute_boundary_mask(
            targets,
            k=self.boundary_k,
            num_classes=logits.shape[1],
        )

        g_miss = self.margin_miss - energy

        g_corr = energy - self.margin_correct

        def masked_mean(x, mask):
            if mask.sum() == 0:
                return torch.tensor(0.0, device=x.device)
            return x[mask].mean()

        corr_boundary = correct_mask & boundary_mask
        corr_center = correct_mask & (~boundary_mask)

        miss_boundary = (~correct_mask) & boundary_mask
        miss_center = (~correct_mask) & (~boundary_mask)

        g_corr_boundary = masked_mean(g_corr, corr_boundary)
        g_corr_center = masked_mean(g_corr, corr_center)

        g_miss_boundary = masked_mean(g_miss, miss_boundary)
        g_miss_center = masked_mean(g_miss, miss_center)

        new_corr_boundary = torch.clamp(
            self.lambda_corr_boundary + self.rho * g_corr_boundary,
            min=0.0,
        )

        new_corr_center = torch.clamp(
            self.lambda_corr_center + self.rho * g_corr_center,
            min=0.0,
        )

        new_miss_boundary = torch.clamp(
            self.lambda_miss_boundary + self.rho * g_miss_boundary,
            min=0.0,
        )

        new_miss_center = torch.clamp(
            self.lambda_miss_center + self.rho * g_miss_center,
            min=0.0,
        )

        self.lambda_corr_boundary_raw.copy_(self._inverse_softplus(new_corr_boundary))

        self.lambda_corr_center_raw.copy_(self._inverse_softplus(new_corr_center))

        self.lambda_miss_boundary_raw.copy_(self._inverse_softplus(new_miss_boundary))

        self.lambda_miss_center_raw.copy_(self._inverse_softplus(new_miss_center))

class BoundEBOAugLagLoss2(nn.Module):
    """
    Version pure Augmented Lagrangian, sans Log-Barrier.

    Contraintes :
        correct sample  -> energy <= margin_correct   (g_corr = energy - margin_correct <= 0)
        wrong sample    -> energy >= margin_miss      (g_miss = margin_miss - energy   <= 0)

    Les multiplicateurs de Lagrange sont mis a jour explicitement apres chaque batch.
    """

    def __init__(
        self,
        base_loss: nn.Module,

        boundary_k: int = 1,

        margin_correct: float = -25.0,
        margin_miss: float = -5.0,

        temperature: float = 1.0,

        rho: float = 1.0,

        lambda_init_corr_center: float = 0.01,
        lambda_init_corr_boundary: float = 0.01,

        lambda_init_miss_center: float = 0.01,
        lambda_init_miss_boundary: float = 0.01,
    ):
        super().__init__()

        self.base_loss = base_loss

        self.boundary_k = boundary_k

        self.margin_correct = margin_correct
        self.margin_miss = margin_miss

        self.temperature = temperature

        self.rho = rho

        self.lambda_corr_center_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_corr_center)),
            requires_grad=False,
        )
        self.lambda_corr_boundary_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_corr_boundary)),
            requires_grad=False,
        )
        self.lambda_miss_center_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_miss_center)),
            requires_grad=False,
        )
        self.lambda_miss_boundary_raw = nn.Parameter(
            self._inverse_softplus(torch.tensor(lambda_init_miss_boundary)),
            requires_grad=False,
        )

    @property
    def lambda_corr_center(self):
        return F.softplus(self.lambda_corr_center_raw)

    @property
    def lambda_corr_boundary(self):
        return F.softplus(self.lambda_corr_boundary_raw)

    @property
    def lambda_miss_center(self):
        return F.softplus(self.lambda_miss_center_raw)

    @property
    def lambda_miss_boundary(self):
        return F.softplus(self.lambda_miss_boundary_raw)

    @staticmethod
    def _inverse_softplus(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        value = torch.clamp(value, min=eps)
        return value + torch.log(-torch.expm1(-value))

    def forward(self, logits, targets, image=None):

        base = self.base_loss(logits, targets)

        energy = energy_fn(logits, self.temperature)

        pred = torch.argmax(logits, dim=1)
        correct_mask = pred == targets

        boundary_mask = compute_boundary_mask(
            targets,
            k=self.boundary_k,
            num_classes=logits.shape[1],
        )

        g_corr = energy - self.margin_correct
        g_miss = self.margin_miss - energy

        lambda_corr = torch.where(
            boundary_mask,
            self.lambda_corr_boundary.expand_as(energy),
            self.lambda_corr_center.expand_as(energy),
        )

        lambda_miss = torch.where(
            boundary_mask,
            self.lambda_miss_boundary.expand_as(energy),
            self.lambda_miss_center.expand_as(energy),
        )

        corr_term = lambda_corr * g_corr
        miss_term = lambda_miss * g_miss

        alm_term = torch.where(correct_mask, corr_term, miss_term)

        return base + alm_term.mean()

    @torch.no_grad()
    def update_lambdas(self, logits, targets):
        """
        Mise a jour duale classique :
            lambda <- max(0, lambda + rho * g(x))
        """

        energy = energy_fn(logits, self.temperature)

        pred = torch.argmax(logits, dim=1)
        correct_mask = pred == targets

        boundary_mask = compute_boundary_mask(
            targets,
            k=self.boundary_k,
            num_classes=logits.shape[1],
        )

        g_corr = energy - self.margin_correct
        g_miss = self.margin_miss - energy

        def masked_mean(x, mask):
            if mask.sum() == 0:
                return torch.tensor(0.0, device=x.device)
            return x[mask].mean()

        corr_boundary = correct_mask & boundary_mask
        corr_center = correct_mask & (~boundary_mask)

        miss_boundary = (~correct_mask) & boundary_mask
        miss_center = (~correct_mask) & (~boundary_mask)

        g_corr_boundary = masked_mean(g_corr, corr_boundary)
        g_corr_center = masked_mean(g_corr, corr_center)

        g_miss_boundary = masked_mean(g_miss, miss_boundary)
        g_miss_center = masked_mean(g_miss, miss_center)

        new_corr_boundary = torch.clamp(
            self.lambda_corr_boundary + self.rho * g_corr_boundary,
            min=0.0,
        )

        new_corr_center = torch.clamp(
            self.lambda_corr_center + self.rho * g_corr_center,
            min=0.0,
        )

        new_miss_boundary = torch.clamp(
            self.lambda_miss_boundary + self.rho * g_miss_boundary,
            min=0.0,
        )

        new_miss_center = torch.clamp(
            self.lambda_miss_center + self.rho * g_miss_center,
            min=0.0,
        )

        self.lambda_corr_boundary_raw.copy_(self._inverse_softplus(new_corr_boundary))

        self.lambda_corr_center_raw.copy_(self._inverse_softplus(new_corr_center))

        self.lambda_miss_boundary_raw.copy_(self._inverse_softplus(new_miss_boundary))

        self.lambda_miss_center_raw.copy_(self._inverse_softplus(new_miss_center))

def normalize_loss_name(loss_name: str) -> str:
    normalized = loss_name.lower()
    aliases = {
        'eboloss': 'ebo_ce',
        'ebo_loss': 'ebo_ce',
        'boundeboloss': 'bound_ebo_ce',
        'bound_ebo_loss': 'bound_ebo_ce',
        'ebolosslogbarrier': 'log_ebo',
        'ebo_loss_log_barrier': 'log_ebo',
        'logeboloss': 'log_ebo',
        'boundebologbarrierloss': 'bound_log_ebo',
        'bound_ebo_log_barrier_loss': 'bound_log_ebo',
        'bound_log_ebo_loss': 'bound_log_ebo',
        'boundlogebo': 'bound_log_ebo',
        'bound_ebo_log_barrier': 'bound_log_ebo',
        'boundary_log_ebo': 'bound_log_ebo',
        'boundeboauglagloss': 'bound_ebo_aug_lag',
        'bound_ebo_aug_lag_loss': 'bound_ebo_aug_lag',
        'bound_aug_lag_ebo': 'bound_ebo_aug_lag',
        'boundary_aug_lag_ebo': 'bound_ebo_aug_lag',
        'boundeboauglogloss': 'bound_ebo_aug_log',
        'bound_ebo_aug_log_loss': 'bound_ebo_aug_log',
        'bound_aug_log_ebo': 'bound_ebo_aug_log',
        'boundary_aug_log_ebo': 'bound_ebo_aug_log',
    }
    return aliases.get(normalized, normalized)

def build_loss(
    loss_name: str,
    num_classes: int,
    lambda_ebo_in: float = 0.1,
    lambda_ebo_corr: float = 0.1,
    lambda_ebo_cen_in: float | None = None,
    lambda_ebo_out_in: float | None = None,
    lambda_ebo_cen_corr: float | None = None,
    lambda_ebo_out_corr: float | None = None,
    boundary_k: int = 1,
    margin_correct: float = -35.0,
    margin_miss: float = -5.0,
    barrier_t: float = 1.0,
    rho: float = 1.0,
) -> nn.Module:
    loss_name = normalize_loss_name(loss_name)

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

    if loss_name in {'hybrid_ebo_ce', 'hybrid_ebo_cross_entropy'}:
        if num_classes < 2:
            raise ValueError('hybrid_ebo_cross_entropy requires num_classes >= 2')
        return HybridEBOLoss(
            CEDiceLoss(num_classes),
            lambda_ebo_in=lambda_ebo_in,
            lambda_ebo_corr=lambda_ebo_corr,
            margin_correct=margin_correct,
            margin_miss=margin_miss,
        )
    if loss_name in {'bound_ebo', 'bound_ebo_cross_entropy', 'bound_ebo_ce'}:
        if num_classes < 2:
            raise ValueError('bound_ebo_cross_entropy requires num_classes >= 2')
        return BoundEBOLoss(
            CEDiceLoss(num_classes),
            lambda_ebo_cen_in=lambda_ebo_in if lambda_ebo_cen_in is None else lambda_ebo_cen_in,
            lambda_ebo_out_in=lambda_ebo_in if lambda_ebo_out_in is None else lambda_ebo_out_in,
            lambda_ebo_cen_corr=lambda_ebo_corr if lambda_ebo_cen_corr is None else lambda_ebo_cen_corr,
            lambda_ebo_out_corr=lambda_ebo_corr if lambda_ebo_out_corr is None else lambda_ebo_out_corr,
            boundary_k=boundary_k,
            margin_correct=margin_correct,
            margin_miss=margin_miss,
        )
    if loss_name in {'log_ebo'}:
        if num_classes < 2:
            raise ValueError('log_ebo requires num_classes >= 2')
        return EBOLossLogBarrier(
            CEDiceLoss(num_classes),
            lambda_ebo_in=lambda_ebo_in,
            lambda_ebo_corr=lambda_ebo_corr,
            margin_correct=margin_correct,
            margin_miss=margin_miss,
            t=barrier_t,
        )

    if loss_name in {'bound_log_ebo', 'bound_ebo_log_barrier', 'boundary_log_ebo'}:
        if num_classes < 2:
            raise ValueError('bound_log_ebo requires num_classes >= 2')
        return BoundEBOLogBarrierLoss(
            CEDiceLoss(num_classes),
            lambda_ebo_cen_in=lambda_ebo_in if lambda_ebo_cen_in is None else lambda_ebo_cen_in,
            lambda_ebo_out_in=lambda_ebo_in if lambda_ebo_out_in is None else lambda_ebo_out_in,
            lambda_ebo_cen_corr=lambda_ebo_corr if lambda_ebo_cen_corr is None else lambda_ebo_cen_corr,
            lambda_ebo_out_corr=lambda_ebo_corr if lambda_ebo_out_corr is None else lambda_ebo_out_corr,
            boundary_k=boundary_k,
            margin_correct=margin_correct,
            margin_miss=margin_miss,
            t=barrier_t,
        )

    if loss_name in {'bound_ebo_aug_lag'}:
        if num_classes < 2:
            raise ValueError('bound_ebo_aug_lag requires num_classes >= 2')
        return BoundEBOAugLagLoss(
            CEDiceLoss(num_classes),
            boundary_k=boundary_k,
            margin_correct=margin_correct,
            margin_miss=margin_miss,
            rho=rho,
            lambda_init_corr_center=lambda_ebo_corr if lambda_ebo_cen_corr is None else lambda_ebo_cen_corr,
            lambda_init_corr_boundary=lambda_ebo_corr if lambda_ebo_out_corr is None else lambda_ebo_out_corr,
            lambda_init_miss_center=lambda_ebo_in if lambda_ebo_cen_in is None else lambda_ebo_cen_in,
            lambda_init_miss_boundary=lambda_ebo_in if lambda_ebo_out_in is None else lambda_ebo_out_in,
        )

    if loss_name in {'bound_ebo_aug_log'}:
        if num_classes < 2:
            raise ValueError('bound_ebo_aug_log requires num_classes >= 2')
        return BoundEBOAugLogLoss(
            CEDiceLoss(num_classes),
            boundary_k=boundary_k,
            margin_correct=margin_correct,
            margin_miss=margin_miss,
            rho=rho,
            lambda_init_corr_center=lambda_ebo_corr if lambda_ebo_cen_corr is None else lambda_ebo_cen_corr,
            lambda_init_corr_boundary=lambda_ebo_corr if lambda_ebo_out_corr is None else lambda_ebo_out_corr,
            lambda_init_miss_center=lambda_ebo_in if lambda_ebo_cen_in is None else lambda_ebo_cen_in,
            lambda_init_miss_boundary=lambda_ebo_in if lambda_ebo_out_in is None else lambda_ebo_out_in,
            t=barrier_t,
        )

    raise ValueError(f'Unsupported loss: {loss_name}')
