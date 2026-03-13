"""
training.py
-----------
Joint loss function and training loop for the dependent frequency-severity model.

The joint loss is:

    L(θ) = -ℓ_Poisson(N; λ) - w · ℓ_Gamma(Y_pos; μ_pos, φ)

where ℓ_Gamma is summed over rows with n_claims > 0 only.  The weight w is
either fixed (``loss_weight_sev``) or set automatically each epoch to equalise
the two loss magnitudes.

The critical training constraint: frequency gradients flow through the trunk from
ALL rows; severity gradients flow through the trunk only from rows where
n_claims > 0.  Both gradient streams update the SAME trunk weights, which is
where implicit frequency-severity dependence is learned.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from insurance_frequency_severity.dependent.model import DependentFreqSevNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class JointLoss(nn.Module):
    """Joint Poisson-Gamma negative log-likelihood.

    Poisson NLL for frequency (all rows) plus Gamma NLL for average severity
    (positive-claim rows only).

    Parameters
    ----------
    loss_weight_sev:
        Fixed weight applied to the Gamma loss.  If ``auto_balance=True`` this
        is ignored and the weight is updated each batch to equalise the two loss
        scales.
    auto_balance:
        When True, the severity weight is set to
        ``|ℓ_freq| / max(|ℓ_sev|, ε)`` each forward call, so both losses
        contribute roughly equally.  Useful when you have no prior intuition
        about the relative scales.
    eps:
        Small constant for numerical stability in auto-balancing.
    """

    def __init__(
        self,
        loss_weight_sev: float = 1.0,
        auto_balance: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.loss_weight_sev = loss_weight_sev
        self.auto_balance = auto_balance
        self.eps = eps

    def forward(
        self,
        log_lambda: Tensor,
        log_mu: Tensor,
        phi: Tensor,
        n_claims: Tensor,
        avg_severity: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute joint loss.

        Parameters
        ----------
        log_lambda:
            Log Poisson rate (including log exposure), shape (batch,).
        log_mu:
            Log expected average severity, shape (batch,).
        phi:
            Gamma dispersion scalar.
        n_claims:
            Observed claim counts, shape (batch,).
        avg_severity:
            Observed average severity (total loss / n_claims), shape (batch,).
            Only rows where n_claims > 0 are used.

        Returns
        -------
        total_loss: scalar Tensor.
        freq_loss: scalar Tensor (for logging).
        sev_loss: scalar Tensor (for logging).
        """
        # -- Poisson NLL (all rows) --
        # ℓ_freq = Σ [n_i·log λ_i - λ_i - log(n_i!)]
        # We drop the log-factorial term (constant w.r.t. parameters).
        lambda_ = torch.exp(log_lambda)
        freq_loss = -torch.mean(n_claims.float() * log_lambda - lambda_)

        # -- Gamma NLL (positive-claim rows only) --
        pos_mask = n_claims > 0
        n_pos = pos_mask.sum()

        if n_pos == 0:
            sev_loss = torch.tensor(0.0, device=log_lambda.device)
        else:
            log_mu_pos = log_mu[pos_mask]
            y_pos = avg_severity[pos_mask]
            n_pos_vals = n_claims[pos_mask].float()
            mu_pos = torch.exp(log_mu_pos)

            # Average severity ȳ_i = Σ Y_ij / n_i follows Gamma with:
            #   shape = n_i / φ,  scale = φ·μ_i / n_i   (mean = μ_i)
            # Gamma NLL = -Σ [α·log β - log Γ(α) + (α-1)·log ȳ - β·ȳ]
            # where α = n_i/φ, β = n_i/(φ·μ_i)
            alpha = n_pos_vals / phi
            # Gamma NLL (up to log-Gamma constant):
            # = Σ [α·log(α/μ) - (α-1)·log(ȳ) + α·ȳ/μ - log Γ(α)]
            # Simplified form (standard Gamma deviance contribution):
            # nll_i = α·(log μ - log ȳ + ȳ/μ - 1)  +  log Γ(α) - α·log α
            log_y_pos = torch.log(y_pos + 1e-10)
            ratio = y_pos / (mu_pos + 1e-10)
            # Using Gamma log-likelihood: log p(y|α,β) = α log β - lgamma(α) + (α-1) log y - β y
            # with β = α/μ
            beta = alpha / (mu_pos + 1e-10)
            log_p = (
                alpha * torch.log(beta + 1e-10)
                - torch.lgamma(alpha)
                + (alpha - 1.0) * log_y_pos
                - beta * y_pos
            )
            sev_loss = -torch.mean(log_p)

        # -- Combine --
        if self.auto_balance:
            w = (freq_loss.detach().abs() / (sev_loss.detach().abs() + self.eps)).clamp(0.01, 100.0)
        else:
            w = self.loss_weight_sev

        total_loss = freq_loss + w * sev_loss
        return total_loss, freq_loss.detach(), sev_loss.detach()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

@dataclass
class _EarlyStoppingState:
    patience: int
    min_delta: float
    best_loss: float = math.inf
    wait: int = 0
    should_stop: bool = False
    best_state: Optional[dict] = None

    def step(self, val_loss: float, model_state: dict) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            self.best_state = {k: v.cpu().clone() for k, v in model_state.items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters for the training loop.

    Parameters
    ----------
    max_epochs:
        Maximum number of training epochs.
    batch_size:
        Mini-batch size.  Larger batches give more stable gradient estimates
        but use more memory.  For most UK motor datasets 512-2048 works well.
    lr:
        Learning rate for the Adam optimiser.
    trunk_lr_multiplier:
        If not 1.0, the trunk uses ``lr * trunk_lr_multiplier`` while the heads
        use ``lr``.  A value < 1.0 (e.g. 0.3) slows trunk updates, which can
        help when the heads need to adapt faster.
    weight_decay:
        L2 regularisation applied to all parameters.
    loss_weight_sev:
        Fixed severity loss weight (used when ``auto_balance=False``).
    auto_balance:
        Equalise Poisson and Gamma loss magnitudes automatically each step.
    patience:
        Early stopping patience (epochs without improvement on validation loss).
        Set to None to disable early stopping.
    min_delta:
        Minimum improvement in validation loss to count as an improvement.
    lr_reduce_factor:
        Factor by which the LR scheduler reduces the learning rate on plateau.
    lr_patience:
        Epochs without improvement before the LR scheduler fires.
    verbose:
        Whether to print epoch-level training summaries.
    device:
        PyTorch device string.  ``"auto"`` selects CUDA if available, else CPU.
    """

    max_epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    trunk_lr_multiplier: float = 1.0
    weight_decay: float = 1e-4
    loss_weight_sev: float = 1.0
    auto_balance: bool = True
    patience: Optional[int] = 15
    min_delta: float = 1e-4
    lr_reduce_factor: float = 0.5
    lr_patience: int = 5
    verbose: bool = True
    device: str = "auto"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DependentFSTrainer:
    """Trains a ``DependentFreqSevNet`` using the joint Poisson-Gamma loss.

    Parameters
    ----------
    model:
        The network to train.
    config:
        Training hyperparameters.
    """

    def __init__(
        self,
        model: DependentFreqSevNet,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self._device = self._resolve_device(self.config.device)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "freq_loss": [],
            "sev_loss": [],
            "gamma": [],
        }

    @staticmethod
    def _resolve_device(spec: str) -> torch.device:
        if spec == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(spec)

    def _build_optimizer(self) -> Optimizer:
        cfg = self.config
        if abs(cfg.trunk_lr_multiplier - 1.0) < 1e-9:
            return Adam(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        trunk_params = list(self.model.trunk.parameters())
        head_params = (
            list(self.model.freq_head.parameters())
            + list(self.model.sev_head.parameters())
            + [self.model.log_phi]
        )
        return Adam(
            [
                {"params": trunk_params, "lr": cfg.lr * cfg.trunk_lr_multiplier},
                {"params": head_params, "lr": cfg.lr},
            ],
            weight_decay=cfg.weight_decay,
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: JointLoss,
        optimizer: Optional[Optimizer],
        train: bool,
    ) -> Tuple[float, float, float]:
        """Run one epoch.  Returns (mean_loss, mean_freq_loss, mean_sev_loss)."""
        self.model.train(train)
        total_loss = freq_loss_sum = sev_loss_sum = 0.0
        n_batches = 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                x = batch["x"].to(self._device)
                log_exp = batch["log_exposure"].to(self._device)
                n = batch["n_claims"].to(self._device)
                y = batch["avg_severity"].to(self._device)

                log_lambda, log_mu, phi = self.model(x, log_exp, n)
                loss, fl, sl = criterion(log_lambda, log_mu, phi, n, y)

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    optimizer.step()

                total_loss += loss.item()
                freq_loss_sum += fl.item()
                sev_loss_sum += sl.item()
                n_batches += 1

        return total_loss / n_batches, freq_loss_sum / n_batches, sev_loss_sum / n_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> "DependentFSTrainer":
        """Train the model.

        Parameters
        ----------
        train_loader:
            DataLoader yielding batches with keys ``x``, ``log_exposure``,
            ``n_claims``, ``avg_severity``.
        val_loader:
            Optional validation DataLoader for early stopping.

        Returns
        -------
        self
        """
        cfg = self.config
        self.model.to(self._device)
        criterion = JointLoss(
            loss_weight_sev=cfg.loss_weight_sev,
            auto_balance=cfg.auto_balance,
        )
        optimizer = self._build_optimizer()
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=cfg.lr_reduce_factor,
            patience=cfg.lr_patience,
            min_lr=1e-6,
        )
        early_stop: Optional[_EarlyStoppingState] = None
        if cfg.patience is not None:
            early_stop = _EarlyStoppingState(
                patience=cfg.patience, min_delta=cfg.min_delta
            )

        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.time()
            train_loss, fl, sl = self._run_epoch(
                train_loader, criterion, optimizer, train=True
            )
            self.history["train_loss"].append(train_loss)
            self.history["freq_loss"].append(fl)
            self.history["sev_loss"].append(sl)
            if self.model.gamma is not None:
                self.history["gamma"].append(self.model.gamma.item())

            val_loss = train_loss
            if val_loader is not None:
                val_loss, _, _ = self._run_epoch(
                    val_loader, criterion, optimizer=None, train=False
                )
            self.history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            if cfg.verbose and (epoch % 10 == 0 or epoch == 1):
                gamma_str = ""
                if self.model.gamma is not None:
                    gamma_str = f"  γ={self.model.gamma.item():.4f}"
                elapsed = time.time() - t0
                logger.info(
                    "Epoch %3d/%d  train=%.4f  val=%.4f  "
                    "freq=%.4f  sev=%.4f%s  (%.1fs)",
                    epoch, cfg.max_epochs,
                    train_loss, val_loss, fl, sl, gamma_str, elapsed,
                )

            if early_stop is not None:
                early_stop.step(val_loss, self.model.state_dict())
                if early_stop.should_stop:
                    if cfg.verbose:
                        logger.info(
                            "Early stopping at epoch %d (best val=%.4f)",
                            epoch, early_stop.best_loss,
                        )
                    break

        # Restore best weights if early stopping was used.
        if early_stop is not None and early_stop.best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self._device) for k, v in early_stop.best_state.items()}
            )

        self.model.eval()
        return self
