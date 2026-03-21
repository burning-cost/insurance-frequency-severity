"""
wrapper.py
----------
Scikit-learn-compatible estimator wrapping ``DependentFreqSevNet``.

``DependentFSModel`` follows the sklearn estimator protocol, so it works with
``cross_val_score``, ``GridSearchCV``, and pipeline-style workflows.  It hides
all PyTorch plumbing — callers work with numpy arrays and DataFrames throughout.

Fitting requires three target arrays alongside the feature matrix:

  - ``n_claims``: non-negative integer claim counts per policy
  - ``avg_severity``: average claim amount (total_loss / n_claims); 0 for
    zero-claim policies
  - ``exposure``: years at risk, strictly positive

The ``score`` method returns the negative mean pure premium Poisson deviance as
a maximisation target compatible with sklearn conventions.

Requires torch. Install with: pip install insurance-frequency-severity[neural]
"""

from __future__ import annotations

import logging
import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    from torch import Tensor
    from torch.utils.data import DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from insurance_frequency_severity.dependent.model import DependentFreqSevNet, SharedTrunkConfig
from insurance_frequency_severity.dependent.training import DependentFSTrainer, TrainingConfig
from insurance_frequency_severity.dependent.data import FreqSevDataset, make_train_val_loaders
from insurance_frequency_severity.dependent.premium import PurePremiumEstimator

logger = logging.getLogger(__name__)


def _require_torch(feature: str = "This feature") -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"{feature} requires torch. "
            "Install with: pip install insurance-frequency-severity[neural]"
        )


class DependentFSModel(BaseEstimator):
    """Sklearn-compatible dependent frequency-severity neural model.

    Requires torch. Install with: pip install insurance-frequency-severity[neural]

    Parameters
    ----------
    trunk_config:
        Shared encoder architecture.  Defaults to ``SharedTrunkConfig()``.
    training_config:
        Training hyperparameters.  Defaults to ``TrainingConfig()``.
    use_explicit_gamma:
        Whether to include the GGS γ·N conditional covariate in the severity
        head.  When True, ``gamma_`` is populated after fitting and analytical
        pure premium is available.
    n_mc:
        Number of Monte Carlo samples per policy for ``predict_pure_premium``
        (used when ``method="mc"`` or always if ``use_explicit_gamma=False``).
    val_fraction:
        Fraction of training data held out for early stopping.  Set to 0.0 to
        disable validation split (and early stopping).
    batch_size:
        Mini-batch size for training DataLoader.
    random_state:
        Seed for reproducibility of data splitting and MC sampling.

    Attributes
    ----------
    model_:
        Fitted ``DependentFreqSevNet`` (set after ``fit``).
    trainer_:
        Fitted ``DependentFSTrainer`` (holds training history).
    n_features_in_:
        Number of input features.
    gamma_:
        Estimated explicit dependence parameter γ (None if
        ``use_explicit_gamma=False``).
    """

    def __init__(
        self,
        trunk_config: Optional[SharedTrunkConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        use_explicit_gamma: bool = True,
        n_mc: int = 1000,
        val_fraction: float = 0.1,
        batch_size: int = 512,
        random_state: int = 42,
    ) -> None:
        self.trunk_config = trunk_config
        self.training_config = training_config
        self.use_explicit_gamma = use_explicit_gamma
        self.n_mc = n_mc
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        n_claims: np.ndarray,
        avg_severity: np.ndarray,
        exposure: np.ndarray,
    ) -> "DependentFSModel":
        """Fit the joint frequency-severity model.

        Parameters
        ----------
        X:
            Feature matrix, shape (n_policies, n_features).  Must be numeric
            float32.  Use ``prepare_features`` to encode raw DataFrames.
        n_claims:
            Claim count per policy, shape (n_policies,).
        avg_severity:
            Average claim size per policy, shape (n_policies,).  Zero for
            policies with zero claims.
        exposure:
            Years at risk per policy, shape (n_policies,).  Must be > 0.

        Returns
        -------
        self
        """
        _require_torch("DependentFSModel.fit")

        X = np.asarray(X, dtype=np.float32)
        n_claims = np.asarray(n_claims, dtype=np.float32)
        avg_severity = np.asarray(avg_severity, dtype=np.float32)
        exposure = np.asarray(exposure, dtype=np.float32)

        self.n_features_in_ = X.shape[1]
        trunk_config = self.trunk_config or SharedTrunkConfig()
        train_config = self.training_config or TrainingConfig()
        # Propagate batch_size into training config if not set by caller.
        if self.training_config is None:
            train_config = TrainingConfig(
                batch_size=self.batch_size,
                verbose=train_config.verbose,
            )

        self.model_ = DependentFreqSevNet(
            in_features=self.n_features_in_,
            trunk_config=trunk_config,
            use_explicit_gamma=self.use_explicit_gamma,
        )

        dataset = FreqSevDataset(X, n_claims, avg_severity, exposure)

        if self.val_fraction > 0.0:
            train_loader, val_loader = make_train_val_loaders(
                dataset,
                val_fraction=self.val_fraction,
                batch_size=train_config.batch_size,
                seed=self.random_state,
            )
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=train_config.batch_size,
                shuffle=True,
            )
            val_loader = None

        self.trainer_ = DependentFSTrainer(self.model_, train_config)
        self.trainer_.fit(train_loader, val_loader)

        if self.use_explicit_gamma and self.model_.gamma is not None:
            self.gamma_ = self.model_.gamma.item()
        else:
            self.gamma_ = None

        return self

    # ------------------------------------------------------------------
    # Predict helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, arr: np.ndarray, dtype=None) -> "Tensor":
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(np.asarray(arr, dtype=np.float32), dtype=dtype)

    def _device(self) -> "torch.device":
        return next(self.model_.parameters()).device

    def _forward_numpy(
        self,
        X: np.ndarray,
        exposure: np.ndarray,
        n_claims_for_gamma: Optional[np.ndarray] = None,
    ) -> Tuple["Tensor", "Tensor", "Tensor"]:
        """Run forward pass and return (log_lambda, log_mu, phi) as CPU tensors."""
        check_is_fitted(self, "model_")
        self.model_.eval()
        device = self._device()
        x_t = self._to_tensor(X).to(device)
        log_exp_t = self._to_tensor(np.log(exposure.astype(np.float32))).to(device)
        n_t: Optional["Tensor"] = None
        if self.use_explicit_gamma:
            # When use_explicit_gamma=True the severity head requires n_claims.
            # Default to zeros (N=0) when not provided — gives baseline severity
            # without the gamma adjustment, which is correct for predict_frequency
            # and predict_severity calls.
            if n_claims_for_gamma is not None:
                n_t = self._to_tensor(n_claims_for_gamma).to(device)
            else:
                n_t = torch.zeros(len(X), dtype=torch.float32, device=device)

        with torch.no_grad():
            log_lambda, log_mu, phi = self.model_(x_t, log_exp_t, n_t)

        return log_lambda.cpu(), log_mu.cpu(), phi.cpu()

    # ------------------------------------------------------------------
    # Public prediction methods
    # ------------------------------------------------------------------

    def predict_frequency(
        self,
        X: np.ndarray,
        exposure: np.ndarray,
    ) -> np.ndarray:
        """Predict expected claim frequency (per unit exposure).

        Parameters
        ----------
        X:
            Feature matrix, shape (n, p).
        exposure:
            Exposure for each policy, shape (n,).

        Returns
        -------
        np.ndarray of shape (n,)
            Expected claims per unit exposure for each policy.
        """
        _require_torch("DependentFSModel.predict_frequency")
        exposure = np.asarray(exposure, dtype=np.float32)
        log_lambda, _, _ = self._forward_numpy(X, exposure)
        lambda_with_exp = torch.exp(log_lambda).numpy()
        return lambda_with_exp / exposure

    def predict_severity(
        self,
        X: np.ndarray,
        exposure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict expected average severity (at N=0 for γ·N term).

        When ``use_explicit_gamma=True``, the severity head includes γ·N.  In
        this method N is set to zero, giving the baseline severity
        exp(SevHead(h)).  This is the severity for a policy with zero prior
        claims — which is useful for comparing relative risk, but not for
        computing the pure premium (use ``predict_pure_premium`` for that).

        Parameters
        ----------
        X:
            Feature matrix, shape (n, p).
        exposure:
            Needed internally; defaults to all-ones if not provided.

        Returns
        -------
        np.ndarray of shape (n,)
            Expected average severity (baseline, N=0).
        """
        _require_torch("DependentFSModel.predict_severity")
        n = len(X)
        if exposure is None:
            exposure = np.ones(n, dtype=np.float32)
        exposure = np.asarray(exposure, dtype=np.float32)
        n_zero = np.zeros(n, dtype=np.float32)
        _, log_mu, _ = self._forward_numpy(X, exposure, n_claims_for_gamma=n_zero)
        return torch.exp(log_mu).numpy()

    def predict_pure_premium(
        self,
        X: np.ndarray,
        exposure: np.ndarray,
        method: str = "auto",
        n_mc: Optional[int] = None,
    ) -> np.ndarray:
        """Predict pure premium (expected total loss per unit exposure).

        Parameters
        ----------
        X:
            Feature matrix, shape (n, p).
        exposure:
            Exposure, shape (n,).
        method:
            ``"auto"`` uses analytical when γ is available, MC otherwise.
            ``"mc"`` forces Monte Carlo regardless.
            ``"analytical"`` uses the GGS MGF formula (requires γ).
        n_mc:
            Override the MC sample size.

        Returns
        -------
        np.ndarray of shape (n,)
            Pure premium per unit exposure.
        """
        _require_torch("DependentFSModel.predict_pure_premium")
        check_is_fitted(self, "model_")
        exposure = np.asarray(exposure, dtype=np.float32)
        n = len(X)

        use_analytical = (
            method == "analytical"
            or (method == "auto" and self.use_explicit_gamma)
        )

        estimator = PurePremiumEstimator(
            n_mc=n_mc or self.n_mc,
            seed=self.random_state,
        )

        # Zero out n_claims for the forward pass (base severity, no look-ahead)
        n_zero = np.zeros(n, dtype=np.float32)
        log_lambda, log_mu_base, phi = self._forward_numpy(
            X, exposure, n_claims_for_gamma=n_zero
        )
        exp_t = torch.tensor(exposure, dtype=torch.float32)

        if use_analytical and self.model_.gamma is not None:
            gamma_t = self.model_.gamma.detach().cpu()
            pp = estimator.analytical(log_lambda, log_mu_base, gamma_t, exp_t)
        else:
            pp = estimator.monte_carlo(log_lambda, log_mu_base, phi, exp_t)

        return pp.numpy()

    def predict(self, X: np.ndarray, exposure: Optional[np.ndarray] = None) -> np.ndarray:
        """Sklearn-compatible predict: returns pure premium with unit exposure."""
        if exposure is None:
            exposure = np.ones(len(X), dtype=np.float32)
        return self.predict_pure_premium(X, exposure)

    def latent_repr(self, X: np.ndarray) -> np.ndarray:
        """Return the shared latent representation h for each policy.

        Useful for diagnostics: inspect whether the latent space captures
        meaningful risk structure, or compute correlation between freq-direction
        and sev-direction in h.

        Parameters
        ----------
        X:
            Feature matrix, shape (n, p).

        Returns
        -------
        np.ndarray of shape (n, latent_dim)
        """
        _require_torch("DependentFSModel.latent_repr")
        check_is_fitted(self, "model_")
        device = self._device()
        x_t = self._to_tensor(X).to(device)
        h = self.model_.latent(x_t)
        return h.cpu().numpy()

    def score(
        self,
        X: np.ndarray,
        n_claims: np.ndarray,
        avg_severity: np.ndarray,
        exposure: np.ndarray,
    ) -> float:
        """Negative mean Poisson deviance on frequency (higher is better).

        This is the sklearn convention: ``score`` returns a value to maximise.

        The metric is the mean Poisson deviance on frequency predictions, negated
        so that cross_val_score() sees it as a maximisation problem.

        Parameters
        ----------
        X, n_claims, avg_severity, exposure:
            Evaluation data in the same format as ``fit``.

        Returns
        -------
        float
            Negative mean Poisson deviance.
        """
        _require_torch("DependentFSModel.score")
        exposure = np.asarray(exposure, dtype=np.float32)
        freq_pred = self.predict_frequency(X, exposure)
        freq_actual_per_unit = n_claims.astype(np.float32) / exposure

        # Poisson deviance = 2 * Σ [y log(y/μ) - (y - μ)]
        eps = 1e-10
        y = freq_actual_per_unit
        mu = freq_pred + eps
        dev = 2.0 * np.where(y > 0, y * np.log(y / mu) - (y - mu), mu)
        return -float(np.mean(dev))

    def training_history(self) -> dict:
        """Return the training history dict from the trainer."""
        check_is_fitted(self, "trainer_")
        return self.trainer_.history
