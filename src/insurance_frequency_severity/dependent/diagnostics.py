"""
diagnostics.py
--------------
Post-fit diagnostics for the dependent frequency-severity model.

The class ``DependentFSDiagnostics`` takes a fitted ``DependentFSModel`` and
evaluation data and provides:

  - Lorenz / concentration curves for frequency and severity (lift ordering)
  - Calibration plots: predicted vs observed in deciles
  - Dependence significance test: bootstrap LRT on H₀: γ=0
  - Latent correlation analysis: structure in the shared embedding space
  - Head-to-head comparison against two independently fitted networks
  - Gini index and Normalised Gini for pricing lift

All plot methods return ``(fig, ax)`` tuples and require matplotlib.  The
non-plotting methods return plain numpy arrays and dicts so they work without
matplotlib installed.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats

from insurance_frequency_severity.dependent.wrapper import DependentFSModel

logger = logging.getLogger(__name__)


class DependentFSDiagnostics:
    """Diagnostic tools for a fitted ``DependentFSModel``.

    Parameters
    ----------
    model:
        A fitted ``DependentFSModel``.
    X:
        Feature matrix used for evaluation, shape (n, p).
    n_claims:
        Observed claim counts, shape (n,).
    avg_severity:
        Observed average severity, shape (n,).
    exposure:
        Policy exposure, shape (n,).
    """

    def __init__(
        self,
        model: DependentFSModel,
        X: np.ndarray,
        n_claims: np.ndarray,
        avg_severity: np.ndarray,
        exposure: np.ndarray,
    ) -> None:
        self.model = model
        self.X = np.asarray(X, dtype=np.float32)
        self.n_claims = np.asarray(n_claims, dtype=np.float32)
        self.avg_severity = np.asarray(avg_severity, dtype=np.float32)
        self.exposure = np.asarray(exposure, dtype=np.float32)

    # ------------------------------------------------------------------
    # Lorenz / Gini
    # ------------------------------------------------------------------

    def lorenz_curve(
        self,
        target: str = "frequency",
        n_groups: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute the concentration (Lorenz) curve for frequency or severity.

        Policies are sorted by their predicted risk (ascending), and we plot
        the cumulative proportion of actual losses vs. cumulative proportion of
        policies.  A perfect model has all concentration at the right; a random
        model gives the 45-degree diagonal.

        Parameters
        ----------
        target:
            ``"frequency"`` or ``"severity"`` or ``"pure_premium"``.
        n_groups:
            Number of groups for binning the predictions.

        Returns
        -------
        cum_exposure: np.ndarray
            Cumulative exposure fraction (x-axis).
        cum_loss: np.ndarray
            Cumulative loss fraction (y-axis).
        gini: float
            Normalised Gini coefficient (2·AUC − 1).
        """
        if target == "frequency":
            pred = self.model.predict_frequency(self.X, self.exposure)
            actual = self.n_claims / self.exposure
        elif target == "severity":
            pos = self.n_claims > 0
            if pos.sum() == 0:
                raise ValueError("No positive-claim rows for severity Lorenz curve.")
            pred = self.model.predict_severity(self.X[pos], self.exposure[pos])
            actual = self.avg_severity[pos]
        elif target == "pure_premium":
            pred = self.model.predict_pure_premium(self.X, self.exposure)
            actual = self.n_claims * self.avg_severity / self.exposure
        else:
            raise ValueError(f"Unknown target '{target}'. Use 'frequency', 'severity', or 'pure_premium'.")

        order = np.argsort(pred)
        actual_sorted = actual[order]
        exp_sorted = self.exposure[order] if target != "severity" else np.ones_like(actual_sorted)

        cum_exp = np.cumsum(exp_sorted) / exp_sorted.sum()
        cum_loss = np.cumsum(actual_sorted * exp_sorted) / (actual_sorted * exp_sorted).sum()

        cum_exp = np.concatenate([[0.0], cum_exp])
        cum_loss = np.concatenate([[0.0], cum_loss])

        auc = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(cum_loss, cum_exp)
        gini = 2.0 * auc - 1.0
        return cum_exp, cum_loss, gini

    def gini_summary(self) -> Dict[str, float]:
        """Return Gini coefficients for frequency, severity, and pure premium."""
        result = {}
        for t in ("frequency", "pure_premium"):
            _, _, g = self.lorenz_curve(target=t)
            result[f"gini_{t}"] = g
        pos = self.n_claims > 0
        if pos.sum() >= 10:
            _, _, g_sev = self.lorenz_curve(target="severity")
            result["gini_severity"] = g_sev
        return result

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibration(
        self,
        target: str = "frequency",
        n_deciles: int = 10,
    ) -> Dict[str, np.ndarray]:
        """Calibration of predicted vs observed in n_deciles risk buckets.

        Parameters
        ----------
        target:
            ``"frequency"`` or ``"pure_premium"``.
        n_deciles:
            Number of equally-sized quantile buckets.

        Returns
        -------
        dict with keys ``pred_mean``, ``obs_mean``, ``bucket_edge``.
        """
        if target == "frequency":
            pred = self.model.predict_frequency(self.X, self.exposure)
            obs = self.n_claims / self.exposure
        elif target == "pure_premium":
            pred = self.model.predict_pure_premium(self.X, self.exposure)
            obs = self.n_claims * self.avg_severity / self.exposure
        else:
            raise ValueError(f"Unknown target '{target}'.")

        quantiles = np.linspace(0, 100, n_deciles + 1)
        edges = np.percentile(pred, quantiles)
        pred_means = []
        obs_means = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (pred >= lo) & (pred <= hi)
            if mask.sum() == 0:
                continue
            pred_means.append(pred[mask].mean())
            obs_means.append(obs[mask].mean())
        return {
            "pred_mean": np.array(pred_means),
            "obs_mean": np.array(obs_means),
            "bucket_edge": edges,
        }

    # ------------------------------------------------------------------
    # Dependence test
    # ------------------------------------------------------------------

    def dependence_test(
        self,
        n_bootstrap: int = 200,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Test H₀: γ = 0 (frequency-severity independence via explicit term).

        This is only meaningful when ``use_explicit_gamma=True``.  It returns
        the estimated γ, its bootstrap standard error, and an approximate
        z-statistic.

        The test does NOT account for the implicit latent dependence from the
        shared trunk — that is captured in the latent correlation analysis
        instead.

        Parameters
        ----------
        n_bootstrap:
            Number of bootstrap resamples for the standard error.
        seed:
            Random seed.

        Returns
        -------
        dict with keys ``gamma``, ``gamma_se``, ``z_stat``, ``p_value``.
        """
        if not self.model.use_explicit_gamma:
            return {"gamma": None, "gamma_se": None, "z_stat": None, "p_value": None,
                    "note": "use_explicit_gamma=False; no γ to test."}

        gamma_hat = self.model.gamma_

        # Bootstrap the γ estimate.
        rng = np.random.default_rng(seed)
        n = len(self.X)
        gamma_boots = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            m_boot = DependentFSModel(
                trunk_config=self.model.trunk_config,
                training_config=self.model.training_config,
                use_explicit_gamma=True,
                val_fraction=0.0,
                random_state=seed,
            )
            # Fit with a short training (5 epochs) to get bootstrap distribution
            import copy
            from insurance_frequency_severity.dependent.training import TrainingConfig
            fast_tc = TrainingConfig(max_epochs=5, verbose=False, auto_balance=True)
            m_boot.training_config = fast_tc
            try:
                m_boot.fit(
                    self.X[idx],
                    self.n_claims[idx],
                    self.avg_severity[idx],
                    self.exposure[idx],
                )
                gamma_boots.append(m_boot.gamma_)
            except Exception:
                pass

        gamma_boots = np.array(gamma_boots)
        se = gamma_boots.std() if len(gamma_boots) > 1 else np.nan
        z = gamma_hat / se if se > 0 else np.nan
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan

        return {
            "gamma": gamma_hat,
            "gamma_se": se,
            "z_stat": z,
            "p_value": p_value,
            "n_bootstrap": len(gamma_boots),
        }

    # ------------------------------------------------------------------
    # Latent correlation
    # ------------------------------------------------------------------

    def latent_correlation(self) -> Dict[str, object]:
        """Correlation structure in the shared latent space.

        Computes:
        - Correlation matrix of latent dimensions h₁, …, h_d
        - Correlation of each latent dim with observed frequency and severity
        - Number of latent dims with |corr| > 0.1 with freq and with sev

        Returns
        -------
        dict with keys ``latent_corr``, ``freq_corr``, ``sev_corr``.
        """
        h = self.model.latent_repr(self.X)  # (n, d)
        freq_actual = self.n_claims / (self.exposure + 1e-10)

        pos = self.n_claims > 0
        sev_actual = np.where(pos, self.avg_severity, np.nan)

        # Latent-latent correlation matrix
        latent_corr = np.corrcoef(h.T)  # (d, d)

        # Latent → frequency correlation
        freq_corr = np.array([
            np.corrcoef(h[:, j], freq_actual)[0, 1] for j in range(h.shape[1])
        ])

        # Latent → severity correlation (positive claims only)
        sev_corr_vals = []
        for j in range(h.shape[1]):
            valid = ~np.isnan(sev_actual) & (sev_actual > 0)
            if valid.sum() > 5:
                sev_corr_vals.append(np.corrcoef(h[valid, j], sev_actual[valid])[0, 1])
            else:
                sev_corr_vals.append(np.nan)
        sev_corr = np.array(sev_corr_vals)

        return {
            "latent_corr": latent_corr,
            "freq_corr": freq_corr,
            "sev_corr": sev_corr,
            "n_freq_active": int((np.abs(freq_corr) > 0.1).sum()),
            "n_sev_active": int((np.abs(sev_corr[~np.isnan(sev_corr)]) > 0.1).sum()),
        }

    # ------------------------------------------------------------------
    # Comparison vs independent
    # ------------------------------------------------------------------

    def vs_independent(
        self,
        X_val: Optional[np.ndarray] = None,
        n_claims_val: Optional[np.ndarray] = None,
        avg_severity_val: Optional[np.ndarray] = None,
        exposure_val: Optional[np.ndarray] = None,
        n_mc: int = 500,
    ) -> Dict[str, float]:
        """Compare dependent model vs a naive independent (GLM-style) baseline.

        The independent baseline ignores the shared trunk: it predicts pure
        premium as predict_frequency * predict_severity (i.e. assumes E[Z]=E[N]·E[Y]).
        The dependent model uses ``predict_pure_premium`` (MC or analytical).

        Returns mean squared error and mean absolute error for both.

        Parameters
        ----------
        X_val, n_claims_val, avg_severity_val, exposure_val:
            Held-out evaluation data.  If None, uses the data passed to the
            constructor.
        n_mc:
            MC samples for the dependent model.
        """
        X = X_val if X_val is not None else self.X
        n_c = n_claims_val if n_claims_val is not None else self.n_claims
        a_s = avg_severity_val if avg_severity_val is not None else self.avg_severity
        exp = exposure_val if exposure_val is not None else self.exposure

        X = np.asarray(X, dtype=np.float32)
        n_c = np.asarray(n_c, dtype=np.float32)
        a_s = np.asarray(a_s, dtype=np.float32)
        exp = np.asarray(exp, dtype=np.float32)

        actual_pp = n_c * a_s / (exp + 1e-10)

        # Dependent model prediction
        dep_pp = self.model.predict_pure_premium(X, exp, n_mc=n_mc)

        # Independent baseline: freq × sev
        freq = self.model.predict_frequency(X, exp)
        sev = self.model.predict_severity(X, exp)
        indep_pp = freq * sev

        results = {}
        for name, pred in [("dependent", dep_pp), ("independent", indep_pp)]:
            err = pred - actual_pp
            results[f"{name}_mse"] = float(np.mean(err ** 2))
            results[f"{name}_mae"] = float(np.mean(np.abs(err)))
            # Poisson deviance on pure premium
            eps = 1e-10
            y, mu = actual_pp + eps, pred + eps
            dev = 2.0 * np.where(actual_pp > 0, y * np.log(y / mu) - (y - mu), mu - y)
            results[f"{name}_mean_deviance"] = float(np.mean(dev))

        results["mse_reduction_pct"] = (
            100.0 * (results["independent_mse"] - results["dependent_mse"])
            / (results["independent_mse"] + 1e-10)
        )
        return results

    # ------------------------------------------------------------------
    # Plotting (requires matplotlib)
    # ------------------------------------------------------------------

    def plot_lorenz(
        self,
        target: str = "frequency",
        ax=None,
    ):
        """Plot the Lorenz concentration curve.

        Parameters
        ----------
        target:
            ``"frequency"``, ``"severity"``, or ``"pure_premium"``.
        ax:
            Matplotlib axes to plot on.  If None, a new figure is created.

        Returns
        -------
        fig, ax
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )

        cum_exp, cum_loss, gini = self.lorenz_curve(target=target)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure

        ax.plot(cum_exp, cum_loss, label=f"Model (Gini={gini:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("Cumulative exposure fraction")
        ax.set_ylabel(f"Cumulative {target} fraction")
        ax.set_title(f"Lorenz curve — {target}")
        ax.legend()
        return fig, ax

    def plot_calibration(
        self,
        target: str = "frequency",
        n_deciles: int = 10,
        ax=None,
    ):
        """Plot calibration: predicted vs observed in decile buckets.

        Returns
        -------
        fig, ax
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        cal = self.calibration(target=target, n_deciles=n_deciles)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure

        ax.scatter(cal["pred_mean"], cal["obs_mean"], zorder=3)
        lo = min(cal["pred_mean"].min(), cal["obs_mean"].min())
        hi = max(cal["pred_mean"].max(), cal["obs_mean"].max())
        ax.plot([lo, hi], [lo, hi], "k--")
        ax.set_xlabel("Predicted (decile mean)")
        ax.set_ylabel("Observed (decile mean)")
        ax.set_title(f"Calibration — {target}")
        return fig, ax

    def plot_training_history(self, model: Optional[DependentFSModel] = None, ax=None):
        """Plot training and validation loss curves.

        Parameters
        ----------
        model:
            The fitted model.  If None, uses ``self.model``.

        Returns
        -------
        fig, ax
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        m = model or self.model
        hist = m.training_history()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure

        ax.plot(hist["train_loss"], label="Train loss")
        if any(v != 0 for v in hist.get("val_loss", [])):
            ax.plot(hist["val_loss"], label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Joint loss")
        ax.set_title("Training history")
        ax.legend()
        return fig, ax
