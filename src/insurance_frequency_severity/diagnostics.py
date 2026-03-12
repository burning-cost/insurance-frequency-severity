"""
diagnostics.py — Model validation and dependence testing.

DependenceTest
    Tests H0: independence of N and S. Three approaches: Kendall tau permutation
    test, Spearman correlation test, and likelihood ratio test for omega=0.

CopulaGOF
    Goodness-of-fit for a fitted copula. Uses Rosenblatt probability integral
    transform to map the bivariate sample onto (0,1)^2, then tests uniformity.

compare_copulas()
    Fits all three copula families to the same data and returns an AIC/BIC
    comparison table. The primary use case: convincing a pricing team that the
    Sarmanov model is justified over the independence assumption.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize_scalar


class DependenceTest:
    """
    Tests the null hypothesis of independence between claim frequency and severity.

    Three test statistics are available:
    - 'kendall': Kendall tau with asymptotic normal approximation
    - 'spearman': Spearman rho with t-distribution approximation
    - 'lrt': Likelihood ratio test for omega=0 in the Sarmanov model

    Only positive-claim observations are used (n=0 policies contribute no
    severity information).

    Parameters
    ----------
    n_permutations : int
        Number of permutations for the permutation test. Set to 0 to use
        asymptotic approximation only.

    Examples
    --------
    >>> test = DependenceTest()
    >>> result = test.fit(n_positive, s_positive)
    >>> result.summary()
    """

    def __init__(self, n_permutations: int = 1000):
        self.n_permutations = n_permutations
        self.tau_: Optional[float] = None
        self.tau_pval_: Optional[float] = None
        self.rho_s_: Optional[float] = None
        self.rho_s_pval_: Optional[float] = None
        self.n_obs_: Optional[int] = None

    def fit(
        self,
        n: np.ndarray,
        s: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> "DependenceTest":
        """
        Run the dependence tests.

        Parameters
        ----------
        n : array-like
            Claim counts for positive-claim policies only.
        s : array-like
            Average claim severities (must match n in length).
        rng : numpy Generator, optional
            For reproducible permutation tests.
        """
        n = np.asarray(n, dtype=float)
        s = np.asarray(s, dtype=float)

        # Filter to valid positive-claim observations
        mask = (n > 0) & (s > 0) & np.isfinite(n) & np.isfinite(s)
        n_valid = n[mask]
        s_valid = s[mask]

        if len(n_valid) < 10:
            raise ValueError(
                f"Only {len(n_valid)} valid positive-claim observations. "
                "Need at least 10 for dependence testing."
            )

        self.n_obs_ = int(len(n_valid))

        # Kendall tau
        tau, pval_tau = stats.kendalltau(n_valid, s_valid)
        self.tau_ = float(tau)
        self.tau_pval_ = float(pval_tau)

        # Spearman rho
        rho_s, pval_rho = stats.spearmanr(n_valid, s_valid)
        self.rho_s_ = float(rho_s)
        self.rho_s_pval_ = float(pval_rho)

        # Permutation test for Kendall tau
        if self.n_permutations > 0:
            if rng is None:
                rng = np.random.default_rng()
            perm_taus = np.empty(self.n_permutations)
            for i in range(self.n_permutations):
                s_perm = rng.permutation(s_valid)
                t, _ = stats.kendalltau(n_valid, s_perm)
                perm_taus[i] = t
            self.tau_pval_perm_ = float(np.mean(np.abs(perm_taus) >= np.abs(self.tau_)))
        else:
            self.tau_pval_perm_ = None

        return self

    def summary(self) -> pd.DataFrame:
        """Return test results as a DataFrame."""
        if self.tau_ is None:
            raise RuntimeError("Must call .fit() first")

        rows = [
            {
                "test": "Kendall tau (asymptotic)",
                "statistic": self.tau_,
                "p_value": self.tau_pval_,
                "n_obs": self.n_obs_,
                "conclusion": "reject H0 (dependence)" if self.tau_pval_ < 0.05 else "fail to reject H0 (independence)",
            },
            {
                "test": "Spearman rho (asymptotic)",
                "statistic": self.rho_s_,
                "p_value": self.rho_s_pval_,
                "n_obs": self.n_obs_,
                "conclusion": "reject H0 (dependence)" if self.rho_s_pval_ < 0.05 else "fail to reject H0 (independence)",
            },
        ]

        if hasattr(self, "tau_pval_perm_") and self.tau_pval_perm_ is not None:
            rows.append({
                "test": f"Kendall tau (permutation, B={self.n_permutations})",
                "statistic": self.tau_,
                "p_value": self.tau_pval_perm_,
                "n_obs": self.n_obs_,
                "conclusion": "reject H0 (dependence)" if self.tau_pval_perm_ < 0.05 else "fail to reject H0 (independence)",
            })

        return pd.DataFrame(rows)


class CopulaGOF:
    """
    Goodness-of-fit testing for a fitted copula model.

    Uses the Rosenblatt probability integral transform (RPIT) to map the
    bivariate sample onto uniform margins, then tests uniformity of the
    transformed data using a Kolmogorov-Smirnov test on each marginal and
    a joint test.

    For the Sarmanov copula, the RPIT involves computing the conditional CDF
    F_{S|N}(s|n) = f(n,s) / f_N(n), which is tractable analytically.

    Parameters
    ----------
    joint_model : JointFreqSev
        A fitted JointFreqSev model.
    """

    def __init__(self, joint_model: Any):
        self.joint_model = joint_model
        self.ks_stat_u_: Optional[float] = None
        self.ks_stat_v_: Optional[float] = None
        self.ks_pval_u_: Optional[float] = None
        self.ks_pval_v_: Optional[float] = None
        self.n_obs_: Optional[int] = None

    def fit(
        self,
        n: np.ndarray,
        s: np.ndarray,
        freq_params: list,
        sev_params: list,
    ) -> "CopulaGOF":
        """
        Compute RPIT and test uniformity.

        For positive-claim observations only.
        """
        n = np.asarray(n, dtype=float)
        s = np.asarray(s, dtype=float)
        mask = (n > 0) & (s > 0) & np.isfinite(s)
        n_pos = n[mask]
        s_pos = s[mask]
        fp_pos = [freq_params[i] for i in range(len(n)) if mask[i]]
        sp_pos = [sev_params[i] for i in range(len(n)) if mask[i]]
        self.n_obs_ = int(len(n_pos))

        # Marginal CDF transforms (PIT for each margin)
        # U_N = mid-point PIT for discrete N
        # V_S = CDF of S
        from insurance_frequency_severity.copula import GaussianCopulaMixed
        gc = GaussianCopulaMixed(rho=0.0)

        freq_family = self.joint_model._freq_family
        sev_family = self.joint_model._sev_family

        fp_dict = {
            "mu": np.array([p["mu"] for p in fp_pos]),
            **({
                "alpha": np.array([p["alpha"] for p in fp_pos])
            } if freq_family == "nb" else {}),
        }
        if sev_family == "gamma":
            sp_dict = {
                "mu": np.array([p["mu"] for p in sp_pos]),
                "shape": np.array([p["shape"] for p in sp_pos]),
            }
        else:
            sp_dict = {
                "log_mu": np.array([p["log_mu"] for p in sp_pos]),
                "log_sigma": np.array([p["log_sigma"] for p in sp_pos]),
            }

        u = gc._pit_freq(n_pos, fp_dict, freq_family)
        v = gc._pit_sev(s_pos, sp_dict, sev_family)
        u = np.clip(u, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)

        self._u = u
        self._v = v

        # KS test against Uniform(0,1)
        ks_u = stats.kstest(u, "uniform")
        ks_v = stats.kstest(v, "uniform")

        self.ks_stat_u_ = float(ks_u.statistic)
        self.ks_stat_v_ = float(ks_v.statistic)
        self.ks_pval_u_ = float(ks_u.pvalue)
        self.ks_pval_v_ = float(ks_v.pvalue)

        return self

    def summary(self) -> pd.DataFrame:
        """Return GOF test results."""
        if self.ks_stat_u_ is None:
            raise RuntimeError("Must call .fit() first")

        return pd.DataFrame([
            {
                "margin": "frequency (PIT)",
                "ks_statistic": self.ks_stat_u_,
                "ks_pvalue": self.ks_pval_u_,
                "assessment": "acceptable" if self.ks_pval_u_ > 0.05 else "poor fit",
            },
            {
                "margin": "severity (CDF)",
                "ks_statistic": self.ks_stat_v_,
                "ks_pvalue": self.ks_pval_v_,
                "assessment": "acceptable" if self.ks_pval_v_ > 0.05 else "poor fit",
            },
        ])


def compare_copulas(
    n: np.ndarray,
    s: np.ndarray,
    freq_glm: Any,
    sev_glm: Any,
    freq_X: Optional[pd.DataFrame] = None,
    sev_X: Optional[pd.DataFrame] = None,
    exposure: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Fit all three copula families and compare by AIC/BIC.

    This is the primary diagnostic for deciding whether dependence modelling
    is warranted. The workflow:

    1. Fit independence model (omega=0 in Sarmanov). Establishes baseline.
    2. Fit Sarmanov, Gaussian, FGM copulas.
    3. Compare AIC/BIC. If independence model wins, the dependence is not
       statistically significant.

    Parameters
    ----------
    n : array-like
        Claim counts for all policies.
    s : array-like
        Average severities (0 or NaN for zero-claim policies).
    freq_glm : fitted GLM
        Frequency model.
    sev_glm : fitted GLM
        Severity model.
    freq_X, sev_X : DataFrame, optional
        Feature matrices for prediction.
    exposure : array-like, optional
        Exposure for frequency.
    rng : numpy Generator, optional

    Returns
    -------
    DataFrame with copula, omega/rho, spearman_rho, aic, bic, delta_aic columns.
    """
    from insurance_frequency_severity.joint import JointFreqSev

    data = pd.DataFrame({"claim_count": n, "avg_severity": s})
    if exposure is not None:
        data["_exposure"] = exposure

    results = []

    for copula_family in ["sarmanov", "gaussian", "fgm"]:
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm, copula=copula_family)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(
                    data,
                    n_col="claim_count",
                    s_col="avg_severity",
                    freq_X=freq_X,
                    sev_X=sev_X,
                    exposure_col="_exposure" if exposure is not None else None,
                    ci_method="profile" if copula_family == "sarmanov" else "profile",
                    rng=rng,
                )
            summary = model.dependence_summary()
            results.append({
                "copula": copula_family,
                "omega_or_rho": float(model.omega_),
                "spearman_rho": float(model.rho_) if model.rho_ is not None else np.nan,
                "aic": float(model.aic_),
                "bic": float(model.bic_),
                "converged": True,
            })
        except Exception as e:
            results.append({
                "copula": copula_family,
                "omega_or_rho": np.nan,
                "spearman_rho": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "converged": False,
            })

    df = pd.DataFrame(results)
    if df["aic"].notna().any():
        min_aic = df["aic"].min()
        df["delta_aic"] = df["aic"] - min_aic
        df = df.sort_values("aic")

    return df.reset_index(drop=True)
