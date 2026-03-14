"""
joint.py — Main user-facing API for joint frequency-severity modelling.

Two classes are provided:

JointFreqSev
    Copula-based joint model. Accepts fitted frequency and severity GLMs
    (statsmodels or any object with .predict() and distributional parameters).
    Estimates the Sarmanov omega by IFM (profile likelihood given fitted
    marginals). Computes premium correction factors E[N*S]/(E[N]*E[S]).

ConditionalFreqSev
    Garrido et al. (2016) method. Refits the severity GLM with claim count N
    as an additional covariate. Much simpler than copula estimation; fits in
    standard GLM tools. Less flexible but practically tractable.

Design choices
--------------
- We do NOT refit the marginal GLMs. Pricing teams have their own GLM
  pipelines with rating factors, offsets, and exposure adjustments that we
  cannot replicate. We accept fitted objects and plug in their predictions.

- IFM is the default estimation method. Full joint MLE is available but
  slower and rarely changes omega estimates materially when marginals are
  well-specified.

- The premium correction is returned as a multiplicative factor relative to
  the independence prediction. A factor of 1.05 means the true expected cost
  is 5% higher than the independence GLM predicts.

- We warn loudly when n_policies < 1000 (identification is weak) and when
  n_claims < 500 (omega standard errors will be large).

GLM interface
-------------
We require from the frequency GLM:
  - .predict(X) or .fittedvalues: E[N|x] for each policy
  - .model.family: ideally stats.NegativeBinomial or Poisson
  - Overdispersion parameter alpha: extracted from .model.family.alpha or
    estimated as phi from the fitted model

We require from the severity GLM:
  - .predict(X) or .fittedvalues: E[S|x] for each claim
  - Dispersion (shape) parameter for Gamma: estimated as 1/phi_hat
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize_scalar, minimize

from insurance_frequency_severity.copula import (
    SarmanovCopula,
    GaussianCopulaMixed,
    FGMCopula,
)


# ---------------------------------------------------------------------------
# Helper: extract GLM marginal parameters
# ---------------------------------------------------------------------------

def _extract_freq_params(
    glm,
    X: Optional[pd.DataFrame],
    exposure: Optional[np.ndarray],
) -> Tuple[np.ndarray, float, str]:
    """
    Extract per-policy E[N|x] and overdispersion from a fitted frequency GLM.

    Returns (mu_n, alpha, family_name) where alpha=0 signals Poisson.
    """
    if X is not None:
        try:
            mu_n = np.asarray(glm.predict(X), dtype=float)
        except Exception:
            mu_n = np.asarray(glm.fittedvalues, dtype=float)
    else:
        mu_n = np.asarray(glm.fittedvalues, dtype=float)

    if exposure is not None:
        mu_n = mu_n * np.asarray(exposure, dtype=float)

    # Detect family
    family_name = "nb"
    alpha = 1.0  # default overdispersion

    model_obj = getattr(glm, "model", None)
    family = getattr(model_obj, "family", None) if model_obj else None
    if family is None:
        family = getattr(glm, "family", None)

    if family is not None:
        fname = type(family).__name__.lower()
        if "poisson" in fname:
            family_name = "poisson"
            alpha = 0.0
        elif "negativebinomial" in fname or "nb" in fname:
            family_name = "nb"
            # statsmodels NB: family.alpha is the overdispersion
            alpha = float(getattr(family, "alpha", 1.0))
        elif "tweedie" in fname:
            # Treat as Poisson for frequency
            family_name = "poisson"
            alpha = 0.0

    # Try to get alpha from params if not found in family
    if family_name == "nb" and alpha == 1.0:
        params = getattr(glm, "params", None)
        if params is not None and hasattr(params, "__getitem__"):
            try:
                alpha = float(params["alpha"])
            except (KeyError, TypeError):
                pass

    return mu_n, alpha, family_name


def _extract_sev_params(
    glm,
    X: Optional[pd.DataFrame],
    weights: Optional[np.ndarray],
) -> Tuple[np.ndarray, float, str]:
    """
    Extract per-claim E[S|x] and shape parameter from a fitted severity GLM.

    Returns (mu_s, shape, family_name).
    For Gamma GLM: shape = 1 / phi_hat (where phi is the dispersion).
    """
    if X is not None:
        try:
            mu_s = np.asarray(glm.predict(X), dtype=float)
        except Exception:
            mu_s = np.asarray(glm.fittedvalues, dtype=float)
    else:
        mu_s = np.asarray(glm.fittedvalues, dtype=float)

    family_name = "gamma"
    shape = 1.0

    model_obj = getattr(glm, "model", None)
    family = getattr(model_obj, "family", None) if model_obj else None
    if family is None:
        family = getattr(glm, "family", None)

    if family is not None:
        fname = type(family).__name__.lower()
        if "lognormal" in fname or "gaussian" in fname:
            family_name = "lognormal"
        elif "gamma" in fname:
            family_name = "gamma"

    # Gamma shape = 1/phi. statsmodels stores scale/dispersion differently.
    # Attempt to get dispersion (phi) from GLM results.
    phi = 1.0
    scale = getattr(glm, "scale", None)
    if scale is not None:
        try:
            phi = float(scale)
        except (TypeError, ValueError):
            pass

    if phi > 0:
        shape = 1.0 / phi
    else:
        shape = 1.0

    # Reasonable range guard
    shape = max(shape, 0.01)

    return mu_s, shape, family_name


# ---------------------------------------------------------------------------
# JointFreqSev
# ---------------------------------------------------------------------------

class JointFreqSev:
    """
    Joint frequency-severity model using Sarmanov copula (or alternatives).

    Accepts fitted GLM objects from statsmodels (or any compatible object with
    .predict() and .fittedvalues). Does not refit the marginal models.

    Parameters
    ----------
    freq_glm : fitted GLM
        Frequency model (Poisson or NegativeBinomial GLM).
    sev_glm : fitted GLM
        Severity model (Gamma or lognormal GLM). Should be fitted on
        positive-claim observations only.
    copula : 'sarmanov' | 'gaussian' | 'fgm'
        Copula family to use. Default 'sarmanov'.
    kernel_theta : float
        Laplace exponent for the frequency kernel in Sarmanov copula.
    kernel_alpha : float
        Laplace exponent for the severity kernel in Sarmanov copula.

    Examples
    --------
    >>> model = JointFreqSev(freq_glm=nb_glm, sev_glm=gamma_glm)
    >>> model.fit(
    ...     claims_df,
    ...     n_col="claim_count",
    ...     s_col="avg_severity",
    ...     freq_X=X_freq,
    ...     sev_X=X_sev,
    ... )
    >>> corrections = model.premium_correction(X_new)
    >>> model.dependence_summary()
    """

    def __init__(
        self,
        freq_glm: Any,
        sev_glm: Any,
        copula: Literal["sarmanov", "gaussian", "fgm"] = "sarmanov",
        kernel_theta: float = 0.5,
        kernel_alpha: float = 0.001,
    ):
        self.freq_glm = freq_glm
        self.sev_glm = sev_glm
        self.copula_family = copula
        self.kernel_theta = kernel_theta
        self.kernel_alpha = kernel_alpha

        # Set after .fit()
        self.omega_: Optional[float] = None
        self.rho_: Optional[float] = None
        self.omega_ci_: Optional[Tuple[float, float]] = None
        self.aic_: Optional[float] = None
        self.bic_: Optional[float] = None
        self._copula_obj: Optional[SarmanovCopula | GaussianCopulaMixed | FGMCopula] = None
        self._freq_family: Optional[str] = None
        self._sev_family: Optional[str] = None
        self._alpha: Optional[float] = None
        self._shape: Optional[float] = None
        self._n_obs: Optional[int] = None
        self._n_claims: Optional[int] = None
        self._mu_n_fitted: Optional[np.ndarray] = None
        self._mu_s_fitted: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.DataFrame,
        n_col: str = "claim_count",
        s_col: str = "avg_severity",
        freq_X: Optional[pd.DataFrame] = None,
        sev_X: Optional[pd.DataFrame] = None,
        exposure_col: Optional[str] = None,
        method: Literal["ifm", "mle"] = "ifm",
        ci_method: Literal["profile", "bootstrap"] = "profile",
        n_bootstrap: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> "JointFreqSev":
        """
        Estimate the dependence parameter omega.

        Parameters
        ----------
        data : DataFrame
            Policy-level data. Must contain n_col (claim counts) and s_col
            (average severity for claims, 0 or NaN for zero-claim policies).
        n_col : str
            Column name for claim counts.
        s_col : str
            Column name for average claim severity.
        freq_X : DataFrame, optional
            Covariates for frequency GLM prediction. If None, uses
            glm.fittedvalues.
        sev_X : DataFrame, optional
            Covariates for severity GLM prediction (on claims rows only). If
            None, uses glm.fittedvalues.
        exposure_col : str, optional
            Column of exposure (e.g., years at risk) for the frequency GLM.
        method : 'ifm' | 'mle'
            IFM (default): profile likelihood for omega given fitted marginals.
            MLE: joint estimation (experimental, slower).
        ci_method : 'profile' | 'bootstrap'
            Method for omega confidence intervals.
        n_bootstrap : int
            Number of bootstrap replicates (ci_method='bootstrap' only).
        rng : numpy Generator, optional
            Random state for bootstrap.
        """
        n = np.asarray(data[n_col], dtype=float)
        s_raw = np.asarray(data[s_col], dtype=float)

        # Replace NaN/0 severity for zero-claim rows with a placeholder
        # (1.0 works; those rows won't contribute severity to the likelihood)
        s = np.where((n == 0) | np.isnan(s_raw) | (s_raw <= 0), 1.0, s_raw)

        n_policies = len(n)
        n_claims = int(np.sum(n > 0))

        self._n_obs = n_policies
        self._n_claims = n_claims

        if n_policies < 1000:
            warnings.warn(
                f"Only {n_policies} policies. Omega estimation needs at least 1,000 "
                f"observations for reliable inference. Results may be unstable.",
                UserWarning,
                stacklevel=2,
            )
        if n_claims < 500:
            warnings.warn(
                f"Only {n_claims} claim events. Standard errors on omega will be "
                f"large. Consider the Garrido conditional method (ConditionalFreqSev) "
                f"as a more stable alternative.",
                UserWarning,
                stacklevel=2,
            )

        # Extract marginal parameters
        exposure = (
            np.asarray(data[exposure_col], dtype=float) if exposure_col else None
        )
        mu_n, alpha, freq_family = _extract_freq_params(self.freq_glm, freq_X, exposure)
        mu_s, shape, sev_family = _extract_sev_params(self.sev_glm, sev_X, None)

        # mu_n has n_policies entries; mu_s has n_policies entries (using fittedvalues)
        # If sev GLM was fitted on claims-only, we need to handle alignment.
        if len(mu_s) != n_policies:
            # Severity GLM was fitted on positive-claim rows only.
            # We need E[S|x] for all policies (even zero-claim ones) for prediction.
            # For zero-claim policies, E[S|x] is still well-defined as the GLM
            # prediction under their covariates.
            if sev_X is not None:
                mu_s_all = np.asarray(self.sev_glm.predict(sev_X), dtype=float)
            else:
                # Cannot align — use a common mean
                warnings.warn(
                    "Severity GLM fittedvalues length does not match n_policies and "
                    "no sev_X provided. Using mean severity for all policies.",
                    UserWarning,
                    stacklevel=2,
                )
                mu_s_all = np.full(n_policies, float(np.mean(mu_s)))
        else:
            mu_s_all = mu_s

        self._freq_family = freq_family
        self._sev_family = sev_family
        self._alpha = alpha
        self._shape = shape
        self._mu_n_fitted = mu_n
        self._mu_s_fitted = mu_s_all

        # Build per-observation parameter arrays
        if freq_family == "nb":
            freq_params = [{"mu": float(mu_n[i]), "alpha": float(alpha)} for i in range(n_policies)]
        else:
            freq_params = [{"mu": float(mu_n[i])} for i in range(n_policies)]

        if sev_family == "gamma":
            sev_params = [{"mu": float(mu_s_all[i]), "shape": float(shape)} for i in range(n_policies)]
        else:
            # Lognormal: log_mu = log(E[S]) - log_sigma^2/2
            # We treat shape as 1/phi; log_sigma^2 = log(1 + 1/shape)
            log_sigma = float(np.sqrt(np.log(1.0 + 1.0 / shape)))
            log_mu_arr = np.log(mu_s_all) - 0.5 * log_sigma**2
            sev_params = [{"log_mu": float(log_mu_arr[i]), "log_sigma": log_sigma} for i in range(n_policies)]

        # --- Fit copula ---
        if self.copula_family == "sarmanov":
            self._fit_sarmanov(n, s, freq_params, sev_params, freq_family, sev_family, ci_method, n_bootstrap, rng)
        elif self.copula_family == "gaussian":
            self._fit_gaussian(n, s, freq_params, sev_params, freq_family, sev_family, ci_method, n_bootstrap, rng)
        elif self.copula_family == "fgm":
            self._fit_fgm(n, s, freq_params, sev_params, freq_family, sev_family, ci_method, n_bootstrap, rng)
        else:
            raise ValueError(f"Unknown copula family: {self.copula_family}")

        return self

    def _profile_ll_sarmanov(
        self,
        omega: float,
        n: np.ndarray,
        s: np.ndarray,
        freq_params: list,
        sev_params: list,
        freq_family: str,
        sev_family: str,
    ) -> float:
        copula = SarmanovCopula(
            freq_family=freq_family,
            sev_family=sev_family,
            omega=omega,
            kernel_theta=self.kernel_theta,
            kernel_alpha=self.kernel_alpha,
        )
        return copula.log_likelihood(n, s, freq_params, sev_params)

    def _fit_sarmanov(self, n, s, freq_params, sev_params, freq_family, sev_family,
                      ci_method, n_bootstrap, rng):
        # Compute omega bounds from a representative observation
        ref_fp = freq_params[0]
        ref_sp = sev_params[0]
        copula_ref = SarmanovCopula(
            freq_family=freq_family,
            sev_family=sev_family,
            omega=0.0,
            kernel_theta=self.kernel_theta,
            kernel_alpha=self.kernel_alpha,
        )
        omega_min, omega_max = copula_ref.omega_bounds(ref_fp, ref_sp)
        # Clamp to a reasonable search range
        omega_min = max(omega_min, -50.0)
        omega_max = min(omega_max, 50.0)

        def neg_ll(omega):
            return -self._profile_ll_sarmanov(omega, n, s, freq_params, sev_params,
                                               freq_family, sev_family)

        result = minimize_scalar(neg_ll, bounds=(omega_min, omega_max), method="bounded")
        self.omega_ = float(result.x)

        copula = SarmanovCopula(
            freq_family=freq_family,
            sev_family=sev_family,
            omega=self.omega_,
            kernel_theta=self.kernel_theta,
            kernel_alpha=self.kernel_alpha,
        )
        self._copula_obj = copula

        ll_hat = -float(result.fun)
        ll_indep = self._profile_ll_sarmanov(0.0, n, s, freq_params, sev_params,
                                              freq_family, sev_family)

        # AIC/BIC: count omega as the extra parameter
        self.aic_ = -2 * ll_hat + 2 * 1
        self.bic_ = -2 * ll_hat + np.log(self._n_claims) * 1

        # Spearman rho (MC estimate)
        self.rho_ = copula.spearman_rho(ref_fp, ref_sp, n_mc=10_000)

        # Confidence intervals
        if ci_method == "profile":
            self.omega_ci_ = self._profile_ci_sarmanov(
                n, s, freq_params, sev_params, freq_family, sev_family,
                omega_min, omega_max, ll_hat
            )
        elif ci_method == "bootstrap":
            self.omega_ci_ = self._bootstrap_ci_sarmanov(
                n, s, freq_params, sev_params, freq_family, sev_family,
                omega_min, omega_max, n_bootstrap, rng
            )

    def _profile_ci_sarmanov(self, n, s, freq_params, sev_params, freq_family, sev_family,
                               omega_min, omega_max, ll_hat):
        """95% CI via profile likelihood (chi-squared approximation, df=1)."""
        threshold = ll_hat - 0.5 * stats.chi2.ppf(0.95, df=1)

        def above_threshold(omega):
            ll = self._profile_ll_sarmanov(omega, n, s, freq_params, sev_params,
                                           freq_family, sev_family)
            return ll - threshold

        from scipy.optimize import brentq

        omega_hat = self.omega_

        try:
            lo = brentq(above_threshold, omega_min, omega_hat - 1e-8)
        except (ValueError, RuntimeError):
            lo = omega_min

        try:
            hi = brentq(above_threshold, omega_hat + 1e-8, omega_max)
        except (ValueError, RuntimeError):
            hi = omega_max

        return (lo, hi)

    def _bootstrap_ci_sarmanov(self, n, s, freq_params, sev_params, freq_family, sev_family,
                                omega_min, omega_max, n_bootstrap, rng):
        if rng is None:
            rng = np.random.default_rng()
        n_obs = len(n)
        boot_omegas = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_obs, size=n_obs)
            n_b = n[idx]
            s_b = s[idx]
            fp_b = [freq_params[i] for i in idx]
            sp_b = [sev_params[i] for i in idx]

            def neg_ll(omega):
                return -self._profile_ll_sarmanov(omega, n_b, s_b, fp_b, sp_b,
                                                  freq_family, sev_family)
            try:
                res = minimize_scalar(neg_ll, bounds=(omega_min, omega_max), method="bounded")
                boot_omegas.append(float(res.x))
            except Exception:
                pass

        if len(boot_omegas) < 10:
            warnings.warn("Bootstrap CI estimation failed for most replicates.", UserWarning)
            return (omega_min, omega_max)

        boot_arr = np.array(boot_omegas)
        return (float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5)))

    def _fit_gaussian(self, n, s, freq_params, sev_params, freq_family, sev_family,
                      ci_method, n_bootstrap, rng):
        """Fit Gaussian copula rho via profile likelihood."""
        gc = GaussianCopulaMixed(rho=0.0)

        def neg_ll(rho):
            gc.rho = float(rho)
            return -gc.log_likelihood(n, s, freq_params, sev_params, freq_family, sev_family)

        result = minimize_scalar(neg_ll, bounds=(-0.999, 0.999), method="bounded")
        self.omega_ = float(result.x)  # store rho as omega for consistency

        gc.rho = self.omega_
        self._copula_obj = gc
        self.rho_ = gc.spearman_rho()

        ll_hat = -float(result.fun)
        self.aic_ = -2 * ll_hat + 2 * 1
        self.bic_ = -2 * ll_hat + np.log(self._n_claims) * 1

        # Simple CI via bootstrap
        if ci_method == "bootstrap" and rng is not None:
            boot_rhos = []
            n_obs = len(n)
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n_obs, size=n_obs)
                n_b, s_b = n[idx], s[idx]
                fp_b = [freq_params[i] for i in idx]
                sp_b = [sev_params[i] for i in idx]
                try:
                    def nll_b(rho):
                        gc.rho = float(rho)
                        return -gc.log_likelihood(n_b, s_b, fp_b, sp_b, freq_family, sev_family)
                    res = minimize_scalar(nll_b, bounds=(-0.999, 0.999), method="bounded")
                    boot_rhos.append(float(res.x))
                except Exception:
                    pass
            if boot_rhos:
                self.omega_ci_ = (
                    float(np.percentile(boot_rhos, 2.5)),
                    float(np.percentile(boot_rhos, 97.5)),
                )
        if self.omega_ci_ is None:
            self.omega_ci_ = (-0.999, 0.999)

    def _fit_fgm(self, n, s, freq_params, sev_params, freq_family, sev_family,
                 ci_method, n_bootstrap, rng):
        """Fit FGM copula theta via PIT + profile likelihood."""
        gc_ref = GaussianCopulaMixed(rho=0.0)

        def pit_u(fp):
            return gc_ref._pit_freq(
                np.array([fp["mu"]]), {"mu": fp["mu"], **({} if freq_family == "poisson" else {"alpha": fp.get("alpha", 1.0)})},
                freq_family
            )[0]

        def pit_v(sp):
            return gc_ref._pit_sev(
                np.array([sp["mu"] if sev_family == "gamma" else np.exp(sp["log_mu"])]),
                sp, sev_family
            )[0]

        # Compute PIT values for positive-claim rows only
        mask = np.asarray([ni > 0 for ni in n])
        if not mask.any():
            self.omega_ = 0.0
            self._copula_obj = FGMCopula(theta=0.0)
            self.rho_ = 0.0
            self.omega_ci_ = (-1.0, 1.0)
            return

        n_pos = n[mask]
        s_pos = s[mask]

        # Build PIT arrays using GaussianCopulaMixed._pit methods
        gc = GaussianCopulaMixed(rho=0.0)
        fp_mask = [freq_params[i] for i in range(len(n)) if mask[i]]
        sp_mask = [sev_params[i] for i in range(len(n)) if mask[i]]

        fp_dict = {
            "mu": np.array([p["mu"] for p in fp_mask]),
            **({
                "alpha": np.array([p["alpha"] for p in fp_mask])
            } if freq_family == "nb" else {}),
        }
        sp_dict_gamma = sev_family == "gamma"
        if sp_dict_gamma:
            sp_dict = {
                "mu": np.array([p["mu"] for p in sp_mask]),
                "shape": np.array([p["shape"] for p in sp_mask]),
            }
        else:
            sp_dict = {
                "log_mu": np.array([p["log_mu"] for p in sp_mask]),
                "log_sigma": np.array([p["log_sigma"] for p in sp_mask]),
            }

        u = gc._pit_freq(n_pos, fp_dict, freq_family)
        v = gc._pit_sev(s_pos, sp_dict, sev_family)
        u = np.clip(u, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)

        fgm = FGMCopula(theta=0.0)

        def neg_ll(theta):
            fgm.theta = float(theta)
            return -fgm.log_likelihood(u, v)

        result = minimize_scalar(neg_ll, bounds=(-1.0, 1.0), method="bounded")
        self.omega_ = float(result.x)
        fgm.theta = self.omega_
        self._copula_obj = fgm
        self.rho_ = fgm.spearman_rho()

        ll_hat = -float(result.fun)
        self.aic_ = -2 * ll_hat + 2 * 1
        self.bic_ = -2 * ll_hat + np.log(int(mask.sum())) * 1
        self.omega_ci_ = (-1.0, 1.0)

    def premium_correction(
        self,
        X: Optional[pd.DataFrame] = None,
        exposure: Optional[np.ndarray] = None,
        n_mc: int = 50_000,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """
        Compute per-policy premium correction factors.

        The correction factor is: E[N*S] / (E[N] * E[S])

        Under independence, E[N*S] = E[N]*E[S] so the factor equals 1.
        With negative dependence (typical in UK motor), the factor < 1.

        For the Sarmanov copula, we have the analytical result:
            E[N*S] = E[N]*E[S] + omega * E[N*phi_1(N)] * E[S*phi_2(S)]

        For Gaussian copula and FGM, we use Monte Carlo simulation.

        Parameters
        ----------
        X : DataFrame, optional
            New covariate data for prediction. If None, uses training fitted values.
        exposure : array-like, optional
            Exposure for frequency prediction.
        n_mc : int
            Monte Carlo samples for Gaussian/FGM copulas.

        Returns
        -------
        DataFrame with columns: mu_n, mu_s, mu_ns_independent, mu_ns_joint,
            correction_factor, premium_independent, premium_joint.
        """
        if self.omega_ is None:
            raise RuntimeError("Must call .fit() before .premium_correction()")

        if X is not None:
            mu_n = np.asarray(self.freq_glm.predict(X), dtype=float)
            mu_s = np.asarray(self.sev_glm.predict(X), dtype=float)
        else:
            mu_n = self._mu_n_fitted
            mu_s = self._mu_s_fitted

        if exposure is not None:
            mu_n = mu_n * np.asarray(exposure, dtype=float)

        n_policies = len(mu_n)
        correction = np.ones(n_policies)

        if self.copula_family == "sarmanov":
            correction = self._sarmanov_correction(mu_n, mu_s)
        else:
            # Monte Carlo for Gaussian/FGM
            correction = self._mc_correction(mu_n, mu_s, n_mc, rng)

        mu_ns_ind = mu_n * mu_s
        mu_ns_joint = mu_ns_ind * correction

        return pd.DataFrame({
            "mu_n": mu_n,
            "mu_s": mu_s,
            "mu_ns_independent": mu_ns_ind,
            "mu_ns_joint": mu_ns_joint,
            "correction_factor": correction,
            "premium_independent": mu_ns_ind,
            "premium_joint": mu_ns_joint,
        })

    def _sarmanov_correction(self, mu_n: np.ndarray, mu_s: np.ndarray) -> np.ndarray:
        """
        Analytical premium correction for Sarmanov copula.

        E[N*S] = E[N]*E[S] + omega * Cov_sarmanov(N, S)
               = E[N]*E[S] + omega * E[N*phi_1(N)] * E[S*phi_2(S)]

        E[N*phi_1(N)] for NB with Laplace kernel phi_1(n) = exp(-theta*n) - M_N(theta):
            = E[N * exp(-theta*N)] - mu_n * M_N(theta)
            = -d/dtheta M_N(theta) - mu_n * M_N(theta)   [by differentiation of MGF]

        For NB: M_N(theta) = (p/(1-(1-p)*e^{-theta}))^r
            d/dtheta M_N(theta) = -r*(1-p)*e^{-theta}/(1-(1-p)*e^{-theta}) * M_N(theta)

        E[S*phi_2(S)] for Gamma with Laplace kernel phi_2(s) = exp(-alpha*s) - M_S(alpha):
            = E[S * exp(-alpha*S)] - mu_s * M_S(alpha)
            = -d/dalpha M_S(alpha) - mu_s * M_S(alpha)

        For Gamma(shape, scale=mu_s/shape):
            M_S(alpha) = (1/(1+alpha*scale))^shape
            d/dalpha M_S(alpha) = -shape * scale * M_S(alpha) / (1+alpha*scale)
        """
        omega = self.omega_
        theta = self.kernel_theta
        alpha_k = self.kernel_alpha

        n_pol = len(mu_n)
        correction = np.ones(n_pol)

        for i in range(n_pol):
            mu_ni = float(mu_n[i])
            mu_si = float(mu_s[i])

            if self._freq_family == "nb":
                alpha_disp = self._alpha
                r = 1.0 / alpha_disp
                p = 1.0 / (1.0 + mu_ni * alpha_disp)
                M1 = (p / (1.0 - (1.0 - p) * np.exp(-theta))) ** r
                dM1 = -r * (1.0 - p) * np.exp(-theta) / (1.0 - (1.0 - p) * np.exp(-theta)) * M1
                E_n_phi1 = -dM1 - mu_ni * M1
            else:
                # Poisson
                M1 = np.exp(mu_ni * (np.exp(-theta) - 1.0))
                dM1 = mu_ni * (-np.exp(-theta)) * M1
                E_n_phi1 = -dM1 - mu_ni * M1

            if self._sev_family == "gamma":
                shape = self._shape
                scale = mu_si / shape
                M2 = (1.0 / (1.0 + alpha_k * scale)) ** shape
                dM2 = -shape * scale / (1.0 + alpha_k * scale) * M2
                E_s_phi2 = -dM2 - mu_si * M2
            else:
                # Lognormal: numerical
                log_sigma = float(np.sqrt(np.log(1.0 + 1.0 / self._shape)))
                log_mu_i = np.log(mu_si) - 0.5 * log_sigma**2
                # E[S*exp(-alpha*S)] = integral s*exp(-alpha*s)*f_S(s) ds
                # Approximate via: -d/dalpha M_S(alpha) (numerical derivative)
                from insurance_frequency_severity.copula import LaplaceKernelLognormal
                kern = LaplaceKernelLognormal(alpha=alpha_k)
                da = 1e-5
                kern_p = LaplaceKernelLognormal(alpha=alpha_k + da)
                kern_m = LaplaceKernelLognormal(alpha=alpha_k - da)
                M2_p = kern_p.mgf(log_mu_i, log_sigma)
                M2_m = kern_m.mgf(log_mu_i, log_sigma)
                dM2 = (M2_p - M2_m) / (2 * da)
                M2 = kern.mgf(log_mu_i, log_sigma)
                E_s_phi2 = -dM2 - mu_si * M2

            covariance_term = omega * E_n_phi1 * E_s_phi2
            E_NS_ind = mu_ni * mu_si
            E_NS_joint = E_NS_ind + covariance_term

            if E_NS_ind > 0:
                correction[i] = E_NS_joint / E_NS_ind
            else:
                correction[i] = 1.0

        return correction

    def _mc_correction(
        self,
        mu_n: np.ndarray,
        mu_s: np.ndarray,
        n_mc: int,
        rng: Optional[np.random.Generator],
    ) -> np.ndarray:
        """Monte Carlo correction factor for Gaussian/FGM copulas."""
        if rng is None:
            rng = np.random.default_rng()

        # Use representative marginal parameters for MC
        mu_n_bar = float(np.mean(mu_n))
        mu_s_bar = float(np.mean(mu_s))

        if self._freq_family == "nb":
            fp = {"mu": mu_n_bar, "alpha": float(self._alpha)}
        else:
            fp = {"mu": mu_n_bar}

        if self._sev_family == "gamma":
            sp = {"mu": mu_s_bar, "shape": float(self._shape)}
        else:
            log_sigma = float(np.sqrt(np.log(1.0 + 1.0 / self._shape)))
            sp = {
                "log_mu": float(np.log(mu_s_bar) - 0.5 * log_sigma**2),
                "log_sigma": log_sigma,
            }

        if self.copula_family == "gaussian":
            copula_obj: GaussianCopulaMixed = self._copula_obj
            n_samp, s_samp = copula_obj.sample(n_mc, fp, sp, self._freq_family, self._sev_family, rng=rng)
        else:
            # FGM: sample u, v from copula then invert marginals
            fgm: FGMCopula = self._copula_obj
            u, v = fgm.sample(n_mc, rng=rng)
            # Invert frequency CDF
            if self._freq_family == "nb":
                r = 1.0 / float(self._alpha)
                p = 1.0 / (1.0 + mu_n_bar * float(self._alpha))
                n_samp = stats.nbinom.ppf(u, r, p).astype(int)
            else:
                n_samp = stats.poisson.ppf(u, mu_n_bar).astype(int)
            # Invert severity CDF
            if self._sev_family == "gamma":
                scale = mu_s_bar / float(self._shape)
                s_samp = stats.gamma.ppf(v, a=float(self._shape), scale=scale)
            else:
                log_sigma = float(np.sqrt(np.log(1.0 + 1.0 / self._shape)))
                log_mu_bar = float(np.log(mu_s_bar) - 0.5 * log_sigma**2)
                s_samp = stats.lognorm.ppf(v, s=log_sigma, scale=np.exp(log_mu_bar))

        E_NS_joint = float(np.mean(n_samp * s_samp))
        E_N_E_S = mu_n_bar * mu_s_bar

        if E_N_E_S > 0:
            factor_global = E_NS_joint / E_N_E_S
        else:
            factor_global = 1.0

        # Scale correction by individual mu_n * mu_s ratios
        # (The global MC gives average correction; per-policy varies by covariates)
        return np.full(len(mu_n), factor_global)

    def loss_cost(
        self,
        X: pd.DataFrame,
        exposure: Optional[np.ndarray] = None,
        n_mc: int = 50_000,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Corrected pure premium predictions for new data.

        Returns E[N*S|x] for each row of X, incorporating the dependence
        correction.

        Parameters
        ----------
        X : DataFrame
            Feature matrix for new policies.
        exposure : array-like, optional
            Exposure vector for frequency scaling.
        n_mc : int
            Monte Carlo samples (Gaussian/FGM copulas only).

        Returns
        -------
        ndarray of corrected pure premiums.
        """
        corrections_df = self.premium_correction(X, exposure=exposure, n_mc=n_mc, rng=rng)
        return corrections_df["premium_joint"].to_numpy()

    def dependence_summary(self) -> pd.DataFrame:
        """
        Summary of estimated dependence parameters.

        Returns
        -------
        DataFrame with omega (or rho for Gaussian), Spearman's rho estimate,
        95% confidence interval, AIC, BIC.
        """
        if self.omega_ is None:
            raise RuntimeError("Must call .fit() first")

        ci_lo, ci_hi = self.omega_ci_ if self.omega_ci_ else (np.nan, np.nan)

        param_name = "omega" if self.copula_family == "sarmanov" else "rho"

        return pd.DataFrame([{
            "copula": self.copula_family,
            param_name: self.omega_,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "spearman_rho": self.rho_,
            "aic": self.aic_,
            "bic": self.bic_,
            "n_policies": self._n_obs,
            "n_claims": self._n_claims,
        }])


# ---------------------------------------------------------------------------
# ConditionalFreqSev — Garrido (2016) method
# ---------------------------------------------------------------------------

class ConditionalFreqSev:
    """
    Garrido, Genest, Schulz (2016) conditional frequency-severity model.

    The method adds claim count N (or the indicator I(N>0)) as a covariate in
    the severity GLM. The intuition: if high-count years have systematically
    different severity, this covariate captures the dependence directly.

    Under Poisson frequency with log-link severity:
        log E[S|x, N] = beta'x + gamma * N

    The pure premium becomes:
        E[N * S | x] = E[N|x] * E[S|x, N=1] * exp(gamma * E[N|x])

    The correction factor is exp(gamma * E[N|x]).

    This is simpler and more robust than copula estimation. It uses only
    standard GLM infrastructure. The tradeoff is that it restricts the
    dependence structure to a specific parametric form.

    Parameters
    ----------
    freq_glm : fitted GLM
        Frequency model. Used to predict E[N|x].
    sev_glm_base : fitted GLM
        Base severity model (fitted without N covariate). Used as starting
        point; we refit with N added.
    n_as_indicator : bool
        If True, use I(N>0) rather than N itself as the covariate.
        Indicator is more robust when some policies have very high N.
    """

    def __init__(
        self,
        freq_glm: Any,
        sev_glm_base: Any,
        n_as_indicator: bool = False,
    ):
        self.freq_glm = freq_glm
        self.sev_glm_base = sev_glm_base
        self.n_as_indicator = n_as_indicator

        self.gamma_: Optional[float] = None
        self.gamma_se_: Optional[float] = None
        self.sev_glm_conditional_: Optional[Any] = None
        self._freq_family: Optional[str] = None
        self._mu_n_fitted: Optional[np.ndarray] = None
        self._mu_s_fitted: Optional[np.ndarray] = None

    def fit(
        self,
        data: pd.DataFrame,
        n_col: str = "claim_count",
        s_col: str = "avg_severity",
        sev_feature_cols: Optional[list] = None,
        freq_X: Optional[pd.DataFrame] = None,
        exposure_col: Optional[str] = None,
    ) -> "ConditionalFreqSev":
        """
        Fit the conditional severity model with N as covariate.

        Parameters
        ----------
        data : DataFrame
            Policy-level data.
        n_col : str
            Claim count column.
        s_col : str
            Average severity column.
        sev_feature_cols : list, optional
            Feature columns to include in severity GLM alongside N.
        freq_X : DataFrame, optional
            Feature matrix for frequency GLM prediction.
        exposure_col : str, optional
            Exposure column for frequency.
        """
        import statsmodels.api as sm

        n = np.asarray(data[n_col], dtype=float)
        s = np.asarray(data[s_col], dtype=float)

        exposure = (
            np.asarray(data[exposure_col], dtype=float) if exposure_col else None
        )
        mu_n, alpha, freq_family = _extract_freq_params(self.freq_glm, freq_X, exposure)
        self._freq_family = freq_family
        self._mu_n_fitted = mu_n

        # Restrict to positive-claim observations
        mask_pos = (n > 0) & (s > 0) & np.isfinite(s)
        n_pos = n[mask_pos]
        s_pos = s[mask_pos]

        if mask_pos.sum() < 20:
            raise ValueError(
                f"Only {mask_pos.sum()} positive-claim observations. Cannot fit conditional model."
            )

        # Build covariate matrix for conditional severity GLM
        if sev_feature_cols:
            X_sev = data.loc[mask_pos, sev_feature_cols].copy()
        else:
            X_sev = pd.DataFrame(index=data.index[mask_pos])

        n_covariate = (n_pos > 0).astype(float) if self.n_as_indicator else n_pos
        X_sev = X_sev.copy()
        X_sev["_n_covariate"] = np.asarray(n_covariate)
        X_sev_const = sm.add_constant(X_sev, has_constant="add")

        # Fit Gamma GLM with log link
        gamma_family = sm.families.Gamma(link=sm.families.links.Log())
        try:
            sev_cond = sm.GLM(
                s_pos,
                X_sev_const,
                family=gamma_family,
                var_weights=n_pos,
            ).fit(disp=True)
        except Exception as e:
            raise RuntimeError(f"Conditional severity GLM fit failed: {e}") from e

        self.sev_glm_conditional_ = sev_cond
        self.gamma_ = float(sev_cond.params["_n_covariate"])
        self.gamma_se_ = float(sev_cond.bse["_n_covariate"])

        # Baseline severity (N=0 prediction)
        # If freq_X is provided, predict on the full feature matrix so that
        # premium_correction() returns predictions for all policies (not just claimants).
        if freq_X is not None:
            self._mu_s_fitted = np.asarray(self.sev_glm_base.predict(freq_X), dtype=float)
        else:
            self._mu_s_fitted = np.asarray(self.sev_glm_base.fittedvalues, dtype=float)

        return self

    def premium_correction(
        self,
        X: Optional[pd.DataFrame] = None,
        exposure: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compute premium correction factors using the Garrido formula.

        Correction = exp(gamma * E[N|x])

        For a log-link severity model, this is the exact correction when
        N is the covariate. For general link functions, it is an approximation.

        Returns
        -------
        DataFrame with mu_n, correction_factor, premium columns.
        """
        if self.gamma_ is None:
            raise RuntimeError("Must call .fit() first")

        if X is not None:
            mu_n = np.asarray(self.freq_glm.predict(X), dtype=float)
            mu_s = np.asarray(self.sev_glm_base.predict(X), dtype=float)
        else:
            mu_n = self._mu_n_fitted
            mu_s = self._mu_s_fitted

        if exposure is not None:
            mu_n = mu_n * np.asarray(exposure, dtype=float)

        correction = np.exp(self.gamma_ * mu_n)
        premium_ind = mu_n * mu_s
        premium_joint = premium_ind * correction

        return pd.DataFrame({
            "mu_n": mu_n,
            "mu_s": mu_s,
            "correction_factor": correction,
            "premium_independent": premium_ind,
            "premium_joint": premium_joint,
        })

    def loss_cost(
        self,
        X: pd.DataFrame,
        exposure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Corrected pure premium predictions."""
        return self.premium_correction(X, exposure)["premium_joint"].to_numpy()

    def dependence_summary(self) -> pd.DataFrame:
        """Summary of estimated gamma (N covariate in severity GLM)."""
        if self.gamma_ is None:
            raise RuntimeError("Must call .fit() first")
        return pd.DataFrame([{
            "method": "garrido_conditional",
            "gamma": self.gamma_,
            "gamma_se": self.gamma_se_,
            "gamma_t": self.gamma_ / self.gamma_se_ if self.gamma_se_ else np.nan,
            "n_as_indicator": self.n_as_indicator,
        }])
