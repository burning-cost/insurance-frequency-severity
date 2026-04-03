"""
copula.py — Copula classes for mixed discrete-continuous frequency-severity modelling.

Three families are implemented:

1. SarmanovCopula  — Primary choice. Handles discrete (count) frequency margin
   without the probability integral transform (PIT) identifiability problem that
   afflicts standard copulas with discrete margins. Sarmanov copulas are
   distributions in their own right; they do not require a unique copula
   representation of the joint law.

2. GaussianCopulaMixed — Standard Gaussian copula with PIT approximation for the
   discrete margin. Familiar to practitioners; appropriate when the PIT
   approximation error is negligible (typically when frequency is not highly
   concentrated on zero).

3. FGMCopula — Farlie-Gumbel-Morgenstern baseline. Closed-form moments,
   analytically clean, but limited to Spearman rho in [-1/3, 1/3]. Useful for
   confirming that dependence is weak enough not to matter.

Sarmanov theory
---------------
The bivariate Sarmanov distribution has joint density (or PMF x PDF):

    f(n, s) = f_N(n) * f_S(s) * [1 + omega * phi_1(n) * phi_2(s)]

where phi_i are bounded kernel functions satisfying E[phi_i(X_i)] = 0 under the
respective marginal. The non-negativity constraint is:

    1 + omega * phi_1(n) * phi_2(s) >= 0  for all (n, s) in support

which bounds omega:  -1/B <= omega <= 1/B  where B = sup |phi_1 * phi_2|.

Kernel choices
--------------
For NB/Poisson frequency: phi_1(n) = exp(-theta * n) - M_N(theta)
For Gamma/Lognormal severity: phi_2(s) = exp(-alpha * s) - M_S(alpha)

where M_X(t) = E[exp(-tX)] is the moment generating function (Laplace transform
evaluated at t). These are the exponential (Laplace) kernels. Polynomial kernels
phi_1(n) = n - E[N] and phi_2(s) = s - E[S] are simpler but have larger omega
bounds that may be harder to satisfy pointwise.

References
----------
Vernic, Bolance, Alemany (2022). IME 102, 111-125.
Lee, Shi (2019). IME 87, 115-129.
Blier-Wong (2026). arXiv:2601.09016.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize_scalar
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Kernel base class and standard kernels
# ---------------------------------------------------------------------------

class Kernel:
    """Abstract kernel phi(x) satisfying E[phi(X)] = 0 under marginal distribution."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def expected_value(self, dist_params: dict) -> float:
        """E[phi(X)] — should equal 0 for a valid kernel, verified numerically."""
        raise NotImplementedError

    def sup_abs(self, dist_params: dict) -> float:
        """Supremum of |phi(x)| over the support."""
        raise NotImplementedError


class LaplaceKernelNB(Kernel):
    """
    Exponential (Laplace) kernel for Negative Binomial frequency.

    phi(n) = exp(-theta * n) - M_N(theta)

    where M_N(theta) = E[exp(-theta * N)] is the NB Laplace transform.

    For NB(mu, alpha) with PMF P(N=n) = C(n+r-1, n) * p^r * (1-p)^n:
      - r = 1/alpha (overdispersion parameterisation)
      - p = 1/(1 + mu * alpha)
      - M_N(theta) = (p / (1 - (1-p)*exp(-theta)))^r
    """

    def __init__(self, theta: float = 0.5):
        if theta <= 0:
            raise ValueError("theta must be positive for Laplace kernel")
        self.theta = theta

    def __call__(self, n: np.ndarray) -> np.ndarray:
        n = np.asarray(n, dtype=float)
        return np.exp(-self.theta * n)  # raw; subtract M_N below in copula

    def mgf(self, mu: float, alpha: float) -> float:
        """NB Laplace transform E[exp(-theta*N)] at self.theta."""
        r = 1.0 / alpha
        p = 1.0 / (1.0 + mu * alpha)
        # M_N(theta) = (p / (1 - (1-p)*exp(-theta)))^r
        denom = 1.0 - (1.0 - p) * np.exp(-self.theta)
        if np.any(denom <= 0):
            raise ValueError(
                "MGF domain condition violated for NB kernel: "
                "1 - (1-p)*exp(-theta) <= 0. theta is too small for these parameters."
            )
        val = (p / denom) ** r
        return np.asarray(val, dtype=float)

    def centred(self, n: np.ndarray, mu: float, alpha: float) -> np.ndarray:
        """phi(n) = exp(-theta*n) - M_N(theta). Zero-mean under NB(mu, alpha)."""
        return self(n) - self.mgf(mu, alpha)

    def sup_abs(self, mu: float, alpha: float) -> float:
        """
        sup |phi(n)|. For Laplace kernel:
        - phi(0) = 1 - M_N(theta) > 0 (since exp(0)=1 > M_N(theta) for theta>0)
        - phi(n) -> -M_N(theta) as n -> inf
        So sup = max(phi(0), M_N(theta)) = max(1 - M_N(theta), M_N(theta))
        """
        m = float(np.asarray(self.mgf(mu, alpha)).flat[0])
        return float(max(1.0 - m, m))


class LaplaceKernelPoisson(Kernel):
    """
    Exponential (Laplace) kernel for Poisson frequency.

    phi(n) = exp(-theta * n) - M_N(theta)
    M_N(theta) = exp(mu * (exp(-theta) - 1))
    """

    def __init__(self, theta: float = 0.5):
        if theta <= 0:
            raise ValueError("theta must be positive for Laplace kernel")
        self.theta = theta

    def __call__(self, n: np.ndarray) -> np.ndarray:
        return np.exp(-self.theta * np.asarray(n, dtype=float))

    def mgf(self, mu: float) -> float:
        """Poisson Laplace transform E[exp(-theta*N)]."""
        return np.asarray(np.exp(mu * (np.exp(-self.theta) - 1.0)), dtype=float)

    def centred(self, n: np.ndarray, mu: float) -> np.ndarray:
        return self(n) - self.mgf(mu)

    def sup_abs(self, mu: float) -> float:
        m = float(np.asarray(self.mgf(mu)).flat[0])
        return float(max(1.0 - m, m))


class LaplaceKernelGamma(Kernel):
    """
    Exponential (Laplace) kernel for Gamma severity.

    phi(s) = exp(-alpha * s) - M_S(alpha)

    For Gamma(shape=k, scale=beta): M_S(alpha) = (1/(1 + alpha*beta))^k
    In GLM parameterisation: E[S] = mu_s, shape parameter k (dispersion phi=1/k).
    So beta = mu_s / k.
    """

    def __init__(self, alpha: float = 0.01):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return np.exp(-self.alpha * np.asarray(s, dtype=float))

    def mgf(self, mu_s: float, shape: float) -> float:
        """Gamma Laplace transform E[exp(-alpha*S)]."""
        beta = mu_s / shape
        # M_S(alpha) = (1 / (1 + alpha*beta))^shape — valid only for alpha < 1/beta
        denom = 1.0 + self.alpha * beta
        if np.any(denom <= 0):
            raise ValueError("alpha too large: alpha * beta >= 1, MGF undefined")
        return np.asarray(denom ** (-shape), dtype=float)

    def centred(self, s: np.ndarray, mu_s: float, shape: float) -> np.ndarray:
        return self(s) - self.mgf(mu_s, shape)

    def sup_abs(self, mu_s: float, shape: float) -> float:
        m = float(np.asarray(self.mgf(mu_s, shape)).flat[0])
        # phi(s) ranges from (1 - m) at s=0 down to (-m) as s->inf
        return float(max(1.0 - m, m))


class LaplaceKernelLognormal(Kernel):
    """
    Exponential (Laplace) kernel for Lognormal severity.

    phi(s) = exp(-alpha*s) - M_S(alpha)

    M_S(alpha) = E[exp(-alpha*S)] for Lognormal has no closed form,
    so we approximate via moment expansion or numerical integration.
    For small alpha: M_S(alpha) ≈ 1 - alpha*mu_s + alpha^2*(mu_s^2 + sigma_s^2)/2
    We use numerical quadrature.
    """

    def __init__(self, alpha: float = 0.001):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return np.exp(-self.alpha * np.asarray(s, dtype=float))

    def mgf(self, log_mu: float, log_sigma: float) -> float:
        """
        Numerical E[exp(-alpha*S)] for S ~ Lognormal(log_mu, log_sigma).
        Uses Gauss-Laguerre quadrature via scipy.

        log_mu and log_sigma must be scalars — use centred() for array inputs.
        """
        from scipy.integrate import quad
        # Ensure scalar inputs: quad requires the integrand to return a scalar
        _lmu = float(np.asarray(log_mu).flat[0])
        _lsig = float(np.asarray(log_sigma).flat[0])
        alpha = self.alpha
        # Inline pdf to guarantee scalar return from the integrand
        result, _ = quad(
            lambda s_val: (
                float(np.exp(-alpha * s_val))
                * float(stats.lognorm.pdf(s_val, s=_lsig, scale=np.exp(_lmu)))
            ),
            0, np.inf,
            limit=200,
        )
        return float(result)

    def centred(self, s: np.ndarray, log_mu, log_sigma) -> np.ndarray:
        """phi(s) = exp(-alpha*s) - E[exp(-alpha*S)], element-wise when parameters are arrays."""
        s_arr = self(s)  # exp(-alpha * s), shape (n,)
        lmu = np.asarray(log_mu, dtype=float)
        lsig = np.asarray(log_sigma, dtype=float)
        if lmu.ndim == 0 and lsig.ndim == 0:
            # Scalar path: single mgf call
            return s_arr - self.mgf(float(lmu), float(lsig))
        # Array path: per-observation parameters — compute mgf element-wise
        lmu_flat = np.broadcast_to(lmu, s_arr.shape).ravel()
        lsig_flat = np.broadcast_to(lsig, s_arr.shape).ravel()
        mgf_vals = np.array([self.mgf(float(m), float(sg)) for m, sg in zip(lmu_flat, lsig_flat)])
        return s_arr - mgf_vals.reshape(s_arr.shape)

    def sup_abs(self, log_mu: float, log_sigma: float) -> float:
        m = self.mgf(log_mu, log_sigma)
        return float(max(1.0 - m, m))


# ---------------------------------------------------------------------------
# Sarmanov copula
# ---------------------------------------------------------------------------

@dataclass
class SarmanovCopula:
    """
    Sarmanov bivariate distribution for mixed discrete-continuous margins.

    Joint density/PMF:
        f(n, s) = f_N(n) * f_S(s) * [1 + omega * phi_1(n) * phi_2(s)]

    Parameters
    ----------
    freq_family : 'nb' | 'poisson'
        Marginal family for claim frequency.
    sev_family : 'gamma' | 'lognormal'
        Marginal family for claim severity.
    omega : float
        Dependence parameter. Must satisfy the non-negativity constraint.
        Negative omega indicates that high-frequency claims tend to have
        lower severity (the typical finding in UK motor via NCD suppression).
    kernel_theta : float
        Laplace exponent for the frequency kernel phi_1.
    kernel_alpha : float
        Laplace exponent for the severity kernel phi_2.

    Notes
    -----
    omega=0 corresponds to independence. The Sarmanov family reduces to
    the product of marginals when omega=0.

    The parameter space for omega is bounded:
        -1 / (sup_phi1 * sup_phi2) <= omega <= 1 / (sup_phi1 * sup_phi2)

    In practice, for moderate mu_N and mu_S, the feasible omega range is
    roughly [-10, 10] or wider, so the bound rarely binds.
    """

    freq_family: Literal["nb", "poisson"] = "nb"
    sev_family: Literal["gamma", "lognormal"] = "gamma"
    omega: float = 0.0
    kernel_theta: float = 0.5
    kernel_alpha: float = 0.001

    def __post_init__(self):
        self._freq_kernel: Optional[LaplaceKernelNB | LaplaceKernelPoisson] = None
        self._sev_kernel: Optional[LaplaceKernelGamma | LaplaceKernelLognormal] = None
        self._build_kernels()

    def _build_kernels(self):
        if self.freq_family == "nb":
            self._freq_kernel = LaplaceKernelNB(theta=self.kernel_theta)
        else:
            self._freq_kernel = LaplaceKernelPoisson(theta=self.kernel_theta)

        if self.sev_family == "gamma":
            self._sev_kernel = LaplaceKernelGamma(alpha=self.kernel_alpha)
        else:
            self._sev_kernel = LaplaceKernelLognormal(alpha=self.kernel_alpha)

    def _phi1(self, n: np.ndarray, freq_params: dict) -> np.ndarray:
        """Centred frequency kernel phi_1(n)."""
        if self.freq_family == "nb":
            return self._freq_kernel.centred(n, freq_params["mu"], freq_params["alpha"])
        else:
            return self._freq_kernel.centred(n, freq_params["mu"])

    def _phi2(self, s: np.ndarray, sev_params: dict) -> np.ndarray:
        """Centred severity kernel phi_2(s)."""
        if self.sev_family == "gamma":
            return self._sev_kernel.centred(s, sev_params["mu"], sev_params["shape"])
        else:
            return self._sev_kernel.centred(s, sev_params["log_mu"], sev_params["log_sigma"])

    def _log_freq_pmf(self, n: np.ndarray, freq_params: dict) -> np.ndarray:
        """log P(N=n) under marginal frequency distribution."""
        mu = freq_params["mu"]
        if self.freq_family == "nb":
            alpha = freq_params["alpha"]
            r = 1.0 / alpha
            p = 1.0 / (1.0 + mu * alpha)
            n_arr = np.asarray(n, dtype=float)
            # NB PMF: Gamma(n+r)/(n! Gamma(r)) * p^r * (1-p)^n
            log_pmf = (
                gammaln(n_arr + r) - gammaln(r) - gammaln(n_arr + 1)
                + r * np.log(p)
                + n_arr * np.log1p(-p)
            )
        else:
            n_arr = np.asarray(n, dtype=float)
            log_pmf = n_arr * np.log(mu) - mu - gammaln(n_arr + 1)
        return log_pmf

    def _log_sev_pdf(self, s: np.ndarray, sev_params: dict) -> np.ndarray:
        """log f_S(s) under marginal severity distribution."""
        s_arr = np.asarray(s, dtype=float)
        if self.sev_family == "gamma":
            mu_s = sev_params["mu"]
            shape = sev_params["shape"]
            scale = mu_s / shape
            log_pdf = stats.gamma.logpdf(s_arr, a=shape, scale=scale)
        else:
            log_mu = sev_params["log_mu"]
            log_sigma = sev_params["log_sigma"]
            log_pdf = stats.lognorm.logpdf(s_arr, s=log_sigma, scale=np.exp(log_mu))
        return log_pdf

    def log_joint_density(
        self,
        n: np.ndarray,
        s: np.ndarray,
        freq_params: dict,
        sev_params: dict,
    ) -> np.ndarray:
        """
        log f(n, s) = log f_N(n) + log f_S(s) + log(1 + omega * phi_1(n) * phi_2(s))

        Parameters
        ----------
        n : array-like
            Claim counts (non-negative integers).
        s : array-like
            Average claim severities (positive reals). Only used for policies
            with n > 0.
        freq_params : dict
            Distribution parameters for frequency margin.
            For NB: {'mu': float, 'alpha': float}
            For Poisson: {'mu': float}
        sev_params : dict
            Distribution parameters for severity margin.
            For Gamma: {'mu': float, 'shape': float}
            For Lognormal: {'log_mu': float, 'log_sigma': float}

        Returns
        -------
        ndarray of log joint densities. Shape matches n/s.
        """
        n = np.asarray(n, dtype=float)
        s = np.asarray(s, dtype=float)

        log_f_n = self._log_freq_pmf(n, freq_params)
        log_f_s = self._log_sev_pdf(s, sev_params)
        phi1 = self._phi1(n, freq_params)
        phi2 = self._phi2(s, sev_params)

        interaction = 1.0 + self.omega * phi1 * phi2
        # Guard against numerical zero or negative values
        interaction = np.clip(interaction, 1e-300, None)

        return log_f_n + log_f_s + np.log(interaction)

    def log_likelihood(
        self,
        n: np.ndarray,
        s: np.ndarray,
        freq_params: dict | list,
        sev_params: dict | list,
    ) -> float:
        """
        Total log-likelihood over a sample.

        For observations with n=0, the severity does not exist; the contribution
        is just log f_N(0). Pass s=0 or any positive value for these rows;
        the function handles them correctly by treating the joint as a marginal
        frequency contribution only when n=0.

        Implementation note: when n=0, the joint is f(0, s) = f_N(0) * f_S(s) *
        [1 + omega*phi_1(0)*phi_2(s)]. But since s is unobserved for n=0, we
        must integrate over s, recovering f_N(0). This is correct automatically
        since E[phi_2(S)] = 0, so:

            integral f_S(s) * [1 + omega*phi_1(0)*phi_2(s)] ds
            = 1 + omega * phi_1(0) * E[phi_2(S)]
            = 1 + omega * phi_1(0) * 0  = 1

        So for n=0 rows: contribution = log f_N(0).
        For n>0 rows: contribution = log f_N(n) + log f_S(s) + log(1 + omega*phi1*phi2).
        """
        n = np.asarray(n, dtype=float)
        s = np.asarray(s, dtype=float)

        # Handle per-observation parameters (arrays of dicts) or single dict
        fp_alpha = None  # initialised before conditional so it is always defined
        if isinstance(freq_params, dict):
            fp_mu = np.full_like(n, freq_params["mu"])
            if self.freq_family == "nb":
                if "alpha" not in freq_params:
                    raise ValueError(
                        "freq_params must contain 'alpha' for NB family"
                    )
                fp_alpha = np.full_like(n, freq_params["alpha"])
        else:
            fp_mu = np.array([p["mu"] for p in freq_params])
            if self.freq_family == "nb":
                fp_alpha = np.array([p["alpha"] for p in freq_params])

        if isinstance(sev_params, dict):
            if self.sev_family == "gamma":
                sp_mu = np.full_like(s, sev_params["mu"])
                sp_shape = np.full_like(s, sev_params["shape"])
            else:
                sp_lmu = np.full_like(s, sev_params["log_mu"])
                sp_lsigma = np.full_like(s, sev_params["log_sigma"])
        else:
            if self.sev_family == "gamma":
                sp_mu = np.array([p["mu"] for p in sev_params])
                sp_shape = np.array([p["shape"] for p in sev_params])
            else:
                sp_lmu = np.array([p["log_mu"] for p in sev_params])
                sp_lsigma = np.array([p["log_sigma"] for p in sev_params])

        # Frequency log-PMF for all observations
        if self.freq_family == "nb":
            fp = {"mu": fp_mu, "alpha": fp_alpha}
        else:
            fp = {"mu": fp_mu}
        log_f_n = self._log_freq_pmf(n, fp)

        # For n=0 rows: contribution is just log f_N(0)
        mask_pos = n > 0
        ll = np.where(mask_pos, 0.0, log_f_n)

        if mask_pos.any():
            n_pos = n[mask_pos]
            s_pos = s[mask_pos]

            if self.freq_family == "nb":
                fp_pos = {"mu": fp_mu[mask_pos], "alpha": fp_alpha[mask_pos]}
            else:
                fp_pos = {"mu": fp_mu[mask_pos]}

            if self.sev_family == "gamma":
                sp_pos = {"mu": sp_mu[mask_pos], "shape": sp_shape[mask_pos]}
            else:
                sp_pos = {"log_mu": sp_lmu[mask_pos], "log_sigma": sp_lsigma[mask_pos]}

            log_jd = self.log_joint_density(n_pos, s_pos, fp_pos, sp_pos)
            ll[mask_pos] = log_jd

        return float(np.sum(ll))

    def omega_bounds(
        self,
        freq_params: dict,
        sev_params: dict,
        n_grid: int = 50,
        s_grid: int = 200,
    ) -> Tuple[float, float]:
        """
        Compute feasible omega range for given marginal parameters.

        Returns (omega_min, omega_max) such that
            1 + omega * phi_1(n) * phi_2(s) >= 0 everywhere.

        Uses a grid search over (n, s) to find the extremes of phi1*phi2.
        """
        # Frequency support: n = 0, 1, ..., n_grid
        n_vals = np.arange(n_grid)
        phi1 = self._phi1(n_vals, freq_params)

        # Severity support: quantiles of marginal distribution
        if self.sev_family == "gamma":
            mu_s = sev_params["mu"]
            shape = sev_params["shape"]
            scale = mu_s / shape
            s_vals = stats.gamma.ppf(np.linspace(0.001, 0.999, s_grid), a=shape, scale=scale)
        else:
            log_mu = sev_params["log_mu"]
            log_sigma = sev_params["log_sigma"]
            s_vals = stats.lognorm.ppf(
                np.linspace(0.001, 0.999, s_grid),
                s=log_sigma, scale=np.exp(log_mu)
            )
        phi2 = self._phi2(s_vals, sev_params)

        # Product grid
        products = np.outer(phi1, phi2)
        max_prod = float(np.max(products))
        min_prod = float(np.min(products))

        omega_max = (1.0 / max_prod) if max_prod > 0 else np.inf
        omega_min = (-1.0 / (-min_prod)) if min_prod < 0 else -np.inf

        return (omega_min, omega_max)

    def spearman_rho(
        self,
        freq_params: dict,
        sev_params: dict,
        n_mc: int = 50_000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Monte Carlo estimate of Spearman's rho for this Sarmanov distribution.

        Simulates n_mc samples and computes the rank correlation.
        """
        n_samp, s_samp = self.sample(n_mc, freq_params, sev_params, rng=rng)
        rho = float(np.corrcoef(
            stats.rankdata(n_samp),
            stats.rankdata(s_samp)
        )[0, 1])
        return rho

    def kendall_tau(
        self,
        freq_params: dict,
        sev_params: dict,
        n_mc: int = 20_000,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Monte Carlo estimate of Kendall's tau."""
        from scipy.stats import kendalltau
        n_samp, s_samp = self.sample(n_mc, freq_params, sev_params, rng=rng)
        tau, _ = kendalltau(n_samp, s_samp)
        return float(tau)

    def sample(
        self,
        size: int,
        freq_params: dict,
        sev_params: dict,
        rng: Optional[np.random.Generator] = None,
        max_iter: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from the Sarmanov distribution using acceptance-rejection.

        The proposal distribution is the product f_N * f_S. The acceptance
        probability is proportional to 1 + omega * phi1(n) * phi2(s).

        The bound on the acceptance ratio is:
            M = 1 + |omega| * sup|phi1| * sup|phi2|

        For typical insurance parameters with small omega, acceptance rate is
        very high (> 90%).

        Returns
        -------
        (n_samples, s_samples) : arrays of shape (size,)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Compute acceptance bound M
        if self.freq_family == "nb":
            sup1 = self._freq_kernel.sup_abs(freq_params["mu"], freq_params["alpha"])
        else:
            sup1 = self._freq_kernel.sup_abs(freq_params["mu"])

        if self.sev_family == "gamma":
            sup2 = self._sev_kernel.sup_abs(sev_params["mu"], sev_params["shape"])
        else:
            sup2 = self._sev_kernel.sup_abs(sev_params["log_mu"], sev_params["log_sigma"])

        M = 1.0 + abs(self.omega) * sup1 * sup2

        n_collected = np.zeros(size, dtype=int)
        s_collected = np.zeros(size, dtype=float)
        filled = 0

        for _ in range(max_iter):
            remaining = size - filled
            if remaining <= 0:
                break
            # Oversample to reduce iterations
            n_propose = int(remaining * M * 1.5) + 100

            # Sample from marginals
            if self.freq_family == "nb":
                mu = freq_params["mu"]
                alpha = freq_params["alpha"]
                r = 1.0 / alpha
                p = r / (r + mu)
                n_prop = rng.negative_binomial(r, p, size=n_propose)
            else:
                n_prop = rng.poisson(freq_params["mu"], size=n_propose)

            if self.sev_family == "gamma":
                mu_s = sev_params["mu"]
                shape = sev_params["shape"]
                scale = mu_s / shape
                s_prop = rng.gamma(shape, scale, size=n_propose)
            else:
                log_mu = sev_params["log_mu"]
                log_sigma = sev_params["log_sigma"]
                s_prop = np.exp(rng.normal(log_mu, log_sigma, size=n_propose))

            # Acceptance ratio
            if self.freq_family == "nb":
                p1 = self._phi1(n_prop, freq_params)
            else:
                p1 = self._phi1(n_prop, freq_params)

            if self.sev_family == "gamma":
                p2 = self._phi2(s_prop, sev_params)
            else:
                p2 = self._phi2(s_prop, sev_params)

            ratio = (1.0 + self.omega * p1 * p2) / M
            ratio = np.clip(ratio, 0, 1)

            accept = rng.uniform(size=n_propose) < ratio
            n_acc = n_prop[accept]
            s_acc = s_prop[accept]

            take = min(len(n_acc), remaining)
            n_collected[filled:filled + take] = n_acc[:take]
            s_collected[filled:filled + take] = s_acc[:take]
            filled += take

        if filled < size:
            warnings.warn(
                f"Acceptance-rejection sampler: only filled {filled}/{size} samples "
                f"after {max_iter} iterations. Acceptance rate may be very low. "
                f"Check omega bounds (M={M:.2f})."
            )

        return n_collected[:filled], s_collected[:filled]


# ---------------------------------------------------------------------------
# Gaussian copula with PIT approximation for discrete margins
# ---------------------------------------------------------------------------

@dataclass
class GaussianCopulaMixed:
    """
    Gaussian copula linking discrete frequency and continuous severity.

    This is the standard approach of Czado et al. (2012) and the R package
    CopulaRegression. The discrete margin creates an identifiability issue
    (Sklar's theorem is not unique for discrete distributions), addressed by
    the mid-point PIT approximation:

        U_N = (F_N(n) + F_N(n-1)) / 2    for n > 0
        U_N = F_N(0) / 2                  for n = 0

    This maps count data onto (0,1) for use with the Gaussian copula.
    The approximation is accurate when the Poisson or NB mass at each point
    is small (not too much probability on any single value).

    Parameters
    ----------
    rho : float
        Gaussian copula correlation parameter in (-1, 1).
        This is NOT Spearman's rho (though they are closely related for the
        Gaussian copula: rho_S ≈ (6/pi) * arcsin(rho/2)).
    """

    rho: float = 0.0

    def _pit_freq(self, n: np.ndarray, freq_params: dict, freq_family: str) -> np.ndarray:
        """Mid-point PIT for discrete frequency margin."""
        n = np.asarray(n, dtype=float)
        if freq_family == "nb":
            mu = freq_params["mu"]
            alpha = freq_params["alpha"]
            r = 1.0 / alpha
            p = 1.0 / (1.0 + mu * alpha)
            cdf_n = stats.nbinom.cdf(n, r, p)
            cdf_nm1 = np.where(n > 0, stats.nbinom.cdf(n - 1, r, p), 0.0)
        else:
            mu = freq_params["mu"]
            cdf_n = stats.poisson.cdf(n, mu)
            cdf_nm1 = np.where(n > 0, stats.poisson.cdf(n - 1, mu), 0.0)
        return 0.5 * (cdf_n + cdf_nm1)

    def _pit_sev(self, s: np.ndarray, sev_params: dict, sev_family: str) -> np.ndarray:
        """Standard PIT for continuous severity margin."""
        s = np.asarray(s, dtype=float)
        if sev_family == "gamma":
            mu_s = sev_params["mu"]
            shape = sev_params["shape"]
            scale = mu_s / shape
            return stats.gamma.cdf(s, a=shape, scale=scale)
        else:
            log_mu = sev_params["log_mu"]
            log_sigma = sev_params["log_sigma"]
            return stats.lognorm.cdf(s, s=log_sigma, scale=np.exp(log_mu))

    def log_likelihood(
        self,
        n: np.ndarray,
        s: np.ndarray,
        freq_params: dict | list,
        sev_params: dict | list,
        freq_family: str = "nb",
        sev_family: str = "gamma",
    ) -> float:
        """
        Gaussian copula log-likelihood (positive-claim observations only).

        For n=0 observations, there is no severity, so only the marginal
        frequency contributes.
        """
        n = np.asarray(n, dtype=float)
        s = np.asarray(s, dtype=float)

        mask_pos = n > 0
        if not mask_pos.any():
            return 0.0

        n_pos = n[mask_pos]
        s_pos = s[mask_pos]

        if isinstance(freq_params, dict):
            fp_pos = freq_params
        else:
            fp_pos = {
                "mu": np.array([p["mu"] for p in freq_params])[mask_pos],
                **({
                    "alpha": np.array([p["alpha"] for p in freq_params])[mask_pos]
                } if freq_family == "nb" else {}),
            }

        if isinstance(sev_params, dict):
            sp_pos = sev_params
        else:
            if sev_family == "gamma":
                sp_pos = {
                    "mu": np.array([p["mu"] for p in sev_params])[mask_pos],
                    "shape": np.array([p["shape"] for p in sev_params])[mask_pos],
                }
            else:
                sp_pos = {
                    "log_mu": np.array([p["log_mu"] for p in sev_params])[mask_pos],
                    "log_sigma": np.array([p["log_sigma"] for p in sev_params])[mask_pos],
                }

        u = self._pit_freq(n_pos, fp_pos, freq_family)
        v = self._pit_sev(s_pos, sp_pos, sev_family)

        # Clip to avoid numerical issues at boundaries
        u = np.clip(u, 1e-8, 1 - 1e-8)
        v = np.clip(v, 1e-8, 1 - 1e-8)

        # Gaussian copula density: c(u,v) = phi2(Phi^{-1}(u), Phi^{-1}(v); rho) /
        #                                    (phi(Phi^{-1}(u)) * phi(Phi^{-1}(v)))
        z1 = stats.norm.ppf(u)
        z2 = stats.norm.ppf(v)

        rho = np.clip(self.rho, -0.9999, 0.9999)
        rho2 = rho ** 2

        log_c = (
            -0.5 * np.log(1 - rho2)
            - (rho2 * (z1**2 + z2**2) - 2 * rho * z1 * z2) / (2 * (1 - rho2))
        )

        return float(np.sum(log_c))

    def spearman_rho(self) -> float:
        """Approximate Spearman rho from Gaussian copula parameter."""
        return float(6.0 / np.pi * np.arcsin(self.rho / 2.0))

    def sample(
        self,
        size: int,
        freq_params: dict,
        sev_params: dict,
        freq_family: str = "nb",
        sev_family: str = "gamma",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample using Gaussian copula via conditional normal."""
        if rng is None:
            rng = np.random.default_rng()

        rho = np.clip(self.rho, -0.9999, 0.9999)
        z1 = rng.standard_normal(size)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(size)
        u = stats.norm.cdf(z1)
        v = stats.norm.cdf(z2)

        # Invert marginal CDFs
        if freq_family == "nb":
            mu = freq_params["mu"]
            alpha = freq_params["alpha"]
            r = 1.0 / alpha
            p = 1.0 / (1.0 + mu * alpha)
            n_samp = stats.nbinom.ppf(u, r, p).astype(int)
        else:
            n_samp = stats.poisson.ppf(u, freq_params["mu"]).astype(int)

        if sev_family == "gamma":
            mu_s = sev_params["mu"]
            shape = sev_params["shape"]
            scale = mu_s / shape
            s_samp = stats.gamma.ppf(v, a=shape, scale=scale)
        else:
            log_mu = sev_params["log_mu"]
            log_sigma = sev_params["log_sigma"]
            s_samp = stats.lognorm.ppf(v, s=log_sigma, scale=np.exp(log_mu))

        return n_samp, s_samp


# ---------------------------------------------------------------------------
# FGM copula — simple baseline
# ---------------------------------------------------------------------------

@dataclass
class FGMCopula:
    """
    Farlie-Gumbel-Morgenstern (FGM) copula.

    C(u, v) = u * v * (1 + theta * (1-u) * (1-v))

    theta in [-1, 1] corresponds to Spearman rho in [-1/3, 1/3].

    The FGM copula is a special case of the Sarmanov family with
    phi_1(x) = F_1(x) - E[F_1(X)] and phi_2(y) = F_2(y) - E[F_2(Y)].

    This is implemented here as a simple sanity-check baseline. If the FGM
    copula fits as well as Sarmanov, the dependence is weak and the correction
    factor is small. If FGM is inadequate (theta hits ±1), the data likely
    has moderate dependence and Sarmanov is justified.
    """

    theta: float = 0.0

    def __post_init__(self):
        if not -1.0 <= self.theta <= 1.0:
            raise ValueError(f"FGM theta must be in [-1, 1], got {self.theta}")

    def cdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """C(u, v) = uv[1 + theta(1-u)(1-v)]"""
        u = np.asarray(u)
        v = np.asarray(v)
        return u * v * (1.0 + self.theta * (1.0 - u) * (1.0 - v))

    def pdf(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """c(u, v) = 1 + theta(1 - 2u)(1 - 2v)"""
        u = np.asarray(u)
        v = np.asarray(v)
        return 1.0 + self.theta * (1.0 - 2.0 * u) * (1.0 - 2.0 * v)

    def log_likelihood(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> float:
        """Log-likelihood given PIT-transformed observations."""
        density = self.pdf(u, v)
        density = np.clip(density, 1e-300, None)
        return float(np.sum(np.log(density)))

    def spearman_rho(self) -> float:
        """Exact Spearman rho = theta / 3."""
        return self.theta / 3.0

    def kendall_tau(self) -> float:
        """Exact Kendall tau = 2*theta/9."""
        return 2.0 * self.theta / 9.0

    def sample(
        self,
        size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample (U, V) from FGM copula using conditional CDF method.

        C_{V|U}(v|u) = v * [1 + theta*(1-2u)*(1-v)]
        Solved analytically via quadratic: b*v^2 - (1+b)*v + w = 0,
        where b = theta*(1-2u).
        """
        if rng is None:
            rng = np.random.default_rng()
        u = rng.uniform(size=size)
        w = rng.uniform(size=size)
        # C_{V|U}(v|u) = w => solve quadratic in v
        # b*v^2 - (1+b)*v + w = 0, where b = theta*(1-2u)
        # v = [(1+b) - sqrt((1+b)^2 - 4*b*w)] / (2*b)
        b = self.theta * (1.0 - 2.0 * u)
        with np.errstate(invalid="ignore"):
            disc = (1.0 + b) ** 2 - 4.0 * b * w
            disc = np.maximum(disc, 0.0)
            v = np.where(
                np.abs(b) < 1e-10,
                w,
                ((1.0 + b) - np.sqrt(disc)) / (2.0 * b),
            )
        return u, np.clip(v, 0, 1)
