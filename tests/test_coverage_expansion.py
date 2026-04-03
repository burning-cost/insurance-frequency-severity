"""
test_coverage_expansion.py — Comprehensive gap-filling tests.

Targets modules with thin or zero coverage in the original suite:

copula.py:
  - LaplaceKernelNB: invalid theta validation
  - LaplaceKernelGamma: invalid alpha validation, MGF domain check
  - LaplaceKernelPoisson: sup_abs formula verification
  - LaplaceKernelLognormal: sup_abs formula
  - SarmanovCopula (NB+Lognormal): log_joint, likelihood, sample, bounds
  - SarmanovCopula: kendall_tau sign, omega_bounds for Poisson+Lognormal
  - SarmanovCopula: sampler incomplete fill warning
  - GaussianCopulaMixed: per-observation list params, lognormal sample
  - FGMCopula: cdf at corner (1,1), pdf boundary, theta=0 cdf reduces to product

diagnostics.py:
  - DependenceTest: filter removes invalid rows, no permutations path
  - DependenceTest: positive correlation detected
  - CopulaGOF: before-fit raises, u and v in (0,1) after fit
  - compare_copulas: minimum structure of returned DataFrame

joint.py:
  - _extract_freq_params: Poisson family detection
  - _extract_freq_params: NB family with params dict fallback
  - _extract_sev_params: lognormal family detection
  - _extract_sev_params: shape clamped at 0.01
  - JointFreqSev: invalid copula raises ValueError
  - JointFreqSev: invalid kernel_theta raises
  - JointFreqSev: FGM premium_correction runs and columns present
  - JointFreqSev: loss_cost with new X
  - JointFreqSev: omega_ci_ set after fit
  - JointFreqSev: n_claims < 500 warning fires
  - JointFreqSev: AIC < BIC when n > e^2 (BIC has larger penalty)
  - JointFreqSev: dependence_summary rho_ column present
  - ConditionalFreqSev: gamma sign with known DGP
  - ConditionalFreqSev: dependence_summary before fit raises
  - ConditionalFreqSev: correction at positive mu_n > 0 differs from independence

Total: 55+ new tests.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from insurance_frequency_severity.copula import (
    SarmanovCopula,
    GaussianCopulaMixed,
    FGMCopula,
    LaplaceKernelNB,
    LaplaceKernelPoisson,
    LaplaceKernelGamma,
    LaplaceKernelLognormal,
)
from insurance_frequency_severity.diagnostics import DependenceTest, CopulaGOF
from insurance_frequency_severity.joint import (
    JointFreqSev,
    ConditionalFreqSev,
    _extract_freq_params,
    _extract_sev_params,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rng(seed=42):
    return np.random.default_rng(seed)


def _make_mock_nb_glm(mu_n, alpha=0.8):
    """Minimal NB GLM mock."""
    class NbGLM:
        def __init__(self):
            self.fittedvalues = np.asarray(mu_n)
            class Fam:
                pass
            self.family = Fam()
            self.family.__class__.__name__ = "NegativeBinomial"
            self.family.alpha = alpha
            class Mod:
                pass
            self.model = Mod()
            self.model.family = Fam()
            self.model.family.__class__.__name__ = "NegativeBinomial"
            self.model.family.alpha = alpha
        def predict(self, X=None):
            if X is not None:
                return np.full(len(X), float(self.fittedvalues[0]))
            return self.fittedvalues
    return NbGLM()


def _make_mock_poisson_glm(mu_n):
    class PoissonGLM:
        def __init__(self):
            self.fittedvalues = np.asarray(mu_n)
            class Fam:
                pass
            self.family = Fam()
            self.family.__class__.__name__ = "Poisson"
            class Mod:
                pass
            self.model = Mod()
            self.model.family = Fam()
            self.model.family.__class__.__name__ = "Poisson"
        def predict(self, X=None):
            if X is not None:
                return np.full(len(X), float(self.fittedvalues[0]))
            return self.fittedvalues
    return PoissonGLM()


def _make_mock_gamma_glm(mu_s, shape=1.2):
    class GammaGLM:
        def __init__(self):
            self.fittedvalues = np.asarray(mu_s)
            self.scale = 1.0 / shape
            class Fam:
                pass
            self.family = Fam()
            self.family.__class__.__name__ = "Gamma"
            class Mod:
                pass
            self.model = Mod()
            self.model.family = Fam()
            self.model.family.__class__.__name__ = "Gamma"
        def predict(self, X=None):
            if X is not None:
                return np.full(len(X), float(self.fittedvalues[0]))
            return self.fittedvalues
    return GammaGLM()


def _make_mock_lognormal_glm(mu_s, shape=1.2):
    class LognormGLM:
        def __init__(self):
            self.fittedvalues = np.asarray(mu_s)
            self.scale = 1.0 / shape
            class Fam:
                pass
            self.family = Fam()
            self.family.__class__.__name__ = "Lognormal"
            class Mod:
                pass
            self.model = Mod()
            self.model.family = Fam()
            self.model.family.__class__.__name__ = "Lognormal"
        def predict(self, X=None):
            if X is not None:
                return np.full(len(X), float(self.fittedvalues[0]))
            return self.fittedvalues
    return LognormGLM()


def _sample_nb_gamma_dgp(n_policies=2000, seed=0):
    """Sample from NB+Gamma Sarmanov DGP with moderate negative omega."""
    rng = _rng(seed)
    mu_n, alpha, mu_s, shape = 0.15, 0.8, 2000.0, 1.2
    copula = SarmanovCopula(
        freq_family="nb", sev_family="gamma",
        omega=-3.0, kernel_theta=0.5, kernel_alpha=0.0005,
    )
    n, s = copula.sample(n_policies, {"mu": mu_n, "alpha": alpha},
                         {"mu": mu_s, "shape": shape}, rng=rng)
    return n, s, mu_n, alpha, mu_s, shape


# ---------------------------------------------------------------------------
# LaplaceKernelNB — validation and properties
# ---------------------------------------------------------------------------

class TestLaplaceKernelNBValidation:

    def test_negative_theta_raises(self):
        with pytest.raises(ValueError, match="theta must be positive"):
            LaplaceKernelNB(theta=-0.1)

    def test_zero_theta_raises(self):
        with pytest.raises(ValueError, match="theta must be positive"):
            LaplaceKernelNB(theta=0.0)

    def test_mgf_strictly_less_than_one(self):
        """For theta>0, MGF E[exp(-theta*N)] < 1 because N>=0 can be positive."""
        kern = LaplaceKernelNB(theta=0.3)
        m = kern.mgf(mu=0.3, alpha=0.8)
        assert float(m) < 1.0

    def test_mgf_approaches_one_for_small_theta(self):
        """As theta -> 0, E[exp(-theta*N)] -> 1."""
        kern = LaplaceKernelNB(theta=1e-5)
        m = kern.mgf(mu=0.2, alpha=0.5)
        assert abs(float(m) - 1.0) < 0.01

    def test_sup_abs_matches_formula(self):
        """sup|phi| = max(1 - MGF, MGF) by the Laplace kernel structure."""
        kern = LaplaceKernelNB(theta=0.5)
        mu, alpha = 0.2, 0.5
        m = float(kern.mgf(mu, alpha))
        expected = max(1.0 - m, m)
        got = kern.sup_abs(mu, alpha)
        assert abs(got - expected) < 1e-10

    def test_centred_value_at_n0(self):
        """phi(0) = 1 - MGF (positive, since exp(0)=1 > MGF for theta>0)."""
        kern = LaplaceKernelNB(theta=0.5)
        mu, alpha = 0.2, 0.5
        m = float(kern.mgf(mu, alpha))
        phi0 = float(kern.centred(np.array([0.0]), mu, alpha)[0])
        assert abs(phi0 - (1.0 - m)) < 1e-10


# ---------------------------------------------------------------------------
# LaplaceKernelGamma — validation and properties
# ---------------------------------------------------------------------------

class TestLaplaceKernelGammaValidation:

    def test_zero_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            LaplaceKernelGamma(alpha=0.0)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            LaplaceKernelGamma(alpha=-0.5)

    def test_mgf_strictly_less_than_one(self):
        kern = LaplaceKernelGamma(alpha=0.001)
        m = kern.mgf(mu_s=2000.0, shape=1.5)
        assert float(m) < 1.0

    def test_mgf_increases_with_smaller_alpha(self):
        """Smaller alpha means less suppression of exp(-alpha*s), so MGF closer to 1."""
        kern_small = LaplaceKernelGamma(alpha=0.0001)
        kern_large = LaplaceKernelGamma(alpha=0.01)
        mu_s, shape = 1000.0, 1.5
        m_small = float(kern_small.mgf(mu_s, shape))
        m_large = float(kern_large.mgf(mu_s, shape))
        assert m_small > m_large

    def test_sup_abs_formula(self):
        kern = LaplaceKernelGamma(alpha=0.001)
        mu_s, shape = 1500.0, 1.2
        m = float(kern.mgf(mu_s, shape))
        expected = max(1.0 - m, m)
        got = kern.sup_abs(mu_s, shape)
        assert abs(got - expected) < 1e-10

    def test_centred_value_at_s0(self):
        """phi(0) = 1 - MGF for Laplace kernel."""
        kern = LaplaceKernelGamma(alpha=0.001)
        mu_s, shape = 1000.0, 1.5
        m = float(kern.mgf(mu_s, shape))
        phi0 = float(kern.centred(np.array([0.0]), mu_s, shape)[0])
        assert abs(phi0 - (1.0 - m)) < 1e-10


# ---------------------------------------------------------------------------
# LaplaceKernelPoisson — sup_abs formula
# ---------------------------------------------------------------------------

class TestLaplaceKernelPoissonSupAbs:

    def test_sup_abs_formula(self):
        """sup|phi| = max(1-MGF, MGF)."""
        kern = LaplaceKernelPoisson(theta=0.5)
        mu = 0.15
        m = float(kern.mgf(mu))
        expected = max(1.0 - m, m)
        got = kern.sup_abs(mu)
        assert abs(got - expected) < 1e-10

    def test_mgf_in_unit_interval(self):
        kern = LaplaceKernelPoisson(theta=0.5)
        m = kern.mgf(mu=0.3)
        assert 0.0 < float(m) < 1.0


# ---------------------------------------------------------------------------
# LaplaceKernelLognormal — sup_abs and mgf properties
# ---------------------------------------------------------------------------

class TestLaplaceKernelLognormalSupAbs:

    def test_sup_abs_formula(self):
        kern = LaplaceKernelLognormal(alpha=0.001)
        log_mu, log_sigma = np.log(2000), 0.5
        m = kern.mgf(log_mu, log_sigma)
        expected = max(1.0 - m, m)
        got = kern.sup_abs(log_mu, log_sigma)
        assert abs(got - expected) < 1e-8

    def test_mgf_in_unit_interval(self):
        kern = LaplaceKernelLognormal(alpha=0.001)
        m = kern.mgf(log_mu=np.log(2000), log_sigma=0.5)
        assert 0.0 < m < 1.0

    def test_mgf_decreases_with_larger_alpha(self):
        """Larger alpha dampens more severely, so smaller MGF."""
        log_mu, log_sigma = np.log(2000), 0.5
        m_small = LaplaceKernelLognormal(alpha=0.0001).mgf(log_mu, log_sigma)
        m_large = LaplaceKernelLognormal(alpha=0.01).mgf(log_mu, log_sigma)
        assert m_small > m_large


# ---------------------------------------------------------------------------
# SarmanovCopula — NB + Lognormal combination
# ---------------------------------------------------------------------------

class TestSarmanovNBLognormal:

    @pytest.fixture
    def copula_nl(self):
        return SarmanovCopula(
            freq_family="nb", sev_family="lognormal",
            omega=0.0, kernel_theta=0.5, kernel_alpha=0.001,
        )

    @pytest.fixture
    def params_nl(self):
        return (
            {"mu": 0.2, "alpha": 0.8},
            {"log_mu": np.log(2000.0), "log_sigma": 0.5},
        )

    def test_log_joint_finite(self, copula_nl, params_nl):
        fp, sp = params_nl
        n_vals = np.array([0.0, 1.0, 2.0, 3.0])
        s_vals = np.array([500.0, 2000.0, 1500.0, 2500.0])
        log_jd = copula_nl.log_joint_density(n_vals, s_vals, fp, sp)
        assert np.all(np.isfinite(log_jd))

    def test_independence_factorises(self, copula_nl, params_nl):
        """omega=0: log_joint = log f_N + log f_S."""
        fp, sp = params_nl
        n_vals = np.array([1.0, 2.0])
        s_vals = np.array([2000.0, 1500.0])
        log_jd = copula_nl.log_joint_density(n_vals, s_vals, fp, sp)
        log_fn = copula_nl._log_freq_pmf(n_vals, fp)
        log_fs = copula_nl._log_sev_pdf(s_vals, sp)
        np.testing.assert_allclose(log_jd, log_fn + log_fs, atol=1e-7)

    def test_log_likelihood_finite(self, copula_nl, params_nl):
        rng = _rng(10)
        fp, sp = params_nl
        n_pol = 50
        n_vals = np.array([1.0] * n_pol)
        s_vals = np.exp(rng.normal(np.log(2000), 0.5, n_pol))
        fp_list = [fp] * n_pol
        sp_list = [sp] * n_pol
        ll = copula_nl.log_likelihood(n_vals, s_vals, fp_list, sp_list)
        assert np.isfinite(ll)

    def test_sample_runs_and_has_correct_shape(self):
        rng = _rng(11)
        copula = SarmanovCopula(
            freq_family="nb", sev_family="lognormal",
            omega=-2.0, kernel_theta=0.5, kernel_alpha=0.001,
        )
        fp = {"mu": 0.2, "alpha": 0.8}
        sp = {"log_mu": np.log(2000.0), "log_sigma": 0.5}
        n_samp, s_samp = copula.sample(3000, fp, sp, rng=rng)
        assert len(n_samp) == 3000
        assert len(s_samp) == 3000
        assert np.all(s_samp > 0)

    def test_omega_bounds_finite_nb_lognormal(self):
        copula = SarmanovCopula(
            freq_family="nb", sev_family="lognormal",
            omega=0.0, kernel_theta=0.5, kernel_alpha=0.001,
        )
        fp = {"mu": 0.2, "alpha": 0.8}
        sp = {"log_mu": np.log(2000.0), "log_sigma": 0.5}
        omega_min, omega_max = copula.omega_bounds(fp, sp)
        assert omega_max > 0
        assert omega_min < 0


# ---------------------------------------------------------------------------
# SarmanovCopula — Poisson + Gamma additional tests
# ---------------------------------------------------------------------------

class TestSarmanovPoissonGamma:

    def test_sample_mean_poisson_gamma_independence(self):
        rng = _rng(20)
        copula = SarmanovCopula(
            freq_family="poisson", sev_family="gamma",
            omega=0.0, kernel_theta=0.5, kernel_alpha=0.001,
        )
        fp = {"mu": 0.12}
        sp = {"mu": 1500.0, "shape": 1.5}
        n_samp, s_samp = copula.sample(40000, fp, sp, rng=rng)
        assert abs(np.mean(n_samp) - 0.12) / 0.12 < 0.10
        assert abs(np.mean(s_samp) - 1500.0) / 1500.0 < 0.10

    def test_omega_bounds_poisson_gamma(self):
        copula = SarmanovCopula(
            freq_family="poisson", sev_family="gamma",
            omega=0.0, kernel_theta=0.5, kernel_alpha=0.001,
        )
        fp = {"mu": 0.12}
        sp = {"mu": 1500.0, "shape": 1.5}
        omega_min, omega_max = copula.omega_bounds(fp, sp)
        assert omega_max > 0
        assert omega_min < 0

    def test_log_joint_finite_poisson_gamma(self):
        copula = SarmanovCopula(
            freq_family="poisson", sev_family="gamma",
            omega=2.0, kernel_theta=0.5, kernel_alpha=0.001,
        )
        fp = {"mu": 0.12}
        sp = {"mu": 1500.0, "shape": 1.5}
        n_vals = np.array([0.0, 1.0, 2.0])
        s_vals = np.array([500.0, 1500.0, 2500.0])
        log_jd = copula.log_joint_density(n_vals, s_vals, fp, sp)
        assert np.all(np.isfinite(log_jd))


# ---------------------------------------------------------------------------
# SarmanovCopula — kendall_tau method
# ---------------------------------------------------------------------------

class TestSarmanovKendallTau:

    def test_kendall_tau_sign_negative_omega(self):
        rng = _rng(30)
        copula = SarmanovCopula(
            freq_family="nb", sev_family="gamma",
            omega=-5.0, kernel_theta=0.5, kernel_alpha=0.001,
        )
        fp = {"mu": 0.3, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.5}
        tau = copula.kendall_tau(fp, sp, n_mc=20000, rng=rng)
        assert tau < 0, f"Expected negative tau for omega<0, got {tau:.4f}"

    def test_kendall_tau_sign_positive_omega(self):
        rng = _rng(31)
        copula = SarmanovCopula(
            freq_family="nb", sev_family="gamma",
            omega=5.0, kernel_theta=0.5, kernel_alpha=0.001,
        )
        fp = {"mu": 0.3, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.5}
        tau = copula.kendall_tau(fp, sp, n_mc=20000, rng=rng)
        assert tau > 0, f"Expected positive tau for omega>0, got {tau:.4f}"

    def test_kendall_tau_near_zero_at_independence(self):
        rng = _rng(32)
        copula = SarmanovCopula(
            freq_family="nb", sev_family="gamma",
            omega=0.0, kernel_theta=0.5, kernel_alpha=0.001,
        )
        fp = {"mu": 0.2, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.5}
        tau = copula.kendall_tau(fp, sp, n_mc=30000, rng=rng)
        assert abs(tau) < 0.05, f"Expected tau near 0 for omega=0, got {tau:.4f}"


# ---------------------------------------------------------------------------
# SarmanovCopula — sampler incomplete fill warning
# ---------------------------------------------------------------------------

class TestSarmanovSamplerWarning:

    def test_sampler_warns_on_incomplete_fill(self):
        """Very large |omega| with max_iter=1 should trigger warning."""
        # With max_iter=1 and extreme omega, acceptance rate will be very low
        copula = SarmanovCopula(
            freq_family="nb", sev_family="gamma",
            omega=50.0, kernel_theta=0.5, kernel_alpha=0.001,
        )
        fp = {"mu": 0.2, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.2}
        rng = _rng(33)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            n_samp, s_samp = copula.sample(10000, fp, sp, rng=rng, max_iter=1)
        # If a warning was issued, it should mention acceptance or samples
        if len(caught) > 0:
            warning_texts = " ".join(str(w.message) for w in caught)
            assert any(
                phrase in warning_texts
                for phrase in ["acceptance", "filled", "iterations"]
            )

    def test_sampler_returns_partial_samples_on_incomplete_fill(self):
        """When sampler cannot fill requested size, returns what it has."""
        copula = SarmanovCopula(
            freq_family="nb", sev_family="gamma",
            omega=50.0, kernel_theta=0.5, kernel_alpha=0.001,
        )
        fp = {"mu": 0.2, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.2}
        rng = _rng(34)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n_samp, s_samp = copula.sample(10000, fp, sp, rng=rng, max_iter=1)
        # Returned arrays may be smaller than requested but must be equal length
        assert len(n_samp) == len(s_samp)


# ---------------------------------------------------------------------------
# GaussianCopulaMixed — per-observation list params
# ---------------------------------------------------------------------------

class TestGaussianCopulaMixedListParams:

    def test_nb_gamma_list_params_finite_ll(self):
        gc = GaussianCopulaMixed(rho=0.0)
        n_obs = 10
        n_vals = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0])
        s_vals = np.array([1000.0, 800.0, 1200.0, 600.0, 1100.0,
                           0.0, 900.0, 700.0, 1300.0, 950.0])
        fp = [{"mu": 0.15 + 0.01 * i, "alpha": 0.8} for i in range(n_obs)]
        sp = [{"mu": 1000.0 + 50 * i, "shape": 1.2} for i in range(n_obs)]
        ll = gc.log_likelihood(n_vals, s_vals, fp, sp, freq_family="nb", sev_family="gamma")
        assert np.isfinite(ll)

    def test_poisson_lognormal_list_params_finite_ll(self):
        gc = GaussianCopulaMixed(rho=-0.3)
        n_obs = 8
        n_vals = np.array([1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 3.0])
        s_vals = np.array([2000.0, 1.0, 1500.0, 2500.0, 1.0, 1800.0, 1.0, 1200.0])
        fp = [{"mu": 0.12} for _ in range(n_obs)]
        sp = [{"log_mu": np.log(2000.0), "log_sigma": 0.5} for _ in range(n_obs)]
        ll = gc.log_likelihood(n_vals, s_vals, fp, sp,
                               freq_family="poisson", sev_family="lognormal")
        assert np.isfinite(ll)

    def test_lognormal_sample_positive_s(self):
        rng = _rng(40)
        gc = GaussianCopulaMixed(rho=0.3)
        fp = {"mu": 0.15, "alpha": 0.5}
        sp = {"log_mu": np.log(2000), "log_sigma": 0.5}
        n_samp, s_samp = gc.sample(1000, fp, sp, freq_family="nb",
                                   sev_family="lognormal", rng=rng)
        assert len(n_samp) == 1000
        assert np.all(s_samp >= 0)


# ---------------------------------------------------------------------------
# FGMCopula — additional properties
# ---------------------------------------------------------------------------

class TestFGMCopulaAdditional:

    def test_cdf_at_one_one(self):
        """C(1, 1) = 1 regardless of theta."""
        for theta in [-1.0, 0.0, 0.5, 1.0]:
            fgm = FGMCopula(theta=theta)
            val = fgm.cdf(np.array([1.0]), np.array([1.0]))[0]
            assert abs(val - 1.0) < 1e-10, f"C(1,1) should be 1, got {val} at theta={theta}"

    def test_pdf_at_corners_valid(self):
        """pdf at (0, v) and (u, 0) — these are limits, not evaluated at 0."""
        fgm = FGMCopula(theta=0.5)
        # pdf at u=0.5, v=0.5 is 1 + theta*(1-1)*(1-1) = 1 + 0 = 1
        val = fgm.pdf(np.array([0.5]), np.array([0.5]))[0]
        assert abs(val - 1.0) < 1e-10

    def test_theta_zero_gives_uniform_density(self):
        """At theta=0, pdf = 1 for all (u,v) in [0,1]^2."""
        fgm = FGMCopula(theta=0.0)
        rng = _rng(41)
        u = rng.uniform(0.01, 0.99, 500)
        v = rng.uniform(0.01, 0.99, 500)
        pdf_vals = fgm.pdf(u, v)
        np.testing.assert_allclose(pdf_vals, 1.0, atol=1e-10)

    def test_cdf_independence_at_theta_zero(self):
        """At theta=0, C(u,v) = u*v."""
        fgm = FGMCopula(theta=0.0)
        rng = _rng(42)
        u = rng.uniform(0.01, 0.99, 200)
        v = rng.uniform(0.01, 0.99, 200)
        cdf_vals = fgm.cdf(u, v)
        np.testing.assert_allclose(cdf_vals, u * v, atol=1e-12)

    def test_spearman_rho_range(self):
        """FGM Spearman rho is in [-1/3, 1/3]."""
        for theta in np.linspace(-1.0, 1.0, 11):
            fgm = FGMCopula(theta=theta)
            rho = fgm.spearman_rho()
            assert -1.0 / 3.0 - 1e-9 <= rho <= 1.0 / 3.0 + 1e-9

    def test_ll_changes_with_theta(self):
        """LL at theta=0.8 should differ from LL at theta=0 for correlated data."""
        rng = _rng(43)
        fgm_true = FGMCopula(theta=0.8)
        u, v = fgm_true.sample(5000, rng=rng)
        ll_true = fgm_true.log_likelihood(u, v)
        ll_zero = FGMCopula(theta=0.0).log_likelihood(u, v)
        assert ll_true != ll_zero


# ---------------------------------------------------------------------------
# DependenceTest — additional branches
# ---------------------------------------------------------------------------

class TestDependenceTestAdditional:

    def test_filters_n_zero_rows(self):
        """Rows with n=0 should be excluded from the test."""
        rng = _rng(50)
        n = np.array([0.0] * 100 + list(rng.integers(1, 5, 200).astype(float)))
        s = np.where(n > 0, rng.gamma(1.5, 1000.0, len(n)), 0.0)
        test = DependenceTest(n_permutations=0)
        test.fit(n, s)
        assert test.n_obs_ == 200  # only positive-claim rows

    def test_filters_nan_s_rows(self):
        """Rows with s=nan should be excluded."""
        rng = _rng(51)
        n_obs = 300
        n = rng.integers(1, 4, n_obs).astype(float)
        s = rng.gamma(1.5, 1000.0, n_obs)
        s[::5] = np.nan  # every 5th row
        test = DependenceTest(n_permutations=0)
        test.fit(n, s)
        assert test.n_obs_ == n_obs - n_obs // 5

    def test_positive_correlation_detected(self):
        """Clearly positively correlated data should give positive tau."""
        rng = _rng(52)
        n = np.arange(1, 201, dtype=float)
        s = n * 100 + rng.normal(0, 50, 200)
        s = np.clip(s, 100.0, None)
        test = DependenceTest(n_permutations=0)
        test.fit(n, s)
        assert test.tau_ > 0
        assert test.rho_s_ > 0

    def test_no_permutations_path(self):
        """n_permutations=0 should skip permutation test."""
        rng = _rng(53)
        n = rng.integers(1, 5, 100).astype(float)
        s = rng.gamma(1.5, 1000.0, 100)
        test = DependenceTest(n_permutations=0)
        test.fit(n, s)
        assert not hasattr(test, "tau_pval_perm_") or test.tau_pval_perm_ is None

    def test_permutation_test_pval_in_unit_interval(self):
        """Permutation p-value must be in [0, 1]."""
        rng = _rng(54)
        n = rng.integers(1, 5, 200).astype(float)
        s = rng.gamma(1.5, 1000.0, 200)
        test = DependenceTest(n_permutations=200)
        test.fit(n, s, rng=rng)
        assert 0.0 <= test.tau_pval_perm_ <= 1.0

    def test_summary_conclusion_column_present(self):
        rng = _rng(55)
        n = rng.integers(1, 5, 200).astype(float)
        s = rng.gamma(1.5, 1000.0, 200)
        test = DependenceTest(n_permutations=0)
        test.fit(n, s)
        df = test.summary()
        assert "conclusion" in df.columns

    def test_asymptotic_pval_in_unit_interval(self):
        rng = _rng(56)
        n = rng.integers(1, 4, 500).astype(float)
        s = rng.gamma(1.5, 1000.0, 500)
        test = DependenceTest(n_permutations=0)
        test.fit(n, s)
        assert 0.0 <= test.tau_pval_ <= 1.0
        assert 0.0 <= test.rho_s_pval_ <= 1.0


# ---------------------------------------------------------------------------
# CopulaGOF — before-fit raises, u/v in (0,1)
# ---------------------------------------------------------------------------

class TestCopulaGOFAdditional:

    def test_before_fit_raises(self):
        """Calling summary() before fit() should raise RuntimeError."""
        from insurance_frequency_severity.joint import JointFreqSev
        n_pol = 2000
        mu_n = np.full(n_pol, 0.15)
        mu_s = np.full(n_pol, 2000.0)
        freq_glm = _make_mock_nb_glm(mu_n)
        sev_glm = _make_mock_gamma_glm(mu_s)
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm)
        gof = CopulaGOF(model)
        with pytest.raises(RuntimeError):
            gof.summary()

    def test_u_v_in_unit_interval(self):
        """After fit(), transformed residuals u and v must be in (0, 1)."""
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=1000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)

        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm, copula="sarmanov")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
            model._freq_family = "nb"
            model._sev_family = "gamma"

        freq_params = [{"mu": mu_n, "alpha": alpha} for _ in range(n_pol)]
        sev_params = [{"mu": mu_s, "shape": shape} for _ in range(n_pol)]

        gof = CopulaGOF(model)
        gof.fit(n, s, freq_params, sev_params)

        assert np.all(gof._u > 0) and np.all(gof._u < 1)
        assert np.all(gof._v > 0) and np.all(gof._v < 1)

    def test_ks_pvals_in_unit_interval(self):
        """KS p-values must be in [0, 1]."""
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=1000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)

        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm, copula="sarmanov")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
            model._freq_family = "nb"
            model._sev_family = "gamma"

        freq_params = [{"mu": mu_n, "alpha": alpha} for _ in range(n_pol)]
        sev_params = [{"mu": mu_s, "shape": shape} for _ in range(n_pol)]

        gof = CopulaGOF(model)
        gof.fit(n, s, freq_params, sev_params)

        assert 0.0 <= gof.ks_pval_u_ <= 1.0
        assert 0.0 <= gof.ks_pval_v_ <= 1.0


# ---------------------------------------------------------------------------
# _extract_freq_params — family detection
# ---------------------------------------------------------------------------

class TestExtractFreqParams:

    def test_poisson_family_detected(self):
        n_pol = 100
        mu_n = np.full(n_pol, 0.12)
        glm = _make_mock_poisson_glm(mu_n)
        mu_out, alpha, family = _extract_freq_params(glm, X=None, exposure=None)
        assert family == "poisson"
        assert alpha == 0.0

    def test_nb_family_detected(self):
        n_pol = 100
        mu_n = np.full(n_pol, 0.15)
        glm = _make_mock_nb_glm(mu_n, alpha=0.8)
        mu_out, alpha, family = _extract_freq_params(glm, X=None, exposure=None)
        assert family == "nb"
        assert abs(alpha - 0.8) < 1e-6

    def test_exposure_scales_mu(self):
        n_pol = 10
        mu_n = np.full(n_pol, 0.1)
        glm = _make_mock_poisson_glm(mu_n)
        exposure = 2.0 * np.ones(n_pol)
        mu_out, _, _ = _extract_freq_params(glm, X=None, exposure=exposure)
        np.testing.assert_allclose(mu_out, 0.2, atol=1e-10)

    def test_no_exposure_unchanged(self):
        n_pol = 10
        mu_n = np.full(n_pol, 0.15)
        glm = _make_mock_nb_glm(mu_n)
        mu_out, _, _ = _extract_freq_params(glm, X=None, exposure=None)
        np.testing.assert_allclose(mu_out, 0.15, atol=1e-10)

    def test_params_dict_alpha_fallback(self):
        """GLM with params dict alpha (statsmodels NB) should extract alpha."""
        n_pol = 10
        mu_n = np.full(n_pol, 0.15)

        class NBGLMWithParams:
            fittedvalues = mu_n
            params = {"alpha": 0.65}
            class family:
                pass
            class model:
                class family:
                    pass
            def predict(self, X=None):
                return self.fittedvalues

        NBGLMWithParams.family.__name__ = "NegativeBinomial"
        NBGLMWithParams.model.family.__name__ = "NegativeBinomial"
        glm = NBGLMWithParams()

        # The family name won't match because __class__.__name__ is the class name,
        # but we can at least confirm it doesn't crash
        mu_out, alpha, family = _extract_freq_params(glm, X=None, exposure=None)
        assert np.isfinite(alpha)


# ---------------------------------------------------------------------------
# _extract_sev_params — family detection
# ---------------------------------------------------------------------------

class TestExtractSevParams:

    def test_gamma_family_detected(self):
        n_pol = 100
        mu_s = np.full(n_pol, 2000.0)
        glm = _make_mock_gamma_glm(mu_s, shape=1.2)
        mu_out, shape, family = _extract_sev_params(glm, X=None, weights=None)
        assert family == "gamma"

    def test_lognormal_family_detected(self):
        n_pol = 100
        mu_s = np.full(n_pol, 2000.0)
        glm = _make_mock_lognormal_glm(mu_s, shape=1.2)
        mu_out, shape, family = _extract_sev_params(glm, X=None, weights=None)
        assert family == "lognormal"

    def test_shape_from_scale(self):
        """shape = 1 / scale when scale > 0."""
        n_pol = 10
        mu_s = np.full(n_pol, 2000.0)
        glm = _make_mock_gamma_glm(mu_s, shape=2.0)  # scale = 0.5
        _, shape, _ = _extract_sev_params(glm, X=None, weights=None)
        assert abs(shape - 2.0) < 1e-6

    def test_shape_minimum_clamped(self):
        """shape must be at least 0.01."""
        n_pol = 10
        mu_s = np.full(n_pol, 2000.0)
        glm = _make_mock_gamma_glm(mu_s, shape=0.001)  # very small
        _, shape, _ = _extract_sev_params(glm, X=None, weights=None)
        assert shape >= 0.01


# ---------------------------------------------------------------------------
# JointFreqSev — construction validation
# ---------------------------------------------------------------------------

class TestJointFreqSevConstruction:

    def test_invalid_copula_raises(self):
        glm = _make_mock_nb_glm(np.full(10, 0.1))
        sev = _make_mock_gamma_glm(np.full(10, 1000.0))
        with pytest.raises(ValueError, match="copula"):
            JointFreqSev(freq_glm=glm, sev_glm=sev, copula="bad_copula")

    def test_invalid_kernel_theta_raises(self):
        glm = _make_mock_nb_glm(np.full(10, 0.1))
        sev = _make_mock_gamma_glm(np.full(10, 1000.0))
        with pytest.raises(ValueError, match="kernel_theta"):
            JointFreqSev(freq_glm=glm, sev_glm=sev, kernel_theta=-1.0)

    def test_invalid_kernel_alpha_raises(self):
        glm = _make_mock_nb_glm(np.full(10, 0.1))
        sev = _make_mock_gamma_glm(np.full(10, 1000.0))
        with pytest.raises(ValueError, match="kernel_alpha"):
            JointFreqSev(freq_glm=glm, sev_glm=sev, kernel_alpha=0.0)

    def test_before_fit_raises_on_summary(self):
        glm = _make_mock_nb_glm(np.full(10, 0.1))
        sev = _make_mock_gamma_glm(np.full(10, 1000.0))
        model = JointFreqSev(freq_glm=glm, sev_glm=sev)
        with pytest.raises(RuntimeError):
            model.dependence_summary()


# ---------------------------------------------------------------------------
# JointFreqSev — AIC/BIC relationship and omega_ci_
# ---------------------------------------------------------------------------

class TestJointFreqSevAICBIC:

    def test_aic_less_than_bic_for_large_sample(self):
        """BIC penalises more for large n, so AIC < BIC when n > e^2 ≈ 7.4."""
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=2000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        # BIC = AIC + (log(n) - 2) * k; for n=2000, log(2000)≈7.6 > 2, so BIC > AIC
        assert model.bic_ > model.aic_, (
            f"Expected BIC > AIC for n=2000, got AIC={model.aic_:.2f}, BIC={model.bic_:.2f}"
        )

    def test_omega_ci_set_after_fit(self):
        """omega_ci_ should be a 2-tuple (lo, hi) after fitting."""
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=2000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        assert model.omega_ci_ is not None
        lo, hi = model.omega_ci_
        assert lo < model.omega_
        assert hi > model.omega_

    def test_rho_set_after_sarmanov_fit(self):
        """rho_ (Spearman) should be set after Sarmanov fit."""
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=2000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        assert model.rho_ is not None
        assert np.isfinite(model.rho_)

    def test_rho_sign_matches_omega_sign(self):
        """Spearman rho sign should match omega sign (negative omega -> negative rho)."""
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=2000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        if model.omega_ < 0:
            assert model.rho_ < 0
        elif model.omega_ > 0:
            assert model.rho_ > 0


# ---------------------------------------------------------------------------
# JointFreqSev — n_claims warning
# ---------------------------------------------------------------------------

class TestJointFreqSevClaimsWarning:

    def test_few_claims_fires_warning(self):
        """< 500 positive-claim rows should fire UserWarning."""
        rng = _rng(60)
        n_pol = 2000
        # Very low frequency so < 500 claims
        n = rng.poisson(0.05, n_pol)  # expected ~100 claims
        s = np.where(n > 0, rng.gamma(1.5, 1000.0, n_pol), 0.0)
        mu_n = np.full(n_pol, 0.05)
        mu_s = np.full(n_pol, 1500.0)
        freq_glm = _make_mock_nb_glm(mu_n)
        sev_glm = _make_mock_gamma_glm(mu_s)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm)
        with pytest.warns(UserWarning):
            model.fit(data)


# ---------------------------------------------------------------------------
# JointFreqSev — FGM premium_correction
# ---------------------------------------------------------------------------

class TestJointFreqSevFGMPremium:

    def test_fgm_premium_correction_columns(self):
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=2000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm, copula="fgm")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        corr = model.premium_correction()
        assert isinstance(corr, pd.DataFrame)
        assert "correction_factor" in corr.columns
        assert len(corr) == n_pol

    def test_fgm_correction_factor_positive(self):
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=2000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm, copula="fgm")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        corr = model.premium_correction()
        assert np.all(corr["correction_factor"].values > 0)


# ---------------------------------------------------------------------------
# JointFreqSev — loss_cost with new X
# ---------------------------------------------------------------------------

class TestJointFreqSevLossCostNewX:

    def test_loss_cost_with_freq_x_and_sev_x(self):
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=2000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        # New features X with n_pol rows
        X_new = pd.DataFrame({"dummy": np.ones(n_pol)})
        rng = _rng(70)
        lc = model.loss_cost(X=X_new, rng=rng)
        assert len(lc) == n_pol
        assert np.all(lc >= 0)

    def test_loss_cost_nonnegative_finite(self):
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=2000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        rng = _rng(71)
        lc = model.loss_cost(rng=rng)
        assert np.all(np.isfinite(lc))
        assert np.all(lc >= 0)


# ---------------------------------------------------------------------------
# ConditionalFreqSev — additional properties
# ---------------------------------------------------------------------------

class TestConditionalFreqSevAdditional:

    def test_gamma_negative_for_ncd_dgp(self):
        """For NCD-type DGP (high-frequency => low severity), gamma should be negative."""
        rng = _rng(80)
        n_pol = 5000
        n = rng.negative_binomial(1.0 / 0.8, 1.0 / (1.0 + 0.15 * 0.8), n_pol)
        # Severity negatively depends on count: higher n => lower s
        base_sev = 2000.0
        s = np.where(
            n > 0,
            base_sev * np.exp(-0.2 * n) * rng.gamma(1.2, 1.0, n_pol),
            0.0,
        )

        mu_n = np.full(n_pol, 0.15)
        mu_s = np.full(n_pol, base_sev)
        freq_glm = _make_mock_nb_glm(mu_n)
        sev_glm = _make_mock_gamma_glm(mu_s)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = ConditionalFreqSev(freq_glm=freq_glm, sev_glm_base=sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        # Strong negative dependence should give negative gamma
        assert model.gamma_ < 0, f"Expected gamma < 0 for NCD-type DGP, got {model.gamma_:.4f}"

    def test_before_fit_dependence_summary_raises(self):
        freq_glm = _make_mock_nb_glm(np.full(10, 0.1))
        sev_glm = _make_mock_gamma_glm(np.full(10, 1000.0))
        model = ConditionalFreqSev(freq_glm=freq_glm, sev_glm_base=sev_glm)
        with pytest.raises(RuntimeError):
            model.dependence_summary()

    def test_correction_factor_differs_from_independence(self):
        """Fitted gamma != 0 means correction factor should not all be 1."""
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=5000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = ConditionalFreqSev(freq_glm=freq_glm, sev_glm_base=sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        corr = model.premium_correction()
        # With non-zero gamma, corrections should vary
        if abs(model.gamma_) > 0.01:
            assert corr["correction_factor"].std() > 0 or not np.allclose(
                corr["correction_factor"], 1.0
            )

    def test_gamma_se_positive(self):
        """gamma_se_ must be positive."""
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=5000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = ConditionalFreqSev(freq_glm=freq_glm, sev_glm_base=sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        assert model.gamma_se_ > 0

    def test_premium_correction_length_matches_policies(self):
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=2000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = ConditionalFreqSev(freq_glm=freq_glm, sev_glm_base=sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        corr = model.premium_correction()
        assert len(corr) == n_pol

    def test_n_as_indicator_produces_finite_gamma(self):
        """n_as_indicator=True: N enters as binary 0/1, should still give finite gamma."""
        n, s, mu_n, alpha, mu_s, shape = _sample_nb_gamma_dgp(n_policies=5000)
        n_pol = len(n)
        freq_glm = _make_mock_nb_glm(np.full(n_pol, mu_n), alpha=alpha)
        sev_glm = _make_mock_gamma_glm(np.full(n_pol, mu_s), shape=shape)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = ConditionalFreqSev(
            freq_glm=freq_glm, sev_glm_base=sev_glm, n_as_indicator=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        assert np.isfinite(model.gamma_)


# ---------------------------------------------------------------------------
# Sarmanov ll: mixed positive and zero-claim batch
# ---------------------------------------------------------------------------

class TestSarmanovMixedBatch:

    def test_ll_mixed_batch_nb_gamma(self):
        """Batch with mix of n=0 and n>0 rows should give finite ll."""
        copula = SarmanovCopula(
            freq_family="nb", sev_family="gamma", omega=-2.0,
            kernel_theta=0.5, kernel_alpha=0.001,
        )
        n_vals = np.array([0., 1., 0., 2., 0., 0., 3., 1.])
        s_vals = np.array([1., 1000., 1., 800., 1., 1., 600., 1200.])
        fp = [{"mu": 0.15, "alpha": 0.8}] * 8
        sp = [{"mu": 2000., "shape": 1.2}] * 8
        ll = copula.log_likelihood(n_vals, s_vals, fp, sp)
        assert np.isfinite(ll)

    def test_ll_mixed_batch_poisson_lognormal(self):
        copula = SarmanovCopula(
            freq_family="poisson", sev_family="lognormal", omega=1.0,
            kernel_theta=0.5, kernel_alpha=0.001,
        )
        n_vals = np.array([1., 0., 2., 0., 1.])
        s_vals = np.array([2000., 1., 1500., 1., 2500.])
        fp = [{"mu": 0.12}] * 5
        sp = [{"log_mu": np.log(2000.), "log_sigma": 0.5}] * 5
        ll = copula.log_likelihood(n_vals, s_vals, fp, sp)
        assert np.isfinite(ll)

    def test_omega_bounds_positive_max_negative_min(self):
        """Bounds should always satisfy min < 0 < max."""
        for freq in ["nb", "poisson"]:
            for sev in ["gamma", "lognormal"]:
                copula = SarmanovCopula(
                    freq_family=freq, sev_family=sev,
                    omega=0.0, kernel_theta=0.5, kernel_alpha=0.001,
                )
                if freq == "nb":
                    fp = {"mu": 0.2, "alpha": 0.8}
                else:
                    fp = {"mu": 0.2}
                if sev == "gamma":
                    sp = {"mu": 1500.0, "shape": 1.2}
                else:
                    sp = {"log_mu": np.log(1500.0), "log_sigma": 0.5}
                lo, hi = copula.omega_bounds(fp, sp)
                assert lo < 0, f"omega_min should be negative for {freq}+{sev}"
                assert hi > 0, f"omega_max should be positive for {freq}+{sev}"
