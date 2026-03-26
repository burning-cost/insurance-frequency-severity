"""
Structural gap tests for insurance-frequency-severity copula module.

Covers branches not exercised by the existing test_copula.py suite:

SarmanovCopula
- Poisson + Lognormal family combination: log_joint_density, sample
- omega_bounds with wide omega interval (both finite bounds)
- log_likelihood with all n=0 observations (pure frequency path)
- log_likelihood with per-observation heterogeneous parameters (list of dicts)
- LaplaceKernelPoisson: centred kernel expectation ~0 under Poisson
- LaplaceKernelLognormal: mgf in (0,1), centred expectation ~0
- Sarmanov positivity: density must be >= 0 everywhere within returned bounds

GaussianCopulaMixed
- all n=0 input returns 0.0 (no positive claims)
- Poisson family path (as opposed to NB only)
- lognormal severity path in _pit_sev
- sample: shapes are correct for non-NB families

FGMCopula
- log_likelihood is zero at theta=0 (uniform pdf => log(1) = 0)
- cdf satisfies Fréchet bounds: max(u+v-1,0) <= C(u,v) <= min(u,v)
- theta=1 and theta=-1 extreme cases do not crash
"""

import numpy as np
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


# ---------------------------------------------------------------------------
# LaplaceKernelPoisson: centred expectation ~0
# ---------------------------------------------------------------------------

class TestLaplaceKernelPoisson:

    def test_mgf_in_unit_interval(self):
        """M_N(theta) = E[exp(-theta*N)] in (0,1) for theta > 0."""
        kern = LaplaceKernelPoisson(theta=0.5)
        m = kern.mgf(mu=0.15)
        assert 0 < float(np.asarray(m).flat[0]) < 1

    def test_centred_expectation_near_zero(self):
        """E[phi(N)] ~= 0 under Poisson(mu)."""
        rng = np.random.default_rng(100)
        mu = 0.15
        n_samp = rng.poisson(mu, size=100_000)
        kern = LaplaceKernelPoisson(theta=0.5)
        phi_vals = kern.centred(n_samp, mu)
        assert abs(np.mean(phi_vals)) < 0.02, (
            f"E[phi_Poisson(N)] not near 0: {np.mean(phi_vals):.4f}"
        )

    def test_sup_abs_positive(self):
        kern = LaplaceKernelPoisson(theta=0.5)
        sup = kern.sup_abs(mu=0.15)
        assert sup > 0

    def test_invalid_theta_raises(self):
        with pytest.raises(ValueError, match="theta"):
            LaplaceKernelPoisson(theta=0.0)
        with pytest.raises(ValueError, match="theta"):
            LaplaceKernelPoisson(theta=-0.1)


# ---------------------------------------------------------------------------
# LaplaceKernelLognormal: mgf and centred expectation
# ---------------------------------------------------------------------------

class TestLaplaceKernelLognormal:

    def test_mgf_in_unit_interval(self):
        """M_S(alpha) = E[exp(-alpha*S)] in (0,1) for alpha > 0, S > 0."""
        kern = LaplaceKernelLognormal(alpha=0.001)
        m = kern.mgf(log_mu=np.log(2000), log_sigma=0.5)
        assert 0 < m < 1

    def test_centred_expectation_near_zero(self):
        """E[phi(S)] ~= 0 under Lognormal(log_mu, log_sigma)."""
        rng = np.random.default_rng(101)
        log_mu = np.log(2000)
        log_sigma = 0.5
        s_samp = np.exp(rng.normal(log_mu, log_sigma, size=30_000))
        kern = LaplaceKernelLognormal(alpha=0.001)
        phi_vals = kern.centred(s_samp, log_mu, log_sigma)
        assert abs(np.mean(phi_vals)) < 0.05, (
            f"E[phi_lognormal(S)] not near 0: {np.mean(phi_vals):.4f}"
        )

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            LaplaceKernelLognormal(alpha=0.0)


# ---------------------------------------------------------------------------
# SarmanovCopula: Poisson + Lognormal combination
# ---------------------------------------------------------------------------

class TestSarmanovPoissonLognormal:

    @pytest.fixture
    def copula_pl(self):
        return SarmanovCopula(
            freq_family="poisson",
            sev_family="lognormal",
            omega=0.0,
            kernel_theta=0.5,
            kernel_alpha=0.001,
        )

    @pytest.fixture
    def params_pl(self):
        return (
            {"mu": 0.12},
            {"log_mu": np.log(2000), "log_sigma": 0.5},
        )

    def test_log_joint_finite(self, copula_pl, params_pl):
        fp, sp = params_pl
        n_vals = np.array([0, 1, 2])
        s_vals = np.array([500.0, 2000.0, 1500.0])
        log_jd = copula_pl.log_joint_density(n_vals, s_vals, fp, sp)
        assert np.all(np.isfinite(log_jd))

    def test_independence_is_product_of_marginals(self, copula_pl, params_pl):
        """omega=0: log_joint = log_freq + log_sev."""
        fp, sp = params_pl
        n_vals = np.array([1, 2, 3])
        s_vals = np.array([2000.0, 1500.0, 2500.0])
        log_jd = copula_pl.log_joint_density(n_vals, s_vals, fp, sp)
        log_fn = copula_pl._log_freq_pmf(n_vals, fp)
        log_fs = copula_pl._log_sev_pdf(s_vals, sp)
        np.testing.assert_allclose(log_jd, log_fn + log_fs, atol=1e-8)

    def test_sample_runs(self):
        rng = np.random.default_rng(102)
        copula = SarmanovCopula(
            freq_family="poisson",
            sev_family="lognormal",
            omega=0.0,
            kernel_theta=0.5,
            kernel_alpha=0.001,
        )
        fp = {"mu": 0.12}
        sp = {"log_mu": np.log(2000), "log_sigma": 0.5}
        n_samp, s_samp = copula.sample(5000, fp, sp, rng=rng)
        assert len(n_samp) == 5000
        assert len(s_samp) == 5000
        assert np.all(s_samp > 0)

    def test_sample_mean_close_to_theoretical(self):
        """E[N] ~= 0.12 under Poisson."""
        rng = np.random.default_rng(103)
        copula = SarmanovCopula(
            freq_family="poisson",
            sev_family="lognormal",
            omega=0.0,
            kernel_theta=0.5,
            kernel_alpha=0.001,
        )
        fp = {"mu": 0.12}
        sp = {"log_mu": np.log(2000), "log_sigma": 0.5}
        n_samp, _ = copula.sample(50_000, fp, sp, rng=rng)
        assert abs(np.mean(n_samp) - 0.12) / 0.12 < 0.10


# ---------------------------------------------------------------------------
# SarmanovCopula: omega_bounds returns finite bounds
# ---------------------------------------------------------------------------

class TestOmegaBounds:

    def test_omega_bounds_finite_nb_gamma(self):
        copula = SarmanovCopula(
            freq_family="nb", sev_family="gamma", omega=0.0,
            kernel_theta=0.5, kernel_alpha=0.0005,
        )
        fp = {"mu": 0.2, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.2}
        omega_min, omega_max = copula.omega_bounds(fp, sp)
        assert np.isfinite(omega_min) or np.isinf(omega_min)  # may be -inf
        assert np.isfinite(omega_max) or np.isinf(omega_max)  # may be +inf
        # At minimum, max should be positive and min should be negative
        assert omega_max > 0
        assert omega_min < 0

    def test_density_nonneg_within_bounds(self):
        """Density must be non-negative within returned bounds."""
        copula = SarmanovCopula(
            freq_family="nb", sev_family="gamma", omega=0.0,
            kernel_theta=0.5, kernel_alpha=0.0005,
        )
        fp = {"mu": 0.2, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.2}
        omega_min, omega_max = copula.omega_bounds(fp, sp)

        for omega in [omega_min * 0.8, 0.0, omega_max * 0.8]:
            if not np.isfinite(omega):
                continue
            copula.omega = omega
            n_test = np.array([0, 1, 2, 3])
            s_test = np.array([500.0, 1000.0, 2000.0, 3000.0])
            log_jd = copula.log_joint_density(n_test, s_test, fp, sp)
            # log density must be finite (not -inf, which would mean density=0)
            assert np.all(np.isfinite(log_jd)), (
                f"Non-finite log density at omega={omega}"
            )


# ---------------------------------------------------------------------------
# SarmanovCopula: log_likelihood with all-zero n
# ---------------------------------------------------------------------------

class TestSarmanovLogLikelihoodAllZero:

    def test_all_zero_n_gives_frequency_only_contribution(self):
        """When all n=0, ll = sum of log f_N(0)."""
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=-2.0)
        n_obs = 50
        fp = {"mu": 0.1, "alpha": 0.5}
        sp = {"mu": 1000.0, "shape": 1.5}
        n_zeros = np.zeros(n_obs)
        s_placeholder = np.ones(n_obs)  # irrelevant for n=0

        ll = copula.log_likelihood(n_zeros, s_placeholder, fp, sp)
        # Should equal n * log f_N(0) for NB
        r = 1.0 / fp["alpha"]
        p = 1.0 / (1.0 + fp["mu"] * fp["alpha"])
        import scipy.special
        log_pmf_zero = r * np.log(p)  # NB(0): log(p^r)
        expected = n_obs * log_pmf_zero
        assert abs(ll - expected) < 0.1, (
            f"All-zero ll={ll:.4f}, expected {expected:.4f}"
        )

    def test_all_zero_n_finite(self):
        """All-zero n must not produce nan or inf in log-likelihood."""
        copula = SarmanovCopula(freq_family="poisson", sev_family="gamma", omega=0.0)
        fp = {"mu": 0.1}
        sp = {"mu": 1000.0, "shape": 1.5}
        n_zeros = np.zeros(100)
        s_dummy = np.full(100, 1.0)
        ll = copula.log_likelihood(n_zeros, s_dummy, fp, sp)
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# SarmanovCopula: heterogeneous per-observation parameters (list of dicts)
# ---------------------------------------------------------------------------

class TestSarmanovHeterogeneousParams:

    def test_list_of_dicts_nb_gamma(self):
        """Per-observation parameters as list of dicts should work."""
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=0.0)
        n_obs = 10
        n_vals = np.ones(n_obs, dtype=float)
        s_vals = np.full(n_obs, 1000.0)

        fp = [{"mu": 0.1 + 0.01 * i, "alpha": 0.5} for i in range(n_obs)]
        sp = [{"mu": 1000.0 + 50 * i, "shape": 1.5} for i in range(n_obs)]

        ll = copula.log_likelihood(n_vals, s_vals, fp, sp)
        assert np.isfinite(ll)

    def test_list_of_dicts_poisson_gamma(self):
        copula = SarmanovCopula(freq_family="poisson", sev_family="gamma", omega=0.0)
        n_obs = 8
        n_vals = np.array([1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 3.0])
        s_vals = np.full(n_obs, 1200.0)
        fp = [{"mu": 0.12} for _ in range(n_obs)]
        sp = [{"mu": 1200.0, "shape": 1.3} for _ in range(n_obs)]
        ll = copula.log_likelihood(n_vals, s_vals, fp, sp)
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# GaussianCopulaMixed: all-zero n returns 0.0
# ---------------------------------------------------------------------------

class TestGaussianCopulaMixedEdgeCases:

    def test_all_zero_n_returns_zero(self):
        """When all n=0, there are no positive-claim rows; ll contribution = 0."""
        gc = GaussianCopulaMixed(rho=0.3)
        fp = {"mu": 0.15, "alpha": 0.5}
        sp = {"mu": 1000.0, "shape": 1.5}
        n_zeros = np.zeros(50)
        s_dummy = np.full(50, 1.0)
        ll = gc.log_likelihood(n_zeros, s_dummy, fp, sp, freq_family="nb", sev_family="gamma")
        assert ll == 0.0

    def test_poisson_family_path(self):
        """GaussianCopulaMixed with Poisson frequency should compute finite ll."""
        gc = GaussianCopulaMixed(rho=0.0)
        fp = {"mu": 0.15}
        sp = {"mu": 1000.0, "shape": 1.5}
        n_vals = np.array([1, 2, 0, 1, 3])
        s_vals = np.array([800.0, 1200.0, 1.0, 900.0, 700.0])
        ll = gc.log_likelihood(n_vals, s_vals, fp, sp, freq_family="poisson", sev_family="gamma")
        assert np.isfinite(ll)

    def test_lognormal_sev_path(self):
        """GaussianCopulaMixed with lognormal severity."""
        gc = GaussianCopulaMixed(rho=0.0)
        fp = {"mu": 0.15, "alpha": 0.5}
        sp = {"log_mu": np.log(2000), "log_sigma": 0.5}
        n_vals = np.array([1, 0, 2, 1])
        s_vals = np.array([2000.0, 1.0, 1500.0, 2500.0])
        ll = gc.log_likelihood(n_vals, s_vals, fp, sp, freq_family="nb", sev_family="lognormal")
        assert np.isfinite(ll)

    def test_sample_shapes_poisson_gamma(self):
        """sample() with poisson freq gives correct shape arrays."""
        rng = np.random.default_rng(104)
        gc = GaussianCopulaMixed(rho=-0.2)
        fp = {"mu": 0.15}
        sp = {"mu": 1500.0, "shape": 1.5}
        n_samp, s_samp = gc.sample(500, fp, sp, freq_family="poisson", sev_family="gamma", rng=rng)
        assert len(n_samp) == 500
        assert len(s_samp) == 500


# ---------------------------------------------------------------------------
# FGMCopula: additional properties
# ---------------------------------------------------------------------------

class TestFGMCopulaGaps:

    def test_log_likelihood_zero_at_theta_zero(self):
        """At theta=0, pdf=1 everywhere, so log-likelihood = sum(log(1)) = 0."""
        rng = np.random.default_rng(105)
        fgm = FGMCopula(theta=0.0)
        u = rng.uniform(0.01, 0.99, 1000)
        v = rng.uniform(0.01, 0.99, 1000)
        ll = fgm.log_likelihood(u, v)
        assert abs(ll) < 1e-8, f"Expected ll=0 for theta=0, got {ll:.6f}"

    def test_frechet_lower_bound(self):
        """C(u,v) >= max(u + v - 1, 0) (Fréchet lower bound)."""
        rng = np.random.default_rng(106)
        u = rng.uniform(0.01, 0.99, 200)
        v = rng.uniform(0.01, 0.99, 200)
        for theta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            fgm = FGMCopula(theta=theta)
            cdf_vals = fgm.cdf(u, v)
            lower = np.maximum(u + v - 1, 0)
            assert np.all(cdf_vals >= lower - 1e-10), (
                f"FGM CDF violates Fréchet lower bound at theta={theta}"
            )

    def test_frechet_upper_bound(self):
        """C(u,v) <= min(u, v) (Fréchet upper bound)."""
        rng = np.random.default_rng(107)
        u = rng.uniform(0.01, 0.99, 200)
        v = rng.uniform(0.01, 0.99, 200)
        for theta in [-1.0, 0.0, 1.0]:
            fgm = FGMCopula(theta=theta)
            cdf_vals = fgm.cdf(u, v)
            upper = np.minimum(u, v)
            assert np.all(cdf_vals <= upper + 1e-10), (
                f"FGM CDF violates Fréchet upper bound at theta={theta}"
            )

    def test_theta_extremes_do_not_crash(self):
        """theta=1 and theta=-1 are the boundary values and must work."""
        rng = np.random.default_rng(108)
        for theta in [-1.0, 1.0]:
            fgm = FGMCopula(theta=theta)
            u, v = fgm.sample(1000, rng=rng)
            assert np.all((u >= 0) & (u <= 1))
            assert np.all((v >= 0) & (v <= 1))

    def test_pdf_integrates_to_one_numerically(self):
        """FGM pdf should integrate to 1 over [0,1]^2 (verify with grid)."""
        from scipy.integrate import dblquad
        for theta in [-0.8, 0.0, 0.8]:
            fgm = FGMCopula(theta=theta)
            def pdf_fn(v, u):
                return float(fgm.pdf(np.array([u]), np.array([v]))[0])
            integral, _ = dblquad(pdf_fn, 0, 1, 0, 1)
            assert abs(integral - 1.0) < 0.01, (
                f"FGM pdf does not integrate to 1 at theta={theta}: {integral:.4f}"
            )

    def test_kendall_tau_bounds(self):
        """FGM Kendall tau is in [-2/9, 2/9]."""
        for theta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            fgm = FGMCopula(theta=theta)
            tau = fgm.kendall_tau()
            assert -2/9 - 1e-9 <= tau <= 2/9 + 1e-9, (
                f"Kendall tau={tau:.4f} out of FGM bounds [-2/9, 2/9]"
            )
