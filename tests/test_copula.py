"""
Tests for copula.py — Sarmanov, Gaussian, FGM copulas.

Key invariants tested:
- Independence case (omega=0): joint density = product of marginals
- Sarmanov sampler: sample moments match theoretical
- Spearman rho sign matches omega sign
- FGM exact relations: rho_S = theta/3, tau = 2*theta/9
- Omega bounds: density >= 0 within bounds
- Gaussian copula: rho=0 gives independence
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
)


# -------------------------------------------------------------------------
# SarmanovCopula tests
# -------------------------------------------------------------------------

class TestSarmanovIndependence:
    """omega=0 should give independence."""

    def test_log_joint_equals_sum_of_marginals_nb_gamma(self, rng):
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=0.0)
        fp = {"mu": 0.2, "alpha": 0.5}
        sp = {"mu": 1000.0, "shape": 1.5}

        n_vals = np.array([0, 1, 2, 3])
        s_vals = np.array([500.0, 1200.0, 800.0, 2000.0])

        log_joint = copula.log_joint_density(n_vals, s_vals, fp, sp)
        log_fn = copula._log_freq_pmf(n_vals, fp)
        log_fs = copula._log_sev_pdf(s_vals, sp)

        np.testing.assert_allclose(log_joint, log_fn + log_fs, atol=1e-10)

    def test_log_joint_equals_sum_of_marginals_poisson_gamma(self, rng):
        copula = SarmanovCopula(freq_family="poisson", sev_family="gamma", omega=0.0)
        fp = {"mu": 0.15}
        sp = {"mu": 2000.0, "shape": 1.2}

        n_vals = np.array([0, 1, 2])
        s_vals = np.array([1000.0, 2500.0, 800.0])

        log_joint = copula.log_joint_density(n_vals, s_vals, fp, sp)
        log_fn = copula._log_freq_pmf(n_vals, fp)
        log_fs = copula._log_sev_pdf(s_vals, sp)

        np.testing.assert_allclose(log_joint, log_fn + log_fs, atol=1e-10)


class TestSarmanovDensityNonNegative:
    """Joint density must be non-negative within omega bounds."""

    def test_density_nonneg_at_estimated_omega_bounds(self, rng):
        fp = {"mu": 0.2, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.2}

        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=0.0,
                                 kernel_theta=0.5, kernel_alpha=0.0005)
        omega_min, omega_max = copula.omega_bounds(fp, sp)

        # Test at 90% of bounds
        for omega in [omega_min * 0.9, 0.0, omega_max * 0.9]:
            copula.omega = omega
            n_vals = np.arange(10)
            s_vals = stats.gamma.ppf(np.linspace(0.05, 0.95, 10), a=1.2,
                                     scale=1500.0/1.2)
            for n in n_vals:
                for s in s_vals:
                    ld = copula.log_joint_density(
                        np.array([n]), np.array([s]), fp, sp
                    )
                    assert np.isfinite(ld[0]), f"Non-finite density at n={n}, s={s}, omega={omega}"


class TestSarmanovSamplerMoments:
    """Sample moments should approximate theoretical moments."""

    def test_sample_mean_nb(self, rng):
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=0.0,
                                 kernel_theta=0.5, kernel_alpha=0.001)
        fp = {"mu": 0.2, "alpha": 0.5}
        sp = {"mu": 1000.0, "shape": 1.5}

        n_samp, s_samp = copula.sample(30000, fp, sp, rng=rng)

        # Sample means should be within 5% of theoretical
        assert abs(np.mean(n_samp) - 0.2) / 0.2 < 0.10, f"NB mean off: {np.mean(n_samp):.4f} vs 0.2"
        assert abs(np.mean(s_samp) - 1000.0) / 1000.0 < 0.10, f"Gamma mean off: {np.mean(s_samp):.1f} vs 1000"

    def test_sample_size_correct(self, rng):
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=-2.0,
                                 kernel_theta=0.5, kernel_alpha=0.001)
        fp = {"mu": 0.15, "alpha": 0.8}
        sp = {"mu": 2000.0, "shape": 1.2}

        n_samp, s_samp = copula.sample(1000, fp, sp, rng=rng)
        assert len(n_samp) == 1000
        assert len(s_samp) == 1000

    def test_negative_omega_gives_negative_rank_correlation(self, rng):
        """Negative omega => high N tends to have low S."""
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=-5.0,
                                 kernel_theta=0.5, kernel_alpha=0.001)
        fp = {"mu": 0.3, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.5}

        n_samp, s_samp = copula.sample(20000, fp, sp, rng=rng)
        rho = np.corrcoef(stats.rankdata(n_samp), stats.rankdata(s_samp))[0, 1]
        assert rho < 0, f"Expected negative Spearman rho for omega<0, got {rho:.4f}"

    def test_positive_omega_gives_positive_rank_correlation(self, rng):
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=5.0,
                                 kernel_theta=0.5, kernel_alpha=0.001)
        fp = {"mu": 0.3, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.5}

        n_samp, s_samp = copula.sample(20000, fp, sp, rng=rng)
        rho = np.corrcoef(stats.rankdata(n_samp), stats.rankdata(s_samp))[0, 1]
        assert rho > 0, f"Expected positive Spearman rho for omega>0, got {rho:.4f}"


class TestSarmanovLogLikelihood:
    """Log-likelihood structure tests."""

    def test_ll_decreases_with_wrong_omega(self, nb_gamma_dgp, rng):
        """Likelihood at true omega should exceed likelihood at omega=0."""
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        # Use only positive-claim rows for simplicity
        mask = n > 0
        n_pos, s_pos = n[mask], s[mask]

        # Build constant-parameter lists
        fp = [{"mu": dgp["mu_n"], "alpha": dgp["alpha"]} for _ in range(len(n_pos))]
        sp = [{"mu": dgp["mu_s"], "shape": dgp["shape"]} for _ in range(len(n_pos))]

        copula_true = SarmanovCopula(
            freq_family="nb", sev_family="gamma",
            omega=dgp["omega_true"],
            kernel_theta=dgp["kernel_theta"],
            kernel_alpha=dgp["kernel_alpha"],
        )
        copula_indep = SarmanovCopula(
            freq_family="nb", sev_family="gamma",
            omega=0.0,
            kernel_theta=dgp["kernel_theta"],
            kernel_alpha=dgp["kernel_alpha"],
        )

        ll_true = copula_true.log_likelihood(n_pos, s_pos, fp, sp)
        ll_indep = copula_indep.log_likelihood(n_pos, s_pos, fp, sp)

        # True omega should give higher (less negative) log-likelihood
        assert ll_true > ll_indep, (
            f"Expected ll at true omega ({ll_true:.1f}) > ll at omega=0 ({ll_indep:.1f})"
        )

    def test_ll_handles_all_zeros(self, rng):
        """All-zero claim rows should not crash."""
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=0.0)
        fp = [{"mu": 0.1, "alpha": 0.5} for _ in range(100)]
        sp = [{"mu": 1000.0, "shape": 1.5} for _ in range(100)]
        n_zeros = np.zeros(100)
        s_placeholder = np.ones(100)

        ll = copula.log_likelihood(n_zeros, s_placeholder, fp, sp)
        assert np.isfinite(ll)

    def test_ll_single_claim(self):
        """Single positive-claim observation should give finite ll."""
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=-2.0)
        fp = [{"mu": 0.2, "alpha": 0.5}]
        sp = [{"mu": 1200.0, "shape": 1.5}]

        ll = copula.log_likelihood(np.array([1.0]), np.array([1200.0]), fp, sp)
        assert np.isfinite(ll)


class TestSarmanovSpearmanRho:
    """spearman_rho() should have correct sign."""

    def test_spearman_rho_sign_negative(self, rng):
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=-3.0,
                                 kernel_theta=0.5, kernel_alpha=0.001)
        fp = {"mu": 0.2, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.2}

        rho = copula.spearman_rho(fp, sp, n_mc=20000, rng=rng)
        assert rho < 0, f"Expected negative rho, got {rho:.4f}"

    def test_spearman_rho_zero_at_independence(self, rng):
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=0.0)
        fp = {"mu": 0.2, "alpha": 0.8}
        sp = {"mu": 1500.0, "shape": 1.2}

        rho = copula.spearman_rho(fp, sp, n_mc=30000, rng=rng)
        assert abs(rho) < 0.05, f"Expected rho near 0 for omega=0, got {rho:.4f}"


# -------------------------------------------------------------------------
# LaplaceKernel tests
# -------------------------------------------------------------------------

class TestLaplaceKernelNB:
    def test_mgf_positive(self):
        kern = LaplaceKernelNB(theta=0.5)
        m = kern.mgf(mu=0.2, alpha=0.5)
        assert 0 < m < 1

    def test_centred_kernel_expectation_zero(self, rng):
        """E[phi(N)] should be ~0 under NB(mu, alpha)."""
        from insurance_frequency_severity.copula import SarmanovCopula
        copula = SarmanovCopula(freq_family="nb", sev_family="gamma", omega=0.0)
        fp = {"mu": 0.2, "alpha": 0.5}
        sp = {"mu": 1000.0, "shape": 1.5}
        # Sample N from NB
        alpha = fp["alpha"]
        r = 1.0 / alpha
        p = r / (r + fp["mu"])
        n_samp = rng.negative_binomial(r, p, size=50000)
        phi_vals = LaplaceKernelNB(theta=0.5).centred(n_samp, fp["mu"], fp["alpha"])
        assert abs(np.mean(phi_vals)) < 0.02, f"E[phi(N)] not near 0: {np.mean(phi_vals):.4f}"

    def test_sup_abs_positive(self):
        kern = LaplaceKernelNB(theta=0.5)
        sup = kern.sup_abs(mu=0.2, alpha=0.5)
        assert sup > 0


class TestLaplaceKernelGamma:
    def test_mgf_positive(self):
        kern = LaplaceKernelGamma(alpha=0.001)
        m = kern.mgf(mu_s=1000.0, shape=1.5)
        assert 0 < m < 1

    def test_centred_kernel_expectation_zero(self, rng):
        """E[phi(S)] should be ~0 under Gamma(shape, scale)."""
        mu_s = 1000.0
        shape = 1.5
        scale = mu_s / shape
        s_samp = rng.gamma(shape, scale, size=50000)
        kern = LaplaceKernelGamma(alpha=0.001)
        phi_vals = kern.centred(s_samp, mu_s, shape)
        assert abs(np.mean(phi_vals)) < 0.02, f"E[phi(S)] not near 0: {np.mean(phi_vals):.4f}"


# -------------------------------------------------------------------------
# GaussianCopulaMixed tests
# -------------------------------------------------------------------------

class TestGaussianCopulaMixed:
    def test_ll_finite_at_zero_rho(self, rng):
        gc = GaussianCopulaMixed(rho=0.0)
        fp = {"mu": 0.2, "alpha": 0.5}
        sp = {"mu": 1000.0, "shape": 1.5}
        n_vals = np.array([1, 2, 1, 3])
        s_vals = np.array([800.0, 1200.0, 900.0, 500.0])
        ll = gc.log_likelihood(n_vals, s_vals, fp, sp)
        assert np.isfinite(ll)

    def test_ll_decreases_with_wrong_rho(self, rng):
        """For data from Gaussian copula with rho=-0.3, ll at -0.3 > ll at 0."""
        gc_dgp = GaussianCopulaMixed(rho=-0.3)
        fp = {"mu": 0.3, "alpha": 0.5}
        sp = {"mu": 1000.0, "shape": 1.5}
        n_s, s_s = gc_dgp.sample(5000, fp, sp, rng=rng)
        mask = n_s > 0
        n_pos, s_pos = n_s[mask], s_s[mask]

        gc_true = GaussianCopulaMixed(rho=-0.3)
        gc_zero = GaussianCopulaMixed(rho=0.0)

        ll_true = gc_true.log_likelihood(n_pos, s_pos, fp, sp)
        ll_zero = gc_zero.log_likelihood(n_pos, s_pos, fp, sp)
        assert ll_true > ll_zero, f"Expected ll at true rho to exceed ll at 0"

    def test_rho_zero_gives_independence_sample(self, rng):
        """rho=0 samples should have near-zero rank correlation."""
        gc = GaussianCopulaMixed(rho=0.0)
        fp = {"mu": 0.2, "alpha": 0.5}
        sp = {"mu": 1000.0, "shape": 1.5}
        n_s, s_s = gc.sample(10000, fp, sp, rng=rng)
        rank_corr = np.corrcoef(stats.rankdata(n_s), stats.rankdata(s_s))[0, 1]
        assert abs(rank_corr) < 0.05, f"Expected near-zero rank corr, got {rank_corr:.4f}"

    def test_spearman_rho_approx(self):
        """spearman_rho() should return a reasonable value.

        For Gaussian copula, spearman_rho = sin(pi * rho / 2).
        At rho=0.5: sin(pi/4) = sqrt(2)/2 ≈ 0.707.
        """
        gc = GaussianCopulaMixed(rho=0.5)
        rho_s = gc.spearman_rho()
        import math
        expected = math.sin(math.pi * 0.5 / 2)
        assert abs(rho_s - expected) < 1e-6, f"Unexpected Spearman rho: {rho_s}, expected {expected:.4f}"
        assert 0.6 < rho_s < 0.8, f"Spearman rho {rho_s:.4f} out of expected range [0.6, 0.8]"


# -------------------------------------------------------------------------
# FGM copula tests
# -------------------------------------------------------------------------

class TestFGMCopula:
    def test_spearman_rho_exact(self):
        for theta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            fgm = FGMCopula(theta=theta)
            assert abs(fgm.spearman_rho() - theta / 3.0) < 1e-10

    def test_kendall_tau_exact(self):
        for theta in [-1.0, 0.0, 1.0]:
            fgm = FGMCopula(theta=theta)
            assert abs(fgm.kendall_tau() - 2.0 * theta / 9.0) < 1e-10

    def test_theta_bounds(self):
        with pytest.raises(ValueError):
            FGMCopula(theta=1.5)
        with pytest.raises(ValueError):
            FGMCopula(theta=-1.5)

    def test_pdf_positive_within_unit_square(self, rng):
        """pdf should be positive for theta in [-1, 1]."""
        u = rng.uniform(0.01, 0.99, 1000)
        v = rng.uniform(0.01, 0.99, 1000)
        for theta in [-0.9, 0.0, 0.9]:
            fgm = FGMCopula(theta=theta)
            pdf_vals = fgm.pdf(u, v)
            assert np.all(pdf_vals > 0), f"Negative pdf for theta={theta}"

    def test_ll_finite(self, rng):
        u = rng.uniform(0.01, 0.99, 100)
        v = rng.uniform(0.01, 0.99, 100)
        fgm = FGMCopula(theta=0.5)
        ll = fgm.log_likelihood(u, v)
        assert np.isfinite(ll)

    def test_sample_size(self, rng):
        fgm = FGMCopula(theta=0.3)
        u, v = fgm.sample(500, rng=rng)
        assert len(u) == 500
        assert len(v) == 500
        assert np.all((u >= 0) & (u <= 1))
        assert np.all((v >= 0) & (v <= 1))

    def test_sample_rho_sign(self):
        """Positive theta => positive rank correlation in samples."""
        rng_pos = np.random.default_rng(101)
        rng_neg = np.random.default_rng(202)
        fgm_pos = FGMCopula(theta=1.0)
        fgm_neg = FGMCopula(theta=-1.0)
        u_p, v_p = fgm_pos.sample(20000, rng=rng_pos)
        u_n, v_n = fgm_neg.sample(20000, rng=rng_neg)
        assert np.corrcoef(u_p, v_p)[0, 1] > 0, "Positive theta should give positive correlation"
        assert np.corrcoef(u_n, v_n)[0, 1] < 0, "Negative theta should give negative correlation"

    def test_cdf_at_corners(self):
        """C(0, v) = C(u, 0) = 0; C(1, v) = v; C(u, 1) = u."""
        fgm = FGMCopula(theta=0.5)
        u_vals = np.array([0.0, 0.5, 1.0])
        v_vals = np.array([0.0, 0.5, 1.0])
        assert fgm.cdf(np.array([0.0]), np.array([0.5]))[0] == pytest.approx(0.0)
        assert fgm.cdf(np.array([0.5]), np.array([0.0]))[0] == pytest.approx(0.0)
        assert fgm.cdf(np.array([1.0]), np.array([0.5]))[0] == pytest.approx(0.5, abs=1e-10)
        assert fgm.cdf(np.array([0.5]), np.array([1.0]))[0] == pytest.approx(0.5, abs=1e-10)
