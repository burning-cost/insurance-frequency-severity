"""Tests for dependent.premium — PurePremiumEstimator."""
import numpy as np
import pytest
import torch

from insurance_frequency_severity.dependent.premium import PurePremiumEstimator


class TestPurePremiumEstimatorMC:
    def _make_inputs(self, n=20):
        torch.manual_seed(0)
        log_lambda = torch.full((n,), np.log(0.1))    # λ=0.1 per policy
        log_mu = torch.full((n,), np.log(3000.0))     # μ=3000
        phi = torch.tensor([1.5])
        exposure = torch.ones(n)
        return log_lambda, log_mu, phi, exposure

    def test_output_shape(self):
        est = PurePremiumEstimator(n_mc=100, seed=0)
        log_lambda, log_mu, phi, exposure = self._make_inputs()
        pp = est.monte_carlo(log_lambda, log_mu, phi, exposure)
        assert pp.shape == (20,)

    def test_output_nonnegative(self):
        est = PurePremiumEstimator(n_mc=100, seed=0)
        pp = est.monte_carlo(*self._make_inputs())
        assert (pp >= 0).all()

    def test_reasonable_magnitude(self):
        """MC estimate of E[N·Y] = E[N]·E[Y] under independence ≈ 0.1*3000=300."""
        est = PurePremiumEstimator(n_mc=5000, seed=0)
        # Use large n so averaging over policies smooths MC noise
        log_lambda = torch.full((500,), np.log(0.1))
        log_mu = torch.full((500,), np.log(3000.0))
        phi = torch.tensor([1.0])
        exposure = torch.ones(500)
        pp = est.monte_carlo(log_lambda, log_mu, phi, exposure)
        mean_pp = pp.mean().item()
        # Should be roughly 300 ± 30%
        assert 200 < mean_pp < 450, f"MC estimate {mean_pp:.1f} far from expected 300"

    def test_higher_lambda_gives_higher_pp(self):
        est = PurePremiumEstimator(n_mc=500, seed=0)
        log_mu = torch.full((20,), np.log(3000.0))
        phi = torch.tensor([1.0])
        exposure = torch.ones(20)

        pp_low = est.monte_carlo(torch.full((20,), np.log(0.05)), log_mu, phi, exposure)
        pp_high = est.monte_carlo(torch.full((20,), np.log(0.3)), log_mu, phi, exposure)
        assert pp_high.mean() > pp_low.mean()

    def test_reproducible_with_same_seed(self):
        inputs = self._make_inputs()
        pp1 = PurePremiumEstimator(n_mc=100, seed=42).monte_carlo(*inputs)
        pp2 = PurePremiumEstimator(n_mc=100, seed=42).monte_carlo(*inputs)
        torch.testing.assert_close(pp1, pp2)

    def test_different_seeds_differ(self):
        inputs = self._make_inputs()
        pp1 = PurePremiumEstimator(n_mc=100, seed=1).monte_carlo(*inputs)
        pp2 = PurePremiumEstimator(n_mc=100, seed=2).monte_carlo(*inputs)
        assert not torch.allclose(pp1, pp2)

    def test_exposure_scaling(self):
        """Pure premium per unit exposure should be invariant to exposure level."""
        est = PurePremiumEstimator(n_mc=1000, seed=0)
        n = 20
        log_lambda_1 = torch.full((n,), np.log(0.1 * 1.0))
        log_lambda_2 = torch.full((n,), np.log(0.1 * 2.0))
        log_mu = torch.full((n,), np.log(3000.0))
        phi = torch.tensor([1.0])
        exp_1 = torch.ones(n)
        exp_2 = torch.full((n,), 2.0)

        pp1 = est.monte_carlo(log_lambda_1, log_mu, phi, exp_1)
        pp2 = est.monte_carlo(log_lambda_2, log_mu, phi, exp_2)
        # Both should be ~300; check means are within 50% of each other
        ratio = pp1.mean() / pp2.mean()
        assert 0.5 < ratio.item() < 2.0


class TestPurePremiumEstimatorAnalytical:
    def _make_inputs(self, n=10, lambda_val=0.1, mu_val=3000.0, gamma_val=-0.15):
        log_lambda = torch.full((n,), np.log(lambda_val))
        log_mu_base = torch.full((n,), np.log(mu_val))
        gamma = torch.tensor([gamma_val])
        exposure = torch.ones(n)
        return log_lambda, log_mu_base, gamma, exposure

    def test_output_shape(self):
        est = PurePremiumEstimator()
        pp = est.analytical(*self._make_inputs())
        assert pp.shape == (10,)

    def test_output_nonnegative(self):
        est = PurePremiumEstimator()
        pp = est.analytical(*self._make_inputs())
        assert (pp >= 0).all()

    def test_matches_ggs_formula(self):
        """E[Z|x] = exp(log_mu + γ) · exp(λ(eᵞ-1)) · λ for Poisson-Gamma."""
        lam = 0.08
        mu = 2000.0
        gamma = -0.15
        log_lambda = torch.tensor([np.log(lam)])
        log_mu_base = torch.tensor([np.log(mu)])
        gamma_t = torch.tensor([gamma])
        exposure = torch.tensor([1.0])

        est = PurePremiumEstimator()
        pp = est.analytical(log_lambda, log_mu_base, gamma_t, exposure)

        # Manual calculation of GGS formula
        eg = np.exp(gamma)
        expected = mu * eg * np.exp(lam * (eg - 1)) * lam
        assert pp.item() == pytest.approx(expected, rel=1e-4)

    def test_gamma_zero_equals_independence(self):
        """At γ=0, analytical reduces to E[N]·E[Y] = λ·μ."""
        lam = 0.1
        mu = 3000.0
        log_lambda = torch.tensor([np.log(lam)])
        log_mu_base = torch.tensor([np.log(mu)])
        gamma = torch.tensor([0.0])
        exposure = torch.tensor([1.0])

        est = PurePremiumEstimator()
        pp = est.analytical(log_lambda, log_mu_base, gamma, exposure)
        expected = lam * mu
        assert pp.item() == pytest.approx(expected, rel=1e-4)

    def test_negative_gamma_reduces_pp(self):
        """Negative γ (motor) should give lower pure premium than γ=0."""
        inputs_zero = self._make_inputs(gamma_val=0.0)
        inputs_neg = self._make_inputs(gamma_val=-0.15)
        est = PurePremiumEstimator()
        pp_zero = est.analytical(*inputs_zero)
        pp_neg = est.analytical(*inputs_neg)
        assert pp_neg.mean() < pp_zero.mean()


class TestPurePremiumConfidenceInterval:
    def test_output_shapes(self):
        est = PurePremiumEstimator(n_mc=100, seed=0)
        log_lambda = torch.full((5,), np.log(0.1))
        log_mu = torch.full((5,), np.log(3000.0))
        phi = torch.tensor([1.5])
        exposure = torch.ones(5)
        lo, mid, hi = est.confidence_interval(log_lambda, log_mu, phi, exposure)
        assert lo.shape == (5,)
        assert mid.shape == (5,)
        assert hi.shape == (5,)

    def test_ordering(self):
        est = PurePremiumEstimator(n_mc=200, seed=0)
        log_lambda = torch.full((10,), np.log(0.1))
        log_mu = torch.full((10,), np.log(3000.0))
        phi = torch.tensor([1.5])
        exposure = torch.ones(10)
        lo, mid, hi = est.confidence_interval(log_lambda, log_mu, phi, exposure)
        assert (lo <= mid).all()
        assert (mid <= hi).all()
