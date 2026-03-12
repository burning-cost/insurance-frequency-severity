"""
Tests for joint.py — JointFreqSev and ConditionalFreqSev.

Key tests:
- IFM recovers omega sign correctly from known DGP
- Independence (omega=0) gives correction factor of 1.0
- Negative omega gives correction < 1 on average
- premium_correction() returns valid DataFrame
- loss_cost() shape matches input
- dependence_summary() returns expected columns
- ConditionalFreqSev.fit() runs without error
- Small-sample warnings fire appropriately
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_frequency_severity.joint import JointFreqSev, ConditionalFreqSev


# -------------------------------------------------------------------------
# JointFreqSev — Sarmanov copula
# -------------------------------------------------------------------------

class TestJointFreqSevFit:
    def test_fit_runs_sarmanov(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = JointFreqSev(
            freq_glm=mock_freq_glm,
            sev_glm=mock_sev_glm,
            copula="sarmanov",
            kernel_theta=dgp["kernel_theta"],
            kernel_alpha=dgp["kernel_alpha"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, n_col="claim_count", s_col="avg_severity", rng=rng)

        assert model.omega_ is not None
        assert np.isfinite(model.omega_)
        assert model.aic_ is not None
        assert model.bic_ is not None

    def test_omega_sign_matches_dgp(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """IFM omega should have the same sign as the true omega."""
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = JointFreqSev(
            freq_glm=mock_freq_glm,
            sev_glm=mock_sev_glm,
            copula="sarmanov",
            kernel_theta=dgp["kernel_theta"],
            kernel_alpha=dgp["kernel_alpha"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        # true omega is negative
        assert model.omega_ < 0, f"Expected negative omega, got {model.omega_:.4f}"

    def test_fit_independence_dgp_omega_near_zero(self, poisson_gamma_dgp, rng):
        """For independence DGP, fitted omega should be near zero."""
        dgp = poisson_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        n_pol = len(n)
        mu_n = np.full(n_pol, dgp["mu_n"])
        mu_s = np.full(n_pol, dgp["mu_s"])
        shape = dgp["shape"]

        class MockPoissGLM:
            def __init__(self):
                self.fittedvalues = mu_n
                class Fam:
                    pass
                self.family = Fam()
                self.family.__class__.__name__ = "Poisson"
                class Model:
                    pass
                self.model = Model()
                self.model.family = Fam()
                self.model.family.__class__.__name__ = "Poisson"
            def predict(self, X=None):
                if X is not None:
                    return np.full(len(X), float(mu_n[0]))
                return self.fittedvalues

        class MockGammaGLM2:
            def __init__(self):
                self.fittedvalues = mu_s
                self.scale = 1.0 / shape
                class Fam:
                    pass
                self.family = Fam()
                self.family.__class__.__name__ = "Gamma"
                class Model:
                    pass
                self.model = Model()
                self.model.family = Fam()
                self.model.family.__class__.__name__ = "Gamma"
            def predict(self, X=None):
                if X is not None:
                    return np.full(len(X), float(mu_s[0]))
                return self.fittedvalues

        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(
            freq_glm=MockPoissGLM(),
            sev_glm=MockGammaGLM2(),
            copula="sarmanov",
            kernel_theta=0.5,
            kernel_alpha=0.001,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        # omega should be near 0 (not necessarily exactly, but small)
        assert abs(model.omega_) < 10.0, f"Omega too large for independence DGP: {model.omega_:.4f}"


class TestJointFreqSevPremiumCorrection:
    def test_correction_shape(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = JointFreqSev(
            freq_glm=mock_freq_glm,
            sev_glm=mock_sev_glm,
            copula="sarmanov",
            kernel_theta=dgp["kernel_theta"],
            kernel_alpha=dgp["kernel_alpha"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        corrections = model.premium_correction()
        assert isinstance(corrections, pd.DataFrame)
        expected_cols = {"mu_n", "mu_s", "mu_ns_independent", "mu_ns_joint",
                         "correction_factor", "premium_independent", "premium_joint"}
        assert expected_cols.issubset(set(corrections.columns))
        assert len(corrections) == len(n)

    def test_correction_negative_dependence(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """Negative omega -> average correction factor < 1."""
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = JointFreqSev(
            freq_glm=mock_freq_glm,
            sev_glm=mock_sev_glm,
            copula="sarmanov",
            kernel_theta=dgp["kernel_theta"],
            kernel_alpha=dgp["kernel_alpha"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        corrections = model.premium_correction()
        mean_cf = float(corrections["correction_factor"].mean())

        # With negative omega (high N -> low S), joint < independent
        assert mean_cf < 1.0, f"Expected mean correction < 1 for negative dependence, got {mean_cf:.4f}"

    def test_independence_omega_zero_correction_near_one(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm):
        """Manually setting omega=0 should give correction = 1.0 exactly."""
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = JointFreqSev(
            freq_glm=mock_freq_glm,
            sev_glm=mock_sev_glm,
            copula="sarmanov",
            kernel_theta=dgp["kernel_theta"],
            kernel_alpha=dgp["kernel_alpha"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)
        # Override omega to 0
        model.omega_ = 0.0

        corrections = model.premium_correction()
        np.testing.assert_allclose(
            corrections["correction_factor"].values,
            1.0,
            atol=1e-6,
            err_msg="omega=0 should give correction_factor=1.0 everywhere",
        )

    def test_correction_values_finite(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = JointFreqSev(
            freq_glm=mock_freq_glm,
            sev_glm=mock_sev_glm,
            copula="sarmanov",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        corr = model.premium_correction()
        assert np.all(np.isfinite(corr["correction_factor"].values))
        assert np.all(corr["correction_factor"].values > 0)


class TestJointFreqSevLossCost:
    def test_loss_cost_shape(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = JointFreqSev(
            freq_glm=mock_freq_glm,
            sev_glm=mock_sev_glm,
            copula="sarmanov",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        lc = model.loss_cost(X=None, rng=rng)
        assert len(lc) == len(n)
        assert np.all(lc >= 0)
        assert np.all(np.isfinite(lc))


class TestJointFreqSevDependenceSummary:
    def test_summary_columns(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = JointFreqSev(
            freq_glm=mock_freq_glm,
            sev_glm=mock_sev_glm,
            copula="sarmanov",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        summary = model.dependence_summary()
        assert "omega" in summary.columns
        assert "spearman_rho" in summary.columns
        assert "aic" in summary.columns
        assert "bic" in summary.columns
        assert "n_policies" in summary.columns

    def test_summary_before_fit_raises(self, mock_freq_glm, mock_sev_glm):
        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm)
        with pytest.raises(RuntimeError):
            model.dependence_summary()


class TestJointFreqSevGaussian:
    def test_fit_gaussian_runs(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, copula="gaussian")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        assert model.omega_ is not None
        assert -1.0 < model.omega_ < 1.0

    def test_gaussian_summary_has_rho(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, copula="gaussian")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        summary = model.dependence_summary()
        assert "rho" in summary.columns


class TestJointFreqSevFGM:
    def test_fit_fgm_runs(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, copula="fgm")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        assert model.omega_ is not None
        assert -1.0 <= model.omega_ <= 1.0


class TestJointFreqSevWarnings:
    def test_small_sample_warning(self, rng):
        """< 1000 policies should trigger UserWarning."""
        n_small = 200
        n = np.zeros(n_small, dtype=int)
        n[:20] = 1
        s = np.where(n > 0, 1000.0, 0.0)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        mu_n = np.full(n_small, 0.1)
        mu_s = np.full(n_small, 1000.0)

        class TinyFreqGLM:
            fittedvalues = mu_n
            scale = 1.0

            class family:
                alpha = 1.0
            class model:
                class family:
                    alpha = 1.0
            def predict(self, X=None):
                return self.fittedvalues

        class TinySevGLM:
            fittedvalues = mu_s
            scale = 1.0

            class family:
                pass
            class model:
                class family:
                    pass
            def predict(self, X=None):
                return self.fittedvalues

        model = JointFreqSev(freq_glm=TinyFreqGLM(), sev_glm=TinySevGLM())
        with pytest.warns(UserWarning, match="1,000"):
            model.fit(data, rng=rng)


# -------------------------------------------------------------------------
# ConditionalFreqSev
# -------------------------------------------------------------------------

class TestConditionalFreqSev:
    def test_fit_runs(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """ConditionalFreqSev.fit() should run and produce a gamma estimate."""
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = ConditionalFreqSev(freq_glm=mock_freq_glm, sev_glm_base=mock_sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)

        assert model.gamma_ is not None
        assert np.isfinite(model.gamma_)
        assert model.gamma_se_ is not None

    def test_premium_correction_runs(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = ConditionalFreqSev(freq_glm=mock_freq_glm, sev_glm_base=mock_sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)

        corr = model.premium_correction()
        assert isinstance(corr, pd.DataFrame)
        assert "correction_factor" in corr.columns
        assert len(corr) == len(n)
        assert np.all(np.isfinite(corr["correction_factor"].values))

    def test_dependence_summary(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = ConditionalFreqSev(freq_glm=mock_freq_glm, sev_glm_base=mock_sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)

        summary = model.dependence_summary()
        assert "gamma" in summary.columns
        assert "gamma_se" in summary.columns

    def test_before_fit_raises(self, mock_freq_glm, mock_sev_glm):
        model = ConditionalFreqSev(freq_glm=mock_freq_glm, sev_glm_base=mock_sev_glm)
        with pytest.raises(RuntimeError):
            model.premium_correction()

    def test_n_as_indicator(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = ConditionalFreqSev(
            freq_glm=mock_freq_glm,
            sev_glm_base=mock_sev_glm,
            n_as_indicator=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)

        assert model.gamma_ is not None
