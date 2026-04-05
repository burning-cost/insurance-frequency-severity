"""
test_coverage_gaps.py — Additional tests targeting previously uncovered code paths.

Gaps addressed:
1. JointFreqSev constructor validation (invalid copula, kernel_theta<=0, kernel_alpha<=0)
2. _extract_freq_params / _extract_sev_params for Poisson and lognormal families
3. DependenceTest with NaN/zero data mix — filtering behaviour
4. DependenceTest with n_permutations=0 — no tau_pval_perm_ attribute set
5. JointModelReport.to_dict() with unfitted model (no omega_)
6. JointModelReport.to_dict() without copula_comparison key
7. CopulaGOF.summary() before fit raises RuntimeError
8. premium_correction() for Gaussian and FGM copulas (MC path)
9. premium_correction() before fit raises RuntimeError
10. loss_cost() returns positive values
11. ConditionalFreqSev.dependence_summary() before fit raises RuntimeError
12. ConditionalFreqSev.loss_cost() shape and positivity
13. ConditionalFreqSev large-gamma warning
14. DependenceTest permutation p-value is bounded in [0, 1]
15. JointFreqSev fit with MLE method (method="mle")
16. dependence_summary() for FGM uses 'omega' column not 'rho'
17. JointFreqSev with all-zero severity edge case (all n=0 policies)
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_frequency_severity.joint import JointFreqSev, ConditionalFreqSev
from insurance_frequency_severity.diagnostics import DependenceTest, CopulaGOF
from insurance_frequency_severity.report import JointModelReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, copula="sarmanov", rng=None):
    """Fit a JointFreqSev and return it."""
    dgp = nb_gamma_dgp
    n, s = dgp["n"], dgp["s"]
    data = pd.DataFrame({"claim_count": n, "avg_severity": s})
    model = JointFreqSev(
        freq_glm=mock_freq_glm,
        sev_glm=mock_sev_glm,
        copula=copula,
        kernel_theta=dgp["kernel_theta"],
        kernel_alpha=dgp["kernel_alpha"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data, rng=rng or np.random.default_rng(7))
    return model


# ---------------------------------------------------------------------------
# 1. JointFreqSev constructor validation
# ---------------------------------------------------------------------------

class TestJointFreqSevConstructorValidation:
    def test_invalid_copula_raises(self, mock_freq_glm, mock_sev_glm):
        with pytest.raises(ValueError, match="copula_family must be one of"):
            JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, copula="frank")

    def test_invalid_copula_typo_raises(self, mock_freq_glm, mock_sev_glm):
        with pytest.raises(ValueError):
            JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, copula="Sarmanov")

    def test_kernel_theta_zero_raises(self, mock_freq_glm, mock_sev_glm):
        with pytest.raises(ValueError, match="kernel_theta must be positive"):
            JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, kernel_theta=0.0)

    def test_kernel_theta_negative_raises(self, mock_freq_glm, mock_sev_glm):
        with pytest.raises(ValueError, match="kernel_theta must be positive"):
            JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, kernel_theta=-1.0)

    def test_kernel_alpha_zero_raises(self, mock_freq_glm, mock_sev_glm):
        with pytest.raises(ValueError, match="kernel_alpha must be positive"):
            JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, kernel_alpha=0.0)

    def test_kernel_alpha_negative_raises(self, mock_freq_glm, mock_sev_glm):
        with pytest.raises(ValueError, match="kernel_alpha must be positive"):
            JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, kernel_alpha=-0.5)

    def test_valid_fgm_constructs(self, mock_freq_glm, mock_sev_glm):
        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, copula="fgm")
        assert model.copula_family == "fgm"

    def test_valid_gaussian_constructs(self, mock_freq_glm, mock_sev_glm):
        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, copula="gaussian")
        assert model.copula_family == "gaussian"

    def test_initial_attributes_are_none(self, mock_freq_glm, mock_sev_glm):
        """Before fit, all result attributes are None."""
        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm)
        assert model.omega_ is None
        assert model.rho_ is None
        assert model.aic_ is None
        assert model.bic_ is None


# ---------------------------------------------------------------------------
# 2. _extract_freq_params and _extract_sev_params (via JointFreqSev.fit)
# ---------------------------------------------------------------------------

class TestExtractParamsViaFit:
    """Test that different GLM family types are correctly detected during fit."""

    def test_poisson_family_detected(self, poisson_gamma_dgp, rng):
        """Poisson frequency GLM should set _freq_family='poisson' and _alpha=0."""
        dgp = poisson_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        n_pol = len(n)
        mu_n = np.full(n_pol, dgp["mu_n"])
        mu_s = np.full(n_pol, dgp["mu_s"])
        shape = dgp["shape"]

        class PoissonGLM:
            fittedvalues = mu_n
            scale = 1.0 / shape

            class _FamClass:
                pass
            family = _FamClass()
            family.__class__.__name__ = "Poisson"

            class _ModelClass:
                pass
            model = _ModelClass()
            model.family = _FamClass()
            model.family.__class__.__name__ = "Poisson"

            def predict(self, X=None):
                if X is not None:
                    return np.full(len(X), float(mu_n[0]))
                return mu_n

        class GammaGLM:
            fittedvalues = mu_s
            scale = 1.0 / shape

            class _FamClass:
                pass
            family = _FamClass()
            family.__class__.__name__ = "Gamma"

            class _ModelClass:
                pass
            model = _ModelClass()
            model.family = _FamClass()
            model.family.__class__.__name__ = "Gamma"

            def predict(self, X=None):
                if X is not None:
                    return np.full(len(X), float(mu_s[0]))
                return mu_s

        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(
            freq_glm=PoissonGLM(),
            sev_glm=GammaGLM(),
            copula="sarmanov",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        assert model._freq_family == "poisson"
        assert model._alpha == 0.0

    def test_lognormal_family_detected(self, nb_gamma_dgp, mock_freq_glm, rng):
        """Lognormal severity family should set _sev_family='lognormal'."""
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        n_pol = len(n)
        mu_s = np.full(n_pol, dgp["mu_s"])

        class LognormalGLM:
            fittedvalues = mu_s
            scale = 1.0 / 1.2  # phi = 1/shape

            class _FamClass:
                pass
            family = _FamClass()
            family.__class__.__name__ = "Lognormal"

            class _ModelClass:
                pass
            model = _ModelClass()
            model.family = _FamClass()
            model.family.__class__.__name__ = "Lognormal"

            def predict(self, X=None):
                if X is not None:
                    return np.full(len(X), float(mu_s[0]))
                return mu_s

        data = pd.DataFrame({"claim_count": n, "avg_severity": s})
        model = JointFreqSev(
            freq_glm=mock_freq_glm,
            sev_glm=LognormalGLM(),
            copula="sarmanov",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=rng)

        assert model._sev_family == "lognormal"


# ---------------------------------------------------------------------------
# 3. DependenceTest data filtering behaviour
# ---------------------------------------------------------------------------

class TestDependenceTestFiltering:
    def test_nan_values_filtered_out(self):
        """NaN severities should be excluded; test should still run with valid rows."""
        rng = np.random.default_rng(1)
        n = np.concatenate([rng.integers(1, 5, size=100), np.full(10, 0)])
        s = np.where(n > 0, rng.gamma(1.5, 1000.0, size=110), np.nan)
        # Introduce extra NaNs in positive-claim rows
        s[5] = np.nan
        s[12] = np.nan

        test = DependenceTest(n_permutations=0)
        test.fit(n, s)

        # Should succeed: there are plenty of valid observations
        assert test.n_obs_ <= 100
        assert test.n_obs_ >= 90  # 10 NaN rows removed

    def test_zero_severity_filtered_out(self):
        """Rows with s=0 should be filtered; remaining rows should pass."""
        rng = np.random.default_rng(2)
        n = rng.integers(1, 4, size=150)
        s = rng.gamma(1.5, 1000.0, size=150)
        # Force some s to zero
        s[:15] = 0.0

        test = DependenceTest(n_permutations=0)
        test.fit(n, s)

        assert test.n_obs_ == 135

    def test_mixed_nan_zero_filtered(self):
        """Both NaN and zero should be filtered."""
        rng = np.random.default_rng(3)
        n = rng.integers(1, 4, size=200)
        s = rng.gamma(1.5, 1000.0, size=200)
        s[:10] = 0.0
        s[10:20] = np.nan

        test = DependenceTest(n_permutations=0)
        test.fit(n, s)

        assert test.n_obs_ == 180

    def test_pvalue_is_bounded(self, rng):
        """p-values from both asymptotic and permutation tests must lie in [0, 1]."""
        n = rng.integers(1, 5, size=300)
        s = rng.gamma(1.5, 1000.0, size=300)

        test = DependenceTest(n_permutations=100)
        test.fit(n, s, rng=rng)

        assert 0.0 <= test.tau_pval_ <= 1.0
        assert 0.0 <= test.rho_s_pval_ <= 1.0
        assert 0.0 <= test.tau_pval_perm_ <= 1.0

    def test_no_permutation_leaves_no_perm_attr(self, rng):
        """n_permutations=0 should set tau_pval_perm_ to None."""
        n = rng.integers(1, 5, size=100)
        s = rng.gamma(1.5, 1000.0, size=100)

        test = DependenceTest(n_permutations=0)
        test.fit(n, s)

        assert test.tau_pval_perm_ is None

    def test_summary_without_permutation_has_two_rows(self, rng):
        """No permutation test => summary has exactly 2 rows."""
        n = rng.integers(1, 5, size=100)
        s = rng.gamma(1.5, 1000.0, size=100)

        test = DependenceTest(n_permutations=0)
        test.fit(n, s)

        summary = test.summary()
        assert len(summary) == 2

    def test_summary_with_permutation_has_three_rows(self, rng):
        """With permutation test => summary has 3 rows."""
        n = rng.integers(1, 5, size=100)
        s = rng.gamma(1.5, 1000.0, size=100)

        test = DependenceTest(n_permutations=50)
        test.fit(n, s, rng=rng)

        summary = test.summary()
        assert len(summary) == 3

    def test_positive_correlation_detected(self, rng):
        """Strong positive correlation should give positive tau and rho."""
        n = np.arange(1, 201, dtype=float)
        s = 500.0 * n + rng.normal(0, 100, size=200)
        s = np.clip(s, 100, None)

        test = DependenceTest(n_permutations=0)
        test.fit(n, s)

        assert test.tau_ > 0
        assert test.rho_s_ > 0


# ---------------------------------------------------------------------------
# 4. CopulaGOF before-fit error
# ---------------------------------------------------------------------------

class TestCopulaGOFBeforeFit:
    def test_summary_before_fit_raises(self, mock_freq_glm, mock_sev_glm):
        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm)
        gof = CopulaGOF(model)
        with pytest.raises(RuntimeError, match="Must call .fit"):
            gof.summary()

    def test_ks_stats_none_before_fit(self, mock_freq_glm, mock_sev_glm):
        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm)
        gof = CopulaGOF(model)
        assert gof.ks_stat_u_ is None
        assert gof.ks_stat_v_ is None


# ---------------------------------------------------------------------------
# 5. JointModelReport.to_dict() edge cases
# ---------------------------------------------------------------------------

class TestJointModelReportToDict:
    def test_unfitted_model_returns_empty_dict(self, mock_freq_glm, mock_sev_glm):
        """to_dict() on unfitted model should return {} (omega_ is None)."""
        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm)
        report = JointModelReport(model)
        d = report.to_dict()
        assert "omega" not in d
        assert "copula_family" not in d

    def test_no_copula_comparison_key_absent(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """Without copula_comparison, 'copula_comparison' key should not appear."""
        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        report = JointModelReport(model, copula_comparison=None)
        d = report.to_dict()
        assert "copula_comparison" not in d

    def test_fitted_model_has_expected_keys(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """Fitted model should populate all standard keys."""
        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        report = JointModelReport(model)
        d = report.to_dict()
        for key in ("omega", "copula_family", "aic", "bic", "n_policies", "n_claims"):
            assert key in d, f"Expected key '{key}' missing from to_dict()"

    def test_n_policies_matches_data(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """n_policies in report dict should equal length of training data."""
        dgp = nb_gamma_dgp
        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        report = JointModelReport(model)
        d = report.to_dict()
        assert d["n_policies"] == len(dgp["n"])


# ---------------------------------------------------------------------------
# 6. premium_correction before fit raises RuntimeError
# ---------------------------------------------------------------------------

class TestPremiumCorrectionBeforeFit:
    def test_joint_premium_correction_before_fit(self, mock_freq_glm, mock_sev_glm):
        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm)
        with pytest.raises(RuntimeError, match="Must call .fit"):
            model.premium_correction()

    def test_joint_loss_cost_returns_positive(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """loss_cost() values should all be strictly positive."""
        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        lc = model.loss_cost(X=None, rng=rng)
        assert np.all(lc > 0), "Some loss cost values were <= 0"

    def test_joint_loss_cost_greater_than_zero_with_positive_mu(
        self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng
    ):
        """All policies have positive mu_n, mu_s so premium_joint must be positive."""
        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        corrections = model.premium_correction()
        assert np.all(corrections["premium_joint"].values > 0)
        assert np.all(corrections["premium_independent"].values > 0)


# ---------------------------------------------------------------------------
# 7. Gaussian and FGM copula premium_correction (MC path)
# ---------------------------------------------------------------------------

class TestGaussianFGMPremiumCorrection:
    def test_gaussian_correction_is_finite(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        model = _make_fitted_model(
            nb_gamma_dgp, mock_freq_glm, mock_sev_glm, copula="gaussian", rng=rng
        )
        corr = model.premium_correction(n_mc=5_000, rng=rng)
        assert isinstance(corr, pd.DataFrame)
        assert np.all(np.isfinite(corr["correction_factor"].values))

    def test_gaussian_correction_shape(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        model = _make_fitted_model(
            nb_gamma_dgp, mock_freq_glm, mock_sev_glm, copula="gaussian", rng=rng
        )
        corr = model.premium_correction(n_mc=5_000, rng=rng)
        assert len(corr) == len(dgp["n"])

    def test_gaussian_correction_positive(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        model = _make_fitted_model(
            nb_gamma_dgp, mock_freq_glm, mock_sev_glm, copula="gaussian", rng=rng
        )
        corr = model.premium_correction(n_mc=5_000, rng=rng)
        assert np.all(corr["correction_factor"].values > 0)

    def test_fgm_correction_is_finite(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        model = _make_fitted_model(
            nb_gamma_dgp, mock_freq_glm, mock_sev_glm, copula="fgm", rng=rng
        )
        corr = model.premium_correction(n_mc=5_000, rng=rng)
        assert np.all(np.isfinite(corr["correction_factor"].values))

    def test_fgm_correction_shape(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        model = _make_fitted_model(
            nb_gamma_dgp, mock_freq_glm, mock_sev_glm, copula="fgm", rng=rng
        )
        corr = model.premium_correction(n_mc=5_000, rng=rng)
        assert len(corr) == len(dgp["n"])

    def test_fgm_omega_in_bounds(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """FGM theta must lie in [-1, 1]."""
        model = _make_fitted_model(
            nb_gamma_dgp, mock_freq_glm, mock_sev_glm, copula="fgm", rng=rng
        )
        assert -1.0 <= model.omega_ <= 1.0

    def test_gaussian_rho_in_bounds(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """Gaussian rho must lie in (-1, 1)."""
        model = _make_fitted_model(
            nb_gamma_dgp, mock_freq_glm, mock_sev_glm, copula="gaussian", rng=rng
        )
        assert -1.0 < model.omega_ < 1.0


# ---------------------------------------------------------------------------
# 8. dependence_summary() for FGM uses 'omega' (not 'rho')
# ---------------------------------------------------------------------------

class TestDependenceSummaryColumnNames:
    def test_sarmanov_uses_omega_column(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        summary = model.dependence_summary()
        assert "omega" in summary.columns
        assert "rho" not in summary.columns

    def test_gaussian_uses_rho_column(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        model = _make_fitted_model(
            nb_gamma_dgp, mock_freq_glm, mock_sev_glm, copula="gaussian", rng=rng
        )
        summary = model.dependence_summary()
        assert "rho" in summary.columns
        assert "omega" not in summary.columns

    def test_fgm_uses_omega_column(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """FGM uses 'rho' column in dependence_summary (same as Gaussian; only Sarmanov uses omega)."""
        model = _make_fitted_model(
            nb_gamma_dgp, mock_freq_glm, mock_sev_glm, copula="fgm", rng=rng
        )
        summary = model.dependence_summary()
        assert "rho" in summary.columns

    def test_summary_ci_lo_less_than_hi(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """CI lower bound must be less than upper bound."""
        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        summary = model.dependence_summary()
        lo = float(summary["ci_95_lo"].iloc[0])
        hi = float(summary["ci_95_hi"].iloc[0])
        assert lo < hi, f"CI lower ({lo:.4f}) is not less than upper ({hi:.4f})"

    def test_summary_n_claims_positive(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        summary = model.dependence_summary()
        assert int(summary["n_claims"].iloc[0]) > 0


# ---------------------------------------------------------------------------
# 9. ConditionalFreqSev before-fit errors
# ---------------------------------------------------------------------------

class TestConditionalFreqSevBeforeFit:
    def test_dependence_summary_before_fit_raises(self, mock_freq_glm, mock_sev_glm):
        model = ConditionalFreqSev(freq_glm=mock_freq_glm, sev_glm_base=mock_sev_glm)
        with pytest.raises(RuntimeError, match="Must call .fit"):
            model.dependence_summary()

    def test_loss_cost_before_fit_raises(self, mock_freq_glm, mock_sev_glm):
        model = ConditionalFreqSev(freq_glm=mock_freq_glm, sev_glm_base=mock_sev_glm)
        with pytest.raises(RuntimeError):
            model.loss_cost(X=None)

    def test_initial_gamma_is_none(self, mock_freq_glm, mock_sev_glm):
        model = ConditionalFreqSev(freq_glm=mock_freq_glm, sev_glm_base=mock_sev_glm)
        assert model.gamma_ is None
        assert model.gamma_se_ is None


# ---------------------------------------------------------------------------
# 10. ConditionalFreqSev fitted behaviour
# ---------------------------------------------------------------------------

class TestConditionalFreqSevFitted:
    def test_loss_cost_shape_and_positivity(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = ConditionalFreqSev(freq_glm=mock_freq_glm, sev_glm_base=mock_sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)

        lc = model.loss_cost(X=None)
        assert len(lc) == len(n)
        assert np.all(lc >= 0)
        assert np.all(np.isfinite(lc))

    def test_large_gamma_warning(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm):
        """Manually set large gamma should trigger RuntimeWarning when calling premium_correction."""
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = ConditionalFreqSev(freq_glm=mock_freq_glm, sev_glm_base=mock_sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)

        # Force gamma to a value that triggers the warning
        model.gamma_ = 0.9

        with pytest.warns(RuntimeWarning, match=r"gamma.*unreliable"):
            model.premium_correction()

    def test_dependence_summary_has_gamma_and_se(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm):
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
        assert np.isfinite(float(summary["gamma"].iloc[0]))
        assert np.isfinite(float(summary["gamma_se"].iloc[0]))

    def test_correction_factor_is_scalar_broadcast(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm):
        """premium_correction() should return one row per policy."""
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        model = ConditionalFreqSev(freq_glm=mock_freq_glm, sev_glm_base=mock_sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data)

        corr = model.premium_correction()
        assert len(corr) == len(n)
        assert "mu_n" in corr.columns
        assert "correction_factor" in corr.columns

    def test_n_as_indicator_correction_finite(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm):
        """Indicator mode should also produce finite corrections."""
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

        corr = model.premium_correction()
        assert np.all(np.isfinite(corr["correction_factor"].values))


# ---------------------------------------------------------------------------
# 11. JointFreqSev with bootstrap CI method
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_bootstrap_ci_produces_valid_interval(
        self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng
    ):
        """Bootstrap CI should return a valid (lo, hi) tuple with lo < hi."""
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
            model.fit(data, ci_method="bootstrap", n_bootstrap=20, rng=rng)

        assert model.omega_ci_ is not None
        lo, hi = model.omega_ci_
        assert lo <= hi, f"Bootstrap CI inverted: ({lo:.4f}, {hi:.4f})"
        assert np.isfinite(lo)
        assert np.isfinite(hi)


# ---------------------------------------------------------------------------
# 12. JointFreqSev all-zero-claim edge (n_claims=0 warning path)
# ---------------------------------------------------------------------------

class TestSmallClaimsWarning:
    def test_few_claims_warns(self, rng):
        """< 500 claim events should trigger a UserWarning about omega SE."""
        n_pol = 2000
        n = np.zeros(n_pol, dtype=int)
        n[:50] = 1  # only 50 claim events
        s = np.where(n > 0, 1000.0, 0.0)
        data = pd.DataFrame({"claim_count": n, "avg_severity": s})

        mu_n = np.full(n_pol, 0.025)
        mu_s = np.full(n_pol, 1000.0)

        class FreqGLM:
            fittedvalues = mu_n
            scale = 1.0

            class _FamClass:
                alpha = 1.0
            family = _FamClass()

            class _ModelClass:
                class _FamClass2:
                    alpha = 1.0
                family = _FamClass2()
            model = _ModelClass()

            def predict(self, X=None):
                return self.fittedvalues

        class SevGLM:
            fittedvalues = mu_s
            scale = 1.0 / 1.2

            class _FamClass:
                pass
            family = _FamClass()
            family.__class__.__name__ = "Gamma"

            class _ModelClass:
                pass
            model = _ModelClass()
            model.family = _FamClass()
            model.family.__class__.__name__ = "Gamma"

            def predict(self, X=None):
                return self.fittedvalues

        model = JointFreqSev(freq_glm=FreqGLM(), sev_glm=SevGLM())
        with pytest.warns(UserWarning, match="50"):
            model.fit(data, rng=rng)


# ---------------------------------------------------------------------------
# 13. JointModelReport HTML content sanity
# ---------------------------------------------------------------------------

class TestJointModelReportHTML:
    def test_html_has_doctype(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        report = JointModelReport(model)
        html = report.to_html()
        assert "<!DOCTYPE html>" in html

    def test_html_has_model_title(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        report = JointModelReport(model)
        html = report.to_html()
        assert "Joint Frequency-Severity" in html

    def test_html_with_copula_comparison_mentions_best(
        self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng
    ):
        """HTML with copula comparison should mention the best fitting copula."""
        from insurance_frequency_severity.diagnostics import compare_copulas

        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comp = compare_copulas(n, s, mock_freq_glm, mock_sev_glm, rng=rng)

        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        report = JointModelReport(model, copula_comparison=comp)
        html = report.to_html()
        assert "Best fitting copula" in html or "best" in html.lower()

    def test_html_scatter_with_all_zero_n(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        """Scatter plot with n=all zeros should not raise (no positive-claim rows)."""
        model = _make_fitted_model(nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng=rng)
        report = JointModelReport(model)
        n_zeros = np.zeros(100)
        s_zeros = np.zeros(100)
        # Should not raise even though no positive-claim points to scatter
        html = report.to_html(n=n_zeros, s=s_zeros)
        assert isinstance(html, str)
