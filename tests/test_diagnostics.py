"""
Tests for diagnostics.py — DependenceTest, CopulaGOF, compare_copulas.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_frequency_severity.diagnostics import (
    DependenceTest,
    CopulaGOF,
    compare_copulas,
)


class TestDependenceTest:
    def test_fit_returns_self(self, rng):
        n = rng.integers(0, 5, size=500)
        s = np.where(n > 0, rng.gamma(1.5, 1000.0, size=500), 0.0)
        test = DependenceTest(n_permutations=50)
        result = test.fit(n, s, rng=rng)
        assert result is test

    def test_tau_and_rho_finite(self, rng):
        n = np.array([1, 2, 1, 3, 1, 2, 1, 4, 1, 2] * 50, dtype=float)
        s = np.array([1000, 800, 1200, 600, 1100, 900, 1300, 700, 1050, 950] * 50)
        test = DependenceTest(n_permutations=0)
        test.fit(n, s, rng=rng)
        assert np.isfinite(test.tau_)
        assert np.isfinite(test.rho_s_)

    def test_summary_has_expected_columns(self, rng):
        n = rng.integers(1, 5, size=200)
        s = rng.gamma(1.5, 1000.0, size=200)
        test = DependenceTest(n_permutations=100)
        test.fit(n, s, rng=rng)
        summary = test.summary()
        assert "test" in summary.columns
        assert "statistic" in summary.columns
        assert "p_value" in summary.columns
        assert len(summary) >= 2

    def test_negative_correlation_detected(self, rng):
        """Strong negative correlation should give negative tau."""
        n = np.arange(1, 201, dtype=float)
        s = 10000.0 / n + rng.normal(0, 50, size=200)
        s = np.clip(s, 100, None)
        test = DependenceTest(n_permutations=0)
        test.fit(n, s, rng=rng)
        assert test.tau_ < 0
        assert test.rho_s_ < 0

    def test_too_few_observations_raises(self, rng):
        n = np.array([1, 2, 1])
        s = np.array([1000.0, 800.0, 1200.0])
        test = DependenceTest(n_permutations=0)
        with pytest.raises(ValueError, match="valid positive-claim"):
            test.fit(n, s)

    def test_before_fit_raises(self):
        test = DependenceTest()
        with pytest.raises(RuntimeError):
            test.summary()


class TestCopulaGOF:
    def test_fit_runs(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        n_pol = len(n)

        freq_params = [{"mu": dgp["mu_n"], "alpha": dgp["alpha"]} for _ in range(n_pol)]
        sev_params = [{"mu": dgp["mu_s"], "shape": dgp["shape"]} for _ in range(n_pol)]

        from insurance_frequency_severity.joint import JointFreqSev
        model = JointFreqSev(
            freq_glm=mock_freq_glm,
            sev_glm=mock_sev_glm,
            copula="sarmanov",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pd.DataFrame({"claim_count": n, "avg_severity": s})
            model.fit(data, rng=rng)
            # Manually set internal state needed by GOF
            model._freq_family = "nb"
            model._sev_family = "gamma"

        gof = CopulaGOF(model)
        gof.fit(n, s, freq_params, sev_params)

        assert gof.ks_stat_u_ is not None
        assert gof.ks_stat_v_ is not None
        assert np.isfinite(gof.ks_stat_u_)
        assert np.isfinite(gof.ks_stat_v_)

    def test_summary_structure(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]
        n_pol = len(n)

        freq_params = [{"mu": dgp["mu_n"], "alpha": dgp["alpha"]} for _ in range(n_pol)]
        sev_params = [{"mu": dgp["mu_s"], "shape": dgp["shape"]} for _ in range(n_pol)]

        from insurance_frequency_severity.joint import JointFreqSev
        model = JointFreqSev(freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, copula="sarmanov")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pd.DataFrame({"claim_count": n, "avg_severity": s})
            model.fit(data, rng=rng)
            model._freq_family = "nb"
            model._sev_family = "gamma"

        gof = CopulaGOF(model)
        gof.fit(n, s, freq_params, sev_params)
        summary = gof.summary()
        assert "margin" in summary.columns
        assert "ks_statistic" in summary.columns
        assert len(summary) == 2


class TestCompareCopulas:
    def test_compare_returns_dataframe(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]

        result = compare_copulas(
            n=n, s=s,
            freq_glm=mock_freq_glm,
            sev_glm=mock_sev_glm,
            rng=rng,
        )

        assert isinstance(result, pd.DataFrame)
        assert "copula" in result.columns
        assert "aic" in result.columns
        assert "bic" in result.columns
        assert len(result) == 3

    def test_compare_sorted_by_aic(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]

        result = compare_copulas(n=n, s=s, freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, rng=rng)

        aic_vals = result["aic"].dropna().values
        if len(aic_vals) > 1:
            assert np.all(np.diff(aic_vals) >= 0), "compare_copulas result should be sorted by AIC"

    def test_delta_aic_present_and_nonneg(self, nb_gamma_dgp, mock_freq_glm, mock_sev_glm, rng):
        dgp = nb_gamma_dgp
        n, s = dgp["n"], dgp["s"]

        result = compare_copulas(n=n, s=s, freq_glm=mock_freq_glm, sev_glm=mock_sev_glm, rng=rng)

        assert "delta_aic" in result.columns
        delta = result["delta_aic"].dropna().values
        assert np.all(delta >= 0)
        assert float(delta[0]) == pytest.approx(0.0, abs=1e-10)
