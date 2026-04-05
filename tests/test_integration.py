"""
Integration tests for insurance-frequency-severity.

The unit tests cover individual components. These tests exercise the full
pipeline: synthetic data -> mock GLMs -> JointFreqSev.fit() ->
premium_correction() -> JointModelReport.to_dict(). The goal is to catch
regressions where components work in isolation but break when assembled.

We use small datasets (~150 policies) to keep test runtime low. We don't
test statistical properties here (unit tests handle that with n=5000) — we
test that the full pipeline runs, produces finite positive outputs, and
doesn't silently produce nonsense.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from insurance_frequency_severity import JointFreqSev, ConditionalFreqSev, JointModelReport
from insurance_frequency_severity.diagnostics import DependenceTest, compare_copulas


# ---------------------------------------------------------------------------
# Shared synthetic dataset and mock GLMs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_nb_gamma_dataset():
    """
    150 policies simulated from NB(mu=0.15, alpha=0.8) x Gamma(shape=1.5, mu=1500).
    Negative dependence (omega=-2.5): high-frequency policies have lower severity.

    We use a fixed seed so the tests are deterministic.
    """
    from insurance_frequency_severity.copula import SarmanovCopula

    rng = np.random.default_rng(123)
    n_policies = 150

    copula = SarmanovCopula(
        freq_family="nb",
        sev_family="gamma",
        omega=-2.5,
        kernel_theta=0.5,
        kernel_alpha=0.0005,
    )
    freq_params = {"mu": 0.15, "alpha": 0.8}
    sev_params = {"mu": 1500.0, "shape": 1.5}
    n_samp, s_samp = copula.sample(n_policies, freq_params, sev_params, rng=rng)

    return {
        "n": n_samp,
        "s": s_samp,
        "n_policies": n_policies,
        "mu_n": 0.15,
        "alpha": 0.8,
        "mu_s": 1500.0,
        "shape": 1.5,
    }


def _make_mock_freq_glm(n_policies: int, mu_n: float, alpha: float):
    """NB mock GLM with correct family metadata."""
    fittedvalues = np.full(n_policies, mu_n)

    class MockNBGLM:
        def __init__(self):
            self.fittedvalues = fittedvalues
            self.scale = 1.0 / alpha

            class Fam:
                pass
            self.family = Fam()
            self.family.__class__ = type("NegativeBinomial", (), {"alpha": alpha})
            self.family.alpha = alpha

            class Model:
                pass
            self.model = Model()
            self.model.family = type("NegativeBinomial", (), {"alpha": alpha})()
            self.model.family.alpha = alpha

        def predict(self, X=None):
            if X is not None:
                return np.full(len(X), mu_n)
            return self.fittedvalues

    return MockNBGLM()


def _make_mock_sev_glm(n_policies: int, mu_s: float, shape: float):
    """Gamma mock GLM with correct family metadata."""
    fittedvalues = np.full(n_policies, mu_s)
    phi = 1.0 / shape

    class MockGammaGLM:
        def __init__(self):
            self.fittedvalues = fittedvalues
            self.scale = phi

            class Fam:
                pass
            self.family = Fam()
            self.family.__class__ = type("Gamma", (), {})

            class Model:
                pass
            self.model = Model()
            self.model.family = type("Gamma", (), {})()

        def predict(self, X=None):
            if X is not None:
                return np.full(len(X), mu_s)
            return self.fittedvalues

    return MockGammaGLM()


# ---------------------------------------------------------------------------
# Integration test 1: Full Sarmanov pipeline end-to-end
# ---------------------------------------------------------------------------

class TestSarmanovFullPipeline:
    """
    Fit a JointFreqSev with Sarmanov copula on a small synthetic dataset,
    run premium corrections, and verify outputs are numerically sane.
    """

    @pytest.fixture(autouse=True)
    def fit(self, small_nb_gamma_dataset):
        d = small_nb_gamma_dataset
        self.data = pd.DataFrame({
            "claim_count": d["n"],
            "avg_severity": d["s"],
        })
        freq_glm = _make_mock_freq_glm(d["n_policies"], d["mu_n"], d["alpha"])
        sev_glm = _make_mock_sev_glm(d["n_policies"], d["mu_s"], d["shape"])

        self.model = JointFreqSev(
            freq_glm=freq_glm,
            sev_glm=sev_glm,
            copula="sarmanov",
            kernel_theta=0.5,
            kernel_alpha=0.0005,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                self.data,
                n_col="claim_count",
                s_col="avg_severity",
                rng=np.random.default_rng(42),
            )
        self.n_policies = d["n_policies"]

    def test_omega_is_finite(self):
        assert self.model.omega_ is not None
        assert np.isfinite(self.model.omega_)

    def test_aic_bic_finite(self):
        assert np.isfinite(self.model.aic_)
        assert np.isfinite(self.model.bic_)

    def test_premium_correction_returns_dataframe(self):
        corrections = self.model.premium_correction()
        assert isinstance(corrections, pd.DataFrame)
        assert len(corrections) == self.n_policies

    def test_premium_correction_columns_present(self):
        corrections = self.model.premium_correction()
        required = {"mu_n", "mu_s", "correction_factor", "premium_joint", "premium_independent"}
        assert required.issubset(set(corrections.columns))

    def test_correction_factors_all_positive(self):
        corrections = self.model.premium_correction()
        factors = corrections["correction_factor"].values
        assert np.all(factors > 0), "All correction factors must be positive"

    def test_correction_factors_all_finite(self):
        corrections = self.model.premium_correction()
        factors = corrections["correction_factor"].values
        assert np.all(np.isfinite(factors)), "All correction factors must be finite"

    def test_joint_premium_all_positive(self):
        corrections = self.model.premium_correction()
        assert np.all(corrections["premium_joint"].values > 0)

    def test_joint_premium_all_finite(self):
        corrections = self.model.premium_correction()
        assert np.all(np.isfinite(corrections["premium_joint"].values))

    def test_dependence_summary_columns(self):
        summary = self.model.dependence_summary()
        assert "omega" in summary.columns
        assert "n_policies" in summary.columns
        assert "aic" in summary.columns

    def test_dependence_summary_omega_matches_fitted(self):
        summary = self.model.dependence_summary()
        assert float(summary["omega"].iloc[0]) == pytest.approx(self.model.omega_)


# ---------------------------------------------------------------------------
# Integration test 2: ConditionalFreqSev full pipeline
# ---------------------------------------------------------------------------

class TestConditionalFullPipeline:
    """
    Garrido method: much simpler — no copula estimation, just a conditional
    severity GLM. The pipeline is: fit -> premium_correction -> dependence_summary.
    """

    @pytest.fixture(autouse=True)
    def fit(self, small_nb_gamma_dataset):
        d = small_nb_gamma_dataset
        self.data = pd.DataFrame({
            "claim_count": d["n"],
            "avg_severity": d["s"],
        })
        freq_glm = _make_mock_freq_glm(d["n_policies"], d["mu_n"], d["alpha"])
        sev_glm = _make_mock_sev_glm(d["n_policies"], d["mu_s"], d["shape"])

        self.model = ConditionalFreqSev(freq_glm=freq_glm, sev_glm_base=sev_glm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.data, n_col="claim_count", s_col="avg_severity")
        self.n_policies = d["n_policies"]

    def test_gamma_is_finite(self):
        assert self.model.gamma_ is not None
        assert np.isfinite(self.model.gamma_)

    def test_gamma_se_positive(self):
        assert self.model.gamma_se_ > 0

    def test_premium_correction_shape(self):
        corr = self.model.premium_correction()
        assert len(corr) == self.n_policies

    def test_premium_correction_finite_positive(self):
        corr = self.model.premium_correction()
        factors = corr["correction_factor"].values
        assert np.all(np.isfinite(factors))
        assert np.all(factors > 0)

    def test_dependence_summary_has_gamma(self):
        summary = self.model.dependence_summary()
        assert "gamma" in summary.columns
        assert "gamma_se" in summary.columns


# ---------------------------------------------------------------------------
# Integration test 3: JointModelReport end-to-end
# ---------------------------------------------------------------------------

class TestJointModelReportPipeline:
    """
    Report generation should produce a non-empty dict without raising.
    We don't test HTML content in detail — just that to_dict() is consistent
    with the fitted model and doesn't blow up.
    """

    @pytest.fixture(autouse=True)
    def build_report(self, small_nb_gamma_dataset):
        d = small_nb_gamma_dataset
        data = pd.DataFrame({
            "claim_count": d["n"],
            "avg_severity": d["s"],
        })
        freq_glm = _make_mock_freq_glm(d["n_policies"], d["mu_n"], d["alpha"])
        sev_glm = _make_mock_sev_glm(d["n_policies"], d["mu_s"], d["shape"])

        model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm, copula="sarmanov")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(data, rng=np.random.default_rng(99))

        self.model = model
        self.report = JointModelReport(model)

    def test_to_dict_returns_dict(self):
        result = self.report.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_omega(self):
        result = self.report.to_dict()
        assert "omega" in result
        assert np.isfinite(result["omega"])

    def test_to_dict_n_policies_matches_data(self, small_nb_gamma_dataset):
        result = self.report.to_dict()
        assert result["n_policies"] == small_nb_gamma_dataset["n_policies"]

    def test_to_dict_copula_family_sarmanov(self):
        result = self.report.to_dict()
        assert result["copula_family"] == "sarmanov"

    def test_to_dict_aic_bic_present(self):
        result = self.report.to_dict()
        assert "aic" in result
        assert "bic" in result
        assert np.isfinite(result["aic"])
        assert np.isfinite(result["bic"])


# ---------------------------------------------------------------------------
# Integration test 4: DependenceTest on simulated data
# ---------------------------------------------------------------------------

class TestDependenceTestPipeline:
    """
    DependenceTest should run on real-looking data without raising. We only
    check that it produces finite test statistics — significance thresholds
    are not tested here (n=150 is small for permutation tests).
    """

    def test_kendall_tau_finite(self, small_nb_gamma_dataset):
        d = small_nb_gamma_dataset
        n, s = d["n"], d["s"]
        # Only positive-claim observations carry severity information
        mask = n > 0
        n_pos, s_pos = n[mask].astype(float), s[mask]

        if len(n_pos) < 5:
            pytest.skip("Too few positive-claim observations in this synthetic sample")

        test = DependenceTest(n_permutations=0)  # no permutation for speed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test.fit(n_pos, s_pos)

        assert np.isfinite(test.tau_)
        assert np.isfinite(test.rho_s_)
        assert 0.0 <= test.tau_pval_ <= 1.0
        assert 0.0 <= test.rho_s_pval_ <= 1.0


# ---------------------------------------------------------------------------
# Integration test 5: compare_copulas utility
# ---------------------------------------------------------------------------

class TestCompareCopulasPipeline:
    """
    compare_copulas() fits three copula families and returns a ranked AIC table.
    On small data it may not converge well, but it must return a valid DataFrame.
    """

    def test_compare_copulas_returns_dataframe(self, small_nb_gamma_dataset):
        d = small_nb_gamma_dataset
        data = pd.DataFrame({
            "claim_count": d["n"],
            "avg_severity": d["s"],
        })
        freq_glm = _make_mock_freq_glm(d["n_policies"], d["mu_n"], d["alpha"])
        sev_glm = _make_mock_sev_glm(d["n_policies"], d["mu_s"], d["shape"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compare_copulas(
                data=data,
                freq_glm=freq_glm,
                sev_glm=sev_glm,
                n_col="claim_count",
                s_col="avg_severity",
                rng=np.random.default_rng(7),
            )

        assert isinstance(result, pd.DataFrame)
        assert "aic" in result.columns or "AIC" in result.columns or len(result.columns) > 0
        # Should have at most 3 rows (one per copula family)
        assert len(result) <= 3
        assert len(result) >= 1
