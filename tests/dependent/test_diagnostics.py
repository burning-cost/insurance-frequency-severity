"""Tests for dependent.diagnostics — DependentFSDiagnostics."""
import numpy as np
import pytest

from insurance_frequency_severity.dependent.benchmarks import make_dependent_claims, feature_cols
from insurance_frequency_severity.dependent.diagnostics import DependentFSDiagnostics
from insurance_frequency_severity.dependent.model import SharedTrunkConfig
from insurance_frequency_severity.dependent.training import TrainingConfig
from insurance_frequency_severity.dependent.wrapper import DependentFSModel


def _fast_fitted_model(n=300, p=4, seed=0):
    """Fit a minimal model for diagnostic testing."""
    df_train, df_test = make_dependent_claims(n_policies=n, n_features=p, seed=seed)
    fc = feature_cols(df_train)
    X_train = df_train[fc].values.astype(np.float32)
    X_test = df_test[fc].values.astype(np.float32)

    cfg = SharedTrunkConfig(hidden_dims=[16], latent_dim=8, use_batch_norm=False)
    tc = TrainingConfig(max_epochs=5, verbose=False, patience=None)
    m = DependentFSModel(trunk_config=cfg, training_config=tc, val_fraction=0.0,
                         use_explicit_gamma=True)
    m.fit(
        X_train,
        df_train["n_claims"].values,
        df_train["avg_severity"].values,
        df_train["exposure"].values,
    )
    return m, X_test, df_test


@pytest.fixture(scope="module")
def diag_fixture():
    m, X_test, df_test = _fast_fitted_model()
    diag = DependentFSDiagnostics(
        model=m,
        X=X_test,
        n_claims=df_test["n_claims"].values,
        avg_severity=df_test["avg_severity"].values,
        exposure=df_test["exposure"].values,
    )
    return diag, m, X_test, df_test


class TestLorenzCurve:
    def test_frequency_lorenz(self, diag_fixture):
        diag, *_ = diag_fixture
        cum_exp, cum_loss, gini = diag.lorenz_curve(target="frequency")
        assert cum_exp.shape == cum_loss.shape
        assert -1.0 <= gini <= 1.0  # can be negative with an untrained/poor model

    def test_pure_premium_lorenz(self, diag_fixture):
        diag, *_ = diag_fixture
        cum_exp, cum_loss, gini = diag.lorenz_curve(target="pure_premium")
        assert len(cum_exp) > 2
        assert -1.0 <= gini <= 1.0

    def test_lorenz_starts_and_ends_correctly(self, diag_fixture):
        diag, *_ = diag_fixture
        cum_exp, cum_loss, _ = diag.lorenz_curve(target="frequency")
        assert cum_exp[0] == pytest.approx(0.0)
        assert cum_exp[-1] == pytest.approx(1.0)
        assert cum_loss[0] == pytest.approx(0.0)
        assert cum_loss[-1] == pytest.approx(1.0)

    def test_severity_lorenz_pos_only(self, diag_fixture):
        diag, *_ = diag_fixture
        # Should work if there are positive-claim rows in test set
        nc = diag.n_claims
        if (nc > 0).sum() >= 5:
            cum_exp, cum_loss, gini = diag.lorenz_curve(target="severity")
            assert len(cum_exp) > 2

    def test_unknown_target_raises(self, diag_fixture):
        diag, *_ = diag_fixture
        with pytest.raises(ValueError, match="Unknown target"):
            diag.lorenz_curve(target="bogus")


class TestGiniSummary:
    def test_returns_dict(self, diag_fixture):
        diag, *_ = diag_fixture
        result = diag.gini_summary()
        assert isinstance(result, dict)

    def test_expected_keys(self, diag_fixture):
        diag, *_ = diag_fixture
        result = diag.gini_summary()
        assert "gini_frequency" in result
        assert "gini_pure_premium" in result

    def test_gini_values_in_range(self, diag_fixture):
        diag, *_ = diag_fixture
        result = diag.gini_summary()
        for k, v in result.items():
            assert -1.0 <= v <= 1.0, f"{k}={v} out of range"


class TestCalibration:
    def test_returns_dict(self, diag_fixture):
        diag, *_ = diag_fixture
        cal = diag.calibration(target="frequency")
        assert isinstance(cal, dict)

    def test_calibration_keys(self, diag_fixture):
        diag, *_ = diag_fixture
        cal = diag.calibration(target="frequency")
        assert "pred_mean" in cal
        assert "obs_mean" in cal
        assert "bucket_edge" in cal

    def test_calibration_shapes(self, diag_fixture):
        diag, *_ = diag_fixture
        cal = diag.calibration(target="frequency", n_deciles=5)
        assert len(cal["pred_mean"]) <= 5
        assert len(cal["obs_mean"]) == len(cal["pred_mean"])

    def test_pure_premium_calibration(self, diag_fixture):
        diag, *_ = diag_fixture
        cal = diag.calibration(target="pure_premium")
        assert len(cal["pred_mean"]) > 0

    def test_unknown_target_raises(self, diag_fixture):
        diag, *_ = diag_fixture
        with pytest.raises(ValueError):
            diag.calibration(target="invalid")


class TestLatentCorrelation:
    def test_returns_dict(self, diag_fixture):
        diag, *_ = diag_fixture
        result = diag.latent_correlation()
        assert isinstance(result, dict)

    def test_expected_keys(self, diag_fixture):
        diag, *_ = diag_fixture
        result = diag.latent_correlation()
        assert "latent_corr" in result
        assert "freq_corr" in result
        assert "sev_corr" in result
        assert "n_freq_active" in result
        assert "n_sev_active" in result

    def test_latent_corr_shape(self, diag_fixture):
        diag, m, *_ = diag_fixture
        result = diag.latent_correlation()
        d = m.trunk_config.latent_dim
        assert result["latent_corr"].shape == (d, d)

    def test_freq_corr_shape(self, diag_fixture):
        diag, m, *_ = diag_fixture
        result = diag.latent_correlation()
        d = m.trunk_config.latent_dim
        assert result["freq_corr"].shape == (d,)

    def test_n_active_nonnegative(self, diag_fixture):
        diag, *_ = diag_fixture
        result = diag.latent_correlation()
        assert result["n_freq_active"] >= 0
        assert result["n_sev_active"] >= 0


class TestVsIndependent:
    def test_returns_dict(self, diag_fixture):
        diag, *_ = diag_fixture
        result = diag.vs_independent(n_mc=50)
        assert isinstance(result, dict)

    def test_expected_keys(self, diag_fixture):
        diag, *_ = diag_fixture
        result = diag.vs_independent(n_mc=50)
        assert "dependent_mse" in result
        assert "independent_mse" in result
        assert "mse_reduction_pct" in result

    def test_mse_nonnegative(self, diag_fixture):
        diag, *_ = diag_fixture
        result = diag.vs_independent(n_mc=50)
        assert result["dependent_mse"] >= 0
        assert result["independent_mse"] >= 0

    def test_with_custom_val_data(self, diag_fixture):
        diag, m, X_test, df_test = diag_fixture
        result = diag.vs_independent(
            X_val=X_test,
            n_claims_val=df_test["n_claims"].values,
            avg_severity_val=df_test["avg_severity"].values,
            exposure_val=df_test["exposure"].values,
            n_mc=50,
        )
        assert isinstance(result, dict)
