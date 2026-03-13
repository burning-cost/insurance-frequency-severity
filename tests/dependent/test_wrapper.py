"""Tests for dependent.wrapper — DependentFSModel sklearn interface."""
import numpy as np
import pytest

from insurance_frequency_severity.dependent.model import SharedTrunkConfig
from insurance_frequency_severity.dependent.training import TrainingConfig
from insurance_frequency_severity.dependent.wrapper import DependentFSModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _small_data(n=200, p=4, seed=42):
    """Small synthetic dataset for wrapper tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    n_claims = rng.poisson(0.1, size=n).astype(np.float32)
    avg_sev = np.where(n_claims > 0, rng.exponential(3000, n), 0.0).astype(np.float32)
    exposure = np.ones(n, dtype=np.float32)
    return X, n_claims, avg_sev, exposure


def _fast_config():
    cfg = SharedTrunkConfig(hidden_dims=[16], latent_dim=8, use_batch_norm=False)
    tc = TrainingConfig(max_epochs=5, verbose=False, patience=None, auto_balance=True)
    return cfg, tc


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestDependentFSModelConstruction:
    def test_default_construction(self):
        m = DependentFSModel()
        assert m.use_explicit_gamma is True
        assert m.n_mc == 1000
        assert m.val_fraction == 0.1

    def test_no_gamma(self):
        m = DependentFSModel(use_explicit_gamma=False)
        assert m.use_explicit_gamma is False

    def test_custom_trunk_config(self):
        cfg = SharedTrunkConfig(hidden_dims=[32], latent_dim=16)
        m = DependentFSModel(trunk_config=cfg)
        assert m.trunk_config.latent_dim == 16

    def test_sklearn_get_params(self):
        m = DependentFSModel(n_mc=500, random_state=7)
        params = m.get_params()
        assert params["n_mc"] == 500
        assert params["random_state"] == 7


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestDependentFSModelFit:
    def test_fit_runs(self):
        cfg, tc = _fast_config()
        m = DependentFSModel(trunk_config=cfg, training_config=tc, val_fraction=0.0)
        X, n, s, e = _small_data()
        m.fit(X, n, s, e)
        assert hasattr(m, "model_")
        assert hasattr(m, "trainer_")

    def test_n_features_in_set(self):
        cfg, tc = _fast_config()
        m = DependentFSModel(trunk_config=cfg, training_config=tc, val_fraction=0.0)
        X, n, s, e = _small_data(p=6)
        m.fit(X, n, s, e)
        assert m.n_features_in_ == 6

    def test_gamma_attribute_set(self):
        cfg, tc = _fast_config()
        m = DependentFSModel(trunk_config=cfg, training_config=tc, val_fraction=0.0,
                             use_explicit_gamma=True)
        X, n, s, e = _small_data()
        m.fit(X, n, s, e)
        assert m.gamma_ is not None
        assert isinstance(m.gamma_, float)

    def test_gamma_none_when_not_used(self):
        cfg, tc = _fast_config()
        m = DependentFSModel(trunk_config=cfg, training_config=tc, val_fraction=0.0,
                             use_explicit_gamma=False)
        X, n, s, e = _small_data()
        m.fit(X, n, s, e)
        assert m.gamma_ is None

    def test_fit_with_val_split(self):
        cfg, tc = _fast_config()
        m = DependentFSModel(trunk_config=cfg, training_config=tc, val_fraction=0.2)
        X, n, s, e = _small_data(n=300)
        m.fit(X, n, s, e)
        assert hasattr(m, "model_")

    def test_returns_self(self):
        cfg, tc = _fast_config()
        m = DependentFSModel(trunk_config=cfg, training_config=tc, val_fraction=0.0)
        X, n, s, e = _small_data()
        result = m.fit(X, n, s, e)
        assert result is m


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestDependentFSModelPredict:
    @pytest.fixture(scope="class")
    def fitted_model(self):
        cfg, tc = _fast_config()
        m = DependentFSModel(trunk_config=cfg, training_config=tc, val_fraction=0.0,
                             use_explicit_gamma=True)
        X, n, s, e = _small_data()
        m.fit(X, n, s, e)
        return m, X, n, s, e

    def test_predict_frequency_shape(self, fitted_model):
        m, X, n, s, e = fitted_model
        freq = m.predict_frequency(X, e)
        assert freq.shape == (len(X),)

    def test_predict_frequency_positive(self, fitted_model):
        m, X, n, s, e = fitted_model
        freq = m.predict_frequency(X, e)
        assert (freq > 0).all()

    def test_predict_severity_shape(self, fitted_model):
        m, X, n, s, e = fitted_model
        sev = m.predict_severity(X, e)
        assert sev.shape == (len(X),)

    def test_predict_severity_positive(self, fitted_model):
        m, X, n, s, e = fitted_model
        sev = m.predict_severity(X, e)
        assert (sev > 0).all()

    def test_predict_pure_premium_shape(self, fitted_model):
        m, X, n, s, e = fitted_model
        pp = m.predict_pure_premium(X, e, n_mc=50)
        assert pp.shape == (len(X),)

    def test_predict_pure_premium_positive(self, fitted_model):
        m, X, n, s, e = fitted_model
        pp = m.predict_pure_premium(X, e, n_mc=50)
        assert (pp >= 0).all()

    def test_predict_pure_premium_mc_method(self, fitted_model):
        m, X, n, s, e = fitted_model
        pp = m.predict_pure_premium(X, e, method="mc", n_mc=50)
        assert pp.shape == (len(X),)

    def test_predict_pure_premium_analytical_method(self, fitted_model):
        m, X, n, s, e = fitted_model
        pp = m.predict_pure_premium(X, e, method="analytical")
        assert pp.shape == (len(X),)
        assert (pp >= 0).all()

    def test_predict_alias(self, fitted_model):
        m, X, n, s, e = fitted_model
        pp = m.predict(X, e)
        assert pp.shape == (len(X),)

    def test_predict_no_exposure(self, fitted_model):
        m, X, n, s, e = fitted_model
        pp = m.predict(X)
        assert pp.shape == (len(X),)

    def test_latent_repr_shape(self, fitted_model):
        m, X, n, s, e = fitted_model
        h = m.latent_repr(X)
        assert h.shape == (len(X), m.trunk_config.latent_dim)

    def test_latent_repr_dtype(self, fitted_model):
        m, X, n, s, e = fitted_model
        h = m.latent_repr(X)
        assert h.dtype == np.float32

    def test_score_returns_float(self, fitted_model):
        m, X, n, s, e = fitted_model
        sc = m.score(X, n, s, e)
        assert isinstance(sc, float)

    def test_training_history_keys(self, fitted_model):
        m, X, n, s, e = fitted_model
        hist = m.training_history()
        assert "train_loss" in hist
        assert "freq_loss" in hist
        assert "sev_loss" in hist

    def test_not_fitted_raises(self):
        m = DependentFSModel()
        X, n, s, e = _small_data()
        with pytest.raises(Exception):
            m.predict_frequency(X, e)

    def test_exposure_scaling(self, fitted_model):
        """Doubling exposure should roughly double frequency (in count, not rate)."""
        m, X, n, s, e = fitted_model
        freq_1 = m.predict_frequency(X[:5], np.ones(5))
        freq_2 = m.predict_frequency(X[:5], 2.0 * np.ones(5))
        # Rate should be approximately the same when we normalise by exposure
        # i.e. freq_2 per unit should ≈ freq_1 per unit
        np.testing.assert_allclose(freq_1, freq_2, rtol=0.05)
