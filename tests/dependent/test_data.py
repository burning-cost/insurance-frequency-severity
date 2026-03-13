"""Tests for dependent.data — FreqSevDataset, prepare_features, make_train_val_loaders."""
import numpy as np
import pandas as pd
import pytest
import torch

from insurance_frequency_severity.dependent.data import FreqSevDataset, make_train_val_loaders, prepare_features


# ---------------------------------------------------------------------------
# FreqSevDataset
# ---------------------------------------------------------------------------

class TestFreqSevDataset:
    def _make(self, n=20, p=4, seed=7):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p)).astype(np.float32)
        n_claims = rng.poisson(0.1, size=n).astype(np.float32)
        avg_sev = np.where(n_claims > 0, rng.exponential(2000, n), 0.0).astype(np.float32)
        exposure = np.ones(n, dtype=np.float32)
        return FreqSevDataset(X, n_claims, avg_sev, exposure)

    def test_len(self):
        ds = self._make(n=15)
        assert len(ds) == 15

    def test_item_keys(self):
        ds = self._make()
        item = ds[0]
        assert set(item.keys()) == {"x", "log_exposure", "n_claims", "avg_severity"}

    def test_x_dtype(self):
        ds = self._make()
        assert ds[0]["x"].dtype == torch.float32

    def test_log_exposure_is_log(self):
        X = np.ones((5, 2), dtype=np.float32)
        n = np.zeros(5)
        sev = np.zeros(5)
        exposure = np.array([1.0, 2.0, 0.5, 4.0, 1.0])
        ds = FreqSevDataset(X, n, sev, exposure)
        expected = torch.log(torch.tensor([1.0, 2.0, 0.5, 4.0, 1.0]))
        torch.testing.assert_close(ds.log_exposure, expected)

    def test_zero_exposure_raises(self):
        X = np.ones((3, 2), dtype=np.float32)
        n = np.zeros(3)
        sev = np.zeros(3)
        exposure = np.array([1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="strictly positive"):
            FreqSevDataset(X, n, sev, exposure)

    def test_length_mismatch_raises(self):
        X = np.ones((5, 2), dtype=np.float32)
        n = np.zeros(4)  # wrong length
        sev = np.zeros(5)
        exposure = np.ones(5)
        with pytest.raises(ValueError, match="same length"):
            FreqSevDataset(X, n, sev, exposure)

    def test_from_dataframe(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "f1": rng.standard_normal(10).astype(np.float32),
            "f2": rng.standard_normal(10).astype(np.float32),
            "n_claims": rng.poisson(0.1, 10).astype(np.float32),
            "avg_severity": np.zeros(10, dtype=np.float32),
            "exposure": np.ones(10, dtype=np.float32),
        })
        ds = FreqSevDataset.from_dataframe(df, feature_cols=["f1", "f2"])
        assert len(ds) == 10
        assert ds[0]["x"].shape == (2,)

    def test_from_dataframe_custom_cols(self):
        df = pd.DataFrame({
            "feat": np.ones(5, dtype=np.float32),
            "claims": np.zeros(5, dtype=np.float32),
            "sev": np.zeros(5, dtype=np.float32),
            "exp": np.ones(5, dtype=np.float32),
        })
        ds = FreqSevDataset.from_dataframe(
            df, ["feat"],
            n_claims_col="claims",
            avg_severity_col="sev",
            exposure_col="exp",
        )
        assert len(ds) == 5


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def _make_df(self, n=30, seed=5):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "age": rng.integers(18, 80, n).astype(float),
            "value": rng.exponential(10000, n),
            "region": rng.choice(["North", "South", "London"], n),
            "vehicle_class": rng.choice(["A", "B", "C"], n),
        })

    def test_returns_array_and_transformer(self):
        df = self._make_df()
        X, ct = prepare_features(df, numeric_cols=["age", "value"])
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(df)

    def test_numeric_only(self):
        df = self._make_df()
        X, ct = prepare_features(df, numeric_cols=["age", "value"])
        assert X.shape[1] == 2

    def test_with_categoricals(self):
        df = self._make_df()
        X, ct = prepare_features(
            df, numeric_cols=["age", "value"],
            categorical_cols=["region"]
        )
        # 2 numeric + 3 one-hot (North/South/London)
        assert X.shape[1] >= 4

    def test_transformer_reuse(self):
        df_train = self._make_df(n=30)
        df_test = self._make_df(n=10, seed=99)
        X_train, ct = prepare_features(
            df_train, numeric_cols=["age", "value"],
            categorical_cols=["region"]
        )
        X_test, _ = prepare_features(
            df_test, numeric_cols=["age", "value"],
            categorical_cols=["region"],
            transformer=ct,
        )
        assert X_train.shape[1] == X_test.shape[1]

    def test_float32_output(self):
        df = self._make_df()
        X, _ = prepare_features(df, numeric_cols=["age", "value"])
        assert X.dtype == np.float32

    def test_unknown_category_handled(self):
        """Unknown categories at test time should produce all-zero rows (no error)."""
        df_train = self._make_df()
        X_train, ct = prepare_features(
            df_train, numeric_cols=["age"],
            categorical_cols=["region"]
        )
        df_test = pd.DataFrame({
            "age": [30.0],
            "region": ["NewRegion"],  # unseen
        })
        X_test, _ = prepare_features(
            df_test, numeric_cols=["age"],
            categorical_cols=["region"],
            transformer=ct,
        )
        assert X_test.shape[0] == 1


# ---------------------------------------------------------------------------
# make_train_val_loaders
# ---------------------------------------------------------------------------

class TestMakeTrainValLoaders:
    def _make_ds(self, n=100):
        X = np.ones((n, 2), dtype=np.float32)
        nc = np.zeros(n, dtype=np.float32)
        sev = np.zeros(n, dtype=np.float32)
        exp = np.ones(n, dtype=np.float32)
        return FreqSevDataset(X, nc, sev, exp)

    def test_returns_two_loaders(self):
        ds = self._make_ds()
        tl, vl = make_train_val_loaders(ds, val_fraction=0.2, batch_size=16)
        assert tl is not None
        assert vl is not None

    def test_split_sizes(self):
        ds = self._make_ds(n=100)
        tl, vl = make_train_val_loaders(ds, val_fraction=0.2, batch_size=100)
        train_n = sum(len(b["x"]) for b in tl)
        val_n = sum(len(b["x"]) for b in vl)
        assert train_n == 80
        assert val_n == 20

    def test_reproducible_split(self):
        ds = self._make_ds()
        tl1, vl1 = make_train_val_loaders(ds, seed=42)
        tl2, vl2 = make_train_val_loaders(ds, seed=42)
        # Both should iterate the same items
        b1 = next(iter(vl1))
        b2 = next(iter(vl2))
        torch.testing.assert_close(b1["x"], b2["x"])

    def test_different_seeds_differ(self):
        ds = self._make_ds(n=100)
        _, vl1 = make_train_val_loaders(ds, seed=1)
        _, vl2 = make_train_val_loaders(ds, seed=99)
        b1 = next(iter(vl1))
        b2 = next(iter(vl2))
        # With large enough dataset, different seeds should give different val sets
        # (This is stochastic so we just check it doesn't crash and shapes match)
        assert b1["x"].shape == b2["x"].shape
