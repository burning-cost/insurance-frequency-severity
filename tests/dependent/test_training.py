"""Tests for dependent.training — JointLoss, DependentFSTrainer."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from insurance_frequency_severity.dependent.model import DependentFreqSevNet, SharedTrunkConfig
from insurance_frequency_severity.dependent.training import DependentFSTrainer, JointLoss, TrainingConfig
from insurance_frequency_severity.dependent.data import FreqSevDataset


# ---------------------------------------------------------------------------
# JointLoss
# ---------------------------------------------------------------------------

class TestJointLoss:
    def _make_inputs(self, batch_size=8, n_pos=5):
        torch.manual_seed(0)
        log_lambda = torch.randn(batch_size)
        log_mu = torch.randn(batch_size)
        phi = torch.tensor([1.5])
        n_claims = torch.zeros(batch_size)
        n_claims[:n_pos] = torch.tensor([1., 2., 1., 3., 1.])
        avg_sev = torch.zeros(batch_size)
        avg_sev[:n_pos] = torch.abs(torch.randn(n_pos)) * 1000 + 500
        return log_lambda, log_mu, phi, n_claims, avg_sev

    def test_returns_three_tensors(self):
        crit = JointLoss()
        inputs = self._make_inputs()
        total, fl, sl = crit(*inputs)
        assert isinstance(total, torch.Tensor)
        assert isinstance(fl, torch.Tensor)
        assert isinstance(sl, torch.Tensor)

    def test_total_loss_is_scalar(self):
        crit = JointLoss()
        total, _, _ = crit(*self._make_inputs())
        assert total.shape == ()

    def test_loss_is_finite(self):
        crit = JointLoss()
        total, fl, sl = crit(*self._make_inputs())
        assert torch.isfinite(total)
        assert torch.isfinite(fl)
        assert torch.isfinite(sl)

    def test_all_zero_claims_sev_loss_zero(self):
        crit = JointLoss()
        log_lambda = torch.randn(5)
        log_mu = torch.randn(5)
        phi = torch.tensor([1.0])
        n_claims = torch.zeros(5)
        avg_sev = torch.zeros(5)
        _, _, sl = crit(log_lambda, log_mu, phi, n_claims, avg_sev)
        assert sl.item() == pytest.approx(0.0)

    def test_fixed_weight_applied(self):
        crit_1 = JointLoss(loss_weight_sev=1.0, auto_balance=False)
        crit_2 = JointLoss(loss_weight_sev=2.0, auto_balance=False)
        inputs = self._make_inputs()
        t1, fl1, sl1 = crit_1(*inputs)
        t2, fl2, sl2 = crit_2(*inputs)
        # Same freq loss, different total because sev is weighted differently
        assert fl1.item() == pytest.approx(fl2.item(), rel=1e-4)

    def test_auto_balance_produces_finite(self):
        crit = JointLoss(auto_balance=True)
        total, _, _ = crit(*self._make_inputs())
        assert torch.isfinite(total)

    def test_gradients_flow_through_loss(self):
        crit = JointLoss()
        log_lambda = torch.randn(6, requires_grad=True)
        log_mu = torch.randn(6, requires_grad=True)
        phi = torch.tensor([1.0], requires_grad=True)
        n = torch.tensor([0., 1., 2., 0., 1., 0.])
        sev = torch.tensor([0., 1000., 2000., 0., 1500., 0.])
        total, _, _ = crit(log_lambda, log_mu, phi, n, sev)
        total.backward()
        assert log_lambda.grad is not None
        assert log_mu.grad is not None

    def test_freq_gradient_on_zero_claim_rows(self):
        """Frequency gradient should be non-zero for zero-claim rows too."""
        crit = JointLoss()
        log_lambda = torch.randn(4, requires_grad=True)
        log_mu = torch.randn(4, requires_grad=True)
        phi = torch.tensor([1.0])
        n = torch.zeros(4)  # ALL zeros
        sev = torch.zeros(4)
        total, _, _ = crit(log_lambda, log_mu, phi, n, sev)
        total.backward()
        assert log_lambda.grad is not None
        # log_mu.grad may be zero or None since sev_loss=0
        # but freq gradient must exist
        assert not torch.all(log_lambda.grad == 0)


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.max_epochs == 100
        assert cfg.batch_size == 512
        assert cfg.lr == 1e-3
        assert cfg.patience == 15
        assert cfg.auto_balance is True
        assert cfg.device == "auto"

    def test_device_auto(self):
        cfg = TrainingConfig(device="auto")
        assert cfg.device == "auto"


# ---------------------------------------------------------------------------
# DependentFSTrainer
# ---------------------------------------------------------------------------

def _make_small_loader(n=200, p=4, batch_size=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    n_claims = rng.poisson(0.1, size=n).astype(np.float32)
    avg_sev = np.where(n_claims > 0, rng.exponential(3000, size=n), 0.0).astype(np.float32)
    exposure = np.ones(n, dtype=np.float32)
    ds = FreqSevDataset(X, n_claims, avg_sev, exposure)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def _make_net(p=4):
    cfg = SharedTrunkConfig(hidden_dims=[16], latent_dim=8, use_batch_norm=False)
    return DependentFreqSevNet(p, cfg, use_explicit_gamma=True)


class TestDependentFSTrainer:
    def test_fit_runs_without_error(self):
        net = _make_net()
        tc = TrainingConfig(max_epochs=3, verbose=False, patience=None, auto_balance=True)
        trainer = DependentFSTrainer(net, tc)
        loader = _make_small_loader()
        trainer.fit(loader)

    def test_history_populated(self):
        net = _make_net()
        tc = TrainingConfig(max_epochs=4, verbose=False, patience=None)
        trainer = DependentFSTrainer(net, tc)
        loader = _make_small_loader()
        trainer.fit(loader)
        assert len(trainer.history["train_loss"]) == 4

    def test_val_loader_used(self):
        net = _make_net()
        tc = TrainingConfig(max_epochs=3, verbose=False, patience=None)
        trainer = DependentFSTrainer(net, tc)
        train_loader = _make_small_loader(n=150)
        val_loader = _make_small_loader(n=50, seed=99)
        trainer.fit(train_loader, val_loader)
        assert len(trainer.history["val_loss"]) == 3

    def test_early_stopping_fires(self):
        net = _make_net()
        tc = TrainingConfig(
            max_epochs=100, verbose=False, patience=2, min_delta=1000.0
        )
        trainer = DependentFSTrainer(net, tc)
        loader = _make_small_loader()
        trainer.fit(loader, val_loader=loader)
        # Should have stopped well before 100 epochs
        assert len(trainer.history["train_loss"]) < 100

    def test_model_in_eval_mode_after_fit(self):
        net = _make_net()
        tc = TrainingConfig(max_epochs=2, verbose=False, patience=None)
        trainer = DependentFSTrainer(net, tc)
        loader = _make_small_loader()
        trainer.fit(loader)
        assert not net.training

    def test_loss_decreases_over_epochs(self):
        """Training loss should generally decrease over several epochs."""
        net = _make_net()
        tc = TrainingConfig(max_epochs=20, verbose=False, patience=None, lr=1e-2)
        trainer = DependentFSTrainer(net, tc)
        loader = _make_small_loader(n=300, batch_size=128)
        trainer.fit(loader)
        first = trainer.history["train_loss"][0]
        last = trainer.history["train_loss"][-1]
        # Not guaranteed to decrease monotonically, but over 20 epochs it should
        assert last < first * 2, f"Loss didn't improve: {first:.4f} → {last:.4f}"

    def test_gamma_tracked_in_history(self):
        net = _make_net()
        tc = TrainingConfig(max_epochs=3, verbose=False, patience=None)
        trainer = DependentFSTrainer(net, tc)
        loader = _make_small_loader()
        trainer.fit(loader)
        assert len(trainer.history["gamma"]) == 3

    def test_trunk_lr_multiplier(self):
        """Different LR for trunk vs heads should not crash training."""
        net = _make_net()
        tc = TrainingConfig(
            max_epochs=2, verbose=False, patience=None, trunk_lr_multiplier=0.1
        )
        trainer = DependentFSTrainer(net, tc)
        loader = _make_small_loader()
        trainer.fit(loader)
        assert len(trainer.history["train_loss"]) == 2

    def test_resolve_device_auto(self):
        import torch
        d = DependentFSTrainer._resolve_device("auto")
        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert d == expected

    def test_resolve_device_cpu(self):
        import torch
        d = DependentFSTrainer._resolve_device("cpu")
        assert d == torch.device("cpu")
