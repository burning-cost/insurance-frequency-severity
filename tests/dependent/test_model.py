"""Tests for dependent.model — SharedTrunkConfig, SharedTrunk, Heads, DependentFreqSevNet."""
import numpy as np
import pytest
import torch

from insurance_frequency_severity.dependent.model import (
    DependentFreqSevNet,
    FrequencyHead,
    SeverityHead,
    SharedTrunk,
    SharedTrunkConfig,
    _activation,
)


# ---------------------------------------------------------------------------
# SharedTrunkConfig
# ---------------------------------------------------------------------------

class TestSharedTrunkConfig:
    def test_defaults(self):
        cfg = SharedTrunkConfig()
        assert cfg.hidden_dims == [128, 64]
        assert cfg.latent_dim == 32
        assert cfg.dropout == 0.1
        assert cfg.activation == "elu"
        assert cfg.use_batch_norm is True

    def test_custom(self):
        cfg = SharedTrunkConfig(hidden_dims=[64], latent_dim=16, dropout=0.0)
        assert cfg.hidden_dims == [64]
        assert cfg.latent_dim == 16
        assert cfg.dropout == 0.0

    def test_activation_field(self):
        for act in ("elu", "relu", "tanh"):
            cfg = SharedTrunkConfig(activation=act)
            assert cfg.activation == act


# ---------------------------------------------------------------------------
# _activation helper
# ---------------------------------------------------------------------------

class TestActivationHelper:
    def test_elu(self):
        act = _activation("elu")
        assert isinstance(act, torch.nn.ELU)

    def test_relu(self):
        act = _activation("relu")
        assert isinstance(act, torch.nn.ReLU)

    def test_tanh(self):
        act = _activation("tanh")
        assert isinstance(act, torch.nn.Tanh)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            _activation("sigmoid")


# ---------------------------------------------------------------------------
# SharedTrunk
# ---------------------------------------------------------------------------

class TestSharedTrunk:
    def _make_trunk(self, in_features=8, hidden=[32, 16], latent=8):
        cfg = SharedTrunkConfig(hidden_dims=hidden, latent_dim=latent, dropout=0.0)
        return SharedTrunk(in_features, cfg)

    def test_output_shape(self):
        trunk = self._make_trunk()
        x = torch.randn(10, 8)
        h = trunk(x)
        assert h.shape == (10, 8)

    def test_single_sample_eval_mode(self):
        trunk = self._make_trunk()
        trunk.eval()
        x = torch.randn(1, 8)
        h = trunk(x)
        assert h.shape == (1, 8)

    def test_single_layer_trunk(self):
        cfg = SharedTrunkConfig(hidden_dims=[16], latent_dim=4, use_batch_norm=False)
        trunk = SharedTrunk(5, cfg)
        x = torch.randn(3, 5)
        h = trunk(x)
        assert h.shape == (3, 4)

    def test_no_batch_norm(self):
        cfg = SharedTrunkConfig(hidden_dims=[16], latent_dim=4, use_batch_norm=False)
        trunk = SharedTrunk(5, cfg)
        # Should not contain any BatchNorm layers
        bn_layers = [m for m in trunk.net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(bn_layers) == 0

    def test_with_batch_norm(self):
        cfg = SharedTrunkConfig(hidden_dims=[16], latent_dim=4, use_batch_norm=True)
        trunk = SharedTrunk(5, cfg)
        bn_layers = [m for m in trunk.net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(bn_layers) >= 1

    def test_dropout_present(self):
        cfg = SharedTrunkConfig(hidden_dims=[32, 16], latent_dim=4, dropout=0.3, use_batch_norm=False)
        trunk = SharedTrunk(8, cfg)
        drop_layers = [m for m in trunk.net.modules() if isinstance(m, torch.nn.Dropout)]
        assert len(drop_layers) >= 1

    def test_gradients_flow(self):
        trunk = self._make_trunk()
        x = torch.randn(4, 8, requires_grad=False)
        h = trunk(x)
        loss = h.sum()
        loss.backward()
        for p in trunk.parameters():
            if p.requires_grad:
                assert p.grad is not None


# ---------------------------------------------------------------------------
# FrequencyHead
# ---------------------------------------------------------------------------

class TestFrequencyHead:
    def test_output_shape(self):
        head = FrequencyHead(latent_dim=8)
        h = torch.randn(5, 8)
        log_exp = torch.zeros(5)
        out = head(h, log_exp)
        assert out.shape == (5,)

    def test_exposure_offset_applied(self):
        head = FrequencyHead(latent_dim=4)
        h = torch.zeros(3, 4)
        log_exp_zero = torch.zeros(3)
        log_exp_one = torch.ones(3) * 0.5
        out_zero = head(h, log_exp_zero)
        out_one = head(h, log_exp_one)
        # With h=0 and linear init at zero, difference should equal log_exp difference
        torch.testing.assert_close(out_one - out_zero, torch.ones(3) * 0.5, atol=1e-5, rtol=0)

    def test_exposure_2d_input(self):
        head = FrequencyHead(latent_dim=4)
        h = torch.randn(3, 4)
        log_exp = torch.zeros(3, 1)
        out = head(h, log_exp)
        assert out.shape == (3,)

    def test_gradient_flows(self):
        head = FrequencyHead(latent_dim=4)
        h = torch.randn(3, 4)
        log_exp = torch.zeros(3)
        out = head(h, log_exp)
        out.sum().backward()
        assert head.linear.weight.grad is not None


# ---------------------------------------------------------------------------
# SeverityHead
# ---------------------------------------------------------------------------

class TestSeverityHead:
    def test_output_shape_no_gamma(self):
        head = SeverityHead(latent_dim=8, use_explicit_gamma=False)
        h = torch.randn(5, 8)
        out = head(h)
        assert out.shape == (5,)

    def test_output_shape_with_gamma(self):
        head = SeverityHead(latent_dim=8, use_explicit_gamma=True)
        h = torch.randn(5, 8)
        n = torch.tensor([0., 1., 2., 0., 3.])
        out = head(h, n)
        assert out.shape == (5,)

    def test_gamma_parameter_exists(self):
        head = SeverityHead(latent_dim=4, use_explicit_gamma=True)
        assert head.gamma is not None
        assert head.gamma.shape == (1,)

    def test_no_gamma_parameter(self):
        head = SeverityHead(latent_dim=4, use_explicit_gamma=False)
        assert head.gamma is None

    def test_gamma_initialized_to_zero(self):
        head = SeverityHead(latent_dim=4, use_explicit_gamma=True)
        assert head.gamma.item() == pytest.approx(0.0)

    def test_raises_without_n_when_gamma_used(self):
        head = SeverityHead(latent_dim=4, use_explicit_gamma=True)
        h = torch.randn(3, 4)
        with pytest.raises(ValueError, match="n_claims must be supplied"):
            head(h)

    def test_gamma_effect_on_output(self):
        """When gamma>0 and N>0, log_mu should increase vs N=0."""
        head = SeverityHead(latent_dim=4, use_explicit_gamma=True)
        with torch.no_grad():
            head.gamma.fill_(0.5)
        h = torch.zeros(2, 4)  # zero weights → linear output = 0
        n_zero = torch.zeros(2)
        n_one = torch.ones(2)
        out_zero = head(h, n_zero)
        out_one = head(h, n_one)
        assert (out_one > out_zero).all()

    def test_gradient_flows_through_gamma(self):
        head = SeverityHead(latent_dim=4, use_explicit_gamma=True)
        h = torch.randn(3, 4)
        n = torch.tensor([1., 2., 0.])
        out = head(h, n)
        out.sum().backward()
        assert head.gamma.grad is not None


# ---------------------------------------------------------------------------
# DependentFreqSevNet
# ---------------------------------------------------------------------------

class TestDependentFreqSevNet:
    def _make_net(self, in_features=8, use_gamma=True):
        cfg = SharedTrunkConfig(hidden_dims=[16], latent_dim=8, use_batch_norm=False)
        return DependentFreqSevNet(in_features, cfg, use_explicit_gamma=use_gamma)

    def test_forward_shapes(self):
        net = self._make_net()
        x = torch.randn(10, 8)
        log_exp = torch.zeros(10)
        n = torch.zeros(10)
        ll, lm, phi = net(x, log_exp, n)
        assert ll.shape == (10,)
        assert lm.shape == (10,)
        assert phi.shape == (1,)

    def test_forward_without_gamma(self):
        net = self._make_net(use_gamma=False)
        x = torch.randn(5, 8)
        log_exp = torch.zeros(5)
        ll, lm, phi = net(x, log_exp)
        assert ll.shape == (5,)
        assert lm.shape == (5,)

    def test_phi_is_positive(self):
        net = self._make_net()
        x = torch.randn(3, 8)
        log_exp = torch.zeros(3)
        n = torch.zeros(3)
        _, _, phi = net(x, log_exp, n)
        assert phi.item() > 0.0

    def test_gamma_property(self):
        net = self._make_net(use_gamma=True)
        assert net.gamma is not None

    def test_gamma_none_when_not_used(self):
        net = self._make_net(use_gamma=False)
        assert net.gamma is None

    def test_latent_output_shape(self):
        net = self._make_net()
        net.eval()
        x = torch.randn(6, 8)
        h = net.latent(x)
        assert h.shape == (6, 8)

    def test_count_parameters_keys(self):
        net = self._make_net()
        counts = net.count_parameters()
        assert "trunk" in counts
        assert "freq_head" in counts
        assert "sev_head" in counts
        assert "total" in counts
        assert counts["total"] > 0

    def test_parameter_count_consistency(self):
        net = self._make_net()
        counts = net.count_parameters()
        # total should be sum of components
        assert counts["total"] == sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )

    def test_joint_gradients_through_trunk(self):
        """Both Poisson and Gamma gradients should update trunk weights."""
        net = self._make_net()
        x = torch.randn(4, 8)
        log_exp = torch.zeros(4)
        n = torch.tensor([0., 1., 2., 0.])
        ll, lm, phi = net(x, log_exp, n)

        # Simulate joint loss
        freq_loss = -(n * ll - torch.exp(ll)).mean()
        pos = n > 0
        sev_loss = -lm[pos].mean()
        total_loss = freq_loss + sev_loss
        total_loss.backward()

        trunk_grads = [
            p.grad for p in net.trunk.parameters() if p.requires_grad and p.grad is not None
        ]
        assert len(trunk_grads) > 0, "Trunk received no gradients"

    def test_extra_repr_contains_phi(self):
        net = self._make_net()
        r = net.extra_repr()
        assert "phi" in r

    def test_extra_repr_contains_gamma(self):
        net = self._make_net(use_gamma=True)
        r = net.extra_repr()
        assert "gamma" in r

    def test_default_trunk_config(self):
        net = DependentFreqSevNet(in_features=4)
        # Should use default SharedTrunkConfig without raising
        x = torch.randn(2, 4)
        log_exp = torch.zeros(2)
        n = torch.zeros(2)
        ll, lm, phi = net(x, log_exp, n)
        assert ll.shape == (2,)
