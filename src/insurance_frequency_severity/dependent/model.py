"""
model.py
--------
Neural network architecture for the dependent frequency-severity model.

The shared trunk (encoder) maps covariates x ∈ R^p to a latent vector
h ∈ R^d_latent.  The frequency head maps h → log λ with an exposure offset;
the severity head maps h → log μ with an optional explicit dependence term γ·N.

Because the trunk is shared, backpropagation from both the Poisson loss and the
Gamma loss updates the same trunk weights on each step.  This is the mechanism
through which the model learns a latent representation that is jointly
informative for frequency and severity — i.e. where the implicit dependence
lives.

When ``use_explicit_gamma=True``, the model also learns a scalar γ.  At
inference time this allows a semi-analytical pure-premium correction via the
Poisson moment generating function (Theorem 1 of the NeurFS paper,
arXiv:2106.10770v2), rather than pure Monte Carlo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SharedTrunkConfig:
    """Hyperparameters for the shared encoder trunk.

    Parameters
    ----------
    hidden_dims:
        Width of each hidden layer in the trunk.  The trunk has
        ``len(hidden_dims)`` hidden layers.  A good starting point for motor
        pricing is ``[128, 64]``; reduce to ``[64, 32]`` for smaller datasets.
    latent_dim:
        Dimension of the latent representation h that feeds both heads.
        Typically 16–64.  Higher values give more capacity but slower training.
    dropout:
        Dropout probability applied after each hidden layer (except the last).
        Set to 0.0 to disable.
    activation:
        Activation function for all hidden layers.  ``"elu"`` is preferred for
        count/severity data because it produces smooth gradients near zero.
        ``"relu"`` and ``"tanh"`` are also supported.
    use_batch_norm:
        Whether to apply batch normalisation after each hidden layer.  Helps
        training stability when input features have very different scales, but
        can hurt if batch sizes are very small (< 32).
    """

    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    latent_dim: int = 32
    dropout: float = 0.1
    activation: Literal["elu", "relu", "tanh"] = "elu"
    use_batch_norm: bool = True


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

def _activation(name: str) -> nn.Module:
    mapping = {"elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}
    if name not in mapping:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(mapping)}.")
    return mapping[name]


class SharedTrunk(nn.Module):
    """Shared encoder that maps input features to a latent representation.

    Parameters
    ----------
    in_features:
        Number of input features (after preprocessing / encoding).
    config:
        Architecture hyperparameters.
    """

    def __init__(self, in_features: int, config: SharedTrunkConfig) -> None:
        super().__init__()
        self.config = config
        layers: List[nn.Module] = []
        prev_dim = in_features
        for i, dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(_activation(config.activation))
            if config.dropout > 0.0 and i < len(config.hidden_dims) - 1:
                layers.append(nn.Dropout(p=config.dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, config.latent_dim))
        if config.use_batch_norm:
            layers.append(nn.BatchNorm1d(config.latent_dim))
        layers.append(_activation(config.activation))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            Feature matrix of shape (batch, in_features).

        Returns
        -------
        Tensor of shape (batch, latent_dim).
        """
        return self.net(x)


class FrequencyHead(nn.Module):
    """Poisson frequency head.

    Maps latent h to log λ and adds the log exposure offset so that
    the output is log(expected_claims) = log λ + log t.

    Parameters
    ----------
    latent_dim:
        Dimension of the latent representation coming from the trunk.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: Tensor, log_exposure: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h:
            Latent representation, shape (batch, latent_dim).
        log_exposure:
            Log of the exposure (time at risk), shape (batch,) or (batch, 1).

        Returns
        -------
        log_lambda: Tensor of shape (batch,).
            log(λ_i · t_i) — the Poisson rate including exposure.
        """
        log_exposure = log_exposure.view(-1)
        log_lambda_x = self.linear(h).squeeze(-1)
        return log_lambda_x + log_exposure


class SeverityHead(nn.Module):
    """Gamma severity head.

    Maps latent h to log μ (the expected average severity given x).  Optionally
    includes an explicit dependence term γ·N where N is the realised claim count
    and γ is a learnable scalar.

    Parameters
    ----------
    latent_dim:
        Dimension of the latent representation coming from the trunk.
    use_explicit_gamma:
        If True, a scalar γ is learned and the forward pass accepts an
        ``n_claims`` argument.  When γ=0 at convergence, the latent dependence
        is sufficient to describe the joint structure.
    """

    def __init__(
        self,
        latent_dim: int,
        use_explicit_gamma: bool = True,
    ) -> None:
        super().__init__()
        self.use_explicit_gamma = use_explicit_gamma
        self.linear = nn.Linear(latent_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        if use_explicit_gamma:
            # Initialise γ to zero so the model starts at independence.
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("gamma", None)

    def forward(
        self,
        h: Tensor,
        n_claims: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        h:
            Latent representation, shape (batch, latent_dim).
        n_claims:
            Realised claim counts, shape (batch,).  Required when
            ``use_explicit_gamma=True``; ignored otherwise.

        Returns
        -------
        log_mu: Tensor of shape (batch,).
            log(μ_i), the log expected average severity.
        """
        log_mu = self.linear(h).squeeze(-1)
        if self.use_explicit_gamma:
            if n_claims is None:
                raise ValueError(
                    "n_claims must be supplied when use_explicit_gamma=True."
                )
            log_mu = log_mu + self.gamma * n_claims.float()
        return log_mu


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DependentFreqSevNet(nn.Module):
    """Shared-trunk frequency-severity neural network.

    Architecture::

        x ∈ R^p  →  SharedTrunk  →  h ∈ R^d_latent
                                         │
                         ┌───────────────┴──────────────┐
                    FrequencyHead                  SeverityHead
                    log λ + log t               log μ [+ γ·N]
                         │                             │
                    Poisson loss                  Gamma loss
                         └──────── joint loss ─────────┘

    Both Poisson and Gamma gradients flow through the shared trunk.  This is the
    mechanism by which the model captures frequency-severity dependence in the
    latent space.

    Parameters
    ----------
    in_features:
        Number of input features after preprocessing.
    trunk_config:
        Architecture for the shared encoder.  See ``SharedTrunkConfig``.
    use_explicit_gamma:
        Whether to include the γ·N conditional covariate term in the severity
        head.  Recommended: ``True`` when you want an interpretable dependence
        parameter and a semi-analytical pure premium.

    Attributes
    ----------
    trunk:
        The shared ``SharedTrunk`` module.
    freq_head:
        ``FrequencyHead`` module.
    sev_head:
        ``SeverityHead`` module.
    log_phi:
        Learnable log of the Gamma dispersion parameter φ.  Initialised to 0
        (φ=1).  Shared across observations (one scalar per model).
    """

    def __init__(
        self,
        in_features: int,
        trunk_config: Optional[SharedTrunkConfig] = None,
        use_explicit_gamma: bool = True,
    ) -> None:
        super().__init__()
        if trunk_config is None:
            trunk_config = SharedTrunkConfig()
        self.trunk_config = trunk_config
        self.use_explicit_gamma = use_explicit_gamma
        self.trunk = SharedTrunk(in_features, trunk_config)
        self.freq_head = FrequencyHead(trunk_config.latent_dim)
        self.sev_head = SeverityHead(trunk_config.latent_dim, use_explicit_gamma)
        # Log Gamma dispersion: φ = exp(log_phi).  One scalar for the whole model.
        self.log_phi = nn.Parameter(torch.zeros(1))

    @property
    def phi(self) -> Tensor:
        """Gamma dispersion parameter φ = exp(log φ)."""
        return torch.exp(self.log_phi)

    @property
    def gamma(self) -> Optional[Tensor]:
        """Explicit dependence parameter γ (None if not used)."""
        if self.use_explicit_gamma:
            return self.sev_head.gamma
        return None

    def forward(
        self,
        x: Tensor,
        log_exposure: Tensor,
        n_claims: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x:
            Feature matrix, shape (batch, in_features).
        log_exposure:
            Log exposure, shape (batch,) or (batch, 1).
        n_claims:
            Realised claim counts, shape (batch,).  Only needed when
            ``use_explicit_gamma=True``.

        Returns
        -------
        log_lambda: shape (batch,)
            Log Poisson rate (frequency × exposure).
        log_mu: shape (batch,)
            Log expected average severity.
        phi: shape (1,)
            Gamma dispersion parameter.
        """
        h = self.trunk(x)
        log_lambda = self.freq_head(h, log_exposure)
        log_mu = self.sev_head(h, n_claims)
        return log_lambda, log_mu, self.phi

    def latent(self, x: Tensor) -> Tensor:
        """Return the shared latent representation h for diagnostic purposes.

        Parameters
        ----------
        x:
            Feature matrix, shape (batch, in_features).

        Returns
        -------
        Tensor of shape (batch, latent_dim).
        """
        self.eval()
        with torch.no_grad():
            return self.trunk(x)

    def count_parameters(self) -> dict:
        """Return a breakdown of trainable parameter counts by component."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "trunk": _count(self.trunk),
            "freq_head": _count(self.freq_head),
            "sev_head": _count(self.sev_head),
            "log_phi": 1,
            "total": _count(self),
        }

    def extra_repr(self) -> str:
        cfg = self.trunk_config
        gamma_str = f", gamma={self.sev_head.gamma.item():.4f}" if self.use_explicit_gamma else ""
        return (
            f"hidden_dims={cfg.hidden_dims}, latent_dim={cfg.latent_dim}, "
            f"phi={self.phi.item():.4f}{gamma_str}"
        )
