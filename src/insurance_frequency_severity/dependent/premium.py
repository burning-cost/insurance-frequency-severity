"""
premium.py
----------
Pure premium computation for the dependent frequency-severity model.

Pure premium is E[total loss per unit exposure] = E[N · average_severity / t].
Because frequency and severity are correlated through the shared trunk (and
optionally through the explicit γ parameter), this is NOT simply E[N] · E[Y].

Two computation modes are provided:

1. **Monte Carlo (always available)**:
   Sample N ~ Poisson(λ) per policy, then for each policy with sampled N > 0
   sample average severity from Gamma(shape=N/φ, scale=φ·μ/N).  Average the
   realisations to estimate E[Z | x].

2. **Semi-analytical (only when γ is used)**:
   When the severity head includes γ·N, the Poisson MGF gives a closed form:

       E[Z | x] = exp{SevHead(h) + γ} · exp{λ(e^γ − 1)} · λ · t⁻¹

   which follows from Theorem 1 of arXiv:2106.10770v2 (NeurFS paper), or
   equivalently from Garrido-Genest-Schulz (2016) Eq (5).

   The analytical formula requires γ to be the ONLY source of dependence.  In
   this library, the trunk also induces implicit dependence, so the analytical
   formula is strictly valid only as an approximation.  We call it
   "semi-analytical" to signal this: it gives a fast estimate that corrects for
   the explicit γ term but ignores residual latent dependence.

   For most practical purposes the semi-analytical result is adequate and much
   faster than MC at large portfolio sizes.

Requires torch. Install with: pip install insurance-frequency-severity[neural]
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    import torch
    from torch import Tensor
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    Tensor = None  # type: ignore[assignment]


def _require_torch(feature: str = "This feature") -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"{feature} requires torch. "
            "Install with: pip install insurance-frequency-severity[neural]"
        )


class PurePremiumEstimator:
    """Compute per-policy pure premium predictions from model outputs.

    Requires torch. Install with: pip install insurance-frequency-severity[neural]

    This class is designed to be called after a forward pass of
    ``DependentFreqSevNet``.  It does not hold a reference to the model;
    the caller passes tensors directly.

    Parameters
    ----------
    n_mc:
        Number of Monte Carlo samples per policy for the MC estimator.
        1000 gives reasonable accuracy; use 5000 for narrow confidence intervals.
    seed:
        Random seed for reproducibility of MC samples.
    device:
        Torch device for MC computations.
    """

    def __init__(
        self,
        n_mc: int = 1000,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        _require_torch("PurePremiumEstimator")
        self.n_mc = n_mc
        self.seed = seed
        self.device = torch.device(device)

    def monte_carlo(
        self,
        log_lambda: "Tensor",
        log_mu: "Tensor",
        phi: "Tensor",
        exposure: "Tensor",
    ) -> "Tensor":
        """Monte Carlo pure premium estimate.

        Samples N_i ~ Poisson(λ_i · t_i / t_i) [unit-rate, then N_i is the
        count] and Y_ij ~ Gamma(N_i/φ, φ·μ_i/N_i) for each realisation.  The
        aggregate pure premium per unit exposure is then mean(N·Ȳ) / t.

        This is fully general — it does not assume any particular dependence
        structure and works whether or not explicit γ is used.

        Parameters
        ----------
        log_lambda:
            Log Poisson rate (including log exposure), shape (n,).
        log_mu:
            Log expected severity, shape (n,).
        phi:
            Gamma dispersion parameter, shape (1,) or scalar.
        exposure:
            Exposure, shape (n,).  Used to normalise the result to a per-unit
            pure premium.

        Returns
        -------
        pp: Tensor of shape (n,)
            Pure premium (expected total loss per unit exposure) for each policy.
        """
        torch.manual_seed(self.seed)
        n_policies = log_lambda.shape[0]
        lambda_ = torch.exp(log_lambda).cpu()  # shape (n,)
        mu = torch.exp(log_mu).cpu()            # shape (n,)
        phi_val = phi.cpu().squeeze().item()
        exp_val = exposure.cpu()

        # Shape: (n_mc, n_policies)
        # Sample Poisson counts for each MC iteration and policy.
        poisson_dist = torch.distributions.Poisson(
            lambda_.unsqueeze(0).expand(self.n_mc, -1)
        )
        n_samp = poisson_dist.sample()  # (n_mc, n_policies)

        # Sample Gamma severities only where n_samp > 0.
        # Shape = alpha of Gamma.  We need to handle n_samp=0 carefully.
        positive_mask = n_samp > 0
        total_loss = torch.zeros(self.n_mc, n_policies)

        # Vectorised: concentration = n_samp / phi; rate = n_samp / (phi * mu)
        # mean of Gamma = concentration / rate = mu.
        # For zero claims, loss = 0.
        alpha = n_samp.float() / phi_val
        rate = n_samp.float() / (phi_val * mu.unsqueeze(0) + 1e-10)

        # Sample from Gamma where positive; leave zeros elsewhere.
        # We clamp alpha and rate to be > 0 to avoid numerical errors.
        safe_alpha = alpha.clamp(min=1e-6)
        safe_rate = rate.clamp(min=1e-6)
        gamma_dist = torch.distributions.Gamma(safe_alpha, safe_rate)
        y_samp = gamma_dist.sample()  # (n_mc, n_policies)

        # Total loss = N * avg_severity (with N * avg_sev = sum of claims)
        # n_samp * y_samp = total claim amount for that realisation
        total_loss = torch.where(positive_mask, n_samp.float() * y_samp, torch.zeros_like(y_samp))

        # Pure premium = E[total_loss] / exposure
        mean_total_loss = total_loss.mean(dim=0)  # (n_policies,)
        pp = mean_total_loss / exp_val
        return pp.to(self.device)

    def analytical(
        self,
        log_lambda: "Tensor",
        log_mu_base: "Tensor",
        gamma: "Tensor",
        exposure: "Tensor",
    ) -> "Tensor":
        """Semi-analytical pure premium for the GGS conditional covariate model.

        Uses the closed form from Garrido-Genest-Schulz (2016) / NeurFS Theorem 1:

            E[Z | x] = exp(log_mu_base + γ) · exp(λ(eᵞ − 1)) · λ

        where log_mu_base = SevHead(h) (without the γ·N term), λ = exp(log_lambda)
        is the Poisson rate for the POLICY (not per unit exposure — i.e. λ = rate·t).

        This formula is exact when γ is the only dependence and frequency is
        Poisson.  When the shared trunk also contributes implicit dependence, this
        is an approximation.

        Parameters
        ----------
        log_lambda:
            Log Poisson rate including exposure, shape (n,).
        log_mu_base:
            Log expected severity WITHOUT the γ·N adjustment, shape (n,).
        gamma:
            The scalar dependence parameter γ.
        exposure:
            Policy exposure, shape (n,).

        Returns
        -------
        pp: Tensor of shape (n,)
            Pure premium per unit exposure.
        """
        lambda_ = torch.exp(log_lambda)      # includes exposure
        g = gamma.squeeze()
        eg = torch.exp(g)

        # E[total_loss | x] = E[N·Y | x]
        #   = exp(log_mu_base + γ) · exp(λ(eᵞ - 1)) · λ
        log_pp_times_exp = (
            log_mu_base + g
            + lambda_ * (eg - 1.0)
            + torch.log(lambda_ + 1e-10)
        )
        total_loss = torch.exp(log_pp_times_exp)
        pp = total_loss / exposure
        return pp

    def confidence_interval(
        self,
        log_lambda: "Tensor",
        log_mu: "Tensor",
        phi: "Tensor",
        exposure: "Tensor",
        alpha: float = 0.05,
    ) -> tuple:
        """Bootstrap confidence interval on the pure premium via MC.

        Returns the (lower, point_estimate, upper) pure premiums, shape (n, 3).

        Parameters
        ----------
        alpha:
            Significance level.  0.05 gives 95% CI.
        """
        torch.manual_seed(self.seed)
        n_policies = log_lambda.shape[0]
        lambda_ = torch.exp(log_lambda).cpu()
        mu = torch.exp(log_mu).cpu()
        phi_val = phi.cpu().squeeze().item()
        exp_val = exposure.cpu()

        poisson_dist = torch.distributions.Poisson(
            lambda_.unsqueeze(0).expand(self.n_mc, -1)
        )
        n_samp = poisson_dist.sample()

        alpha_g = n_samp.float() / phi_val
        rate_g = n_samp.float() / (phi_val * mu.unsqueeze(0) + 1e-10)
        safe_alpha = alpha_g.clamp(min=1e-6)
        safe_rate = rate_g.clamp(min=1e-6)
        gamma_dist = torch.distributions.Gamma(safe_alpha, safe_rate)
        y_samp = gamma_dist.sample()

        total_loss = torch.where(
            n_samp > 0, n_samp.float() * y_samp, torch.zeros_like(y_samp)
        )
        pp_mc = total_loss / exp_val.unsqueeze(0)  # (n_mc, n_policies)

        lo = pp_mc.quantile(alpha / 2, dim=0)
        mid = pp_mc.mean(dim=0)
        hi = pp_mc.quantile(1 - alpha / 2, dim=0)
        return lo, mid, hi
