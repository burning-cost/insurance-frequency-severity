"""
insurance_frequency_severity.dependent
=======================================
Dependent frequency-severity neural two-part model for insurance pricing.

The core idea is multi-task learning: a single shared encoder trunk processes
covariates and produces a latent representation that feeds both a Poisson
frequency head and a Gamma severity head.  Gradients from both losses flow
through the trunk simultaneously, so it learns features that are jointly
informative for frequency *and* severity.  That shared information is where the
implicit frequency-severity dependence lives.

On top of the latent dependence you can add the explicit Garrido-Genest-Schulz
conditional covariate (log μ += γ·N), which gives a semi-analytical pure
premium correction and a directly interpretable dependence parameter.

This subpackage is distinct from the Sarmanov copula approach in the parent
package.  Use Sarmanov when you need an analytical bivariate density and a
simple omega parameter for a regulator; use this subpackage when you have a
large dataset, suspect nonlinear interactions, and want a single model that
learns both tasks jointly.

Public API
----------
Model
~~~~~
``DependentFreqSevNet``   – PyTorch nn.Module (shared trunk + heads)
``SharedTrunkConfig``     – dataclass for trunk hyperparameters
``FrequencyHead``         – Poisson head module
``SeverityHead``          – Gamma head module

Training
~~~~~~~~
``JointLoss``             – Poisson + Gamma NLL with configurable balancing
``DependentFSTrainer``    – training loop with early stopping and LR scheduling

Wrapper
~~~~~~~
``DependentFSModel``      – sklearn-compatible estimator (fit/predict/score)

Premium
~~~~~~~
``PurePremiumEstimator``  – Monte Carlo + optional MGF analytical correction

Diagnostics
~~~~~~~~~~~
``DependentFSDiagnostics`` – Lorenz, calibration, dependence tests, latent corr

Data
~~~~
``FreqSevDataset``        – PyTorch Dataset with exposure handling
``prepare_features``      – numeric encoding helper

Benchmarks
~~~~~~~~~~
``make_dependent_claims`` – synthetic claims with known γ dependence
``make_independent_claims`` – synthetic independent baseline
"""

from insurance_frequency_severity.dependent.model import (
    DependentFreqSevNet,
    FrequencyHead,
    SeverityHead,
    SharedTrunkConfig,
)
from insurance_frequency_severity.dependent.training import DependentFSTrainer, JointLoss
from insurance_frequency_severity.dependent.wrapper import DependentFSModel
from insurance_frequency_severity.dependent.premium import PurePremiumEstimator
from insurance_frequency_severity.dependent.diagnostics import DependentFSDiagnostics
from insurance_frequency_severity.dependent.data import FreqSevDataset, prepare_features
from insurance_frequency_severity.dependent.benchmarks import make_dependent_claims, make_independent_claims

__all__ = [
    "DependentFreqSevNet",
    "FrequencyHead",
    "SeverityHead",
    "SharedTrunkConfig",
    "JointLoss",
    "DependentFSTrainer",
    "DependentFSModel",
    "PurePremiumEstimator",
    "DependentFSDiagnostics",
    "FreqSevDataset",
    "prepare_features",
    "make_dependent_claims",
    "make_independent_claims",
]
