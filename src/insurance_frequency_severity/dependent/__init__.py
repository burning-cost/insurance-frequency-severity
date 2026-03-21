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

Requires torch for neural network classes.
Install with: pip install insurance-frequency-severity[neural]

Public API
----------
Model
~~~~~
``DependentFreqSevNet``   – PyTorch nn.Module (shared trunk + heads)
``SharedTrunkConfig``     – dataclass for trunk hyperparameters (no torch needed)
``FrequencyHead``         – Poisson head module
``SeverityHead``          – Gamma head module

Training
~~~~~~~~
``JointLoss``             – Poisson + Gamma NLL with configurable balancing
``TrainingConfig``        – dataclass for training hyperparameters (no torch needed)
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
``prepare_features``      – numeric encoding helper (no torch needed)

Benchmarks
~~~~~~~~~~
``make_dependent_claims`` – synthetic claims with known γ dependence
``make_independent_claims`` – synthetic independent baseline
"""

# SharedTrunkConfig and TrainingConfig are safe to import eagerly — pure dataclasses, no torch.
from insurance_frequency_severity.dependent.model import SharedTrunkConfig
from insurance_frequency_severity.dependent.training import TrainingConfig
# prepare_features is also torch-free.
from insurance_frequency_severity.dependent.data import prepare_features
# Benchmark generators use only numpy/scipy.
from insurance_frequency_severity.dependent.benchmarks import make_dependent_claims, make_independent_claims

# All remaining names require torch and are loaded lazily.
_NEURAL_NAMES = {
    "DependentFreqSevNet": ("insurance_frequency_severity.dependent.model", "DependentFreqSevNet"),
    "FrequencyHead": ("insurance_frequency_severity.dependent.model", "FrequencyHead"),
    "SeverityHead": ("insurance_frequency_severity.dependent.model", "SeverityHead"),
    "JointLoss": ("insurance_frequency_severity.dependent.training", "JointLoss"),
    "DependentFSTrainer": ("insurance_frequency_severity.dependent.training", "DependentFSTrainer"),
    "DependentFSModel": ("insurance_frequency_severity.dependent.wrapper", "DependentFSModel"),
    "PurePremiumEstimator": ("insurance_frequency_severity.dependent.premium", "PurePremiumEstimator"),
    "DependentFSDiagnostics": ("insurance_frequency_severity.dependent.diagnostics", "DependentFSDiagnostics"),
    "FreqSevDataset": ("insurance_frequency_severity.dependent.data", "FreqSevDataset"),
    "make_train_val_loaders": ("insurance_frequency_severity.dependent.data", "make_train_val_loaders"),
}


def __getattr__(name: str):
    if name in _NEURAL_NAMES:
        module_path, attr = _NEURAL_NAMES[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'insurance_frequency_severity.dependent' has no attribute {name!r}")


__all__ = [
    "SharedTrunkConfig",
    "TrainingConfig",
    "prepare_features",
    "make_dependent_claims",
    "make_independent_claims",
    # Neural (torch required):
    "DependentFreqSevNet",
    "FrequencyHead",
    "SeverityHead",
    "JointLoss",
    "DependentFSTrainer",
    "DependentFSModel",
    "PurePremiumEstimator",
    "DependentFSDiagnostics",
    "FreqSevDataset",
    "make_train_val_loaders",
]
