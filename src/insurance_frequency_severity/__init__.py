"""
insurance-frequency-severity: Sarmanov copula joint frequency-severity modelling,
plus neural two-part dependent model.

Challenges the independence assumption in the standard two-model GLM framework.
Implements Sarmanov copula for mixed discrete-continuous margins, Gaussian copula
as a comparison, and Garrido's conditional method as the simplest baseline.

For the neural two-part dependent model with shared encoder trunk, see the
``dependent`` subpackage:

>>> from insurance_frequency_severity.dependent import DependentFSModel

Typical usage (Sarmanov)
------------------------
>>> from insurance_frequency_severity import JointFreqSev
>>> model = JointFreqSev(freq_glm=my_nb_glm, sev_glm=my_gamma_glm)
>>> model.fit(claims_df, n_col="claim_count", s_col="avg_severity")
>>> corrections = model.premium_correction(X_new)
"""

from insurance_frequency_severity.copula import (
    SarmanovCopula,
    GaussianCopulaMixed,
    FGMCopula,
)
from insurance_frequency_severity.joint import (
    JointFreqSev,
    ConditionalFreqSev,
)
from insurance_frequency_severity.diagnostics import (
    DependenceTest,
    CopulaGOF,
    compare_copulas,
)
from insurance_frequency_severity.report import JointModelReport
from insurance_frequency_severity import dependent

__version__ = "0.2.0"

__all__ = [
    "SarmanovCopula",
    "GaussianCopulaMixed",
    "FGMCopula",
    "JointFreqSev",
    "ConditionalFreqSev",
    "DependenceTest",
    "CopulaGOF",
    "compare_copulas",
    "JointModelReport",
    "dependent",
]
