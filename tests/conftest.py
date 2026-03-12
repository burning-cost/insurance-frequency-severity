"""
Test fixtures for insurance-frequency-severity.

All DGPs use known parameters so tests can verify recovery of omega, sign of
correction factors, etc.
"""
import numpy as np
import pytest


RNG_SEED = 42


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="session")
def nb_gamma_dgp(rng):
    """
    Simulate from Sarmanov(NB, Gamma, omega_true) with n=5000 policies.

    NB: mu=0.15, alpha=0.8 (sparse UK motor style)
    Gamma: shape=1.2, mu_s=2000 (average severity)
    omega_true: -3.0 (negative dependence — high-N policies have lower S)
    """
    n_policies = 5000
    mu_n = 0.15
    alpha = 0.8
    mu_s = 2000.0
    shape = 1.2
    omega_true = -3.0
    kernel_theta = 0.5
    kernel_alpha = 0.0005

    from insurance_frequency_severity.copula import SarmanovCopula
    copula = SarmanovCopula(
        freq_family="nb",
        sev_family="gamma",
        omega=omega_true,
        kernel_theta=kernel_theta,
        kernel_alpha=kernel_alpha,
    )

    freq_params = {"mu": mu_n, "alpha": alpha}
    sev_params = {"mu": mu_s, "shape": shape}

    n_samp, s_samp = copula.sample(
        n_policies, freq_params, sev_params, rng=rng
    )
    return {
        "n": n_samp,
        "s": s_samp,
        "mu_n": mu_n,
        "alpha": alpha,
        "mu_s": mu_s,
        "shape": shape,
        "omega_true": omega_true,
        "kernel_theta": kernel_theta,
        "kernel_alpha": kernel_alpha,
        "freq_params": freq_params,
        "sev_params": sev_params,
    }


@pytest.fixture(scope="session")
def poisson_gamma_dgp(rng):
    """
    Simulate from Sarmanov(Poisson, Gamma, omega=0) — independence DGP.
    """
    n_policies = 3000
    mu_n = 0.12
    mu_s = 1500.0
    shape = 1.5

    from insurance_frequency_severity.copula import SarmanovCopula
    copula = SarmanovCopula(
        freq_family="poisson",
        sev_family="gamma",
        omega=0.0,
        kernel_theta=0.5,
        kernel_alpha=0.001,
    )

    freq_params = {"mu": mu_n}
    sev_params = {"mu": mu_s, "shape": shape}

    n_samp, s_samp = copula.sample(n_policies, freq_params, sev_params, rng=rng)
    return {
        "n": n_samp,
        "s": s_samp,
        "mu_n": mu_n,
        "mu_s": mu_s,
        "shape": shape,
        "omega_true": 0.0,
        "freq_params": freq_params,
        "sev_params": sev_params,
    }


@pytest.fixture(scope="session")
def mock_freq_glm(nb_gamma_dgp):
    """Minimal mock GLM object with .fittedvalues and .predict()."""
    dgp = nb_gamma_dgp
    n_policies = len(dgp["n"])
    mu_n = np.full(n_policies, dgp["mu_n"])

    class MockNBGLM:
        def __init__(self, mu, alpha, family_name):
            self.fittedvalues = mu
            self._alpha = alpha
            self._family_name = family_name

            class Family:
                def __init__(self, alpha):
                    self.alpha = alpha
            self.family = Family(alpha)

            class Model:
                def __init__(self, alpha):
                    class Fam:
                        pass
                    self.family = Fam()
                    self.family.alpha = alpha
            self.model = Model(alpha)

        def predict(self, X=None):
            if X is not None:
                return np.full(len(X), float(self.fittedvalues[0]))
            return self.fittedvalues

    return MockNBGLM(mu_n, dgp["alpha"], "nb")


@pytest.fixture(scope="session")
def mock_sev_glm(nb_gamma_dgp):
    """Minimal mock severity GLM with Gamma family."""
    dgp = nb_gamma_dgp
    n_policies = len(dgp["n"])
    mu_s = np.full(n_policies, dgp["mu_s"])

    class MockGammaGLM:
        def __init__(self, mu, shape):
            self.fittedvalues = mu
            self.scale = 1.0 / shape

            class Family:
                pass
            self.family = Family()
            self.family.__class__.__name__ = "Gamma"

            class Model:
                pass
            self.model = Model()
            self.model.family = Family()
            self.model.family.__class__.__name__ = "Gamma"

        def predict(self, X=None):
            if X is not None:
                return np.full(len(X), float(self.fittedvalues[0]))
            return self.fittedvalues

    return MockGammaGLM(mu_s, dgp["shape"])
