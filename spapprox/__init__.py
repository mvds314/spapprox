"""Saddle point approximations of distributions from their cumulant generating functions."""

from importlib.metadata import PackageNotFoundError, version

from .cgf_base import (
    MultivariateCumulantGeneratingFunction,
    UnivariateCumulantGeneratingFunction,
)

from .cgfs import (
    asymmetric_generalized_normal,
    asymmetric_laplace,
    bernoulli,
    binomial,
    bivariate_gamma,
    bivariate_poisson,
    chi2,
    exponential,
    gamma,
    generalized_normal,
    jensen_multivariate_gamma,
    laplace,
    multinomial,
    multivariate_norm,
    negative_binomial,
    norm,
    poisson,
    univariate_empirical,
    univariate_sample_mean,
)
from .domain import Domain
from .spa import (
    BivariateSaddlePointApprox,
    EstimatingEquationSaddlePointApprox,
    MultivariateEstimatingEquationSaddlePointApprox,
    MultivariateSaddlePointApprox,
    UnivariateSaddlePointApprox,
    UnivariateSaddlePointApproxMean,
)
from .util import Timer

try:
    __version__ = version("spapprox")
except PackageNotFoundError:  # pragma: no cover, e.g. running from a source tree
    __version__ = "unknown"

__all__ = [
    "asymmetric_generalized_normal",
    "asymmetric_laplace",
    "bernoulli",
    "binomial",
    "bivariate_gamma",
    "bivariate_poisson",
    "BivariateSaddlePointApprox",
    "chi2",
    "Domain",
    "EstimatingEquationSaddlePointApprox",
    "exponential",
    "gamma",
    "generalized_normal",
    "jensen_multivariate_gamma",
    "laplace",
    "multinomial",
    "multivariate_norm",
    "MultivariateCumulantGeneratingFunction",
    "MultivariateEstimatingEquationSaddlePointApprox",
    "MultivariateSaddlePointApprox",
    "negative_binomial",
    "norm",
    "poisson",
    "Timer",
    "univariate_empirical",
    "univariate_sample_mean",
    "UnivariateCumulantGeneratingFunction",
    "UnivariateSaddlePointApprox",
    "UnivariateSaddlePointApproxMean",
]
