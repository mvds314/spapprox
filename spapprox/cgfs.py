# -*- coding: utf-8 -*-
import numpy as np

from .cgf_base import (
    UnivariateCumulantGeneratingFunction,
    MultivariateCumulantGeneratingFunction,
)
from .domain import Domain
from .util import type_wrapper


def norm(loc=0, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t: t**2 / 2,
        dK=lambda t: t,
        dK_inv=lambda x: x,
        d2K=lambda t: np.ones(t.shape),
        d3K=lambda t: np.zeros(t.shape),
        loc=loc,
        scale=scale,
    )


def multivariate_norm(loc=0, scale=None, dim=None, cov=None):
    """
    Multivariate normal distribution with mean vector `loc` and covariance matrix `scale`.
    Scale can also be a vector of standard deviations.

    Parameters
    ----------
    loc: array_like, or float
       location vector
    scale: matrix, array_like, or float
       scale matrix. Whan an array is provided, it is interpreted as a vector of standard deviations.
    dim: int
       dimension of the multivariate normal distribution. When not specified, it is either
       inferred or defaults to 2 if, loc and scale can be scalars.
    cov: matrix, array_like, or float
       can be used instead of scale to provide the covariance matrix. The scale equals
       the Cholesky decomposition of the covariance matrix.

    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications
    """
    # Initiliaze scale
    if cov is None and scale is None:
        scale = 1
    elif cov is not None:
        assert scale is None, "Cannot specify both scale and cov"
        scale = np.linalg.cholesky(cov)
    assert scale is not None
    scale = np.asanyarray(scale)
    # Initialize loc
    assert loc is not None
    loc = np.asanyarray(loc)
    # Infer dimension
    if dim is None:
        if loc is not None and loc.ndim > 0:
            dim = loc.shape[0]
        elif scale is not None and scale.ndim > 0:
            dim = scale.shape[0]
        else:
            dim = 2
    # Validate input
    assert (
        loc is None or loc.ndim == 0 or (loc.ndim == 1 and loc.shape[0] == dim)
    ), "loc has wrong shape"
    assert (
        scale is None
        or scale.ndim == 0
        or (scale.ndim == 1 and scale.shape[0] == dim)
        or (scale.ndim == 2 and scale.shape[1] == dim)
    ), "scale has wrong shape"
    # Return
    return MultivariateCumulantGeneratingFunction(
        K=lambda t: np.sum(t * t, axis=-1) / 2,
        dim=dim,
        dK=lambda t: t,
        dK_inv=lambda x: x,
        d2K=lambda t: (np.tile(np.eye(dim), (t.shape[0], 1, 1)) if t.ndim == 2 else np.eye(dim)),
        d3K=lambda t: np.zeros(t.shape),
        loc=loc if loc is None or loc.ndim > 0 else loc.tolist(),
        scale=scale if scale is None or scale.ndim > 0 else scale.tolist(),
    )


def exponential(loc=0, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t: np.log(1 / (1 - t)),
        dK=lambda t: 1 / (1 - t),
        dK_inv=lambda x: 1 - 1 / x,
        d2K=lambda t: 1 / (1 - t) ** 2,
        d3K=lambda t: 2 / (1 - t) ** 3,
        domain=Domain(l=1),
        loc=loc,
        scale=scale,
    )


def gamma(a=1, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, a=a: -a * np.log(1 - t),
        dK=lambda t, a=a: a / (1 - t),
        dK_inv=lambda x, a=a: (1 - a / x),
        d2K=lambda t, a=a: a / (1 - t) ** 2,
        d3K=lambda t, a=a: 2 * a / (1 - t) ** 3,
        domain=Domain(l=1),
        scale=scale,
    )


def chi2(df=1):
    return gamma(a=df / 2, scale=2)


def laplace(loc=0, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t: -np.log(1 - t**2),
        dK=lambda t: 2 * t / (1 - t**2),
        dK_inv=lambda x: (1 - np.sqrt(1 + x**2)) / (-1 * x),
        d2K=lambda t: 2 * (1 + t**2) / (1 - t**2) ** 2,
        d3K=lambda t: 4 * t * (3 + t**2) / (1 - t**2) ** 3,
        domain=Domain(g=-1, l=1),
        loc=loc,
        scale=scale,
    )


def univariate_sample_mean(cgf, sample_size):
    r"""
    Given the cumulant generating function of a univariate random variable, this class
    provides the saddle point approximation of the sample mean of the random variable.

    Given :math:`n` i.i.d. random variables :math:`X_1, \ldots, X_n` with
    cumulant generating function :math:`K`, the cumulant generating function :math:`K_n`
    of the sample mean :math:`\bar{X}` is given by

    .. math::
        K_{\bar{X}}(t) = \sum_{i=1}^n 1/n*K_i(t)= \sum_{i=1}^n K_i(t/n) = n K(t/n).
    """
    assert isinstance(cgf, UnivariateCumulantGeneratingFunction)
    assert isinstance(sample_size, int) and sample_size > 0
    return UnivariateCumulantGeneratingFunction(
        lambda t, n=sample_size, cgf=cgf: n * cgf.K(t / n),
        dK=lambda t, n=sample_size, cgf=cgf: cgf.dK(t / n),
        dK_inv=lambda x, n=sample_size, cgf=cgf: n * cgf.dK_inv(x),
        d2K=lambda t, n=sample_size, cgf=cgf: cgf.d2K(t / n) / n,
        d3K=lambda t, n=sample_size, cgf=cgf: cgf.d3K(t / n) / n**2,
        domain=sample_size * cgf.domain,
    )


def univariate_empirical(x):
    """
    Given a vector :math`x` with observations of a univariate random variable,
    draw one of then with equal probability.
    """

    @type_wrapper(xloc=0)
    def K(t, x=x):
        if t.ndim == 0:
            y = np.log(np.exp(t * x).mean())
        else:
            y = np.log(np.exp(np.atleast_2d(t).T.dot(np.atleast_2d(x))).mean(axis=1))
        return y.tolist() if t.ndim == 1 else y

    @type_wrapper(xloc=0)
    def dK(t, x=x):
        if t.ndim == 0:
            y = (x * np.exp(t * x)).mean() / (np.exp(t * x).mean())
        else:
            y = (x * np.exp(np.atleast_2d(t).T.dot(np.atleast_2d(x)))).mean(axis=1) / (
                np.exp(np.atleast_2d(t).T.dot(np.atleast_2d(x))).mean(axis=1)
            )
        return y.tolist() if t.ndim == 1 else y

    return UnivariateCumulantGeneratingFunction(K, dK=dK)


# TODO: add asymmetric laplace?

# TODO: add generalized normal distribution

# TODO: add asymmetric generalized normal distribution

# TODO: multivariate Gamma (See Butler 2007, page 77; Huzurbar 2012)

# TODO: multinoimal distribution (see Butler 2007, page 80)

# TODO: add bivariate poisson distribution (see Butler 2007, page 81)

# TODO: also see Butler 2007, page 90 for higher order expansions

# TODO: add multinomial bootstrap of the mean

# TODO: see the exercise in Butler for negative binomial distribution, and multivariate logistc distribution


def poisson(mu=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, mu=mu: mu * (np.exp(t) - 1),
        dK=lambda t, mu=mu: mu * np.exp(t),
        dK_inv=lambda x, mu=mu: np.log(x / mu),
        d2K=lambda t, mu=mu: mu * np.exp(t),
        d3K=lambda t, mu=mu: mu * np.exp(t),
        domain=Domain(ge=0),
    )


def binomial(n=1, p=0.5):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, n=n, p=p: n * np.log(p * (np.exp(t) - 1) + 1),
        dK=lambda t, n=n, p=p: n * p / ((1 - p) * np.exp(-t) + p),
        dK_inv=lambda x, n=n, p=p: -np.log((n * p / x - p) / (1 - p)),
        d2K=lambda t, n=n, p=p: n * p * (1 - p) * np.exp(-t) / ((1 - p) * np.exp(-t) + p) ** 2,
        d3K=lambda t, n=n, p=p: n
        * p
        * (1 - p)
        * ((1 - p) * np.exp(-2 * t) - np.exp(-t) * p)
        / ((1 - p) * np.exp(-t) + p) ** 3,
    )
