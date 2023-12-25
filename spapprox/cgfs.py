#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from .cgf_base import UnivariateCumulantGeneratingFunction
from .util import type_wrapper


def norm(loc=0, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, loc=loc, scale=scale: loc * t + scale**2 * t**2 / 2,
        dK=lambda t, loc=loc, scale=scale: loc + scale**2 * t,
        dK_inv=lambda x, loc=loc, scale=scale: (x - loc) / scale**2,
        d2K=lambda t, scale=scale: scale**2 + 0 * t,
        d3K=lambda t: 0 * t,
    )


def exponential(scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, scale=scale: np.log(1 / (1 - scale * t)),
        dK=lambda t, scale=scale: scale / (1 - scale * t),
        dK_inv=lambda x, scale=scale: (1 - scale / x) / scale,
        d2K=lambda t, scale=scale: 1 / (1 / scale - t) ** 2,
        d3K=lambda t, scale=scale: 2 / (1 / scale - t) ** 3,
        domain=lambda t: UnivariateCumulantGeneratingFunction._is_in_domain(
            t, g=-np.inf, l=1 / scale
        ),
    )


def gamma(a=1, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, a=a, scale=scale: -a * np.log(1 - scale * t),
        dK=lambda t, a=a, scale=scale: a * scale / (1 - scale * t),
        dK_inv=lambda x, a=a, scale=scale: (1 - a * scale / x) / scale,
        d2K=lambda t, a=a, scale=scale: a * scale**2 / (1 - scale * t) ** 2,
        d3K=lambda t, a=a, scale=scale: 2 * a * scale**3 / (1 - scale * t) ** 3,
        domain=lambda t: UnivariateCumulantGeneratingFunction._is_in_domain(
            t, g=-np.inf, l=1 / scale
        ),
    )


def chi2(df=1):
    return gamma(a=df / 2, scale=2)


def laplace(loc=0, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, loc=loc, scale=scale: loc * t - np.log(1 - scale**2 * t**2),
        dK=lambda t, loc=loc, scale=scale: loc + 2 * scale**2 * t / (1 - scale**2 * t**2),
        dK_inv=lambda x, loc=loc, scale=scale: (scale - np.sqrt(scale**2 + (x - loc) ** 2))
        / (-scale * (x - loc)),
        d2K=lambda t, scale=scale: 2
        * scale**2
        * (1 + scale**2 * t**2)
        / (1 - scale**2 * t**2) ** 2,
        d3K=lambda t, scale=scale: 4
        * scale**4
        * t
        * (3 + scale**2 * t**2)
        / (1 - scale**2 * t**2) ** 3,
        domain=lambda t, scale=scale: UnivariateCumulantGeneratingFunction._is_in_domain(
            t, g=-1 / scale, l=1 / scale
        ),
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
        domain=lambda t, n=sample_size, cgf=cgf: cgf.domain(t / n),
    )


def univariate_empirical(x):
    """
    Given a vector :math`x` with observations of a univariate random variable,
    draw one of then with equal probability.
    """

    @type_wrapper(xloc=0)
    def K(t, x=x):
        if len(t.shape) == 0:
            y = np.log(np.exp(t * x).mean())
        else:
            y = np.log(np.exp(np.atleast_2d(t).T.dot(np.atleast_2d(x))).mean(axis=1))
        return y.tolist() if len(t.shape) == 1 else y

    @type_wrapper(xloc=0)
    def dK(t, x=x):
        if len(t.shape) == 0:
            y = (x * np.exp(t * x)).mean() / (np.exp(t * x).mean())
        else:
            y = (x * np.exp(np.atleast_2d(t).T.dot(np.atleast_2d(x)))).mean(axis=1) / (
                np.exp(np.atleast_2d(t).T.dot(np.atleast_2d(x))).mean(axis=1)
            )
        return y.tolist() if len(t.shape) == 1 else y

    return UnivariateCumulantGeneratingFunction(K, dK=dK)


# TODO: add asymmetric laplace?

# TODO: add generalized normal distribution

# TODO: add asymmetric generalized normal distribution


def poisson(mu=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, mu=mu: mu * (np.exp(t) - 1),
        dK=lambda t, mu=mu: mu * np.exp(t),
        dK_inv=lambda x, mu=mu: np.log(x / mu),
        d2K=lambda t, mu=mu: mu * np.exp(t),
        d3K=lambda t, mu=mu: mu * np.exp(t),
        domain=lambda t: UnivariateCumulantGeneratingFunction._is_in_domain(t, ge=0, l=np.inf),
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
        domain=lambda t: UnivariateCumulantGeneratingFunction._is_in_domain(
            t, g=-np.inf, l=np.inf
        ),
    )
