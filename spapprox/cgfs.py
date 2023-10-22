#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        import numdifftools as nd
    has_numdifftools = True
except ImportError:
    has_numdifftools = False


class cumulant_generating_function:
    r"""
    Base class for cumulant generating function of a distribution

    For a random variable :math:`X` with probability density function :math:`f_X(x)`,
    the cumulant generating function is defined as the logarithm of the moment generating function:

    .. math::
        K_X(t) = \log \mathbb{E} \left[ e^{tX} \right]

    It satisfies the following properties:

    * :math:`\kappa_n = \frac{d^n}{dt^n} K_X(t) \big|_{t=0}'
    * :math:`\kappa_1= \mathbb{E} \left[ X \right]`
    * :math:`\kappa_2= \mathbb{E} \left[ X^2 \right] - \mathbb{E} \left[ X \right]^2= \mathrm{Var} \left[ X \right]`
    * Central moments are polynomial functions of the cumulants
    * Moments are polynomial functions of the cumulants
    * Cumulants are additive for independent random variables
    * Cumulants are homogeneous, i.e., :math:`\kappa_n(\lambda X) = \lambda^n \kappa_n(X)`
    * Cumulants are translation invariant, i.e., :math:`\kappa_n(X + c) = \kappa_n(X)`, for :math:`n=2,3,\dots`

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Cumulant#Cumulant_generating_function
    [2] http://www.scholarpedia.org/article/Cumulants

    Parameters
    ----------
    K : callable
        Cumulant generating function
    dK : callable, optional
        First derivative of the cumulant generating function
    d2K : callable, optional
        Second derivative of the cumulant generating function
    d3K : callable, optional
        Third derivative of the cumulant generating function
    domain : tuple or callable optional
        Domain of the cumulant generating function, either specified through a tuples with greater (less) equal bound for finite values,
        or strictly greater (less) bounds if values are infinite. Alternatively, a callable can be provided that returns True if a value
        is in the domain and False otherwise.
    """

    def __init__(
        self, K, dK=None, d2K=None, d3K=None, dK0=None, d2K0=None, d3K0=None, domain=None
    ):
        self._K = K
        self._dK = dK
        self._d2K = d2K
        self._d3K = d3K
        self._dK0 = dK0
        self._d2K0 = d2K0
        self._d3K0 = d3K0
        if domain is None:
            domain = (-np.inf, np.inf)
        if isinstance(domain, tuple):
            domain = lambda t, domain=domain: self._is_in_domain(
                t, ge=domain[0], le=domain[1], l=np.inf, g=-np.inf
            )
        else:
            assert callable(domain), "domain must be a tuple or callable"
        self.domain = domain

    @property
    def kappa1(self):
        return self.dK(0)

    @property
    def mean(self):
        return self.kappa1

    @property
    def kappa2(self):
        return self.d2K(0)

    @property
    def variance(self):
        return self.kappa2

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def dK0(self):
        if self._dK0 is None:
            self._dK0 = self.dK(0)
        return self._dK0

    @property
    def d2K0(self):
        if self._d2K0 is None:
            self._d2K0 = self.d2K(0)
        return self._d2K0

    @property
    def d3K0(self):
        if self._d3K0 is None:
            self._d3K0 = self.d3K(0)
        return self._d3K0

    @staticmethod
    def _is_in_domain(t, l=None, g=None, le=None, ge=None):
        if ~np.isscalar(t):
            t = np.asanyarray(t)
        val = True
        if l is not None:
            val &= t < l
        if g is not None:
            val &= t > g
        if le is not None:
            val &= t <= le
        if ge is not None:
            val &= t >= ge
        return val

    def K(self, t):
        cond = self.domain(t)
        if np.isscalar(t):
            retval = self._K(t) if cond else np.nan
            return retval if np.isscalar(retval) else retval.item()
        else:
            t = np.asanyarray(t)
            return np.where(cond, self._K(t), np.nan)

    def dK(self, t):
        if self._dK is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._dK = nd.Derivative(self.K, n=1)
        cond = self.domain(t)
        if np.isscalar(t):
            retval = self._dK(t) if cond else np.nan
            return retval if np.isscalar(retval) else retval.item()
        else:
            t = np.asanyarray(t)
            return np.where(cond, self._dK(t), np.nan)

    def d2K(self, t):
        if self._d2K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._d2K = nd.Derivative(self.K, n=2)
        cond = self.domain(t)
        if np.isscalar(t):
            retval = self._d2K(t) if cond else np.nan
            return retval if np.isscalar(retval) else retval.item()
        else:
            t = np.asanyarray(t)
            return np.where(cond, self._d2K(t), np.nan)

    def d3K(self, t):
        if self._d3K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._d3K = nd.Derivative(self.K, n=3)
        cond = self.domain(t)
        if np.isscalar(t):
            retval = self._d3K(t) if cond else np.nan
            return retval if np.isscalar(retval) else retval.item()
        else:
            t = np.asanyarray(t)
            return np.where(cond, self._d3K(t), np.nan)


def norm(loc=0, scale=1):
    return cumulant_generating_function(
        K=lambda t, loc=loc, scale=scale: loc * t + scale**2 * t**2 / 2,
        dK=lambda t, loc=loc, scale=scale: loc + scale**2 * t,
        d2K=lambda t, scale=scale: scale**2 + 0 * t,
        d3K=lambda t: 0 * t,
    )


def exponential(scale=1):
    return cumulant_generating_function(
        K=lambda t, scale=scale: np.log(1 / (1 - scale * t)),
        dK=lambda t, scale=scale: scale / (1 - scale * t),
        d2K=lambda t, scale=scale: 1 / (1 / scale - t) ** 2,
        d3K=lambda t, scale=scale: 2 / (1 / scale - t) ** 3,
        domain=lambda t: cumulant_generating_function._is_in_domain(t, g=-np.inf, l=1 / scale),
    )


def gamma(a=1, scale=1):
    return cumulant_generating_function(
        K=lambda t, a=a, scale=scale: -a * np.log(1 - scale * t),
        dK=lambda t, a=a, scale=scale: a * scale / (1 - scale * t),
        d2K=lambda t, a=a, scale=scale: a * scale**2 / (1 - scale * t) ** 2,
        d3K=lambda t, a=a, scale=scale: 2 * a * scale**3 / (1 - scale * t) ** 3,
        domain=lambda t: cumulant_generating_function._is_in_domain(t, g=-np.inf, l=1 / scale),
    )


def chi2(df=1):
    return gamma(a=df / 2, scale=2)


# TODO: add dgamma
# TODO: add laplace
# TODO: add asymmetric laplace


def poisson(mu=1):
    return cumulant_generating_function(
        K=lambda t, mu=mu: mu * (np.exp(t) - 1),
        dK=lambda t, mu=mu: mu * np.exp(t),
        d2K=lambda t, mu=mu: mu * np.exp(t),
        d3K=lambda t, mu=mu: mu * np.exp(t),
        domain=lambda t: cumulant_generating_function._is_in_domain(t, ge=0, l=np.inf),
    )


def binomial(n=1, p=0.5):
    return cumulant_generating_function(
        K=lambda t, n=n, p=p: n * np.log(p * (np.exp(t) - 1) + 1),
        dK=lambda t, n=n, p=p: n * p / ((1 - p) * np.exp(-t) + p),
        d2K=lambda t, n=n, p=p: n * p * (1 - p) * np.exp(-t) / ((1 - p) * np.exp(-t) + p) ** 2,
        d3K=lambda t, n=n, p=p: n
        * p
        * (1 - p)
        * ((1 - p) * np.exp(-2 * t) - np.exp(-t) * p)
        / ((1 - p) * np.exp(-t) + p) ** 3,
        domain=lambda t: cumulant_generating_function._is_in_domain(t, g=-np.inf, l=np.inf),
    )


# TODO: add multinomial
