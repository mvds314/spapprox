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

    def variance(self):
        return self.kappa2

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


def norm(mu=0, sigma=1):
    return cumulant_generating_function(
        K=lambda t, mu=mu, sigma=sigma: mu * t + sigma**2 * t**2 / 2,
        dK=lambda t, mu=mu, sigma=sigma: mu + sigma**2 * t,
        d2K=lambda t, sigma=sigma: sigma**2 + 0 * t,
        d3K=lambda t: 0 * t,
    )


def poisson(lam=1):
    return cumulant_generating_function(
        K=lambda t, lam=lam: lam * (np.exp(t) - 1),
        dK=lambda t, lam=lam: lam * np.exp(t),
        d2K=lambda t, lam=lam: lam * np.exp(t),
        d3K=lambda t, lam=lam: lam * np.exp(t),
        domain=lambda t: cumulant_generating_function._is_in_domain(t, ge=0, l=np.inf),
    )


def bernoulli(p=0.5):
    return cumulant_generating_function(
        K=lambda t, p=p: np.log(1 - p + p * np.exp(t)),
        dK=lambda t, p=p: p * np.exp(t) / (1 - p + p * np.exp(t)),
        d2K=lambda t, p=p: p**2 * np.exp(2 * t) / (1 - p + p * np.exp(t)) ** 2,
        d3K=lambda t, p=p: 2 * p**3 * np.exp(3 * t) / (1 - p + p * np.exp(t)) ** 3,
    )


def geometric(p=0.5):
    return cumulant_generating_function(
        K=lambda t, p=p: np.log(1 - p) - np.log(1 - p * np.exp(t)),
        dK=lambda t, p=p: p * np.exp(t) / (1 - p * np.exp(t)),
        d2K=lambda t, p=p: p**2 * np.exp(2 * t) / (1 - p * np.exp(t)) ** 2,
        d3K=lambda t, p=p: 2 * p**3 * np.exp(3 * t) / (1 - p * np.exp(t)) ** 3,
    )


def gamma(k=1, theta=1):
    return cumulant_generating_function(
        K=lambda t, k=k, theta=theta: k * np.log(1 - theta * t),
        dK=lambda t, k=k, theta=theta: -k * theta / (1 - theta * t),
        d2K=lambda t, k=k, theta=theta: k * theta**2 / (1 - theta * t) ** 2,
        d3K=lambda t, k=k, theta=theta: -2 * k * theta**3 / (1 - theta * t) ** 3,
    )


def exponential(lam=1):
    return cumulant_generating_function(
        K=lambda t, lam=lam: np.log(lam / (lam - t)),
        dK=lambda t, lam=lam: 1 / (lam - t),
        d2K=lambda t, lam=lam: 1 / (lam - t) ** 2,
        d3K=lambda t, lam=lam: 2 / (lam - t) ** 3,
        domain=lambda t: cumulant_generating_function._is_in_domain(t, g=-np.inf, l=lam),
    )


def chi2(k=1):
    return cumulant_generating_function(
        K=lambda t, k=k: -k / 2 * np.log(1 - 2 * t),
        dK=lambda t, k=k: k / (1 - 2 * t),
        d2K=lambda t, k=k: 2 * k / (1 - 2 * t) ** 2,
        d3K=lambda t, k=k: -8 * k / (1 - 2 * t) ** 3,
    )


def student_t(nu=1):
    return cumulant_generating_function(
        K=lambda t, nu=nu: nu * np.log(1 - 2 * t / nu),
        dK=lambda t, nu=nu: -2 * nu / (nu - 2 * t),
        d2K=lambda t, nu=nu: 4 * nu / (nu - 2 * t) ** 2,
        d3K=lambda t, nu=nu: -16 * nu / (nu - 2 * t) ** 3,
    )
