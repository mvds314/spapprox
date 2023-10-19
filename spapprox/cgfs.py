#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

try:
    import numdifftools as nd

    has_numdifftools = True
except:
    has_numdifftools = False


# # alternatively use https://github.com/maroba/findiff
# def derivative(f, x, method="central", h=1e-5):
#     """Compute the difference formula for f'(a) with step size h.

#     Parameters
#     ----------
#     f : function
#         Vectorized function of one variable
#     a : number
#         Compute derivative at x = a
#     method : string
#         Difference formula: 'forward', 'backward' or 'central'
#     h : number
#         Step size in difference formula

#     Returns
#     -------
#     float
#         Difference formula:
#             central: f(a+h) - f(a-h))/2h
#             forward: f(a+h) - f(a))/h
#             backward: f(a) - f(a-h))/h
#     """
#     if method == "central":
#         return (f(x + h) - f(x - h)) / (2 * h)
#     elif method == "forward":
#         return (f(x + h) - f(x)) / h
#     elif method == "backward":
#         return (f(x) - f(x - h)) / h
#     else:
#         raise ValueError("Method must be 'central', 'forward' or 'backward'.")


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
    """

    def __init__(self, K, dK=None, d2K=None, d3K=None):
        self._K = K
        self._dK = dK
        self._d2K = d2K
        self._d3K = d3K

    def K(self, t):
        return self._K(t)

    def dK(self, t):
        if self._dK is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._dK = nd.Derivative(self.K, n=1)
        return self._dK(t)

    def d2K(self, t):
        if self._d2K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._d2K = nd.Derivative(self.K, n=2)
        return self._d2K(t)

    def d3K(self, t):
        if self._d3K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._dK = nd.Derivative(self.K, n=3)
        return self._d3K(t)


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
