#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    """
    Base class for cumulant generating function of a distribution
    """

    def __init__(self, K, dK=None, d2K=None, d3K=None, **distkwargs):
        self._K = K
        self._dK = dK
        self._d2K = d2K
        self._d3K = d3K
        self._distkwargs = distkwargs

    def K(self, t):
        return self._K(t, **self._distkwargs)

    def dK(self, t):
        if self._dK is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._dK = nd.Derivative(self.K, n=1)
        return self.dK(t, **self._distkwargs)

    def d2K(self, t):
        if self._d2K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._d2K = nd.Derivative(self.K, n=2)
        return self.dK(t, **self._distkwargs)

    def d3K(self, t):
        if self._d3K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._dK = nd.Derivative(self.K, n=3)
        return self.dK(t, **self._distkwargs)


norm = cumulant_generating_function(
    K=lambda t, mu=0, sigma=1: mu * t + sigma**2 * t**2 / 2,
    dK=lambda t, mu=0, sigma=1: mu + sigma**2 * t,
    d2K=lambda t, mu=0, sigma=1: sigma**2 + 0 * t,
    d3K=lambda t, mu=0, sigma=1: 0 * t,
)
