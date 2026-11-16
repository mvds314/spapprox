#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:30:09 2024

@author: martins
"""

import numpy as np
import spapprox as spr
from spapprox import Timer
import numdifftools as ndt
import findiff as fd


cgf = spr.multivariate_norm(dim=3)
cgf.domain = spr.Domain(dim=3, l=[10, 1 + 5e-7, 10])

K = cgf.K
dK = cgf.dK
d2K = cgf.d2K
d3K = cgf.d3K

ts = np.random.randn(1000, 3)
ts = ts[0]
ts = np.array([1, 1, 1]).astype(np.float64)

h = 1e-6

# Gradient
grad = fd.Gradient(h=[h, h, h])

# TODO: continue here, why doesn't this work yet, 1+1e-6 should be outside of the domain
# TODO: but it does work right now, correct?


# TODO: make this into a class as well
def gradf(t):
    global grad
    t = np.asanyarray(t)
    # Handle vectorized evaluation
    if t.ndim == 2:
        return np.array([gradf(tt) for tt in t])
    assert t.ndim == 1, "Only vector or list of vector evaluations are supported"
    assert len(t) == cgf.dim, "Dimension does not match"
    # Handle case where t is not the domain
    if np.isnan(K(t)):
        return np.full(cgf.dim, np.nan)
    # Handle case where t+h or t-h is not in the domain
    assert not np.isnan(K(np.zeros(cgf.dim))), "Zeros is assumed to be in the domain"
    # Use central differences by default
    x = np.array([t - h, t, t + h]).T
    sel = [1] * cgf.dim
    # But adjust if t-h or t+h is not in the domain (for any of the components)
    for i in range(cgf.dim):
        xx = np.zeros((cgf.dim, 3))
        xx[:, i] = x[i]
        Kxx = K(xx)
        if not np.isnan(Kxx).any():
            continue
        assert not np.isnan(Kxx[1]).any(), "Domain is assumed to be rectangular"
        if np.isnan(Kxx[0]).any():
            # Shift to the right
            assert not np.isnan(Kxx[-1]).any(), "Either t-h, or t+h should be in the domain"
            sel[i] += -1
            x[i] += h
        elif np.isnan(Kxx[-1]).any():
            assert not np.isnan(Kxx[0]).any(), "Either t-h, or t+h should be in the domain"
            # Shift to the left
            sel[i] += 1
            x[i] += -h
        else:
            raise RuntimeError("This should never happen, all cases should be handled")
    # TODO: test if the above logic works
    Xis = (
        np.array(np.meshgrid(x[0], x[1], x[2], indexing="ij"))
        .reshape((cgf.dim, cgf.dim**cgf.dim))
        .T
    )
    retval = K(Xis).reshape(tuple([cgf.dim] * cgf.dim))
    retval = grad(retval)
    retval = retval[*sel]
    if not np.isnan(retval).any():
        return retval
    else:
        raise RuntimeError("Not able to handle nan values, probably domain is not rectangular")


# TODO: also create class for the gradient


class PartialDerivative:
    r"""
    Implements tha partial derivative w.r.t. t:

    .. math::
        \sum_k \partial^{\alpha_k}_k f(t),

    where the :math:`\alpha_k` form a tuple with integers, and :math:`\alpha_k`
    differentiates w.r.t. the :math:`k`-th component of :math:`t`.


    Parameters
    ----------
    f : callable
        Function
    t : vector
        point at which to evaluate the derivative
    *orders : tuple with integers
        Derivatives w.r.t. arguments of f.
    """

    def __init__(self, f, *orders, h=1e-6):
        self.f = f
        self.orders = orders
        self.h = h

    @property
    def dim(self):
        return len(self.orders)

    @property
    def _findiff(self):
        """
        See the documentation: https://findiff.readthedocs.io/en/latest/source/examples-basic.html#general-linear-differential-operators
        """
        if not hasattr(self, "_findiff_cache"):
            self._findiff_cache = fd.FinDiff(
                *[(i, h, self.orders[i]) for i in range(self.dim) if self.orders[i] > 0]
            )
        return self._findiff_cache

    def __call__(self, t):
        """
        Evaluate the derivative at t
        """
        t = np.asanyarray(t)
        # Handle vectorized evaluation
        if t.ndim == 2:
            return np.array([self(tt) for tt in t])
        # Process input
        assert t.ndim == 1, "Only vector or list of vector evaluations are supported"
        assert len(t) == self.dim, "Dimension does not match"
        # Handle case where t is not the domain
        if np.isnan(K(t)):
            return np.full(cgf.dim, np.nan)
        # Handle case where t+h or t-h is not in the domain
        assert not np.isnan(K(np.zeros(cgf.dim))), "Zeros is assumed to be in the domain"
        # Use central differences by default
        # TODO: this should work for first derivative, probably we need more grid points for higher ones
        # TODO: add  check
        # assert (
        #     np.max(self.orders) <= 1
        # ), "Only first order derivatives are implemented, grid should be extended for higher order"
        x = np.array([t - h, t, t + h]).T
        sel = [1] * self.dim
        # But adjust if t-h or t+h is not in the domain (for any of the components)
        for i in range(self.dim):
            xx = np.zeros((self.dim, 3))
            xx[:, i] = x[i]
            fxx = self.f(xx)
            if not np.isnan(fxx).any():
                continue
            raise NotImplementedError("Shifts are not implemented yet")
            assert not np.isnan(fxx[1]).any(), "Domain is assumed to be rectangular"
            if np.isnan(fxx[0]).any():
                # Shift to the right
                assert not np.isnan(fxx[-1]).any(), "Either t-h, or t+h should be in the domain"
                sel[i] += -1
                x[i] += h
            elif np.isnan(fxx[-1]).any():
                assert not np.isnan(fxx[0]).any(), "Either t-h, or t+h should be in the domain"
                # Shift to the left
                sel[i] += 1
                x[i] += -h
            else:
                raise RuntimeError("This should never happen, all cases should be handled")
        # TODO: test if the above logic works
        Xis = (
            np.array(np.meshgrid(x[0], x[1], x[2], indexing="ij"))
            .reshape((self.dim, self.dim**self.dim))
            .T
        )
        retval = self.f(Xis).reshape(tuple([self.dim] * self.dim))
        # TODO: there is some error here, why doesn't it work?
        retval = self._findiff(retval)
        retval = retval[*sel]
        if not np.isnan(retval).any():
            return retval
        else:
            raise RuntimeError("Not able to handle nan values, probably domain is not rectangular")


def hessf(f, t):
    """
    The Hessian matrix
    """
    # TODO: implement component wise
    pass


# TODO: continue here and implement higher order derivatives
# TODO: proceed as here: https://findiff.readthedocs.io/en/latest/source/examples-basic.html#general-linear-differential-operators


# TODO: how to deal with higher order derivatives


# # Second order derivative
# d2_dt2 = fd.FinDiff(0, h, 2)


# def d2Kfd(t):
#     global d2_dt2
#     # use central differences by default
#     Ks = K(np.array([t - h, t, t + h, t + 2 * h]))
#     retval = d2_dt2(Ks)[1]
#     sel = np.isnan(retval)
#     if not sel.any():
#         return retval
#     # fall back to left
#     tsel = t[sel]
#     retval[sel] = d2_dt2(K(np.array([tsel, tsel + h, tsel + 2 * h + tsel + 3 * h])))[0]
#     sel = np.isnan(retval)
#     if not sel.any():
#         return retval
#     # or to right
#     tsel = t[sel]
#     retval[sel] = d2_dt2(K(np.array([t[sel] - 3 * h, t[sel] - 2 * h, t[sel] - h, tsel])))[-1]
#     return retval


# # Third order derivative
# d3_dt3 = fd.FinDiff(0, h, 3)


# def d3Kfd(t):
#     global d3_dt3
#     # use central differences by default
#     retval = d3_dt3(K(np.array([t - 3 * h, t - 2 * h, t - h, t, t + h, t + 2 * h])))[3]
#     sel = np.isnan(retval)
#     if not sel.any():
#         return retval
#     # fall back to left
#     tsel = t[sel]
#     retval[sel] = d3_dt3(
#         K(np.array([tsel, tsel + h, tsel + 2 * h, tsel + 3 * h, tsel + 4 * h, tsel + 5 * h]))
#     )[0]
#     retval[sel] = d_dt(K(np.repeat(tsel, 6) + h * np.arange(6)))[0]
#     sel = np.isnan(retval)
#     if not sel.any():
#         return retval
#     # or to right
#     tsel = t[sel]
#     retval[sel] = d3_dt3(
#         K(np.array([tsel - 5 * h, tsel - 4 * h, tsel - 3 * h, tsel - 2 * h, tsel - h, tsel]))
#     )[-1]
#     return retval


with Timer("explicit 1", decimals=6):
    dK(ts)
dKndt = ndt.Gradient(K, n=1)
with Timer("numdifftools 1", decimals=6):
    if ts.ndim == 2:
        np.asanyarray([dKndt(t.T) for t in ts])
    elif ts.ndim == 1:
        dKndt(ts.T)
with Timer("findiff 1", decimals=6):
    gradf(ts)

with Timer("Partial derivative"):
    pd = PartialDerivative(K, 1, 1, 2)
    pd([0, 0, 0])
    # TODO: start by comparing with example on website


# TODO: make some unittests

# with Timer("explicit 2", decimals=6):
#     d2K(ts)
# d2Kndt = ndt.Derivative(K, n=2)
# with Timer("numdifftools 2", decimals=6):
#     d2Kndt(ts)
# with Timer("findiff 2", decimals=6):
#     d2Kfd(ts)

# with Timer("explicit 3", decimals=6):
#     d3K(ts)
# d3Kndt = ndt.Derivative(K, n=3)
# with Timer("numdifftools 3", decimals=6):
#     d2Kndt(ts)
# with Timer("findiff 3", decimals=6):
#     d3Kfd(ts)
