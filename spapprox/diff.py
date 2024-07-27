# -*- coding: utf-8 -*-
"""
Wrappers around the findiff package for fast numerical differentiation
https://pypi.org/project/findiff/
https://findiff.readthedocs.io/en/latest/
"""

from abc import ABC, abstractmethod

import findiff as fd
import numpy as np


class FindiffBase(ABC):
    """
    Base class for numerical differentiation

    Based on the findiff package. Which only works for a grid of points.
    This class creates an appropriate grid around a point where to evaluate the derivative.
    And then applies the findiff package to evaluate the derivative.
    """

    def __init__(self, f, h=1e-6):
        assert callable(f), "f should be callable"
        self.f = f
        assert np.all(np.asanyarray(h) > 0), "h should be positive"
        self.h = h

    @property
    @abstractmethod
    def dim(self):
        """
        Dimension of the function to be evaluated
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _findiff(self):
        raise NotImplementedError

    @property
    def _h_vect(self, t):
        """
        Return the vector of h's
        """
        if not hasattr(self, "_h_vect_cache"):
            h = np.asanyarray(self.h)
            if h.dim > 0:
                assert len(h) == self.dim, "h should be a scalar or a vector of length dim"
                self._h_vect_cache = h
            else:
                self._h_vect_cache = np.array([self.h] * self.dim)
        return self._h_vect_cache

    def _build_grid(self, t):
        t = np.asanyarray(t)
        assert len(t) == self.dim, "Dimension does not match"
        # Handle case where t is not the domain
        if np.isnan(self.f(t)):
            return np.full(self.dim, np.nan)
        # Handle case where t+h or t-h is not in the domain
        assert not np.isnan(self.f(np.zeros(self.dim))), "Zeros is assumed to be in the domain"
        # Use central differences by default
        x = np.array([t - self.h, t, t + self.h]).T
        sel = [1] * self.dim
        # But adjust if t-h or t+h is not in the domain (for any of the components)
        for i in range(self.dim):
            xx = np.zeros((3, self.dim))
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
                x[i] += self.h
            elif np.isnan(fxx[-1]).any():
                assert not np.isnan(fxx[0]).any(), "Either t-h, or t+h should be in the domain"
                # Shift to the left
                sel[i] += 1
                x[i] += -self.h
            else:
                raise RuntimeError("This should never happen, all cases should be handled")
        # TODO: test if the above logic works
        return (
            np.array(np.meshgrid(*[x[i] for i in range(self.dim)], indexing="ij"))
            .reshape((self.dim, 3**self.dim))
            .T
        ), sel

    def __call__(self, t):
        t = np.asanyarray(t)
        # Handle vectorized evaluation
        if t.ndim == 2:
            # TODO: number of return values do not match
            return np.array([self(tt) for tt in t])
        # Process input
        assert t.ndim == 1, "Only vector or list of vector evaluations are supported"
        assert len(t) == self.dim, "Dimension does not match"
        Xis, sel = self._build_grid(t)
        retval = self.f(Xis).reshape(tuple([3] * self.dim))
        # TODO: there is some error here, why doesn't it work?
        retval = self._findiff(retval)
        retval = retval.T[*sel]
        if not np.isnan(retval).any():
            return retval
        else:
            raise RuntimeError("Not able to handle nan values, probably domain is not rectangular")


class Gradient(FindiffBase):
    r"""
    Implements the gradient derivative w.r.t. t:

    .. math::
        [\partial_1 f(t), \partial_2 f(t), \ldots, \partial_d f(t)],

    where :math:`d` is the dimension of :math:`t`.

    Parameters
    ----------
    f : callable
        Function
    t : vector
        point at which to evaluate the derivative
    """

    def __init__(self, f, dim, h=1e-6):
        assert isinstance(dim, int) and dim > 0, "dim should be a positive integer"
        self._dim = dim
        assert np.isscalar(h) or len(h) == dim, "h should be a scalar or a vector of length dim"
        super().__init__(f, h)

    @property
    def dim(self):
        return self._dim

    @property
    def _findiff(self):
        if not hasattr(self, "_findiff_cache"):
            self._findiff_cache = fd.Gradient(
                h=[self.h] * self.dim if np.isscalar(self.h) else self.h
            )
        return self._findiff_cache


class PartialDerivative(FindiffBase):
    r"""
    Implements the partial derivative w.r.t. t:

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
        if not np.isscalar(h):
            assert len(h) == len(orders), "h should be a scalar or a vector of length len(orders)"
        super().__init__(f, h)
        assert np.asanyarray(orders).ndim == 1, "orders should be a vector"
        assert np.all(np.asanyarray(orders) >= 0), "orders should be non-negative"
        self.orders = orders

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
                *[(i, self._h_vect[i], order) for i, order in enumerate(self.orders) if order > 0]
            )
        return self._findiff_cache

    def __call__(self, *args, **kwargs):
        # TODO: this should work for first derivative, probably we need more grid points for higher ones
        # TODO: add  check
        assert (
            np.max(self.orders) <= 1
        ), "Only first order derivatives are implemented, grid should be extended for higher order"
        return super().__call__(*args, **kwargs)
