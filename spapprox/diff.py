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
    r"""
    Base class for numerical differentiation

    Based on the findiff package. Which only works for a grid of points.
    This class creates an appropriate grid around a point where to evaluate the derivative.
    And then applies the findiff package to evaluate the derivative.

    The function :math:`f`, to which the derivative is applied, is assumed to be a scalar-valued function.
    The domain of the function can either be field of scalars, i.e., the real numbers or some interval, or a vector space, i.e.,
    :math:`\mathbb{R}^d`, where :math:`d\geq1`.

    Domains are assumed to be rectangular, i.e., a Cartesian product of intervals.
    Domains are not explicitly defined, but are assumed to be implicitly defined by the function :math:`f`.
    The function should map to NaN if the point is not in the domain.

    The derivative itself can be vector valued, e.g., a gradient, or scalar valued, e.g., a partial derivative.
    Scalar valued derivatives have dimension 0.
    The function :math:`f` is assumed be scalar valued, but should support vector valued evaluation.
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
        Dimension of the domain function to be evaluated.
        For a field of scalars, this is 0.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dim_image(self):
        """
        Dimension of the image of the derivative
        If if the derivative is scalar valued, this is 0.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _findiff(self):
        raise NotImplementedError

    @property
    def _h_vect(self):
        """
        Return the vector of h's
        """
        if not hasattr(self, "_h_vect_cache"):
            h = np.asanyarray(self.h)
            if self.dim == 0:
                assert h.ndim == 0, "h should be scalar if for scalar valued input"
                self._h_vect_cache = self.h
            elif h.ndim == 1:
                assert len(h) == self.dim, "h should be a scalar or a vector of length dim"
                self._h_vect_cache = h
            elif h.ndim == 0:
                self._h_vect_cache = np.array([self.h] * self.dim)
            else:
                raise AssertionError("h can have at most 1 dimension")
        return self._h_vect_cache

    def _build_grid(self, t, dim=None):
        # Initialize
        if dim is None:
            dim = self.dim
        t = np.asanyarray(t)
        assert t.ndim == self.dim == 0 or len(t) == self.dim, "Dimensions do not match"
        t = np.atleast_1d(t)  # Make the below work for vector valued input
        assert not np.isnan(self.f(t)).any(), "t should be in the domain"
        # Handle dim=0 case by handling at as a 1-dim vector case
        if dim == 0:
            # TODO: check return values -> maybe we need to squeeze something
            grid, sel = self._build_grid(t, dim=1)
            return np.sqeeze(grid), sel
        # Continue with the dim>=1 case
        assert dim >= 1, "Domain is assumed to be a vector space at this point"
        # Use central differences by default
        x = np.array([t - self.h, t, t + self.h]).T
        sel = [1] * dim
        # But adjust if t-h or t+h is not in the domain (for any of the components)
        if np.isnan(self.f(x)).any():
            assert not np.isnan(self.f(np.zeros_like(t))), "Zeros is asserted to be in the domain"
            # Because of the rectangular domain?
            for i in range(dim):
                xx = np.zeros((3, dim))
                xx[:, i] = x[i]
                fxx = self.f(xx)
                assert (
                    len(fxx) == 3
                ), "f is assumed to be scalar, 3 retval are expected when feeding t-h, t and t+h"
                if not np.isnan(fxx).any():
                    continue
                assert not np.isnan(fxx[1]).any(), "Domain is assumed to be rectangular"
                if np.isnan(fxx[0]).any():
                    # Shift to the right
                    assert not np.isnan(
                        fxx[-1]
                    ).any(), "Either t-h, or t+h should be in the domain"
                    sel[i] += -1
                    x[i] += self.h
                elif np.isnan(fxx[-1]).any():
                    assert not np.isnan(fxx[0]).any(), "Either t-h, or t+h should be in the domain"
                    # Shift to the left
                    sel[i] += 1
                    x[i] += -self.h
                else:
                    raise RuntimeError("This should never happen, all cases should be handled")
                assert not np.isnan(
                    self.f(x[i])
                ).any(), "Shifts are assumed to fix any domain issues"
            assert not np.isnan(self.f(x.T)).any(), "Shifts are assumed to fix any domain issues"
        return (
            np.array(np.meshgrid(*[x[i] for i in range(dim)], indexing="ij"))
            .reshape((dim, 3**dim))
            .T
        ), sel

    def __call__(self, t):
        t = np.asanyarray(t)
        assert (
            t.ndim <= 2
        ), "Only scalar, vector, or vectorized, as in lists with scalars or lists with vectors, evaluations are supported"
        # Handle vectorized evaluation
        if t.ndim == 2:
            return np.array([self(tt) for tt in t])
        if not (t.ndim == self.dim == 0 or (t.ndim > 0 and len(t) == self.dim)):
            raise ValueError("Dimensions do not match")
        if np.isnan(self.f(t)).any():
            retval = np.nan if self.dim_image == 0 else np.full(self.dim_image, np.nan)
        else:
            Xis, sel = self._build_grid(t)
            retval = self.f(Xis).reshape(tuple([3] * self.dim))
            retval = self._findiff(retval)
            retval = retval.T[*sel]
        return retval


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
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim should be a positive integer")
        self._dim = dim
        assert np.isscalar(h) or len(h) == dim, "h should be a scalar or a vector of length dim"
        super().__init__(f, h)

    @property
    def dim(self):
        return self._dim

    @property
    def dim_image(self):
        """
        Dimension of the image of the gradient
        """
        return self.dim

    @property
    def _findiff(self):
        if not hasattr(self, "_findiff_cache"):
            self._findiff_cache = fd.Gradient(h=self._h_vect)
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
    def dim_image(self):
        """
        Dimension of the image of the partial derivative is scalar valued
        """
        return 0

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

    def __call__(self, t, *args, **kwargs):
        # TODO: this should work for first derivative, probably we need more grid points for higher ones
        assert (
            np.max(self.orders) <= 1
        ), "Only first order derivatives are implemented, grid should be extended for higher order"
        return super().__call__(*args, **kwargs)
