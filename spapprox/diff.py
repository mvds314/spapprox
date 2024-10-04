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
    For a scalar valued domain, we still assume vector valued evaluation is possible!

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
        if not np.all(np.asanyarray(h) > 0):
            raise ValueError("h should be positive")
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
    def orders(self):
        """
        Orders of the derivative per dimension
        """
        raise NotImplementedError

    @property
    def _max_order(self):
        if not hasattr(self, "_max_order_cache"):
            self._max_order_cache = int(np.max(self.orders))
        return self._max_order_cache

    @property
    def _grid_size(self):
        if not hasattr(self, "_grid_size_cache"):
            self._grid_size_cache = 2 * self._max_order + 1
        return self._grid_size_cache

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
        assert not np.isnan(self.f(t)).any(), "t should be in the domain"
        # Handle dim=0 case by handling at as a 1-dim vector case
        if t.ndim == 0 and dim <= 1:
            # Note this case is currently redundant, but we keep it for completeness, and for inheritance
            t = np.atleast_1d(t)  # Make the below work for vector valued input
            grid, sel = self._build_grid(t, dim=1)
            return grid.squeeze(), sel
        # Continue with the dim>=1 case
        assert t.ndim == dim == 0 or len(t) == dim, "Dimensions do not match"
        assert dim >= 1, "Domain is assumed to be a vector space at this point"
        # Use central differences by default
        sel = [self._max_order] * dim
        x = np.array([t + i * self.h for i in range(-self._max_order, self._max_order + 1)]).T
        # But adjust if t-h or t+h is not in the domain (for any of the components)
        if np.isnan(self.f(x.T)).any():
            assert not np.isnan(self.f(np.zeros_like(t))), "Zeros is asserted to be in the domain"
            for i in range(dim):
                xx = np.zeros((self._grid_size, dim))
                xx[:, i] = x[i]
                fxx = self.f(xx)
                assert (
                    len(fxx) == self._grid_size
                ), f"f is assumed to be scalar, {self._grid_size} retvals are expected when feeding t-{self._max_order}h,.., t,.. t+{self._max_order}h"
                if not np.isnan(fxx).any():
                    continue
                if np.isnan(fxx[self._max_order]).any():
                    raise AssertionError("Domain is assumed to be rectangular")
                if np.isnan(fxx[: self._max_order]).any():
                    # Shift to the right
                    if np.isnan(fxx[self._max_order :]).any():
                        raise AssertionError(
                            f"Either t - {self._max_order}h up to t, or t up to t + {self._max_order}h should be in the domain"
                        )
                    shift = next(
                        i
                        for i in range(self._max_order + 1)
                        if not np.isnan(fxx[i : self._max_order]).any()
                    )
                    sel[i] += -shift
                    x[i] += self.h * shift
                elif np.isnan(fxx[-1]).any():
                    if np.isnan(fxx[: self._max_order]).any():
                        raise AssertionError(
                            f"Either t - {self._max_order}h up to t, or t up to t + {self._max_order}h should be in the domain"
                        )
                    shift = next(
                        i
                        for i in range(self._max_order + 1)
                        if not np.isnan(fxx[self._max_order : self._grid_size - i]).any()
                    )
                    # Shift to the left
                    sel[i] += shift
                    x[i] += -self.h * shift
                else:
                    raise RuntimeError("This should never happen, all cases should be handled")
                if np.isnan(self.f(x[i])).any():
                    raise AssertionError("Shifts are assumed to fix any domain issues")
            # TODO: continue here and check the rest of the logic, with the debugger
            if np.isnan(self.f(x.T)).any():
                raise AssertionError("Shifts are assumed to fix any domain issues")
        return (
            np.array(np.meshgrid(*[x[i] for i in range(dim)], indexing="ij"))
            .reshape((dim, self._grid_size**dim))
            .T
        ), sel

    def __call__(self, t):
        t = np.asanyarray(t)
        assert (
            t.ndim <= 2
        ), "Only scalar, vector, or vectorized, as in lists with scalars or lists with vectors, evaluations are supported"
        # Handle vectorized evaluation
        if t.ndim > 1:
            return np.array([self(tt) for tt in t])
        elif t.ndim == 1 and self.dim == 0:
            return np.array([self(tt) for tt in t])
        if self.dim == 0:
            if t.ndim not in [0, 1]:
                raise ValueError(
                    "Only scalar or vector input is supported for scalar valued functions"
                )
        elif t.ndim == 0 or len(t) != self.dim:
            raise ValueError("Dimensions do not match")
        if np.isnan(self.f(t)).any():
            retval = np.nan if self.dim_image == 0 else np.full(self.dim_image, np.nan)
        else:
            if self.dim == 0:
                # Cast to 1-dim vector case
                Xis, sel = self._build_grid(np.expand_dims(t, axis=-1), dim=1)
                retval = self.f(Xis).reshape(self._grid_size)
                retval = self._findiff(retval)
                retval = retval.T[*sel]
                assert (
                    t.ndim > 0 or retval.ndim == 0
                ), "Return value should be scalar for scalar input"
            else:
                Xis, sel = self._build_grid(t)
                retval = self.f(Xis).reshape(tuple([self._grid_size] * self.dim))
                retval = self._findiff(retval)
                retval = retval.T[*sel]
                assert (
                    t.ndim > 0 or retval.ndim == 0
                ), "Return value should be scalar for scalar input"
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
    def orders(self):
        if not hasattr(self, "_orders_cache"):
            self._orders_cache = tuple([1] * self.dim)
        return self._orders_cache

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
        if not np.isscalar(h) and len(h) != len(orders):
            raise ValueError("h should be a scalar or a vector of length len(orders)")
        super().__init__(f, h)
        self.orders = orders

    @property
    def dim(self):
        if not hasattr(self, "_dim_cache"):
            if np.isscalar(self.orders):
                dim = 0
            else:
                dim = len(self.orders)
                # For 1 dim case, we assume scalar instead of vector input
                dim = 0 if dim == 1 else dim
            self._dim_cache = dim
        return self._dim_cache

    @property
    def dim_image(self):
        """
        Dimension of the image of the partial derivative is scalar valued
        """
        return 0

    @property
    def orders(self):
        return self._orders

    @orders.setter
    def orders(self, orders):
        # Validation
        if not (np.isscalar(orders) or np.asanyarray(orders).ndim <= 1):
            raise ValueError("orders should be a scalar or vector")
        if not np.all(np.asanyarray(orders) == np.round(np.asanyarray(orders))):
            raise ValueError("orders should be integers")
        # Set value
        if np.isscalar(orders) or np.asanyarray(orders).ndim == 0:
            self._orders = int(orders)
        elif np.asanyarray(orders).ndim == 1 and len(orders) == 1:
            self._orders = int(orders[0])
        else:
            self._orders = tuple(int(i) for i in orders)

    @property
    def _findiff(self):
        """
        See the documentation: https://findiff.readthedocs.io/en/latest/source/examples-basic.html#general-linear-differential-operators
        """
        if not hasattr(self, "_findiff_cache"):
            if self.dim == 0:
                if not np.isscalar(self.orders):
                    raise RuntimeError("Scalar valued input should have scalar valued orders")
                self._findiff_cache = fd.FinDiff(0, self._h_vect, self.orders)
            else:
                self._findiff_cache = fd.FinDiff(
                    *[
                        (i, self._h_vect[i], order)
                        for i, order in enumerate(self.orders)
                        if order > 0
                    ]
                )
        return self._findiff_cache
