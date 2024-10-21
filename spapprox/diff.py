# -*- coding: utf-8 -*-
"""
Wrappers around the findiff package for fast numerical differentiation
https://pypi.org/project/findiff/
https://findiff.readthedocs.io/en/latest/
"""

from abc import ABC, abstractmethod

import numpy as np
import itertools

try:
    import findiff as fd
except ImportError:
    _has_findiff = False
else:
    _has_findiff = True


def transform_rank3_tensor(T, *A):
    """
    Transform a rank 3 tensor T with a transformation matrix A, along each axis.

    The transformation is given by:

    .. math::
        T_{ijk} = \sum_{l,m,n} A_{il} A_{jm} A_{kn} T_{lmn},

    which in matrix notation can be written as:

    .. math::
        T = A \cdot T \cdot A^T \cdot A^T \cdot A^T.

    Parameters
    ----------
    T : np.ndarray
        Rank 3 tensor
    *A : np.ndarray
        Transformation matrices, one, or three (one for each axis).
        They can also be provided as vectors, in which case they are assumed to be diagonal matrices with the diagonal given.
    """
    assert T.ndim == 3, "Input tensor should be rank 3"
    if len(A) == 1:
        A = [A[0]] * 3
    for i, Ai in enumerate(A):
        if Ai.ndim == 1:
            shape = [1] * T.ndim
            shape[i] = len(Ai)
            T *= Ai.reshape(shape)
        else:
            T = np.tensordot(Ai, T, axes=(1, i))
    return T


def block_diag_3d(*tensors):
    """
    Equivalent to scipy.linalg.block_diag, but for 3d tensors.
    """
    assert all(tensor.ndim == 3 for tensor in tensors), "All tensors should be rank 3"
    # Determine the shape of the resulting tensor
    dim = sum(tensor.shape[0] for tensor in tensors)
    # Initialize the resulting tensor with zeros
    result = np.zeros((dim, dim, dim), dtype=tensors[0].dtype)
    # Place each input tensor along the diagonal
    current_index = 0
    for tensor in tensors:
        dim = tensor.shape[0]
        result[
            current_index : current_index + dim,
            current_index : current_index + dim,
            current_index : current_index + dim,
        ] = tensor
        current_index += dim
    return result


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

    def __init__(self, f, h=None, acc=2):
        if not _has_findiff:
            raise ImportError("The findiff package is required for this functionality")
        if not callable(f):
            raise TypeError("f should be callable")
        self.f = f
        self.h = h
        assert (
            isinstance(acc, (int, np.integer)) and acc >= 2
        ), "accuracy should be an integer >= 2"
        self.acc = int(acc)

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
    def h(self):
        """
        Step size for the derivative
        """
        return self._h

    @h.setter
    def h(self, h):
        """
        Note for an n-th order derivative, one has to divide by :math:`h^n`.
        So, :math:`h^n` should not be too small, as this can lead to numerical instability.
        We aim aim for :math:`h^n~1e-8`, leading to a step size of :math:`h = 1e-8^{1/n}`.
        """
        if h is None:
            sumord = int(np.sum(self.orders))
            self.h = min(np.power(1e-9, 1 / sumord), 1e-4)
        elif np.all(np.asanyarray(h) > 0):
            self._h = h
        else:
            raise ValueError("h should be positive")

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
        if np.isnan(self.f(x.T.squeeze())).any():
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
                retval = self.f(Xis.squeeze())
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
    dim : int
        Dimension of the domain of the function
    h : scalar or vector
        Step size for the derivative
    acc : int
        Accuracy of the finite difference scheme
    """

    def __init__(self, f, dim, h=None, acc=2):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim should be a positive integer")
        self._dim = dim
        super().__init__(f, h=h, acc=acc)
        if not np.isscalar(self.h) and len(self.h) != dim:
            raise ValueError("h should be a scalar or a vector of length dim")

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
            self._findiff_cache = fd.Gradient(h=self._h_vect, acc=self.acc)
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
    h : scalar or vector
        Step size for the derivative
    acc : int
        Accuracy of the finite difference scheme
    """

    def __init__(self, f, *orders, h=None, acc=2):
        self.orders = orders
        super().__init__(f, h=h, acc=acc)
        if not np.isscalar(self.h) and len(self.h) != len(orders):
            raise ValueError(f"h should be a scalar or a vector of length {len(orders)}")

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
                self._findiff_cache = fd.FinDiff(0, self._h_vect, self.orders, acc=self.acc)
            else:
                self._findiff_cache = fd.FinDiff(
                    *[
                        (i, self._h_vect[i], order)
                        for i, order in enumerate(self.orders)
                        if order > 0
                    ],
                    acc=self.acc,
                )
        return self._findiff_cache


class TensorDerivative:
    r"""
    Tensor derivative of order n for a function :math:`f` with a :math:`d`-dimensional domain.
    First order would be gradient, second order a Hessian, Tressian, etc.

    The tensor derivative is a symmetric tensor :math:`T(x)` of order :math:`n`, and has components:

    .. math::
        T(x)_{i_0,\ldots,i_{n-1}} = \partial^n_{i_0,\ldots,i_{n-1}} f(t),

    where :math:`0\leq i_0,\ldots,i_{n-1} < d` are the indices of the matrix and refer to the components of the
    domain of :math:`f`, :math:`\partial^n_{i_0,\ldots,i_{n-1}}` indicates differentiation to both components,
    :math:`i_0,\ldots,i_{n-1}` a number of times equal to their multiplicity.
    So, particularly, :math:`\partial^n_{0,\ldots,0}` indicates differentiating :math:`n` times w.r.t
    to the first component.
    """

    def __init__(self, f, dim, order, h=None, acc=2):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim should be a positive integer")
        if h is None or np.asanyarray(h).ndim == 0:
            h = np.full((dim,) * order, h)
        else:
            h = np.asanyarray(h)
        if h.ndim != order or h.shape != (dim,) * order:
            raise ValueError(f"h should be a scalar or a matrix of size {dim}^{order}")
        if np.asanyarray(acc).ndim == 0:
            acc = np.full((dim,) * order, acc, dtype=int)
        else:
            acc = np.asanyarray(acc, dtype=int)
        if acc.ndim != order or acc.shape != (dim,) * order:
            raise ValueError(f"acc should be a scalar or a matrix of size {dim}^{order}")
        assert isinstance(order, int) and order >= 1, "order should be an integer >= 2"
        self._partials = np.full(tuple([dim] * order), None, dtype=object)
        for ijk in itertools.product(*[range(dim)] * order):
            if self._partials[ijk] is not None:
                continue
            # Note, it is a symmetric tensor, so we only have to instantiate one of the equivalent components
            sorted_ijk = tuple(np.sort(ijk).tolist())
            if self._partials[sorted_ijk] is None:
                orders = [int(np.equal(ijk, i).sum()) for i in range(dim)]
                self._partials[sorted_ijk] = PartialDerivative(f, *orders, h=h[ijk], acc=acc[ijk])
            self._partials[ijk] = self._partials[sorted_ijk]

    @property
    def order(self):
        return self._partials.ndim

    @property
    def dim(self):
        return self._partials.shape[0]

    @property
    def f(self):
        if not hasattr(self, "_f_cache"):
            self._f_cache = self._partials[(0,) * self.order].f
            assert all(p.f is self.f for p in self), "All functions should be the same"
        return self._f_cache

    @property
    def h(self):
        if not hasattr(self, "_h_cache"):
            self._h_cache = np.array([p.h for p in self]).reshape(self._partials.shape)
        return self._h_cache

    @property
    def acc(self):
        if not hasattr(self, "_acc_cache"):
            self._acc_cache = np.array([p.acc for p in self]).reshape(self._partials.shape)
        return self._acc_cache

    @property
    def shape(self):
        if not hasattr(self, "_shape_cache"):
            assert self._partials.ndim == self.order
            assert self._partials.shape == tuple([self.dim] * self.order)
            self._shape_cache = self._partials.shape
        return self._shape_cache

    def __getitem__(self, ijk):
        return self._partials[*ijk]

    def __iter__(self):
        for p in np.nditer(self._partials, flags=["refs_ok"]):
            yield p.tolist()

    def values(self, gen=False):
        if gen:
            return (v for v in self)
        else:
            return list(self.values(gen=True))

    def keys(self, gen=False, unique=False):
        """
        Return the keys of the tensor derivative

        Parameters
        ----------
        gen : bool
            If True, return a generator
        unique : bool
            If True, return the keys are unique upon permutation of the indices
        """
        if gen:
            if unique:
                return itertools.combinations_with_replacement(range(self.dim), self.order)
            else:
                return (k for k in itertools.product(*[range(self.dim)] * self.order))
        else:
            return list(self.keys(gen=True, unique=unique))

    def __len__(self):
        if not hasattr(self, "_len_cache"):
            self._len_cache = len(self._partials)
        return self._len_cache

    def __call__(self, t):
        t = np.asanyarray(t)
        # Handle vectorized evaluation
        if t.ndim > 1:
            return np.array([self(tt) for tt in t])
        elif t.ndim == 1 and self.dim == 0:
            return np.array([self(tt) for tt in t])
        # Some checks
        if self.dim == 0:
            if t.ndim not in [0, 1]:
                raise ValueError(
                    "Only scalar or vector input is supported for scalar valued functions"
                )
        elif t.ndim == 0 or len(t) != self.dim:
            raise ValueError("Dimensions do not match")
        # Evaluate
        retval = np.full(self.shape, np.nan)
        for ijk in self.keys(unique=True):
            val = self[ijk](t)
            # Set all the symmetrically equivalent values
            for ijkp in set(itertools.permutations(ijk, self.order)):
                retval[ijkp] = val
        return retval


class Hessian(TensorDerivative):
    r"""
    Implements the Hessian :math:`H(x)`, i.e., the second order tensor derivative of a
    function :math:`f` with a :math:`d`-dimensional domain.
    The Hessian is a (symmetric) matrix of dimension :math:`d` with components:

    .. math::
        H(x)_{ij} = \partial^2_{ij} f(t),

    where :math:`0\leq i,j < d` are the indices of the matrix and refer to the components of the
    domain of :math:`f`, :math:`\partial^2_{ij}` indicates differentiation to both component :math:`i`,
    and :math:`j`, and, particularly, :math:`\partial^2_{ii}` indicates differentiating twice w.r.t
    to component :math:`i`.

    Parameters
    ----------
    f : callable
        Function
    dim : int
        Dimension of the domain of the function
    h : scalar or vector
        Step size for the derivative
    acc : int
        Accuracy of the finite difference scheme
    """

    def __init__(self, f, dim, h=None, acc=2):
        super().__init__(f, dim, 2, h=h, acc=acc)


class Tressian(TensorDerivative):
    r"""
    Implements the third order tensor derivative :math:`T(x)` of a
    function :math:`f` with a :math:`d`-dimensional domain.

    On math.stackexchange, this derivative is, jokingly, referred to as the Tressian.
    https://math.stackexchange.com/questions/556951/third-order-term-in-taylor-series

    The Tressian is symmetric tensor :math:`T(x)` of order 3 with components:

    .. math::
        T(x)_{ijk} = \partial^3_{ijk} f(t),

    where :math:`0\leq i,j,k < d` are the indices of the tensor and refer to the components of the
    domain of :math:`f`, :math:`\partial^3_{ijk}` indicates differentiation w.r.t. components :math:`i`,
    :math:`j` and :math:`k`. So, for example, both :math:`\partial^3_{010}` and :math:`\partial_{100}`
    indicate differentiating twice w.r.t. the first component, and once w.r.t. to the second component
    in the domain of :math:`f`.

    Parameters
    ----------
    f : callable
        Function
    dim : int
        Dimension of the domain of the function
    h : scalar or vector
        Step size for the derivative
    acc : int
        Accuracy of the finite difference scheme
    """

    def __init__(self, f, dim, h=None, acc=2):
        super().__init__(f, dim, 3, h=h, acc=acc)
