#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import warnings
import numpy as np
import scipy.optimize as spo

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        import numdifftools as nd
    has_numdifftools = True
except ImportError:
    has_numdifftools = False

from .util import type_wrapper, fib


class CumulantGeneratingFunction(ABC):
    r"""
    Base class for cumulant generating function of a distribution
    """

    def __init__(
        self,
        K,
        dK=None,
        dK_inv=None,
        d2K=None,
        d3K=None,
        dK0=None,
        d2K0=None,
        d3K0=None,
        domain=None,
    ):
        self._K = K
        self._dK = dK
        self._dK_inv = dK_inv
        self._d2K = d2K
        self._d3K = d3K
        self._dK0 = dK0
        self._d2K0 = d2K0
        self._d3K0 = d3K0
        assert domain is None or callable(domain), "domain must be a None or callable"
        self.domain = domain

    @staticmethod
    def _is_in_domain(t, l=None, g=None, le=None, ge=None):
        if ~np.isscalar(t):
            t = np.asanyarray(t)
        val = True
        t = np.asanyarray(t)
        if l is not None:
            val &= t < l
        if g is not None:
            val &= t > g
        if le is not None:
            val &= t <= le
        if ge is not None:
            val &= t >= ge
        return val

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

    def __add__(self, other):
        raise NotImplementedError()

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        raise NotImplementedError()

    def __rmul__(self, other):
        return self.__mul__(other)

    @type_wrapper(xloc=1)
    def K(self, t, fillna=np.nan):
        assert self.domain is not None, "domain must be specified"
        cond = self.domain(t)
        t = np.asanyarray(t)
        t = np.where(cond, t, 0)  # prevent outside domain evaluations
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
            return np.where(cond, self._K(t), fillna)

    @type_wrapper(xloc=1)
    def dK(self, t, fillna=np.nan):
        assert self._dK is not None, "dK must be specified"
        assert self.domain is not None, "domain must be specified"
        cond = self.domain(t)
        t = np.asanyarray(t)
        t = np.where(cond, t, 0)  # numdifftolls doesn't work if any evaluetes to NaN
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
            return np.where(cond, self._dK(t), fillna)

    @type_wrapper(xloc=1)
    def dK_inv(self, x, t0=None, **kwargs):
        raise NotImplementedError()

    @type_wrapper(xloc=1)
    def d2K(self, t, fillna=np.nan):
        assert self._d2K is not None, "d2K must be specified"
        assert self.domain is not None, "domain must be specified"
        cond = self.domain(t)
        t = np.asanyarray(t)
        t = np.where(cond, t, 0)  # numdifftolls doesn't work if any evaluetes to NaN
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
            return np.where(cond, self._d2K(t), fillna)

    @type_wrapper(xloc=1)
    def d3K(self, t, fillna=np.nan):
        assert self._d3K is not None, "d3K must be specified"
        assert self.domain is not None, "domain must be specified"
        cond = self.domain(t)
        t = np.asanyarray(t)
        t = np.where(cond, t, 0)  # numdifftolls doesn't work if any evaluetes to NaN
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
            return np.where(cond, self._d3K(t), fillna)


class UnivariateCumulantGeneratingFunction(CumulantGeneratingFunction):
    r"""
    Class for cumulant generating function of a univariate distribution

    For a univariate random variable :math:`X` with probability density function :math:`f_X(x)`,
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
        self,
        K,
        dK=None,
        dK_inv=None,
        d2K=None,
        d3K=None,
        dK0=None,
        d2K0=None,
        d3K0=None,
        domain=None,
    ):
        if domain is None:
            domain = (-np.inf, np.inf)
        if isinstance(domain, tuple):
            domain = lambda t, domain=domain: self._is_in_domain(
                t, ge=domain[0], le=domain[1], l=np.inf, g=-np.inf
            )
        else:
            assert callable(domain), "domain must be a tuple or callable"
        super().__init__(
            K, dK=dK, dK_inv=dK_inv, d2K=d2K, d3K=d3K, domain=domain, dK0=dK0, d2K0=d2K0, d3K0=d3K0
        )

    def __add__(self, other):
        """
        We use the following properties of the cumulant generating function
        for independent random variables :math:`X` and :math:`Y`:

        .. math::
            K_{aX+bY}(t) = K_X(at) + K_Y(bt)

        .. math::
            K_{X+c}(t) = K_X(t) +ct

        """
        if isinstance(other, (int, float)):
            return UnivariateCumulantGeneratingFunction(
                lambda t: self.K(t) + other * t,
                dK=lambda t: self.dK(t) + other,
                dK_inv=lambda x: self.dK_inv(x - other),
                d2K=lambda t: self.d2K(t),
                d3K=lambda t: self.d3K(t),
                domain=lambda t: self.domain(t),
            )
        elif isinstance(other, UnivariateCumulantGeneratingFunction):
            return UnivariateCumulantGeneratingFunction(
                lambda t: self.K(t) + other.K(t),
                dK=lambda t: self.dK(t) + other.dK(t),
                d2K=lambda t: self.d2K(t) + other.d2K(t),
                d3K=lambda t: self.d3K(t) + other.d3K(t),
                domain=lambda t: self.domain(t) & other.domain(t),
            )
        else:
            raise ValueError(
                "Can only add a scalar or another UnivariateCumulantGeneratingFunction"
            )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return UnivariateCumulantGeneratingFunction(
                lambda t: self.K(other * t),
                dK=lambda t: other * self.dK(other * t),
                dK_inv=lambda x: self.dK_inv(x / other) / other,
                d2K=lambda t: other**2 * self.d2K(other * t),
                d3K=lambda t: other**3 * self.d3K(other * t),
                domain=lambda t: self.domain(t),
            )
        else:
            raise ValueError("Can only multiply with a scalar")

    @type_wrapper(xloc=1)
    def dK(self, t, fillna=np.nan):
        if self._dK is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._dK = nd.Derivative(self.K, n=1)
        return super().dK(t, fillna=fillna)

    @type_wrapper(xloc=1)
    def dK_inv(self, x, t0=None, **kwargs):
        """
        Inverse of the derivative of the cumulant generating function.

        .. math::
            x = K'(t).
        """
        if self._dK_inv is not None:
            y = self._dK_inv(x)
        else:
            if len(x.shape) == 0:  # Then it is a scalar "array"
                x = x.tolist()
                kwargs["x0"] = 0 if t0 is None else t0
                kwargs.setdefault("fprime", lambda t: self.d2K(t))
                kwargs.setdefault("fprime2", lambda t: self.d3K(t))
                bracket_methods = [
                    "bisect",
                    "brentq",
                    "brenth",
                    "ridder",
                    "toms748",
                ]
                if "method" in kwargs:
                    methods = [kwargs["method"]]
                else:
                    methods = [
                        "halley",
                        "newton",
                        "secant",
                        "bisect",
                        "brentq",
                        "brenth",
                        "ridder",
                        "toms748",
                    ]
                for method in methods:
                    kwargs["method"] = method
                    if method in bracket_methods and "bracket" not in kwargs:
                        # find valid lb and ub
                        lb = next(
                            -1 * 0.9**i
                            for i in range(10000)
                            if ~np.isnan(self.dK(-1 * 0.9**i))
                        )
                        ub = next(
                            1 * 0.9**i for i in range(10000) if ~np.isnan(self.dK(1 * 0.9**i))
                        )
                        dKlb = self.dK(lb)
                        dKub = self.dK(ub)
                        assert lb < ub and dKlb < dKub, "dK is assumed to be increasing"
                        lb_scalings = (1 - 1 / fib(i) for i in range(3, 100))
                        ub_scalings = (1 - 1 / fib(i) for i in range(3, 100))
                        lb_scaling = next(lb_scalings)
                        ub_scaling = next(ub_scalings)
                        while x < dKlb:
                            lb_new = lb / lb_scaling
                            dKlb_new = self.dK(lb_new)
                            if ~np.isnan(dKlb_new):
                                lb = lb_new
                                dKlb = dKlb_new
                                continue
                            try:
                                lb_scaling = next(lb_scalings)
                            except StopIteration:
                                raise Exception("Could not find valid lb")
                        while x > dKub:
                            ub_new = ub / ub_scaling
                            dKub_new = self.dK(ub_new)
                            if ~np.isnan(dKub_new):
                                ub = ub_new
                                dKub = dKub_new
                                continue
                            try:
                                ub_scaling = next(ub_scalings)
                            except StopIteration:
                                raise Exception("Could not find valid ub")
                        assert self.dK(lb) < x < self.dK(ub)
                        kwargs["bracket"] = [lb, ub]
                    res = spo.root_scalar(lambda t, x=x: self.dK(t) - x, **kwargs)
                    if res.converged:
                        break
                else:
                    raise Exception("Failed to solve the saddle point equation.")
                y = np.asanyarray(res.root)
            else:
                kwargs["x0"] = np.zeros(x.shape) if t0 is None else np.asanayarray(t0)
                kwargs.setdefault("jac", lambda t: np.diag(self.d2K(t)))
                if "method" in kwargs:
                    methods = [kwargs["method"]]
                else:
                    methods = [
                        "hybr",
                        "lm",
                        "broyden1",
                        "broyden2",
                        "anderson",
                        "linearmixing",
                        "diagbroyden",
                        "excitingmixing",
                        "krylov",
                        "df-sane",
                    ]
                for method in methods:
                    kwargs["method"] = method
                    res = spo.root(lambda t, x=x: self.dK(t) - x, **kwargs)
                    if res.success:
                        y = np.asanyarray(res.x)
                        break
                else:
                    y = np.asanyarray(
                        [
                            self.dK_inv(xx, t0=None if t0 is None else t0[i])
                            for i, xx in enumerate(x)
                        ]
                    )
        return y

    @type_wrapper(xloc=1)
    def d2K(self, t, fillna=np.nan):
        if self._d2K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._d2K = nd.Derivative(self.K, n=2)
        return super().d2K(t, fillna=fillna)

    @type_wrapper(xloc=1)
    def d3K(self, t, fillna=np.nan):
        if self._d3K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._d3K = nd.Derivative(self.K, n=3)
        return super().d3K(t, fillna=fillna)


class MultivariateCumulantGeneratingFunction(CumulantGeneratingFunction):
    r"""
    Class for cumulant generating function of a multivariate distribution

    For a random vector :math:`X` with probability density function :math:`f_X(x)`,
    the cumulant generating function is defined as the logarithm of the moment generating function:

    .. math::
        K_X(t) = \log \mathbb{E} \left[ e^{<t,X>} \right],
    where :math:`<t,X>` is the inner product of :math:`t` and :math:`X`.

    The m-th partial derivatives, denoted by:

    .. math::
        \frac{\partial^m}{\partial^i t_i} K_X(t),
    and where in the denominator we multiply over :math:`i` combinations that sum up to m
    are supposed to take the form of numpy arrays with m indices.
    By doing so, the first derivatives, becomes the gradient, the second the Hessian, and the the third, and array with matrices, etc.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Cumulant#Cumulant_generating_function

    Parameters
    ----------
    K : callable
        Cumulant generating function
    dim : int, optional
        Dimension of the random vector
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
        self,
        K,
        dim=2,
        dK=None,
        dK_inv=None,
        d2K=None,
        d3K=None,
        dK0=None,
        d2K0=None,
        d3K0=None,
        domain=None,
    ):
        assert isinstance(dim, int) and dim > 0, "dimimension must be an integer greater than 0"
        self.dim = dim
        if domain is None:
            domain = [(-np.inf, np.inf) for _ in range(dim)]
        if isinstance(domain, list):
            domain = lambda t, domain=domain: all(
                self._is_in_domain(ti, ge=domi[0], le=domi[1], l=np.inf, g=-np.inf)
                for ti, domi in zip(t, domain)
            )
        else:
            assert callable(domain), "domain must be a tuple or callable"
        super().__init__(
            K, dK=dK, dK_inv=dK_inv, d2K=d2K, d3K=d3K, domain=domain, dK0=dK0, d2K0=d2K0, d3K0=d3K0
        )

    def __add__(self, other):
        """
        We use the following properties of the cumulant generating function
        for independent random variables :math:`X` and :math:`Y`:

        .. math::
            K_{aX+bY}(t) = K_X(at) + K_Y(bt)

        .. math::
            K_{X+c}(t) = K_X(t) +ct

        """
        if isinstance(other, (int, float)):
            return MultivariateCumulantGeneratingFunction(
                lambda t: self.K(t) + np.sum(other * t),
                dim=self.dim,
                dK=lambda t: self.dK(t) + other * np.ones(self.dim),
                dK_inv=lambda x: self.dK_inv(x - other),
                d2K=lambda t: self.d2K(t),
                d3K=lambda t: self.d3K(t),
                domain=lambda t: self.domain(t),
            )
        elif isinstance(other, UnivariateCumulantGeneratingFunction):
            # Note, we add them assuming independence, to each component we add a univariate random varibles independent of everything else
            return MultivariateCumulantGeneratingFunction(
                lambda t: self.K(t) + np.sum([other.K(ti) for ti in t]),
                dim=self.dim,
                dK=lambda t: self.dK(t) + [other.dK(ti) for ti in t],
                d2K=lambda t: self.d2K(t) + np.diag([other.d2K(ti) for ti in t]),
                d3K=lambda t: self.d3K(t)
                + np.array(
                    [
                        [
                            [other.d3K(t[i]) if i == j == k else 0 for i in range(self.dim)]
                            for j in range(self.dim)
                        ]
                        for k in range(self.dim)
                    ]
                ),
                domain=lambda t: self.domain(t) & all(other.domain(ti) for ti in t),
            )
        elif isinstance(other, MultivariateCumulantGeneratingFunction):
            assert self.dim == other.dim, "Dimensions must be equal"
            return MultivariateCumulantGeneratingFunction(
                lambda t: self.K(t) + other.K(t),
                dim=self.dim,
                dK=lambda t: self.dK(t) + other.dK(t),
                d2K=lambda t: self.d2K(t) + other.d2K(t),
                d3K=lambda t: self.d3K(t) + other.d3K(t),
                domain=lambda t: self.domain(t) & other.domain(t),
            )
        else:
            raise ValueError("Can only add a scalar or another CumulantGeneratingFunction")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return MultivariateCumulantGeneratingFunction(
                lambda t: self.K(other * t),
                dK=lambda t: other * self.dK(other * t),
                dK_inv=lambda x: self.dK_inv(x / other) / other,
                d2K=lambda t: other**2 * self.d2K(other * t),
                d3K=lambda t: other**3 * self.d3K(other * t),
                domain=lambda t: self.domain(t),
            )
        elif isinstance(other, np.ndarray) and len(other.shape)==1:
            assert len(other)==self.dim, "Vector rescaling should work on all variables"
            #This is simply a rescaling of all the components
            return MultivariateCumulantGeneratingFunction(
                lambda t: self.K(other * t),
                dK=lambda t: other * self.dK(other * t),
                dK_inv=lambda x: self.dK_inv(x / other) / other,
                d2K=lambda t: np.atleast_2d(other).T.dot(np.atleast_2d(other))*(self.d2K(other * t)),
                d3K=lambda t, A=np.array(
                    [
                        [
                            [other[i]*other[j]*other[k] for i in range(self.dim)]
                            for j in range(self.dim)
                        ]
                        for k in range(self.dim)
                    ]
                ):A*self.d3K(other * t),
                domain=lambda t: self.domain(t),
            )
        elif isinstance(other, np.ndarray) and len(other.shape)==2:
            assert other.shape[1] == self.dim, "Dimension must be equal"
            assert np.allclose(self.d2K0-np.diag(self.d2K0),0), "Only linear transformation of indepdent variables are possible"
            # TODO: check dimensions and look up in book
            return MultivariateCumulantGeneratingFunction(
                lambda t: np.sum([self.K(otheri * t[i]) for i,otheri in enumerate(other)]),
                dK=lambda t: other * self.dK(other * t),
                dK_inv=lambda x: self.dK_inv(x / other) / other,
                d2K=lambda t: other**2 * self.d2K(other * t),
                d3K=lambda t: other**3 * self.d3K(other * t),
                domain=lambda t: self.domain(t),
            )
        else:
            raise ValueError("Can only multiply with a scalar")

    # TODO: now check dimensions everywhere
    # TODO: also provide the correlation matrix through the jacobian
    @type_wrapper(xloc=1)
    def dK(self, t, fillna=np.nan):
        if self._dK is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._dK = nd.Derivative(self.K, n=1)
        return super().dK(t, fillna=fillna)

    @type_wrapper(xloc=1)
    def dK_inv(self, x, t0=None, **kwargs):
        """
        Inverse of the derivative of the cumulant generating function.

        .. math::
            x = K'(t).
        """
        # TODO: maybe implement a generic solver, is this the gradient or the innerproduct with the gradient
        raise NotImplementedError()

    # TODO: where is the second derivative?

    @type_wrapper(xloc=1)
    def d3K(self, t, fillna=np.nan):
        # TODO: I'm not sure what this is supposed to be
        if self._d3K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._d3K = nd.Derivative(self.K, n=3)
        return super().d3K(t, fillna=fillna)


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
    """
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
