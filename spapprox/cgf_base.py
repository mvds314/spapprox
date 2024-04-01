#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import warnings
import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize as spo
import statsmodels.api as sm

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        import numdifftools as nd
    has_numdifftools = True
except ImportError:
    has_numdifftools = False

from .domain import Domain
from .util import type_wrapper, fib


class CumulantGeneratingFunction(ABC):
    r"""
    Base class for the cumulant generating function of a random variable (or vector) :math:`X`

    The cumulant generating function :math:`K(t)` (and, optionally, its derivates) of
    a random variable (or vector) :math:`X` should be provided as callable.

    If location and scale parameters are provided, the passed callables should correspond to
    the standardized random variable (vector) :math`Z`, i.e., :math:`X=\text{scale}\times Z + \text{loc}`.

    The class contains logic to:
    * compute and evaluate derivatives
    * compute the inverse of the derivative
    * apply a location and or scale
    * add other random variables (or vectors)

    Parameters
    ----------
    K : callable
      Cumulant generating function, maps t to K(t), and is able to handle vector valued input
    loc : float, or array_like (only for random vectors) optional
      Location parameter of the distribution. If provided, the provided :math:`K` corresponds to the cumulant
      generating function of the standardized random variable :math:`Z`, and :math:`x=\text{scale}\times Z + \text{loc}`.
    scale : float, or matrix (only for random vectors) optional
      Scale parameter of the distribution. If provided, the provided :math:`K` corresponds to the cumulant
      generating function of the standardized random variable :math:`Z`, and :math:`x=\text{scale}\times Z + \text{loc}`.
    dK : callable, optional
      Derivative of the cumulant generating function, maps t to K'(t), and is able to handle vector valued input.
      If not provided, numerical differentiation is used.
    d2K : callable, optional
      Second derivative of the cumulant generating function, maps t to K''(t), and is able to handle vector valued input.
      If not provided, numerical differentiation is used.
    d3K : callable, optional
      Third derivative of the cumulant generating function, maps t to K'''(t), and is able to handle vector valued input.
      If not provided, numerical differentiation is used.
    dK0 : float, or array_like (of for random vectors) optional
      If provided, the derivative of the cumulant generating function at 0, i.e., :math:`K'(0)`.
      If not provided, it is computed and cached once needed.
    d2K0 : float, or array_like (of for random vectors) optional
      If provided, the second derivative of the cumulant generating function at 0, i.e., :math:`K''(0)`.
      If not provided, it is computed and cached once needed.
    d3K0 : float, or array_like (of for random vectors) optional
      If provided, the third derivative of the cumulant generating function at 0, i.e., :math:`K'''(0)`.
      If not provided, it is computed and cached once needed.
    domain : Domain or None, optional
      If not provided, the domain is assumed to be :math:`(-\infty, \infty)`.
    """

    def __init__(
        self,
        K,
        loc=0,
        scale=1,
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
        if domain is None:
            domain = Domain()
        assert isinstance(domain, Domain)
        self.domain = domain
        self.loc = loc
        self.scale = scale

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, loc):
        for att in ["_dK0_cache"]:
            if hasattr(self, att):
                delattr(self, att)
        self._loc = loc

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        for att in ["_dK0_cache", "_d2K0_cache", "_d3K0_cache"]:
            if hasattr(self, att):
                delattr(self, att)
        self._scale = scale

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
    @abstractmethod
    def variance(self):
        raise NotImplementedError()

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def dK0(self):
        if self._dK0 is None:
            self._dK0 = self.dK(0, loc=0, scale=1)
        return self._dK0

    @property
    def d2K0(self):
        if self._d2K0 is None:
            self._d2K0 = self.d2K(0, loc=0, scale=1)
        return self._d2K0

    @property
    def d3K0(self):
        if self._d3K0 is None:
            self._d3K0 = self.d3K(0, loc=0, scale=1)
        return self._d3K0

    @abstractmethod
    def add(self, other, inplace=False):
        raise NotImplementedError()

    def __add__(self, other):
        return self.add(other, inplace=False)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.add(-1 * other, inplace=False)

    def __rsub__(self, other):
        return self.mul(-1, inplace=False).add(other, inplace=False)

    @abstractmethod
    def mul(self, other, inplace=False):
        raise NotImplementedError()

    def __mul__(self, other):
        return self.mul(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    @abstractmethod
    def K(self, t, fillna=np.nan, loc=None, scale=None):
        raise NotImplementedError()

    @abstractmethod
    def dK(self, t, fillna=np.nan, loc=None, scale=None):
        raise NotImplementedError()

    @abstractmethod
    def dK_inv(self, x, loc=None, scale=None, fillna=np.nan):
        raise NotImplementedError()

    @abstractmethod
    def d2K(self, t, fillna=np.nan, loc=None, scale=None):
        raise NotImplementedError()

    @abstractmethod
    def d3K(self, t, fillna=np.nan, loc=None, scale=None):
        raise NotImplementedError()


class UnivariateCumulantGeneratingFunction(CumulantGeneratingFunction):
    r"""
    Class for cumulant generating function of a univariate distribution

    For a univariate random variable :math:`X` with probability density function :math:`f_X(x)`,
    the cumulant generating function is defined as the logarithm of the moment generating function:

    .. math::
        K_X(t) = \log \mathbb{E} \left[ e^{tX} \right].

    It satisfies the following properties:

    * :math:`\kappa_n = \frac{d^n}{dt^n} K_X(t) \big|_{t=0}`
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

    [3] Bertsekas, Tsitsiklis (2000) - Introduction to probability

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
    domain : Domain or None, optional
        If not provided, the domain is assumed to be :math:`(-\infty, \infty)`.
    """

    def __init__(
        self,
        K,
        loc=0,
        scale=1,
        dK=None,
        dK_inv=None,
        d2K=None,
        d3K=None,
        dK0=None,
        d2K0=None,
        d3K0=None,
        domain=None,
    ):
        super().__init__(
            K,
            loc=loc,
            scale=scale,
            dK=dK,
            dK_inv=dK_inv,
            d2K=d2K,
            d3K=d3K,
            domain=domain,
            dK0=dK0,
            d2K0=d2K0,
            d3K0=d3K0,
        )

    @CumulantGeneratingFunction.loc.setter
    def loc(self, loc):
        if isinstance(loc, np.ndarray) and len(loc.shape) == 0:
            loc = loc.tolist()
        assert pd.api.types.is_number(loc), "loc should be a scalar"
        CumulantGeneratingFunction.loc.fset(self, loc)

    @CumulantGeneratingFunction.scale.setter
    def scale(self, scale):
        if isinstance(scale, np.ndarray) and len(scale.shape) == 0:
            scale = scale.tolist()
        assert pd.api.types.is_number(scale), "scale should be a scalar"
        CumulantGeneratingFunction.scale.fset(self, float(scale))

    @property
    def variance(self):
        return self.kappa2

    @type_wrapper(xloc=1)
    def K(self, t, fillna=np.nan, loc=None, scale=None):
        loc = self.loc if loc is None else loc
        scale = self.scale if scale is None else scale
        assert np.isscalar(loc) and np.isscalar(
            scale
        ), "loc and scale should be scalars"
        st = scale * t
        cond = self.domain.is_in_domain(st)
        st = np.where(cond, st, 0)  # prevent outside domain evaluations
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="All-NaN slice encountered"
            )
            val = self._K(st) + loc * t
            return np.where(cond, val, fillna)

    @type_wrapper(xloc=1)
    def dK(self, t, loc=None, scale=None, fillna=np.nan):
        r"""
        Note, the current implementation uses numerical differentiation, but
        an alternative way would be to use the following result:

        .. math::
            K'(t)=\frac{\mathbb{E}\left[ Xe^{tX}\right]}{M(t)},
        where :math:`K(t)` is the cumulant generating function, and :math:`M(t)`
        is the moment generating function of :math:`X`.

        References
        ----------
        [1] Ganesh, O'Connell (2004) - Big Quesues in Probability and Statistics
        """
        # Initialize
        if self._dK is None:
            assert (
                has_numdifftools
            ), "Numdifftools is required if derivatives are not provided"
            self._dK = nd.Derivative(lambda tt: self.K(tt, loc=0, scale=1), n=1)
        loc = self.loc if loc is None else loc
        scale = self.scale if scale is None else scale
        assert np.isscalar(loc) and np.isscalar(
            scale
        ), "loc and scale should be scalars"
        st = scale * t
        cond = self.domain.is_in_domain(st)
        st = np.where(cond, st, 0)  # prevent outside domain evaluations
        # Evaluate
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="All-NaN slice encountered"
            )
            return np.where(cond, scale * self._dK(st) + loc, fillna)

    @type_wrapper(xloc=1)
    def dK_inv(self, x, t0=None, loc=None, scale=None, fillna=np.nan, **kwargs):
        r"""
        Inverse of the derivative of the cumulant generating function.
        It solves:

        .. math::
            x = K'(t).

        Note that the inverse equals of :math:`K(t)`, and all its
        derivatives are given by the Legendre-Fenchel transform :math:`K^*(x)`
        (and all its derivatives):

        .. math::
            K^*(x) = \sup_t \left\{<t,x>-K(t)\right\}

        References
        ----------
        [1] McCullagh (1985) - Tensor methdos in statistics
        """
        # Handle scaling
        loc = self.loc if loc is None else loc
        scale = self.scale if scale is None else scale
        x = np.asanyarray((x - loc) / scale)
        if self._dK_inv is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message="All-NaN slice encountered"
                )
                y = self._dK_inv(x)
        else:
            if len(x.shape) == 0:  # Then it is a scalar "array"
                x = x.tolist()
                kwargs["x0"] = 0 if t0 is None else t0
                kwargs.setdefault("fprime", lambda t: self.d2K(t, loc=0, scale=1))
                kwargs.setdefault("fprime2", lambda t: self.d3K(t, loc=0, scale=1))
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
                            if ~np.isnan(self.dK(-1 * 0.9**i, loc=0, scale=1))
                        )
                        ub = next(
                            1 * 0.9**i
                            for i in range(10000)
                            if ~np.isnan(self.dK(1 * 0.9**i, loc=0, scale=1))
                        )
                        dKlb = self.dK(lb, loc=0, scale=1)
                        dKub = self.dK(ub, loc=0, scale=1)
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
                            dKub_new = self.dK(ub_new, loc=0, scale=1)
                            if ~np.isnan(dKub_new):
                                ub = ub_new
                                dKub = dKub_new
                                continue
                            try:
                                ub_scaling = next(ub_scalings)
                            except StopIteration:
                                raise Exception("Coucld not find valid ub")
                        assert (
                            self.dK(lb, loc=0, scale=1)
                            < x
                            < self.dK(ub, loc=0, scale=1)
                        )
                        kwargs["bracket"] = [lb, ub]
                    try:
                        res = spo.root_scalar(
                            lambda t, x=x: self.dK(t, loc=0, scale=1) - x, **kwargs
                        )
                    except Exception:
                        continue
                    if res.converged:
                        break
                else:
                    raise Exception("Failed to solve the saddle point equation.")
                y = np.asanyarray(res.root)
            else:
                kwargs["x0"] = np.zeros(x.shape) if t0 is None else np.asanayarray(t0)
                kwargs.setdefault("jac", lambda t: np.diag(self.d2K(t, loc=0, scale=1)))
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
                    try:
                        res = spo.root(
                            lambda t, x=x: self.dK(t, loc=0, scale=1) - x, **kwargs
                        )
                    except Exception:
                        continue
                    if res.success:
                        y = np.asanyarray(res.x)
                        break
                else:
                    y = np.asanyarray(
                        [
                            self.dK_inv(
                                xx, loc=0, scale=1, t0=None if t0 is None else t0[i]
                            )
                            for i, xx in enumerate(x)
                        ]
                    )
        cond = self.domain.is_in_domain(y)
        return np.where(cond, y / scale, fillna)

    @type_wrapper(xloc=1)
    def d2K(self, t, fillna=np.nan, loc=None, scale=None):
        """
        Note that higher order derivatives can sometimes
        be found if a generating polynomial exists, i.e.,
        a polynomial that relatates :math:`K(t)` with its
        derivatives.

        For example, the gamma distribution has generating
        polynomial :math:`rK''-(K')^2`.

        References
        ----------
        [1] Pistone, Wynn (1999) - Finitely generated cumulants.
        """
        # Initialize
        if self._d2K is None:
            assert (
                has_numdifftools
            ), "Numdifftools is required if derivatives are not provided"
            self._d2K = nd.Derivative(lambda tt: self.K(tt, loc=0, scale=1), n=2)
        scale = self.scale if scale is None else scale
        assert np.isscalar(scale), "scale should be a scalar"
        st = scale * t
        cond = self.domain.is_in_domain(st)
        st = np.where(cond, st, 0)  # prevent outside domain evaluations
        # Evaluate
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="All-NaN slice encountered"
            )
            return np.where(cond, scale**2 * self._d2K(st), fillna)

    @type_wrapper(xloc=1)
    def d3K(self, t, fillna=np.nan, loc=None, scale=None):
        # Initialize
        if self._d3K is None:
            assert (
                has_numdifftools
            ), "Numdifftools is required if derivatives are not provided"
            self._d3K = nd.Derivative(lambda tt: self.K(tt, loc=0, scale=1), n=3)
        scale = self.scale if scale is None else scale
        assert np.isscalar(scale), "scale should be a scalar"
        st = scale * t
        cond = self.domain.is_in_domain(st)
        st = np.where(cond, st, 0)  # prevent outside domain evaluations
        # Evaluate
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="All-NaN slice encountered"
            )
            return np.where(cond, scale**3 * self._d3K(st), fillna)

    @property
    def dK0(self):
        if not hasattr(self, "_dK0_cache"):
            self._dK0_cache = (
                self.scale * CumulantGeneratingFunction.dK0.fget(self) + self.loc
            )
        return self._dK0_cache

    @property
    def d2K0(self):
        if not hasattr(self, "_d2K0_cache"):
            self._d2K0_cache = self.scale**2 * CumulantGeneratingFunction.d2K0.fget(
                self
            )
        return self._d2K0_cache

    @property
    def d3K0(self):
        if not hasattr(self, "_d3K0_cache"):
            self._d3K0_cache = self.scale**3 * CumulantGeneratingFunction.d3K0.fget(
                self
            )
        return self._d3K0_cache

    def add(self, other, inplace=False):
        """
        We use the following properties of the cumulant generating function
        for independent random variables :math:`X` and :math:`Y`:

        .. math::
            K_{aX+bY+c}(t) = K_X(at)+ K_Y(bt) +ct.

        Parameters
        ----------
        other : int, float or UnivariateCumulantGeneratingFunction
            Object to add
        inplace : bool, optional
            Whether to change the current object or create a new one. Now matter what is chosen,
            the results is always returned.

        References
        ----------
        [1] Bertsekas, Tsitsiklis (2000) - Introduction to probability
        """
        if isinstance(other, (int, float)):
            if inplace:
                self.loc = self.loc + other
                if hasattr(self, "_dK0_cache"):
                    delattr(self, "_dK0_cache")
                return self
            else:
                return UnivariateCumulantGeneratingFunction(
                    self._K,
                    loc=self.loc + other,
                    scale=self.scale,
                    dK=self._dK,
                    dK_inv=self._dK_inv,
                    d2K=self._d2K,
                    d3K=self._d3K,
                    dK0=self._dK0,
                    d2K0=self._d2K0,
                    d3K0=self._d3K0,
                    domain=self.domain,
                )
        elif isinstance(other, UnivariateCumulantGeneratingFunction):
            assert (
                not inplace
            ), "inplace not supported for UnivariateCumulantGeneratingFunction"
            return UnivariateCumulantGeneratingFunction(
                lambda t,
                ss=self.scale,
                so=other.scale,
                ls=self.loc,
                lo=other.loc: self.K(t, scale=ss, loc=ls)
                + other.K(t, scale=so, loc=lo),
                dK=lambda t,
                ss=self.scale,
                so=other.scale,
                ls=self.loc,
                lo=other.loc: self.dK(t, loc=ls, scale=ss)
                + other.dK(t, loc=lo, scale=so),
                d2K=lambda t,
                ss=self.scale,
                so=other.scale,
                ls=self.loc,
                lo=other.loc: self.d2K(t, scale=ss, loc=ls)
                + other.d2K(t, scale=so, loc=lo),
                d3K=lambda t,
                ss=self.scale,
                so=other.scale,
                ls=self.loc,
                lo=other.loc: self.d3K(t, scale=ss, loc=ls)
                + other.d3K(t, scale=so, loc=lo),
                domain=self.domain.intersect(other.domain),
            )
        else:
            raise ValueError(
                "Can only add a scalar or another UnivariateCumulantGeneratingFunction"
            )

    def mul(self, other, inplace=False):
        """
        We use the following properties of the cumulant generating function
        for independent random variables :math:`X` and :math:`Y`:

        .. math::
            K_{aX+bY+c}(t) = K_X(at)+ K_Y(bt) +ct.

        References
        ----------
        [1] Bertsekas, Tsitsiklis (2000) - Introduction to probability
        """
        if isinstance(other, (int, float)):
            if inplace:
                self.loc = other * self.loc
                self.scale = other * self.scale
                for att in ["_dK0_cache", "_d2K0_cache", "_d3K0_cache"]:
                    if hasattr(self, att):
                        delattr(self, att)
                return self
            else:
                return UnivariateCumulantGeneratingFunction(
                    self._K,
                    loc=other * self.loc,
                    scale=other * self.scale,
                    dK=self._dK,
                    dK_inv=self._dK_inv,
                    d2K=self._d2K,
                    d3K=self._d3K,
                    dK0=self._dK0,
                    d2K0=self._d2K0,
                    d3K0=self._d3K0,
                    domain=self.domain,
                )
        else:
            raise ValueError("Can only multiply with a scalar")


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
    and where in the denominator we multiply over :math:`i` combinations that sum up to m,
    are supposed to take the form of numpy arrays with m indices.
    By doing so, the first derivatives, becomes the gradient, the second the Hessian, and the the third, an array with matrices, etc.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Cumulant#Cumulant_generating_function

    [2] Bertsekas, Tsitsiklis (2000) - Introduction to probability

    [3] McCullagh (1995) - Tensor methods in statistics

    [4] Queens university lecture notes: https://mast.queensu.ca/~stat353/slides/5-multivariate_normal17_4.pdf

    [5] Kolassa (2006) - Series approximation methods in statistics, Chapter 6

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
    domain : Domain or None, optional
        Should correspond to the functions :math:`K` as provided above.
        Note that these can be of higher dimension when scale is a projection matrix.
        If not provided, the domain is assumed to be :math:`(-\infty, \infty)` for variables.
    """

    def __init__(
        self,
        K,
        dim=2,
        loc=0,
        scale=1,
        dK=None,
        dK_inv=None,
        d2K=None,
        d3K=None,
        dK0=None,
        d2K0=None,
        d3K0=None,
        domain=None,
    ):
        assert (
            pd.api.types.is_integer(dim) and dim > 0
        ), "dimimension must be an integer greater than 0"
        self.dim = dim
        if domain is None:
            if pd.api.types.is_array_like(loc):
                domain = Domain(dim=len(loc))
            elif pd.api.types.is_array_like(scale):
                domain = Domain(dim=len(scale))
            else:
                domain = Domain(dim=dim)
        super().__init__(
            K,
            loc=loc,
            scale=scale,
            dK=dK,
            dK_inv=dK_inv,
            d2K=d2K,
            d3K=d3K,
            domain=domain,
            dK0=dK0,
            d2K0=d2K0,
            d3K0=d3K0,
        )

    @property
    def loc_vect(self):
        if not hasattr(self, "_loc_vect_cache"):
            if isinstance(self.loc, np.ndarray) and len(self.loc.shape) == 1:
                return self.loc
            elif pd.api.types.is_number(self.loc):
                self._loc_vect_cache = np.full(self.dim, self.loc)
            elif isinstance(self.loc, np.ndarray) and len(self.loc.shape) == 0:
                self._loc_vect_cache = np.full(self.dim, self.loc.tolist())
            else:
                raise ValueError("Invalid type")
        return self._loc_vect_cache

    @property
    def scale_mat(self):
        if not hasattr(self, "_scale_mat_cache"):
            if isinstance(self.scale, np.ndarray) and len(self.scale.shape) == 2:
                self._scale_mat_cache = self.scale
            elif isinstance(self.scale, np.ndarray) and len(self.scale.shape) == 1:
                self._scale_mat_cache = np.diag(self.scale)
            elif isinstance(self.scale, np.ndarray) and len(self.scale.shape) == 0:
                self._scale_mat_cache = np.eye(self.dim) * self.scale.tolist()
            elif pd.api.types.is_number(self.scale):
                self._scale_mat_cache = np.eye(self.dim) * self.scale
            else:
                raise ValueError("Invalid type")
        return self._scale_mat_cache

    @CumulantGeneratingFunction.loc.setter
    def loc(self, loc):
        self._validate_loc(loc)
        if hasattr(self, "_loc_vect_cache"):
            delattr(self, "_loc_vect_cache")
        CumulantGeneratingFunction.loc.fset(self, np.asanyarray(loc))

    def _validate_loc(self, loc):
        assert (
            pd.api.types.is_number(loc)
            or (isinstance(loc, np.ndarray) and len(loc.shape) == 0)
            or (
                isinstance(loc, np.ndarray)
                and len(loc.shape) == 1
                and len(loc) == self.dim
            )
        ), "loc should be a scalar, vector of length {self.dim}"

    @CumulantGeneratingFunction.scale.setter
    def scale(self, scale):
        self._validate_scale(scale)
        for attr in ["_scale_mat_cache", "_scale_inv_cache", "_scale_mat_inv_cache"]:
            if hasattr(self, attr):
                delattr(self, attr)
        CumulantGeneratingFunction.scale.fset(self, np.asanyarray(scale))

    def _validate_scale(self, scale):
        assert (
            pd.api.types.is_number(scale)
            or (isinstance(scale, np.ndarray) and len(scale.shape) == 0)
            or (
                isinstance(scale, np.ndarray)
                and len(scale.shape) == 1
                and len(scale) == self.dim == self.domain.dim
            )
            or (
                isinstance(scale, np.ndarray)
                and len(scale.shape) == 2
                and scale.shape[0] == self.dim
                and scale.shape[1] == self.domain.dim
            )
        ), f"scale should be a scalar, vector of length {self.dim}, or a matrix"

    @property
    def scale_inv(self):
        if not hasattr(self, "_scale_inv_cache"):
            self._scale_inv_cache = self._get_scale_inv()
        return self._scale_inv_cache

    def _validate_scale_inv(self, scale_inv):
        assert (
            pd.api.types.is_number(scale_inv)
            or (isinstance(scale_inv, np.ndarray) and len(scale_inv.shape) == 0)
            or (
                isinstance(scale_inv, np.ndarray)
                and len(scale_inv.shape) == 1
                and len(scale_inv) == self.dim == self.domain.dim
            )
            or (
                isinstance(scale_inv, np.ndarray)
                and len(scale_inv.shape) == 2
                and scale_inv.shape[1] == self.dim
                and scale_inv.shape[0] == self.domain.dim
            )
        ), f"scale_inv should be a scalar, vector of length {self.dim}, or a matrix"

    def _get_scale_inv(self, scale=None):
        if scale is None:
            scale = self.scale
        else:
            self._validate_scale(scale)
        if isinstance(scale, np.ndarray) and len(scale.shape) == 2:
            scale_inv = np.linalg.pinv(scale)
        elif isinstance(scale, np.ndarray) and len(scale.shape) <= 1:
            scale_inv = 1 / scale
        elif pd.api.types.is_number(scale):
            scale_inv = 1 / scale
        else:
            raise ValueError("Invalid type for scale")
        self._validate_scale_inv(scale_inv)
        return scale_inv

    @property
    def scale_mat_inv(self):
        if not hasattr(self, "_scale_mat_inv_cache"):
            self._scale_mat_inv_cache = self._get_scale_mat_inv()
        return self._scale_mat_inv_cache

    def _get_scale_mat_inv(self, scale_inv=None, scale=None, scale_mat=None):
        if scale_inv is None:
            if scale is None and scale_mat is None:
                scale_inv = self.scale_inv
            elif scale is not None:
                scale_inv = self._get_scale_inv(scale)
            elif scale_mat is not None:
                scale_inv = np.linalg.pinv(scale_mat)
            else:
                raise ValueError("Invalid input")
        self._validate_scale_inv(scale_inv)
        if isinstance(scale_inv, np.ndarray) and len(scale_inv.shape) == 2:
            scale_mat_inv = scale_inv
        elif isinstance(scale_inv, np.ndarray) and len(scale_inv.shape) == 1:
            scale_mat_inv = np.diag(self.scale_inv)
        elif isinstance(scale_inv, np.ndarray) and len(scale_inv.shape) == 0:
            scale_mat_inv = np.eye(self.dim) * scale_inv.tolist()
        elif pd.api.types.is_number(scale_inv):
            scale_mat_inv = np.eye(self.dim) * scale_inv
        else:
            raise ValueError("Invalid type for scale_inv")
        return scale_mat_inv

    @type_wrapper(xloc=1)
    def _scale_t(self, t, scale=None):
        """
         Apply the scaling to the :math:`t_Y` variable of the cumulant generating function of
         the transformed random vector :math:`Y`.

         Given a random vector :math:`X` with cumulant generating function :math:`K_X(t_X)`,
         the cumulant generating function of the transformed random vector :math:`Y=AX+b` is given by:

         .. math::
             K_Y(t_Y) = K_X(t_X)+<b,t_Y>,
        were :math:`t_X = A^Tt_Y` is the transformed variable, and :math:`b` is the translation.
        """
        if scale is None:
            scale = self.scale
        else:
            scale = np.asanyarray(scale)
        ts = scale * t if len(scale.shape) <= 1 else np.dot(t, scale)
        assert ts.shape[-1] == self.domain.dim, "Dimensions do not match"
        return ts

    @type_wrapper(xloc=1)
    def _inv_scale_t(self, ts, inv_scale=None):
        """
        Apply the inverse scaling to the :math:`t_Y` variable of the cumulant generating function of
        the transformed random vector :math:`Y`.

        Given a random vector :math:`X` with cumulant generating function :math:`K_X(t_X)`,
        the cumulant generating function of the transformed random vector :math:`Y=AX+b` is given by:

        .. math::
            K_Y(t_Y) = K_X(t_X)+<b,t_Y>,
        were :math:`t_X = A^Tt_Y` is the transformed variable, and :math:`b` is the translation.
        """
        if inv_scale is None:
            inv_scale = self.scale_inv
        else:
            inv_scale = np.asanyarray(inv_scale)
            self._validate_scale_inv(inv_scale)
        t = ts / inv_scale if len(inv_scale.shape) <= 1 else np.dot(ts, inv_scale)
        assert t.shape[-1] == self.dim, "Dimensions do not match"
        return t

    @property
    def cov(self):
        return self.d2K0

    @property
    def cor(self):
        return sm.stats.moment_helpers.cov2corr(self.cov)

    @property
    def variance(self):
        return np.diag(self.cov)

    @property
    def dK0(self):
        if not hasattr(self, "_dK0_cache"):
            self._dK0_cache = (
                self.scale.dot(
                    self.dK(
                        np.zeros(self.domain.dim), loc=0, scale=np.ones(self.domain.dim)
                    )
                )
                + self.loc
            )
        return self._dK0_cache

    @property
    def d2K0(self):
        if not hasattr(self, "_d2K0_cache"):
            self._d2K0_cache = np.dot(
                np.dot(
                    self.scale,
                    self.d2K(
                        np.zeros(self.domain.dim), loc=0, scale=np.ones(self.domain.dim)
                    ),
                ),
                self.scale.T,
            )
        return self._d2K0_cache

    @property
    def d3K0(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        """
        Just set the other components to zero
        See Kolassa (2006) - Series approximation methods in statistics, Chapter 6.8
        """
        if isinstance(item, int):
            assert item < self.dim, "Index out of bounds"
            mcgf = self[[item]]
            return UnivariateCumulantGeneratingFunction(
                lambda t, mcgf=mcgf: mcgf.K(np.expand_dims(t, -1)),
                dK=lambda t, mcgf=mcgf: mcgf.dK(np.expand_dims(t, -1)).squeeze(),
                d2K=lambda t, mcgf=mcgf: mcgf.d2K(np.expand_dims(t, -1)).squeeze(),
                d3K=None,  # TODO: implement this one properly
                domain=mcgf.domain,
                loc=0,
                scale=1,
            )
        elif isinstance(item, slice):
            return self[list(range(self.dim))[item]]
        elif isinstance(item, tuple):
            return self[list(item)]
        elif isinstance(item, list):
            assert all(pd.api.types.is_integer(i) for i in item), "Invalid index"
            assert all(i < self.dim for i in item), "Index out of bounds"
            assert len(set(item)) == len(item), "Index must be unique"
            # return self.ldot(np.eye(self.dim)[item]) #alternative implementation
            idmat = np.eye(self.dim)
            if isinstance(self.scale, np.ndarray) and len(self.scale.shape) == 2:
                scale = self.scale.T.dot(idmat[:, item]).T
            elif isinstance(self.scale, np.ndarray) and len(self.scale.shape) == 1:
                scale = np.diag(self.scale)[item]
            else:
                scale = self.scale * idmat[:, item].T
            if isinstance(self.loc, np.ndarray) and len(self.loc.shape) == 1:
                loc = self.loc[item]
            else:
                loc = self.loc
            return MultivariateCumulantGeneratingFunction(
                lambda t, loc=loc, scale=scale: self.K(t, loc=loc, scale=scale),
                dim=len(item),
                dK=lambda t, loc=loc, scale=scale: self.dK(t, loc=loc, scale=scale)
                if self._dK is not None
                else None,
                d2K=lambda t, loc=loc, scale=scale: self.d2K(t, loc=loc, scale=scale)
                if self._d2K is not None
                else None,
                d3K=None,  # TODO: implement this one properly
                domain=self.domain.ldotinv(scale.T),
                loc=0,
                scale=1,
            )
        else:
            raise ValueError("Invalid index")

    def add(self, other, inplace=True):
        r"""
        We use the following properties of the cumulant generating function
        for independent random vectors :math:`X` and :math:`Y`:

        .. math::
            K_{AX+BY+c}(t) = K_X(A^Tt)+ K_Y(B^Tt) +<c,t>

        Note, when add a univariate random variable :math:`Y`, i.e., in terms of its
        cumulant generating function, we add a univariate random varible
        independent of everything else,

        .. math::
            \text{correlation}(X_i,Y)=0, \forall i
        but the variable will have full correlation with itself.
        Consequently:

        ..math::
            \text{covariance(X_i+Y, X_j+Y)} = covariance(X_i,X_j) + variance(Y).

        References
        ----------
        [1] Bertsekas, Tsitsiklis (2000) - Introduction to probability

        [2] Queens university lecture notes: https://mast.queensu.ca/~stat353/slides/5-multivariate_normal17_4.pdf

        [3] Kolassa (2006) - Series approximation methods in statistics, Chapter 6
        """
        if isinstance(other, (int, float, np.ndarray)):
            if isinstance(other, np.ndarray):
                assert len(other) == self.dim, "Dimensions do not match"
            if inplace:
                self.loc = self.loc + other
                if hasattr(self, "_dK0_cache"):
                    delattr(self, "_dK0_cache")
                return self
            else:
                return MultivariateCumulantGeneratingFunction(
                    self._K,
                    loc=self.loc + other,
                    scale=self.scale,
                    dK=self._dK,
                    dK_inv=self._dK_inv,
                    d2K=self._d2K,
                    d3K=self._d3K,
                    dK0=self._dK0,
                    d2K0=self._d2K0,
                    d3K0=self._d3K0,
                    domain=self.domain,
                )
        elif isinstance(other, UnivariateCumulantGeneratingFunction):
            assert (
                not inplace
            ), "inplace not supported for UnivariateCumulantGeneratingFunction"
            return MultivariateCumulantGeneratingFunction(
                lambda t: self.K(t) + other.K(np.sum(t, axis=-1)),
                dim=self.dim,
                dK=lambda t: np.apply_along_axis(
                    lambda x: np.add(x, other.dK(np.sum(t, axis=-1))), 0, self.dK(t)
                ),
                d2K=lambda t: np.apply_along_axis(
                    lambda x: np.add(x, other.d2K(np.sum(t, axis=-1))), 0, self.d2K(t)
                ),
                d3K=lambda t: np.apply_along_axis(
                    lambda x: np.add(x, other.d3K(np.sum(t, axis=-1))), 0, self.d3K(t)
                ),
                domain=self.domain.intersect(
                    other.domain.ldotinv(np.ones((1, self.dim)))
                ),
            )
        elif isinstance(other, MultivariateCumulantGeneratingFunction):
            assert (
                not inplace
            ), "inplace not supported for MultivariateCumulantGeneratingFunction"
            assert self.dim == other.dim, "Dimensions must be equal"
            return MultivariateCumulantGeneratingFunction(
                lambda t,
                ss=self.scale,
                so=other.scale,
                ls=self.loc,
                lo=other.loc: self.K(t, scale=ss, loc=ls)
                + other.K(t, scale=so, loc=lo),
                dK=lambda t,
                ss=self.scale,
                so=other.scale,
                ls=self.loc,
                lo=other.loc: self.dK(t, loc=ls, scale=ss)
                + other.dK(t, loc=lo, scale=so),
                d2K=lambda t,
                ss=self.scale,
                so=other.scale,
                ls=self.loc,
                lo=other.loc: self.d2K(t, scale=ss, loc=ls)
                + other.d2K(t, scale=so, loc=lo),
                d3K=lambda t,
                ss=self.scale,
                so=other.scale,
                ls=self.loc,
                lo=other.loc: self.d3K(t, scale=ss, loc=ls)
                + other.d3K(t, scale=so, loc=lo),
                domain=self.domain.intersect(other.domain),
            )
        else:
            raise ValueError(
                "Can only add a scalar or another CumulantGeneratingFunction"
            )

    def mul(self, other, inplace=False):
        r"""
        We use the following properties of the cumulant generating function
        for independent random variables :math:`X` and :math:`Y`:

         .. math::
            K_{AX}(t) = K_X(A^Tt)

        References
        ----------
        [1] Bertsekas, Tsitsiklis (2000) - Introduction to probability

        [2] Queens university lecture notes: https://mast.queensu.ca/~stat353/slides/5-multivariate_normal17_4.pdf

        [3] Kolassa (2006) - Series approximation methods in statistics, Chapter 6
        """
        if isinstance(other, (int, float)):
            if inplace:
                self.loc = self.loc * other
                self.scale = self.scale * other
                for att in ["_dK0_cache", "_d2K0_cache", "_d3K0_cache"]:
                    if hasattr(self, att):
                        delattr(self, att)
                return self
            else:
                return MultivariateCumulantGeneratingFunction(
                    self._K,
                    loc=self.loc * other,
                    scale=self.scale * other,
                    dK=self._dK,
                    dK_inv=self._dK_inv,
                    d2K=self._d2K,
                    d3K=self._d3K,
                    dK0=self._dK0,
                    d2K0=self._d2K0,
                    d3K0=self._d3K0,
                    domain=self.domain,
                    dim=self.dim,
                )
        elif isinstance(other, np.ndarray) and len(other.shape) == 1:
            assert (
                len(other) == self.dim
            ), "Vector rescaling should work on all variables"
            # This is simply a rescaling of all the components
            if inplace:
                self.loc = self.loc * other
                self.scale = (other * np.asanyarray(self.scale).T).T
                for att in ["_dK0_cache", "_d2K0_cache", "_d3K0_cache"]:
                    if hasattr(self, att):
                        delattr(self, att)
                return self
            else:
                return MultivariateCumulantGeneratingFunction(
                    self._K,
                    loc=other * self.loc,
                    scale=(other * np.asanyarray(self.scale).T).T,
                    dK=self._dK,
                    dK_inv=self._dK_inv,
                    d2K=self._d2K,
                    d3K=self._d3K,
                    dK0=self._dK0,
                    d2K0=self._d2K0,
                    d3K0=self._d3K0,
                    domain=self.domain,
                    dim=self.dim,
                )
        else:
            raise ValueError("Can only multiply with a scalar or vector")

    def ldot(self, A, inplace=False):
        """
        Dot product with a matrix or vector.

        If :math:`A` is a matrix, it transforms the random vector :math:`X` to :math:`AX`.
        In this case the result is another multivariate cumulant generating function

        If :math:`A` is a vector, it transforms :math:`X` to :math:`<A,x>`, where
        :math:`<.,.>` denotes the inner product.
        In this case, the result is a univariate cumulant generating function.

        We use the following properties of the cumulant generating function
        for independent random variables :math:`X` and :math:`Y`:

         .. math::
            K_{AX}(t) = K_X(A^Tt)

        References
        ----------
        [1] Bertsekas, Tsitsiklis (2000) - Introduction to probability

        [2] Queens university lecture notes: https://mast.queensu.ca/~stat353/slides/5-multivariate_normal17_4.pdf

        [3] Kolassa (2006) - Series approximation methods in statistics, Chapter 6

        """
        if isinstance(A, np.ndarray) and len(A.shape) == 1:
            assert (
                not inplace
            ), "Inplace not possible when projecting to univariate case"
            return self.ldot(np.atleast_2d(A), inplace=inplace)[0]
        elif isinstance(A, np.ndarray) and len(A.shape) == 2 and A.shape[1] == self.dim:
            if inplace:
                assert (
                    A.shape[0] == self.dim
                ), "inplace ldot only possible if dimension remains the same"
                self.loc = A.dot(self.loc_vect)
                self.scale = A.dot(self.scale_mat)
                for att in ["_dK0_cache", "_d2K0_cache", "_d3K0_cache"]:
                    if hasattr(self, att):
                        delattr(self, att)
                return self
            else:
                return MultivariateCumulantGeneratingFunction(
                    self._K,
                    dim=A.shape[0],
                    loc=A.dot(self.loc_vect),
                    scale=A.dot(self.scale_mat),
                    dK=self._dK,
                    dK_inv=self._dK_inv,
                    d2K=self._d2K,
                    d3K=self._d3K,
                    dK0=self._dK0,
                    d2K0=self._d2K0,
                    d3K0=self._d3K0,
                    domain=self.domain,
                )
        else:
            raise ValueError("Invalid shape")

    def stack(self, other):
        return self.from_cgfs(self, other)

    @type_wrapper(xloc=1)
    def K(self, t, fillna=np.nan, loc=None, scale=None):
        # Initialize
        loc = self.loc if loc is None else np.asanyarray(loc)
        scale = self.scale if scale is None else np.asanyarray(scale)
        if len(t.shape) == 0:
            t = np.full(self.dim, t)
        # st = (scale * t.T).T if len(scale.shape) == 1 else np.dot(t, scale)
        st = self._scale_t(t, scale=scale)
        cond = np.squeeze(self.domain.is_in_domain(st))
        st = np.where(cond, st.T, 0).T  # prevent outside domain evaluations
        # Evaluate
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="All-NaN slice encountered"
            )
            val = self._K(st)
            if np.isscalar(val):
                val += np.sum(loc * t)
            elif (
                pd.api.types.is_array_like(val)
                and len(val.shape) == 1
                and len(t.shape) == 1
            ):
                val += np.atleast_2d(t).T.dot(loc).squeeze()
            elif pd.api.types.is_array_like(val) and len(val.shape) == 1:
                val += np.sum((loc * t).T, axis=0)
            else:
                raise RuntimeError(
                    "Only scalar and vector valued return values are supported"
                )
            return np.where(cond, val, fillna)

    @type_wrapper(xloc=1)
    def dK(self, t, loc=None, scale=None, fillna=np.nan):
        r"""
        Note, the current implementation uses numerical differentiation, but
        an alternative way would be to use the following result:

        .. math::
            K'(t)=\frac{E X\exp{<t,X>}}{M(t)},
        where :math:`K(t)` is the cumulant generating function, and :math:`M(t)`
        is the moment generating function of :math:`X`.

        References
        ----------
        [1] Ganesh, O'Connell (2004) - Big Quesues in Probability and Statistics
        """
        # Initialize
        if len(t.shape) == 0:
            t = np.full(self.dim, t)
        if self._dK is None:
            assert (
                has_numdifftools
            ), "Numdifftools is required if derivatives are not provided"
            self._dK = np.vectorize(
                nd.Gradient(lambda tt: self.K(tt, loc=0, scale=1)), signature="(n)->(n)"
            )
        loc = self.loc if loc is None else np.asanyarray(loc)
        scale = self.scale if scale is None else np.asanyarray(scale)
        ts = self._scale_t(t, scale=scale)
        cond = np.squeeze(self.domain.is_in_domain(ts))
        ts = np.where(
            cond, ts.T, 0
        ).T  # numdifftools doesn't work if any evaluates to NaN
        # Evaluate
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="All-NaN slice encountered"
            )
            if len(scale.shape) <= 1:
                y = np.add(scale * self._dK(ts), loc)
            else:
                y = np.add(np.dot(self._dK(ts), scale.T), loc)
            if np.isscalar(cond):
                if cond:
                    return y
                elif np.isscalar(y):
                    return fillna
                else:
                    return np.fill(np.asanyarray(y).shape, fillna)
            else:
                y = y.astype(np.float64)
                y[~cond] = fillna
                return y

    @type_wrapper(xloc=1)
    def dK_inv(self, x, t0=None, loc=None, scale=None, fillna=np.nan, **kwargs):
        """
        Inverse of the derivative of the cumulant generating function.

        It solves:

        .. math::
            x = K'(t).
        """
        # Handle vectorized evaluation
        if len(x.shape) > 2:
            raise ValueError("Invalid shape")
        elif t0 is not None and len(t0.shape) > 2:
            raise ValueError("Invalid shape")
        elif len(x.shape) == 2 and (t0 is None or len(t0.shape) == 1):
            return np.asanyarray(
                [
                    self.dK_inv(
                        xx, t0=t0, loc=loc, scale=scale, fillna=fillna, **kwargs
                    )
                    for xx in x
                ]
            )
        elif len(x.shape) == 2 and len(t0.shape) == 2:
            return np.asanyarray(
                [
                    self.dK_inv(
                        xx, t0=tt0, loc=loc, scale=scale, fillna=fillna, **kwargs
                    )
                    for xx, tt0 in zip(x, t0)
                ]
            )
        # Otherwise just a single evaluation
        # Handle scaling
        loc = self.loc if loc is None else loc
        if scale is not None:
            self._validate_scale(scale)
            scale_inv = self._get_scale_inv(scale)
        else:
            scale_inv = self.scale_inv
        if pd.api.types.is_number(scale_inv):
            x = np.asanyarray(x - loc) * scale_inv
        elif isinstance(scale_inv, np.ndarray) and len(scale_inv.shape) == 1:
            x = np.asanyarray(x - loc) * scale_inv
        else:
            x = self.scale_mat_inv.T.dot(np.asanyarray(x - loc))
        # If dK_inv is provided
        if self._dK_inv is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action="ignore", message="All-NaN slice encountered"
                )
                t = self._dK_inv(x)
        else:
            # Otherwise solve numerically
            kwargs["x0"] = np.zeros(x.shape) if t0 is None else np.asanayarray(t0)
            kwargs.setdefault("jac", lambda tt: self.d2K(tt, loc=0, scale=1))
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
                try:
                    res = spo.root(
                        lambda t, x=x: self.dK(t, loc=0, scale=1) - x, **kwargs
                    )
                except Exception:
                    continue
                if res.success:
                    t = np.asanyarray(res.x)
                    break
            else:
                t = np.asanyarray(
                    [
                        self.dK_inv(
                            xx, loc=0, scale=1, t0=None if t0 is None else t0[i]
                        )
                        for i, xx in enumerate(x)
                    ]
                )
        # Post processing
        cond = np.squeeze(self.domain.is_in_domain(t))
        if pd.api.types.is_number(scale_inv):
            t = t * scale_inv
        elif isinstance(scale_inv, np.ndarray) and len(scale_inv.shape) == 1:
            t = t * scale_inv
        else:
            t = scale_inv.T.dot(t)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="All-NaN slice encountered"
            )
            if np.isscalar(cond):
                if cond:
                    return t
                elif np.isscalar(t):
                    return fillna
                else:
                    return np.fill(np.asanyarray(t).shape, fillna)
            else:
                t = t.astype(np.float64)
                t[~cond] = fillna
                return t

    @type_wrapper(xloc=1)
    def d2K(self, t, loc=None, scale=None, fillna=np.nan):
        """
        This is the Hessian, i.e., the matrix with second order partial derivatives.
        """
        # Initialize
        if len(t.shape) == 0:
            t = np.full(self.dim, t)
        if self._d2K is None:
            assert (
                has_numdifftools
            ), "Numdifftools is required if derivatives are not provided"
            self._d2K = np.vectorize(
                nd.Hessian(lambda tt: self.K(tt, loc=0, scale=1)),
                signature="(n)->(n,n)",
            )
        loc = self.loc if loc is None else np.asanyarray(loc)
        scale = self.scale if scale is None else np.asanyarray(scale)
        assert self._d2K is not None, "d2K must be specified"
        t = np.asanyarray(t)
        ts = self._scale_t(t, scale=scale)
        cond = self.domain.is_in_domain(ts)
        ts = np.where(
            cond, ts.T, 0
        ).T  # numdifftools doesn't work if any evaluates to NaN
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore", message="All-NaN slice encountered"
            )
            y = self._d2K(ts)
            if len(scale.shape) == 0:
                y *= scale**2
            elif len(scale.shape) == 1:
                # This is supposed to be equivalent to the else case
                def diagscalemult(x):
                    x = np.apply_along_axis(lambda z: np.multiply(scale, z), 0, x)
                    return np.apply_along_axis(lambda z: np.multiply(z, scale), 1, x)

                if len(y.shape) == 2:
                    y = diagscalemult(y)
                elif len(y.shape) == 3:
                    y = np.array([diagscalemult(yy) for yy in y])
                else:
                    raise IndexError("retval should a matrix of list of matrices")
            else:
                if len(y.shape) == 2:
                    y = np.dot(np.dot(scale, y), scale.T)
                elif len(y.shape) == 3:
                    y = np.array([np.dot(np.dot(scale, yy), scale.T) for yy in y])
                else:
                    raise IndexError("retval should a matrix of list of matrices")

            if np.isscalar(cond):
                if cond:
                    return y
                elif np.isscalar(y):
                    return fillna
                else:
                    return np.fill(np.asanyarray(y).shape, fillna)
            else:
                y = y.astype(np.float64)
                y[~cond] = fillna
                return y

    @type_wrapper(xloc=1)
    def d3K(self, t, loc=None, scale=None, fillna=np.nan):
        """
        See Kolassa 2006, Chapter 6 for some insights
        """
        raise NotImplementedError()
        # TODO: I'm not sure what this is supposed to be, some kind of tensor?

    @classmethod
    def from_univariate(cls, *cgfs):
        """
        Create a multivariate cgf from a list of univariate cgfs.
        """
        assert (
            len(cgfs) >= 1
        ), "at least 2 Univariate cumulant generating functions should be supplied"
        assert all(
            isinstance(cgf, UnivariateCumulantGeneratingFunction) for cgf in cgfs
        ), "All cgfs must be univariate"
        return cls(
            lambda t, cgfs=cgfs: np.sum(
                [cgf.K(ti) for ti, cgf in zip(t.T, cgfs)], axis=0
            ),
            dim=len(cgfs),
            loc=0,
            scale=1,
            dK=lambda t, cgfs=cgfs: np.array(
                [cgf.dK(ti) for ti, cgf in zip(t.T, cgfs)]
            ).T,
            d2K=lambda t, cgfs=cgfs: np.apply_along_axis(
                np.diag, 0, np.array([cgf.d2K(ti) for ti, cgf in zip(t.T, cgfs)])
            ).swapaxes(0, -1),
            domain=Domain.from_domains(*[cgf.domain for cgf in cgfs]),
        )

    @classmethod
    def from_cgfs(cls, *cgfs):
        """
        Create a multivariate cgf from a list of univariate cgfs.
        """
        assert (
            len(cgfs) > 1
        ), "at least 2 cumulant generating functions should be supplied"
        assert all(
            isinstance(cgf, CumulantGeneratingFunction) for cgf in cgfs
        ), "All cgfs must be CumulantGeneratingFunction"
        if all(isinstance(cgf, UnivariateCumulantGeneratingFunction) for cgf in cgfs):
            return cls.from_univariate(*cgfs)
        else:
            dims = np.array(
                [
                    cgf.dim
                    if isinstance(cgf, MultivariateCumulantGeneratingFunction)
                    else 1
                    for cgf in cgfs
                ]
            )

            def ti(t, i):
                if isinstance(cgfs[i], MultivariateCumulantGeneratingFunction):
                    return t.T[dims[:i].sum() : dims[: i + 1].sum()].T
                else:
                    return t.T[dims[:i].sum() : dims[: i + 1].sum()].T.squeeze()

            @type_wrapper(xloc=0)
            def K(t):
                return np.sum(
                    [np.squeeze(cgf.K(ti(t, i))) for i, cgf in enumerate(cgfs)],
                    axis=0,
                )

            @type_wrapper(xloc=0)
            def dK(t):
                vals = [
                    np.atleast_1d(cgf.dK(ti(t, i)))  # .reshape((dims.sum(), dims[i]))
                    for i, cgf in enumerate(cgfs)
                ]
                if len(t.shape) > 1:
                    vals = [
                        v.reshape((t.shape[0], dims[i])) for i, v in enumerate(vals)
                    ]
                else:
                    vals = [v.reshape(dims[i]) for i, v in enumerate(vals)]
                return np.concatenate(vals, axis=-1)

            @type_wrapper(xloc=0)
            def d2K(t):
                if len(t.shape) == 1:
                    return sp.linalg.block_diag(
                        *[cgf.d2K(ti(t, i)) for i, cgf in enumerate(cgfs)]
                    )
                elif len(t.shape) == 2:
                    return np.array([d2K(tt) for tt in t])
                else:
                    raise

            domain = Domain.from_domains(*[cgf.domain for cgf in cgfs])
            return cls(K, dim=dims.sum(), loc=0, scale=1, dK=dK, d2K=d2K, domain=domain)


# TODO: can we easily construct a conditional cgf using the conditional density? probabliy not?
