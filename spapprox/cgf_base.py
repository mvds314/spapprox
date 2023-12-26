#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import warnings
import numpy as np
import pandas as pd
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
        self._loc = np.asanyarray(loc)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        for att in ["_dK0_cache", "_d2K0_cache", "_d3K0_cache"]:
            if hasattr(self, att):
                delattr(self, att)
        self._scale = np.asanyarray(scale)

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
        if not hasattr(self, "_dK0_cache"):
            self._dK0_cache = self.scale.T.dot(self._dK0) + self.loc
        return self._dK0_cache

    @property
    def d2K0(self):
        if self._d2K0 is None:
            self._d2K0 = self.d2K(0, loc=0, scale=1)
        if not hasattr(self, "_d2K0_cache"):
            self._d2K0_cache = np.dot(np.power(self.scale.T, 2), self._d2K0)
        return self._d2K0_cache

    @property
    def d3K0(self):
        if self._d3K0 is None:
            self._d3K0 = self.d3K(0, loc=0, scale=1)
        if not hasattr(self, "_d3K0_cache"):
            self._d3K0_cache = np.dot(np.power(self.scale.T, 3), self._d3K0)
        return self._d3K0_cache

    # TODO: should we consider a default implementation here?
    @abstractmethod
    def add(self, other, inplace=False):
        raise NotImplementedError()

    def __add__(self, other):
        return self.add(other, inplace=False)

    def __radd__(self, other):
        return self.__add__(other)

    @abstractmethod
    def mul(self, other, inplace=False):
        raise NotImplementedError()

    def __mul__(self, other):
        return self.mul(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    @type_wrapper(xloc=1)
    def K(self, t, fillna=np.nan, loc=None, scale=None):
        loc = self.loc if loc is None else np.asanyarray(loc)
        scale = self.scale if scale is None else np.asanyarray(scale)
        t = np.asanyarray(t)
        tt = scale.T.dot(t)
        cond = self.domain.is_in_domain(tt)
        tt = np.where(cond, tt, 0)  # prevent outside domain evaluations
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
            return np.where(cond, self._K(tt) + loc.dot(t), fillna)

    @type_wrapper(xloc=1)
    def dK(self, t, fillna=np.nan, loc=None, scale=None):
        loc = self.loc if loc is None else np.asanyarray(loc)
        scale = self.scale if scale is None else np.asanyarray(scale)
        assert self._dK is not None, "dK must be specified"
        t = np.asanyarray(t)
        tt = scale.T.dot(t)
        cond = self.domain.is_in_domain(tt)
        tt = np.where(cond, tt, 0)  # numdifftolls doesn't work if any evaluetes to NaN
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
            return np.where(cond, scale.T.dot(self._dK(tt)) + loc, fillna)

    @type_wrapper(xloc=1)
    def dK_inv(self, x, loc=None, scale=None, fillna=np.nan):
        """
        Default implementation
        """
        raise NotImplementedError()

    @type_wrapper(xloc=1)
    def d2K(self, t, fillna=np.nan, loc=None, scale=None):
        loc = self.loc if loc is None else np.asanyarray(loc)
        scale = self.scale if scale is None else np.asanyarray(scale)
        assert self._d2K is not None, "d2K must be specified"
        t = np.asanyarray(t)
        tt = scale.T.dot(t)
        cond = self.domain.is_in_domain(tt)
        tt = np.where(cond, tt, 0)  # numdifftolls doesn't work if any evaluetes to NaN
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
            return np.where(cond, np.dot(np.power(scale.T, 2), self._d2K(tt)), fillna)

    @type_wrapper(xloc=1)
    def d3K(self, t, fillna=np.nan, loc=None, scale=None):
        loc = self.loc if loc is None else np.asanyarray(loc)
        scale = self.scale if scale is None else np.asanyarray(scale)
        assert self._d3K is not None, "d3K must be specified"
        t = np.asanyarray(t)
        tt = scale.T.dot(t)
        cond = tt in self.domain
        tt = np.where(cond, tt, 0)  # numdifftolls doesn't work if any evaluetes to NaN
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
            return np.where(cond, np.dot(np.power(scale.T, 3), self._d3K(tt)), fillna)


# TODO: continue here, and give this one the loc and scale params
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
        assert pd.api.types.is_number(loc) or (
            isinstance(loc, np.ndarray) and len(loc) == 0
        ), "loc should be a scalar"
        CumulantGeneratingFunction.loc.fset(self, loc)

    @CumulantGeneratingFunction.scale.setter
    def scale(self, scale):
        assert pd.api.types.is_number(scale) or (
            isinstance(scale, np.ndarray) and len(scale) == 0
        ), "scale should be a scalar"
        CumulantGeneratingFunction.scale.fset(self, scale)

    @property
    def variance(self):
        return self.kappa2

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
        if isinstance(other, (int, float)) and inplace:
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
            assert not inplace, "inplace not supported for UniariateCumulantGeneratingFunction"
            return UnivariateCumulantGeneratingFunction(
                lambda t, ss=self.scale, so=other.scale, ls=self.loc, loco=other.loc: self.K(
                    t, scale=ss
                )
                + other.K(t, scale=so),
                dK=lambda t, ss=self.scale, so=other.scale, ls=self.loc, lo=other.loc: self.dK(
                    t, loc=ls, scale=ss
                )
                + other.dK(t, loc=lo, scale=so),
                d2K=lambda t, ss=self.scale, so=other.scale, ls=self.loc, lo=other.loc: self.d2K(
                    t, scale=ss, loc=ls
                )
                + other.d2K(t, scale=so, loc=lo),
                d3K=lambda t, ss=self.scale, so=other.scale, ls=self.loc, lo=other.loc: self.d3K(
                    t, scale=ss, loc=ls
                )
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
        if self._dK is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._dK = nd.Derivative(lambda tt: self.K(tt, loc=0, scale=1), n=1)
        return super().dK(t, loc=loc, scale=scale, fillna=fillna)

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
        loc = self.loc if loc is None else np.asanyarray(loc)
        scale = self.scale if scale is None else np.asanyarray(scale)
        x = np.asanyarray((x - loc) / scale)
        if self._dK_inv is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
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
                        assert self.dK(lb, loc=0, scale=1) < x < self.dK(ub, loc=0, scale=1)
                        kwargs["bracket"] = [lb, ub]
                    try:
                        res = spo.root_scalar(
                            lambda t, x=x: self.dK(t, loc=0, scale=1) - x, **kwargs
                        )
                    except:
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
                        res = spo.root(lambda t, x=x: self.dK(t, loc=0, scale=1) - x, **kwargs)
                    except:
                        continue
                    if res.success:
                        y = np.asanyarray(res.x)
                        break
                else:
                    y = np.asanyarray(
                        [
                            self.dK_inv(xx, loc=0, scale=1, t0=None if t0 is None else t0[i])
                            for i, xx in enumerate(x)
                        ]
                    )
        cond = self.domain.is_in_domain(y)
        return np.where(cond, y / scale, fillna)

    @type_wrapper(xloc=1)
    def d2K(self, t, loc=None, scale=None, fillna=np.nan):
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
        if self._d2K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._d2K = nd.Derivative(lambda tt: self.K(tt, loc=0, scale=1), n=2)
        return super().d2K(t, loc=loc, scale=scale, fillna=fillna)

    @type_wrapper(xloc=1)
    def d3K(self, t, loc=None, scale=None, fillna=np.nan):
        if self._d3K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._d3K = nd.Derivative(lambda tt: self.K(tt, loc=0, scale=1), n=3)
        return super().d3K(t, loc=loc, scale=scale, fillna=fillna)


# TODO: give this one loc and scale params
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
            K,
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
    def cov(self):
        return self.d2K0

    @property
    def cor(self):
        return sm.stats.moment_helpers.cov2corr(self.cov)

    @property
    def variance(self):
        return np.diag(self.cov)

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
        """
        raise NotImplementedError()
        # TODO: double check this
        if isinstance(other, (int, float)):
            return (np.ones(self.dim) * other) * self
        elif isinstance(other, np.ndarray):
            return MultivariateCumulantGeneratingFunction(
                lambda t: self.K(t) + np.sum(other * t),
                dim=self.dim,
                dK=lambda t: self.dK(t) + other,
                dK_inv=lambda x: self.dK_inv(x - other),
                d2K=lambda t: self.d2K(t),
                d3K=lambda t: self.d3K(t),
                domain=lambda t: self.domain(t),
            )
        elif isinstance(other, UnivariateCumulantGeneratingFunction):
            return MultivariateCumulantGeneratingFunction(
                lambda t: self.K(t) + np.sum([other.K(ti) for ti in t]),
                dim=self.dim,
                dK=lambda t: self.dK(t) + np.array([other.dK(ti) for ti in t]),
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
        super().__add__(other)

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
        """
        raise NotImplementedError()
        # TODO: double check this
        if isinstance(other, (int, float)):
            return MultivariateCumulantGeneratingFunction(
                lambda t: self.K(other * t),
                dK=lambda t: other * self.dK(other * t),
                dK_inv=lambda x: self.dK_inv(x / other) / other,
                d2K=lambda t: other**2 * self.d2K(other * t),
                d3K=lambda t: other**3 * self.d3K(other * t),
                domain=lambda t: self.domain(t),
            )
        elif isinstance(other, np.ndarray) and len(other.shape) == 1:
            assert len(other) == self.dim, "Vector rescaling should work on all variables"
            # This is simply a rescaling of all the components
            return MultivariateCumulantGeneratingFunction(
                lambda t: self.K(other * t),
                dK=lambda t: other * self.dK(other * t),
                dK_inv=lambda x: self.dK_inv(x / other) / other,
                d2K=lambda t: np.atleast_2d(other).T.dot(np.atleast_2d(other))
                * (self.d2K(other * t)),
                d3K=lambda t, A=np.array(
                    [
                        [
                            [other[i] * other[j] * other[k] for i in range(self.dim)]
                            for j in range(self.dim)
                        ]
                        for k in range(self.dim)
                    ]
                ): A
                * self.d3K(other * t),
                domain=lambda t: self.domain(t),
            )
        elif isinstance(other, np.ndarray) and len(other.shape) == 2:
            assert other.shape[1] == self.dim, "Dimension must be equal"
            assert np.allclose(
                self.d2K0 - np.diag(self.d2K0), 0
            ), "Only linear transformation of indepdent variables are possible"
            return MultivariateCumulantGeneratingFunction(
                lambda t: np.sum([self.K(col * t[i]) for i, col in enumerate(other.T)]),
                # TODO: continue here and fix the other ones
                dK=lambda t: other * self.dK(other * t),
                dK_inv=lambda x: self.dK_inv(x / other) / other,
                d2K=lambda t: other**2 * self.d2K(other * t),
                d3K=lambda t: other**3 * self.d3K(other * t),
                domain=lambda t: self.domain(t),
            )
        else:
            raise ValueError("Can only multiply with a scalar")

    def stack(self, other):
        # TODO: implement this
        raise NotImplementedError()

    @type_wrapper(xloc=1)
    def K(self, t, fillna=np.nan):
        t = np.asanyarray(t)
        assert t.shape[0] == self.dim, "Dimensions do not match"
        return super().K(t, fillna=fillna)

    @type_wrapper(xloc=1)
    def dK(self, t, fillna=np.nan):
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
        t = np.asanyarray(t)
        assert t.shape[0] == self.dim, "Dimensions do not match"
        if self._dK is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._dK = nd.Derivative(self.K, n=1)
        return super().dK(t, fillna=fillna)

    @type_wrapper(xloc=1)
    def dK_inv(self, x, t0=None, **kwargs):
        """
        Inverse of the derivative of the cumulant generating function.

        It solves:

        .. math::
            x = K'(t).
        """
        x = np.asanyarray(x)
        assert x.shape[0] == self.dim, "Dimensions do not match"
        # TODO: maybe implement a generic solver, is this the gradient or the innerproduct with the gradient
        raise NotImplementedError()

    @type_wrapper(xloc=1)
    def d2K(self, t, fillna=np.nan):
        """
        This is supposed to be the Hessian, i.e., the matrix with second order partial derivatives.
        """
        t = np.asanyarray(t)
        assert t.shape[0] == self.dim, "Dimensions do not match"
        if self._d2K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._d2K = nd.Derivative(self.K, n=2)
        return super().d2K(t, fillna=fillna)

    @type_wrapper(xloc=1)
    def d3K(self, t, fillna=np.nan):
        # TODO: I'm not sure what this is supposed to be
        t = np.asanyarray(t)
        assert t.shape[0] == self.dim, "Dimensions do not match"
        if self._d3K is None:
            assert has_numdifftools, "Numdifftools is required if derivatives are not provided"
            self._d3K = nd.Derivative(self.K, n=3)
        return super().d3K(t, fillna=fillna)


# TODO: add affine transformation parameters (maybe through  loc and scale?)
# TODO: write some tests first for the albove, using normal distribution
# TODO: add stacking functionality
# TODO: add some slicing, so that we can extract the marginals
# TODO: implement multivariate saddlepoint approximation
# TODO: can we construct a conditional cgf, or does that always go through the saddlepoint approximation?
