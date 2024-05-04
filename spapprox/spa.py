# -*- coding: utf-8 -*-
import inspect
import itertools
from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
from scipy.integrate import nquad, quad
from scipy.interpolate import RegularGridInterpolator
from statsmodels.tools.validation import PandasWrapper

from .cgfs import (
    MultivariateCumulantGeneratingFunction,
    UnivariateCumulantGeneratingFunction,
    univariate_sample_mean,
)
from .util import fib, type_wrapper

try:
    import fastnorm
except ImportError:
    _has_fastnorm = False
else:
    _has_fastnorm = True


class SaddlePointApprox(ABC):
    """
    Interface for saddle point approximation of probability density functions
    """

    def __init__(self, cgf, pdf_normalization=None):
        assert isinstance(
            cgf, (UnivariateCumulantGeneratingFunction, MultivariateCumulantGeneratingFunction)
        )
        self.cgf = cgf
        self._pdf_normalization_cache = pdf_normalization

    def pdf(self, x=None, t=None, normalize_pdf=True, fillna=np.nan, **solver_kwargs):
        r"""
        Saddle point approximation of the probability density function.
        Given by

        .. math::
            f(x) \approx \frac{1}{\sqrt{2\pi K''(t)}} \exp\left(K(t) - tx\right)

        where :math:`t` solves the saddle point equation.

        Parameters
        ----------
        x : array_like, optional (either x or t must be provided)
            The values at which to evaluate the probability density function.
        t : array_like, optional (either x or t must be provided)
            Solution of the saddle point equation. If not provided, it will be
            computed using numerical root finding.
        normalize_pdf : bool, optional
            Whether to normalize the probability density function. Default is
            True.
        fillna : float, optional
            The value to replace NaNs with.
        """
        assert x is not None or t is not None
        if x is None:
            x = self.cgf.dK(t)
        elif t is None:
            t = self._dK_inv(x, **solver_kwargs)
        wrapper = PandasWrapper(x)
        y = np.asanyarray(self._spapprox_pdf(np.asanyarray(x), np.asanyarray(t)))
        if normalize_pdf:
            y *= 1 / self._pdf_normalization
        y = np.where(np.isnan(y), fillna, y)
        return y.tolist() if y.ndim == 0 else wrapper.wrap(y)

    def cdf(self, *args, x=None, t=None, fillna=np.nan, **solver_kwargs):
        raise NotImplementedError(f"CDF not implemented for {self.__class__.__name__}")

    def ppf(self, q, *args, fillna=np.nan, t0=None, ttol=1e-4, **kwargs):
        raise NotImplementedError(f"PPF not implemented for {self.__class__.__name__}")

    @type_wrapper(xloc=1)
    def _dK_inv(self, x, **solver_kwargs):
        return self.cgf.dK_inv(x, **solver_kwargs)

    @property
    @abstractmethod
    def _pdf_normalization(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self):
        raise NotImplementedError

    def clear_cache(self):
        """
        Clear any information stored from fit functions
        """
        cached_attribs = [
            k
            for k, v in inspect.getmembers(self, lambda a: not inspect.isroutine(a))
            if k.endswith("_cache")
        ]
        for attr in cached_attribs:
            if hasattr(self, attr):
                delattr(self, attr)


class UnivariateSaddlePointApprox(SaddlePointApprox):
    """
    Given the cumulant generating function of a univariate random variable, this class
    provides the saddle point approximation of the probability density function
    and the cumulative distribution function.

    Parameters
    ----------
    cgf : UnivariateCumulantGeneratingFunction
        The cumulant generating function of the random variable.
    pdf_normalization : float, optional
        The normalization constant of the probability density function. If not
        provided, it will be computed using numerical integration.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Saddlepoint_approximation_method

    [2] Butler, R. W. (2007). Saddlepoint approximations with applications.

    [3] Kuonen, D. (2001). Computer-intensive statistical methods: Saddlepoint approximations in bootstrap and inference.
    """

    def __init__(self, cgf, pdf_normalization=None):
        assert isinstance(cgf, UnivariateCumulantGeneratingFunction)
        super().__init__(cgf, pdf_normalization)

    @type_wrapper(xloc=1)
    def _spapprox_pdf(self, x, t, fillna=np.nan):
        t = np.asanyarray(t)
        d2Kt = self.cgf.d2K(t, fillna=fillna)
        with np.errstate(divide="ignore"):
            retval = np.where(
                ~np.isclose(d2Kt, 0) & ~np.isnan(d2Kt),
                np.exp(self.cgf.K(t) - t * x) * np.sqrt(np.divide(1, 2 * np.pi * d2Kt)),
                fillna,
            )
        return np.where(np.isnan(retval), fillna, retval)

    @type_wrapper(xloc=1)
    def _spapprox_cdf_LR(self, x, t, fillna=np.nan):
        r"""
        Lugannani-Rice saddle point approximation of the cumulative distribution as given by

        .. math::
            F(x) \approx \Phi(w) + \phi(w) \left(\frac{1}{w} - \frac{1}{u}\right)

        where :math:`\Phi` and :math:`\phi` are the cumulative and probability density
        of the standard normal distribution, respectively, and :math:`w` and :math:`u`
        are given by

        .. math::
            w = \text{sign}(t)\sqrt{2\left(tx - K(t)\right)},

        .. math::
            u = t\sqrt{K''(t)}.

        For :math:`t = 0`, the approximation is given by

        .. math::
            F(x) \approx \frac{1}{2} + \frac{K'''(0)}{6\sqrt{2\pi K''(0)^3}}.

        Reference
        ---------
        [1] Lugannani, R., & Rice, S. (1980). Saddle point approximation for the distribution of the sum of independent random variables. Advances in applied probability.

        [2] Kuonen, D. (2001). Computer-intensive statistical methods: Saddlepoint approximations in bootstrap and inference.
        """
        t = np.asanyarray(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.sign(t) * np.sqrt(2 * (t * x - self.cgf.K(t)))
            u = t * np.sqrt(self.cgf.d2K(t))
            retval = sps.norm.cdf(w) + sps.norm.pdf(w) * (1 / w - 1 / u)
        retval = np.where(
            ~np.isclose(t, 0),
            retval,
            1 / 2 + self.cgf.d3K0 / 6 / np.sqrt(2 * np.pi) / np.power(self.cgf.d2K0, 3 / 2),
        )
        return np.where(np.isnan(retval), fillna, retval)

    @type_wrapper(xloc=1)
    def _spapprox_cdf_BN(self, x, t, fillna=np.nan):
        r"""
        This is an alternative implementation of the Lugannani-Rice saddle point
        approximation of the cumulative distribution.

        It is given by

        .. math::
            F(x) \approx \Phi(w + \log(u/w)/w)

        where :math:`\Phi` is the cumulative distribution of the standard normal.

        References
        ----------
        [1] Barndorff-Nielsen, O. E. (1986). Inference on full or partial parameters based on the standardized signed log likelihood ratio. Biometrika.

        [2] Bandorff-Nielsen, O. E. (1990) Approximate interval probabilities. Journal of the Royal Statistical Society. Series B (Methodological).

        [3] Kuonen, D. (2001). Computer-intensive statistical methods: Saddlepoint approximations in bootstrap and inference.
        """
        t = np.asanyarray(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.sign(t) * np.sqrt(2 * (t * x - self.cgf.K(t)))
            u = t * np.sqrt(self.cgf.d2K(t))
            retval = sps.norm.cdf(w + np.log(u / w) / w)
        retval = np.where(
            ~np.isclose(t, 0),
            retval,
            1 / 2 + self.cgf.d3K0 / 6 / np.sqrt(2 * np.pi) / np.power(self.cgf.d2K0, 3 / 2),
        )
        return np.where(np.isnan(retval), fillna, retval)

    @property
    def dim(self):
        return 1

    @property
    def _pdf_normalization(self):
        if not hasattr(self, "_pdf_normalization_cache") or self._pdf_normalization_cache is None:
            a, b = self.infer_t_range()
            val = quad(
                lambda t: self.pdf(t=t, normalize_pdf=False, fillna=0) * self.cgf.d2K(t, fillna=0),
                a=a,
                b=b,
            )[0]
            assert not np.isnan(val) and np.isfinite(
                val
            ), "Failed to compute pdf normalization, value is equals NaN or Infinite"
            self._pdf_normalization_cache = val
        return self._pdf_normalization_cache

    def cdf(self, x=None, t=None, fillna=np.nan, backend="LR", **solver_kwargs):
        r"""
        Saddle point approximation of the cumulative distribution function.

        The standard Lugannani-Rice approximation is given by

        .. math::
            F(x) \approx \Phi(w) + \phi(w) \left(\frac{1}{w} - \frac{1}{u}\right)

        where :math:`\Phi` and :math:`\phi` are the cumulative and probability density
        of the standard normal distribution, respectively, and :math:`w` and :math:`u`
        are given by

        .. math::
            w = \text{sign}(t)\sqrt{2\left(tx - K(t)\right)},

        .. math::
            u = t\sqrt{K''(t)}.

        For :math:`t = 0`, the approximation is given by

        .. math::
            F(x) \approx \frac{1}{2} + \frac{K'''(0)}{6\sqrt{2\pi K''(0)^3}}.


        The alternative Barndorff-Nielsen approximation is given by

        .. math::
            F(x) \approx \Phi(w + \log(u/w)/w)

        where :math:`\Phi` is the cumulative distribution of the standard normal.

        Parameters
        ----------
        x : array_like, optional (either x or t must be provided)
            The values at which to evaluate the cumulative distribution function.
        t : array_like, optional (either x or t must be provided)
            Solution of the saddle point equation. If not provided, it will be
            computed using numerical root finding.
        fillna : float, optional
            The value to replace NaNs with.
        backend : str, optional
            The backend to use for the computation. Either 'LR' for Lugannani-Rice approximation
            or 'BN' for Barndorff-Nielsen approximation. Default is 'LR'.
        """
        assert x is not None or t is not None
        if x is None:
            x = self.cgf.dK(t)
        elif t is None:
            t = self._dK_inv(x, **solver_kwargs)
        wrapper = PandasWrapper(x)
        if backend == "LR":
            y = np.asanyarray(self._spapprox_cdf_LR(np.asanyarray(x), np.asanyarray(t)))
        elif backend == "BN":
            y = np.asanyarray(self._spapprox_cdf_BN(np.asanyarray(x), np.asanyarray(t)))
        else:
            raise ValueError("backend must be either 'LR' or 'BN'")
        y = np.where(np.isnan(y), fillna, y)
        return y.tolist() if y.ndim == 0 else wrapper.wrap(y)

    @type_wrapper(xloc=1)
    def ppf(self, q, fillna=np.nan, t0=None, ttol=1e-4, **kwargs):
        r"""
        Percent point function computed as the inverse of the saddle point
        approximation of the cumulative distribution function.

        When available, the ppf interpolated. The method is based on
        a custom implementation. Probably we could have also used the
        Scipy's sampling pinv method.

        Parameters
        ----------
        q : array_like
            The values at which to evaluate the inverse cumulative distribution function.
        fillna : float, optional
            The value to replace NaNs with.
        """
        assert np.all((0 <= np.asanyarray(q)) & (np.asanyarray(q) <= 1))
        if hasattr(self, "_cdf_cache") and hasattr(self, "_t_cache"):
            t = np.interp(q, self._cdf_cache, self._t_cache)
        else:
            if q.ndim == 0:  # Then it is a scalar "array"
                q = q.tolist()
                kwargs["x0"] = 0 if t0 is None else t0
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
                            if ~np.isnan(self.cgf.dK(-1 * 0.9**i))
                        )
                        ub = next(
                            1 * 0.9**i for i in range(10000) if ~np.isnan(self.cgf.dK(1 * 0.9**i))
                        )
                        cdf_lb = self.cdf(x=self.cgf.dK(lb), t=lb, fillna=np.nan)
                        cdf_ub = self.cdf(x=self.cgf.dK(ub), t=ub, fillna=np.nan)
                        assert lb < ub and cdf_lb < cdf_ub, "cdf is assumed to be increasing"
                        lb_scalings = (1 - 1 / fib(i) for i in range(3, 100))
                        ub_scalings = (1 - 1 / fib(i) for i in range(3, 100))
                        lb_scaling = next(lb_scalings)
                        ub_scaling = next(ub_scalings)
                        # find lb through iterative scaling
                        while q < cdf_lb:
                            lb_new = lb / lb_scaling
                            x_new = self.cgf.dK(lb_new)
                            if not np.isnan(x_new):
                                lb = lb_new
                                cdf_lb = self.cdf(x=x_new, t=lb, fillna=np.nan)
                                continue
                            try:
                                lb_scaling = next(lb_scalings)
                            except StopIteration:
                                raise Exception("Could not find valid lb")
                        # find ub through iterative scaling
                        while q > cdf_ub:
                            ub_new = ub / ub_scaling
                            x_new = self.cgf.dK(ub_new)
                            if not np.isnan(x_new):
                                ub = ub_new
                                cdf_ub = self.cdf(x=x_new, t=ub, fillna=np.nan)
                                continue
                            try:
                                ub_scaling = next(ub_scalings)
                            except StopIteration:
                                raise Exception("Could not find valid ub")
                        assert cdf_lb <= q <= cdf_ub
                        kwargs["bracket"] = [lb, ub]
                    try:
                        res = spo.root_scalar(lambda t: self.cdf(t=t) - q, **kwargs)
                    except Exception:
                        continue
                    if res.converged and np.isclose(self.cdf(t=res.root), q, atol=ttol):
                        break
                else:
                    raise Exception("Failed to solve the saddle point equation.")
                t = np.asanyarray(res.root)
            else:
                kwargs["x0"] = np.zeros(q.shape) if t0 is None else np.asanayarray(t0)
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
                    with np.errstate(invalid="ignore"):
                        try:
                            res = spo.root(lambda t, q=q: self.cdf(t=t) - q, **kwargs)
                        except Exception:
                            continue
                    if res.success and np.allclose(self.cdf(t=res.x), q, atol=ttol):
                        t = np.asanyarray(res.x)
                        break
                else:
                    t = np.asanyarray(
                        [self.ppf(qq, t0=None if t0 is None else t0[i]) for i, qq in enumerate(q)]
                    )
        return self.cgf.dK(t)

    def infer_t_range(self, atol=1e-4, rtol=1e-4):
        """
        Infers a suitable range for so that it covers roughly the entire probability mass
        """
        # Determine initial range
        lb = next(-1 * 0.9**i for i in range(10000) if ~np.isnan(self.cgf.dK(-1 * 0.9**i)))
        ub = next(1 * 0.9**i for i in range(10000) if ~np.isnan(self.cgf.dK(1 * 0.9**i)))
        cdf_lb = self.cdf(x=self.cgf.dK(lb), t=lb, fillna=np.nan)
        cdf_ub = self.cdf(x=self.cgf.dK(ub), t=ub, fillna=np.nan)
        assert lb < ub and cdf_lb < cdf_ub, "cdf is assumed to be increasing"
        # Define scaling factors
        lb_scalings = (1 - 1 / fib(i) for i in range(3, 100))
        ub_scalings = (1 - 1 / fib(i) for i in range(3, 100))
        lb_scaling = next(lb_scalings)
        ub_scaling = next(ub_scalings)
        # find lb through iterative scaling
        while not np.isclose(cdf_lb, 0, atol=atol, rtol=rtol):
            lb_new = lb / lb_scaling
            x_new = self.cgf.dK(lb_new)
            if not np.isnan(x_new):
                lb = lb_new
                cdf_lb = self.cdf(x=x_new, t=lb, fillna=np.nan)
                continue
            try:
                lb_scaling = next(lb_scalings)
            except StopIteration:
                raise Exception("Could not find valid lb")
        # find ub through iterative scaling
        while not np.isclose(cdf_ub, 1, atol=atol, rtol=rtol):
            ub_new = ub / ub_scaling
            x_new = self.cgf.dK(ub_new)
            if not np.isnan(x_new):
                ub = ub_new
                cdf_ub = self.cdf(x=x_new, t=ub, fillna=np.nan)
                continue
            try:
                ub_scaling = next(ub_scalings)
            except StopIteration:
                raise Exception("Could not find valid ub")
        assert lb <= 0 <= ub
        return [lb, ub]

    def fit_saddle_point_eqn(self, t_range=None, atol=1e-4, rtol=1e-4, num=1000, **solver_kwargs):
        """
        Evaluate the saddle point equation to the given range of values.
        And use linear interpolation to solve the saddle point equation, form now onwards.

        If t_range is not provided, it will be assumed that the cumulant generating
        function is defined in a neighborhood of zero. The range will be computed
        by scaling iteratively, until we little mass for higher or lower values of x.
        """
        if t_range is None:
            if hasattr(self, "_t_cache"):
                t_range = [self._t_cache[0], self._t_cache[-1]]
            else:
                t_range = self.infer_t_range(atol=atol, rtol=rtol)
        # Solve saddle point equation
        self._x_cache = np.linspace(*self.cgf.dK(t_range), num=num)
        self._t_cache = self.cgf.dK_inv(self._x_cache, **solver_kwargs)

    def fit_ppf(self, t_range=None, atol=1e-4, rtol=1e-4, num=1000):
        """
        Fit the inverse of the cumulative distribution function using linear interpolation.
        """
        if not hasattr(self, "_x_cache"):
            if hasattr(self, "_t_cache"):
                t_range = [self._t_cache[0], self._t_cache[-1]]
            else:
                t_range = self.infer_t_range(atol=atol, rtol=rtol)
            x_range = np.linspace(*self.cgf.dK(t_range), num=num)
        self._cdf_cache = self.cdf(x=x_range)
        self._t_cache = self.cgf.dK_inv(x_range)

    @type_wrapper(xloc=1)
    def _dK_inv(self, x, **solver_kwargs):
        if hasattr(self, "_x_cache") and hasattr(self, "_t_cache"):
            return np.interp(x, self._x_cache, self._t_cache)
        else:
            return super()._dK_inv(x, **solver_kwargs)


class UnivariateSaddlePointApproxMean(UnivariateSaddlePointApprox):
    r"""
    Given the cumulant generating function of a univariate random variable, this class
    provides the saddle point approximation of the sample mean of the random variable.

    Given :math:`n` i.i.d. random variables :math:`X_1, \ldots, X_n` with
    cumulant generating function :math:`K`, the cumulant generating function :math:`K_n`
    of the sample mean :math:`\bar{X}` is given by

    .. math::
        K_{\bar{X}}(t) = \sum_{i=1}^n 1/n*K_i(t)= \sum_{i=1}^n K_i(t/n) = n K(t/n).

    Parameters
    ----------
    cgf : UnivariateCumulantGeneratingFunction
        The cumulant generating function of the random variable.
    sample_size : int
        The sample size on which the sample mean is to be estimated.
    pdf_normalization : float, optional
        The normalization constant of the probability density function. If not
        provided, it will be computed using numerical integration.
    """

    def __init__(self, cgf, sample_size, pdf_normalization=None):
        assert isinstance(cgf, UnivariateCumulantGeneratingFunction)
        self.sample_size = sample_size
        cgf = univariate_sample_mean(cgf, sample_size)
        super().__init__(cgf, pdf_normalization=pdf_normalization)


class MultivariateSaddlePointApprox(SaddlePointApprox):
    r"""
    Given the cumulant generating function of a random vector, this class
    provides the saddle point approximation of the probability density function.

    The multivariate saddle point approximation is given by

    .. math::
        f(\mathbf{x}) \approx \frac{1}{2\pi\sqrt{\left|\det H\right|}}
        \exp\left(K(\mathbf{t}) - \mathbf{x}\cdot\mathbf{t}\right)


    where :math:`H` is the Hessian matrix of the cumulant generating function,
    and :math:`\mathbf{t}` solves the saddle point equation :math:`K(\mathbf{t})=x`.

    We follow the approach as outlined in [4] Section 5.2.


    See the Kolasse book on how to remove singularities
    DasGupta claims the multidimensional case is too notationally complex

    References
    ----------
    [1] Kolassa (2006) - Series approximations in statistics
    [2] DasGupta (2008) - Asymptotic theory of statistics and probability, Chapter 14.9
    [3] Reid (1988) - Saddlepoint methods and statistical inference
    [4] Gatto (2000) - Symbolic computation for approximating distributions of some families of one and two-sample nonparametric test statistics
    [5] Gatto (2019) - Saddlepoint approximations for data in simplices
    """

    def __init__(self, cgf, pdf_normalization=None):
        assert isinstance(cgf, MultivariateCumulantGeneratingFunction)
        self.cgf = cgf
        self._pdf_normalization_cache = pdf_normalization

    def __getitem__(self, i):
        raise NotImplementedError("Return the marginal approximation")

    def condition(self, i, x):
        raise NotImplementedError("Condition the approximation on the ith variable")

    @type_wrapper(xloc=1)
    def _spapprox_pdf(self, x, t, fillna=np.nan):
        t = np.asanyarray(t)
        detd2Kt = np.linalg.det(self.cgf.d2K(t, fillna=fillna))
        with np.errstate(divide="ignore"):
            retval = np.where(
                ~np.isclose(detd2Kt, 0) & ~np.isnan(detd2Kt),
                np.exp(self.cgf.K(t) - np.multiply(t, x).sum(axis=-1))
                / np.power(2 * np.pi, self.dim / 2)
                / np.sqrt(detd2Kt),
                fillna,
            )
        return np.where(np.isnan(retval), fillna, retval)

    @property
    def _pdf_normalization(self):
        if not hasattr(self, "_pdf_normalization_cache") or self._pdf_normalization_cache is None:
            tranges = self.infer_t_ranges()
            val = nquad(
                lambda *args: self.pdf(t=args[: self.dim], normalize_pdf=False, fillna=0)
                * np.linalg.det(self.cgf.d2K(args[: self.dim], fillna=0)),
                tranges,
            )[0]
            assert not np.isnan(val) and np.isfinite(
                val
            ), "Failed to compute pdf normalization, value is equals NaN or Infinite"
            self._pdf_normalization_cache = val
        return self._pdf_normalization_cache

    @property
    def dim(self):
        return self.cgf.dim

    def fit_saddle_point_eqn(self, t_ranges=None, atol=1e-4, rtol=1e-4, num=1000, **solver_kwargs):
        """
        Evaluate the saddle point equation to the given range of values.
        And use linear interpolation to solve the saddle point equation, form now onwards.

        If t_range is not provided, it will be assumed that the cumulant generating
        function is defined in a neighborhood of zero. The range will be computed
        by scaling iteratively, until we little mass for higher or lower values of x.
        """
        if t_ranges is None:
            if hasattr(self, "_t_cache"):
                t_ranges = np.vstack(
                    (self._t_cache.min(axis=0).min(axis=0), self._t_cache.max(axis=0).max(axis=0))
                ).T
            else:
                t_ranges = self.infer_t_ranges(atol=atol, rtol=rtol)
        x_ranges = self.cgf.dK(list(itertools.product(*t_ranges)))
        x_ranges = np.vstack((x_ranges.min(axis=0), x_ranges.max(axis=0))).T
        self._x_cache = [np.linspace(*xr, num=num) for xr in x_ranges]
        xir = np.vstack([xi.ravel() for xi in np.meshgrid(*self._x_cache, indexing="ij")]).T
        self._t_cache = self.cgf.dK_inv(xir, **solver_kwargs)
        self._t_cache = self._t_cache.reshape(*[num] * self.dim, self.dim)
        self._interp_cache = RegularGridInterpolator(self._x_cache, self._t_cache)

    @type_wrapper(xloc=1)
    def _dK_inv(self, x, **solver_kwargs):
        if hasattr(self, "_interp_cache"):
            return self._interp_cache(x)
        else:
            return super()._dK_inv(x, **solver_kwargs)

    def infer_t_ranges(self, atol=1e-4, rtol=1e-4):
        """
        Infers suitable ranges for :math:`t` so that it covers, roughly the entire probability mass.
        The ranges are inferred from the univariat case.
        """
        return [
            UnivariateSaddlePointApprox(self.cgf[i]).infer_t_range(atol=atol, rtol=rtol)
            for i in range(self.dim)
        ]

    def cdf(self, *args, x=None, t=None, fillna=np.nan, **solver_kwargs):
        # TODO: make a basic implementation with numerical integration
        raise NotImplementedError(f"CDF not implemented for {self.__class__.__name__}")

    def fit_cdf(self, t_range=None, atol=1e-4, rtol=1e-4, num=1000):
        """
        Fit the cumulative distribution function using linear interpolation.
        """
        raise NotImplementedError


class BivariateSaddlePointApprox(MultivariateSaddlePointApprox):
    r"""
    Given the cumulant generating function of a bivariate random variable, this class
    provides the saddle point approximation of the probability density function.

    The bivariate saddle point approximation is given by

    .. math::
        f(\mathbf{x}) \approx \frac{1}{2\pi\sqrt{\left|\det H\right|}}
        \exp\left(K(\mathbf{\tilde t}) - \mathbf{x}\cdot\mathbf{\tilde t}\right)

    where :math:`\mathbf{x} = \left[x,y\right]`,
    :math:`H` is the Hessian matrix of the cumulant generating function,
    and :math:`\mathbf{\tilde t}= \left[\tilde s,\tilde t\right]` solves
    the saddle point equation :math:`K(\mathbf{\tilde t})=x`.

    We follow the approach as outlined in [4] Section 5.2.

    References
    ----------
    [1] Broda, Paolella (2011) - Saddlepoint approximations - a review
    [2] Butler, R. W. (2007). Saddlepoint approximations with applications.
    [3] Wang (1990) - Saddlepoint approximations for bivariate distributions
    [4] Paolella (2007) - Intermediate Probability

    """

    def cdf(self, x=None, t=None, fillna=np.nan, **solver_kwargs):
        r"""
        Saddle point approximation of the cumulative distribution function in
        the bivariate case.
        """
        assert x is not None or t is not None
        if x is None:
            x = self.cgf.dK(t)
        elif t is None:
            t = self._dK_inv(x, **solver_kwargs)
        wrapper = PandasWrapper(x)
        x, t = np.asanyarray(x), np.asanyarray(t)
        y = self._spapprox_cdf(x, t)
        y = np.where(np.isnan(y), fillna, y)
        return y.tolist() if y.ndim == 0 else wrapper.wrap(y)

    def _spapprox_cdf(self, x, t, fillna=np.nan, **solver_kwargs):
        r"""
        Saddle point approximation of the cumulative distribution function in
        the bivariate case.

        The approximation is given by

        .. math::
            F(\mathbf{x}) \approx \Phi_2(\mathbf{\tilde x}, \rho)
            + \Phi(\tilde w) n + \Phi(w) \tilde n + n \tilde n,
        
        where vectors are denoted in bold and have components as follows:

        .. math::
            \mathbf{x} = \left[x,y\right],\\
            \mathbf{t} = \left[s, t\right],\\
            \mathbf{\tilde x} = \left[\tilde x,\tilde y\right],\\
            \mathbf{\tilde t_0} = \left[0,\tilde t\right],\\
            \mathbf{s_0} = \left[s,0\right],\\
            \mathbf{t_0} = \left[0,t\right].
            
        And where,
            
        .. math::
            \tilde x = \text{sign}(\tilde t) \sqrt{2(\mathbf{\tilde t_0}\cdot \mathbf{x}
                                                         - K(\mathbf{\tilde t_0}))},\\
            \tilde w = \text{sign}(t) \sqrt{2\left(K(\mathbf{s_0}) - K(\mathbf{t}) + 
                                                     \mathbf{t_0}\cdot\mathbf{x}\right)},\\
            w = \text{sign}(s) \sqrt{2}
                       \sqrt{\mathbf{t}\cdot\mathbf{x} 
                        - \mathbf{\tilde t_0}\cdot\mathbf{x}
                        + K(\mathbf{\tilde t_0}) - K(\mathbf{t})},\\
            b = \frac{\tilde w - \tilde x}{w},\\
            \tilde y = \frac{w - b \tilde x}{\sqrt{1+b^2}},\\
            \rho = \frac{-b}{\sqrt{1+b^2}},\\
            n = \phi(w)\left(\frac{1}{w}-\frac{1}{u}\right),\\
            \tilde n = \phi(\tilde x)\left(\frac{1}{\tilde w}-\frac{1}{\tilde u}\right),\\
            u = s \sqrt{\frac{\text{det} K''(\mathbf{t})}{K''_{tt}(\mathbf{t})}},\\
            \tilde u = t \sqrt{K''_{tt}(\mathbf{t})}

        and :math:`\mathbf{t}=[s,t]` is found by solving the saddle point equation

        .. math::
           \nabla K(\mathbf{t}) = \mathbf{x},

        And, :math:`\mathbf{\tilde t_0}=[0,\tilde t]` is found by solving a second saddlepoint equation:

        .. math::
            \partial_{t} K(\mathbf{\tilde t_0}) = y.
            
        We follow the approach as outlined in [4] Section 5.2.
        Note that, we simplified notation somewhat, and that there is typo in their
        formula for :math:`\tilde n_0` (in their notation): :math:`w_0` is not definiened
        and should probably read :math:`\tilde w_0`.
            
        Parameters
        ----------
        x : array_like, optional (either x or t must be provided)
            The values at which to evaluate the cumulative distribution function.
        t : array_like, optional (either x or t must be provided)
            Solution of the saddle point equation. If not provided, it will be
            computed using numerical root finding.
        fillna : float, optional
            The value to replace NaNs with.
        """
        # Note, slicing a component of a cgf sets the other variables to zero
        tt = self.cgf[1].dK_inv(x[1], **solver_kwargs)
        tt0 = np.hstack((np.zeros(np.shape(tt)), tt))
        t0 = t.copy()
        t0[0] = 0
        s0 = t.copy()
        s0[1] = 0
        tx = np.sign(tt) * np.sqrt(2 * (tt0.dot(x) - self.cgf.K(tt0)))
        tw = np.sign(t[0]) * np.sqrt(2 * (self.cgf.K(s0) - self.cgf.K(t) + t0.dot(x)))
        w = np.sign(t[1]) * np.sqrt(2 * ((t - tt0).dot(x) + self.cgf.K(tt0) - self.cgf.K(t)))
        b = (tw - tx) / w
        ty = (w - b * tx) / np.sqrt(1 + np.square(b))
        tx = np.hstack((tx, ty))
        rho = -b / np.sqrt(1 + np.square(b))
        u = t[0] * np.sqrt(np.linalg.det(self.cgf.d2K(t)) / self.cgf.d2K(t)[1, 1])
        tu = t[1] * np.sqrt(self.cgf.d2K(t)[1, 1])
        n = sps.norm.pdf(w) * (1 / w - 1 / u)
        tn = sps.norm.pdf(tx[0]) * (1 / tw - 1 / tu)
        if _has_fastnorm:
            retval = fastnorm.bivar_norm_cdf(tx, rho)
        else:
            retval = sps.multivariate_normal([0, 0], np.array([[1, rho], [rho, 1]])).cdf(tx)
        retval += sps.norm.cdf(tw) * tn + sps.norm.cdf(w) * n + n * tn
        return retval

    @type_wrapper(xloc=1)
    def ppf(self, q, fillna=np.nan, t0=None, ttol=1e-4, **kwargs):
        r"""
        Percent point function computed as the inverse of the saddle point
        approximation of the cumulative distribution function.

        Parameters
        ----------
        q : array_like
            The values at which to evaluate the inverse cumulative
            distribution function.
        fillna : float, optional
            The value to replace NaNs with.
        """
        raise NotImplementedError("Copy dKinv logic?")
        assert np.all((0 <= np.asanyarray(q)) & (np.asanyarray(q) <= 1))
        if hasattr(self, "_cdf_cache") and hasattr(self, "_t_cache"):
            t = np.interp(q, self._cdf_cache, self._t_cache)
        else:
            if q.ndim == 0:  # Then it is a scalar "array"
                q = q.tolist()
                kwargs["x0"] = 0 if t0 is None else t0
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
                            if ~np.isnan(self.cgf.dK(-1 * 0.9**i))
                        )
                        ub = next(
                            1 * 0.9**i for i in range(10000) if ~np.isnan(self.cgf.dK(1 * 0.9**i))
                        )
                        cdf_lb = self.cdf(x=self.cgf.dK(lb), t=lb, fillna=np.nan)
                        cdf_ub = self.cdf(x=self.cgf.dK(ub), t=ub, fillna=np.nan)
                        assert lb < ub and cdf_lb < cdf_ub, "cdf is assumed to be increasing"
                        lb_scalings = (1 - 1 / fib(i) for i in range(3, 100))
                        ub_scalings = (1 - 1 / fib(i) for i in range(3, 100))
                        lb_scaling = next(lb_scalings)
                        ub_scaling = next(ub_scalings)
                        # find lb through iterative scaling
                        while q < cdf_lb:
                            lb_new = lb / lb_scaling
                            x_new = self.cgf.dK(lb_new)
                            if not np.isnan(x_new):
                                lb = lb_new
                                cdf_lb = self.cdf(x=x_new, t=lb, fillna=np.nan)
                                continue
                            try:
                                lb_scaling = next(lb_scalings)
                            except StopIteration:
                                raise Exception("Could not find valid lb")
                        # find ub through iterative scaling
                        while q > cdf_ub:
                            ub_new = ub / ub_scaling
                            x_new = self.cgf.dK(ub_new)
                            if not np.isnan(x_new):
                                ub = ub_new
                                cdf_ub = self.cdf(x=x_new, t=ub, fillna=np.nan)
                                continue
                            try:
                                ub_scaling = next(ub_scalings)
                            except StopIteration:
                                raise Exception("Could not find valid ub")
                        assert cdf_lb <= q <= cdf_ub
                        kwargs["bracket"] = [lb, ub]
                    try:
                        res = spo.root_scalar(lambda t: self.cdf(t=t) - q, **kwargs)
                    except Exception:
                        continue
                    if res.converged and np.isclose(self.cdf(t=res.root), q, atol=ttol):
                        break
                else:
                    raise Exception("Failed to solve the saddle point equation.")
                t = np.asanyarray(res.root)
            else:
                kwargs["x0"] = np.zeros(q.shape) if t0 is None else np.asanayarray(t0)
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
                    with np.errstate(invalid="ignore"):
                        try:
                            res = spo.root(lambda t, q=q: self.cdf(t=t) - q, **kwargs)
                        except Exception:
                            continue
                    if res.success and np.allclose(self.cdf(t=res.x), q, atol=ttol):
                        t = np.asanyarray(res.x)
                        break
                else:
                    t = np.asanyarray(
                        [self.ppf(qq, t0=None if t0 is None else t0[i]) for i, qq in enumerate(q)]
                    )
        return self.cgf.dK(t)

    def fit_ppf(self, t_range=None, atol=1e-4, rtol=1e-4, num=1000):
        """
        Fit the inverse of the cumulative distribution function using linear interpolation.
        """
        raise NotImplementedError
        if not hasattr(self, "_x_cache"):
            if hasattr(self, "_t_cache"):
                t_range = [self._t_cache[0], self._t_cache[-1]]
            else:
                t_range = self.infer_t_range(atol=atol, rtol=rtol)
            x_range = np.linspace(*self.cgf.dK(t_range), num=num)
        self._cdf_cache = self.cdf(x=x_range)
        self._t_cache = self.cgf.dK_inv(x_range)

    def _dK_inv(self, x, **solver_kwargs):
        # TODO: implement multivariate interpolation with radial basis functions or something like that
        return super()._dK_inv(x, **solver_kwargs)


# TODO: implement Dirichlet bootstrap
# TODO: implement multivariate saddle point approximations
# TODO: implement other bootstraps
# TODO: jacknife stuff
# TODO: conditional distributions
# TODO: approximation for discrete distributions
