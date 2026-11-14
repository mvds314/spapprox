import inspect
import itertools
import warnings
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

    # Half-width, in standardized units t * sqrt(K''(0)), of the window around t=0
    # in which cdf approximations are interpolated rather than evaluated directly.
    # The direct formulas suffer from catastrophic cancellation as t -> 0.
    _cdf_singular_width = 1e-2

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
        rho3 = self.cgf.d3K0 / np.power(self.cgf.d2K0, 3 / 2)
        retval = self._interpolate_cdf_near_zero(
            self._lugannani_rice, x, t, 1 / 2 + rho3 / 6 / np.sqrt(2 * np.pi)
        )
        return np.where(np.isnan(retval), fillna, retval)

    def _lugannani_rice(self, x, t):
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.sign(t) * np.sqrt(2 * (t * x - self.cgf.K(t)))
            u = t * np.sqrt(self.cgf.d2K(t))
            return sps.norm.cdf(w) + sps.norm.pdf(w) * (1 / w - 1 / u)

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
        rho3 = self.cgf.d3K0 / np.power(self.cgf.d2K0, 3 / 2)
        retval = self._interpolate_cdf_near_zero(
            self._barndorff_nielsen, x, t, sps.norm.cdf(rho3 / 6)
        )
        return np.where(np.isnan(retval), fillna, retval)

    def _barndorff_nielsen(self, x, t):
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.sign(t) * np.sqrt(2 * (t * x - self.cgf.K(t)))
            u = t * np.sqrt(self.cgf.d2K(t))
            return sps.norm.cdf(w + np.log(u / w) / w)

    def _interpolate_cdf_near_zero(self, formula, x, t, limit):
        r"""
        Evaluate ``formula(x, t)``, but replace values with :math:`|t|<h` by quadratic
        interpolation through :math:`(-h, F(-h))`, :math:`(0, \text{limit})` and
        :math:`(h, F(h))`, where :math:`F` is the formula as a function of :math:`t`
        and :math:`h` is ``_cdf_singular_width`` in standardized units.
        """
        x, t = np.asanyarray(x), np.asanyarray(t)
        retval = formula(x, t)
        h = self._cdf_singular_width / np.sqrt(self.cgf.d2K0)
        near = np.abs(t) < h
        if np.any(near):
            nodes = np.array([-h, h])
            fm, fp = formula(self.cgf.dK(nodes), nodes)
            interp = (
                limit + (fp - fm) / (2 * h) * t + (fp - 2 * limit + fm) / (2 * h**2) * np.square(t)
            )
            retval = np.where(near, interp, retval)
        return retval

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
            assert not np.isnan(val) and np.isfinite(val), ()
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
        For :math:`t = 0`, it is given by its limit

        .. math::
            F(x) \approx \Phi\left(\frac{K'''(0)}{6 K''(0)^{3/2}}\right).

        Close to :math:`t=0` both approximations suffer from catastrophic cancellation,
        so they are quadratically interpolated between the limit and the values at
        :math:`t=\pm h`, with :math:`h` given by ``_cdf_singular_width`` in
        standardized units :math:`t\sqrt{K''(0)}`.

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
        if hasattr(self, "_cdf_cache") and hasattr(self, "_ppf_t_cache"):
            t = np.interp(q, self._cdf_cache, self._ppf_t_cache)
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
                kwargs["x0"] = np.zeros(q.shape) if t0 is None else np.asanyarray(t0)
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

    def infer_t_range(self, atol=1e-4, rtol=1e-4, max_iter=500):
        """
        Infers a suitable range for so that it covers roughly the entire probability mass

        The bounds are expanded until the cdf is close to zero (lower bound) or one (upper
        bound). For lattice or bounded-support distributions the saddle point cdf may never
        get there: it can turn around, leave :math:`[0, 1]`, or become ``nan``. Such a step is
        treated as the end of the usable range, and the last valid bound is returned with a
        warning.
        """
        # Determine initial range
        lb = next(-1 * 0.9**i for i in range(10000) if ~np.isnan(self.cgf.dK(-1 * 0.9**i)))
        ub = next(1 * 0.9**i for i in range(10000) if ~np.isnan(self.cgf.dK(1 * 0.9**i)))
        cdf_lb = self.cdf(x=self.cgf.dK(lb), t=lb, fillna=np.nan)
        cdf_ub = self.cdf(x=self.cgf.dK(ub), t=ub, fillna=np.nan)
        if not (lb < ub and 0 <= cdf_lb < cdf_ub <= 1):
            raise ValueError(
                "Could not infer a t range: the cdf is not valid and increasing at the initial "
                f"bounds t={lb}, t={ub} (cdf {cdf_lb}, {cdf_ub})"
            )

        def expand(bound, cdf_bound, target):
            # Expand bound (away from zero) until the cdf is close to target; a step is only
            # accepted if the cdf stays valid and moves towards target
            scalings = (1 - 1 / fib(i) for i in range(3, 100))
            scaling = next(scalings)
            for _ in range(max_iter):
                if np.isclose(cdf_bound, target, atol=atol, rtol=rtol):
                    return bound
                new = bound / scaling
                x_new = self.cgf.dK(new)
                if np.isfinite(x_new):
                    cdf_new = self.cdf(x=x_new, t=new, fillna=np.nan)
                    if 0 <= cdf_new <= 1 and abs(cdf_new - target) < abs(cdf_bound - target):
                        bound, cdf_bound = new, cdf_new
                        continue
                try:
                    scaling = next(scalings)
                except StopIteration:
                    break
            warnings.warn(
                f"Could not expand t range until cdf={target}, stopped at t={bound} with "
                f"cdf={cdf_bound}; the saddle point cdf may not reach {target}, e.g., for "
                "lattice or bounded-support distributions",
                RuntimeWarning,
                stacklevel=3,
            )
            return bound

        lb = expand(lb, cdf_lb, 0)
        ub = expand(ub, cdf_ub, 1)
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

        If t_range is not provided, it is taken from a previous fit (of the ppf or the
        saddle point equation), or inferred otherwise.
        """
        if t_range is None:
            if hasattr(self, "_ppf_t_cache"):
                t_range = [self._ppf_t_cache[0], self._ppf_t_cache[-1]]
            elif hasattr(self, "_t_cache"):
                t_range = [self._t_cache[0], self._t_cache[-1]]
            else:
                t_range = self.infer_t_range(atol=atol, rtol=rtol)
        x_range = np.linspace(*self.cgf.dK(t_range), num=num)
        t = self.cgf.dK_inv(x_range)
        self._cdf_cache = self.cdf(x=x_range, t=t)
        self._ppf_t_cache = t

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
                lambda *args: (
                    self.pdf(t=args[: self.dim], normalize_pdf=False, fillna=0)
                    * np.linalg.det(self.cgf.d2K(args[: self.dim], fillna=0))
                ),
                tranges,
            )[0]
            assert not np.isnan(val) and np.isfinite(val), (
                "Failed to compute pdf normalization, value is equals NaN or Infinite"
            )
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

    [5] Kolassa (2006) - Serie approximations in statistics

    [6] Kolassa, Li (2011) - Series approximations to multivariate saddlepoint approximations

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
        y = self._spapprox_cdf(x, t, **solver_kwargs)
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

        And, :math:`\mathbf{\tilde t_0}=[0,\tilde t]` is found by solving a second saddlepoint
        equation:

        .. math::
            \partial_{t} K(\mathbf{\tilde t_0}) = y.

        We follow the approach as outlined in [4] Section 5.2.
        Note that, we simplified notation somewhat, and that there is typo in their
        formula for :math:`\tilde n_0` (in their notation): :math:`w_0` is not defined
        and should probably read :math:`\tilde w_0`.

        There are several cases that lead to singularities.
        First, following [4], if :math:`t=0` and :math:`s\neq0`, then :math:`\tilde n` should be replaced
        by its limiting value

        .. math::
            \tilde n = \frac{\phi(\tilde x)}{6}\frac{K_{ttt}(\mathbf{s_0})}{K_{tt}(\mathbf{s_0})^{3/2}}.

        Second, following [4], if :math:`s=0` and :math:`t\neq0`, then we can reverse the role of :math:`X`
        and :math:`Y`, to reduce it to the first case. Basically, in the notation above, we swap the roles of
        :math:`s` and :math:`t`, :math:`\tilde s` and :math:`\tilde t`, :math:`x` and :math:`y`,
        :math:`\tilde x` and :math:`\tilde y`, and :math:`\mathbf{s_0}` and :math:`\mathbf{t_0}`.
        This leads to

        .. math::
            \tilde y = \text{sign}(\tilde s) \sqrt{2(\mathbf{\tilde s_0}\cdot \mathbf{x}
                                                         - K(\mathbf{\tilde s_0}))},\\
            \tilde w = \text{sign}(s) \sqrt{2\left(K(\mathbf{t_0}) - K(\mathbf{t}) +
                                                     \mathbf{s_0}\cdot\mathbf{x}\right)},\\
            w = \text{sign}(t) \sqrt{2}
                       \sqrt{\mathbf{t}\cdot\mathbf{x}
                        - \mathbf{\tilde s_0}\cdot\mathbf{x}
                        + K(\mathbf{\tilde s_0}) - K(\mathbf{t})},\\
            \tilde x = \frac{w - b \tilde y}{\sqrt{1+b^2}},\\
            u = t \sqrt{\frac{\text{det} K''(\mathbf{t})}{K''_{ss}(\mathbf{t})}},\\
            \tilde u = s \sqrt{K''_{ss}(\mathbf{t})},

        And, where :math:`b`, :math:`\rho`, :math:`n`, and :math:`\tilde n` are as above
        Additionally, we introduce :math:`\mathbf{\tilde s_0}=[\tilde s,0]` as the solution of the alternative
        second saddepoint equation:

        .. math::
            \partial_{s} K(\mathbf{\tilde s_0}) = x.

        The limits above are only used when a component of :math:`\mathbf{t}` is exactly
        zero. Close to zero, the formulas suffer from catastrophic cancellation in the
        terms :math:`1/w - 1/u` and :math:`1/\tilde w - 1/\tilde u`; the roundoff error
        grows roughly like :math:`|t_j|^{-3}`. Therefore, the following is done, which goes
        beyond [3] and [4]. Let

        .. math::
            h_j = \frac{\delta}{\sqrt{K''_{jj}(\mathbf{0})}}, \quad j=1,2,

        where :math:`\delta` is ``_cdf_singular_width`` (0.01 by default), i.e., the
        width is fixed in standardized units. For every component with
        :math:`|t_j| < h_j`, the approximation is replaced by quadratic Lagrange
        interpolation in that component through the nodes :math:`t_j \in \{-h_j, 0, h_j\}`.
        If both components are near zero, a tensor product of these interpolations is
        used, so that 9 nodes are evaluated. At the nodes, the formula is evaluated
        directly, using the exact limits above for nodes on an axis. The interpolation
        error is :math:`O(h^3)`.

        At the double singularity :math:`\mathbf{t}=\mathbf{0}` no closed-form limit is
        available. Instead, let :math:`A(h)` denote the average of the approximation at
        the four points :math:`(\pm h_1, 0)` and :math:`(0, \pm h_2)`, which are evaluated
        using the exact one-component limits. By symmetry, the odd-order terms cancel, so
        that :math:`A(h) = F(\mathbf{0}) + c h^2 + O(h^4)`. The value at the origin is then
        obtained by Richardson extrapolation:

        .. math::
            F(\mathbf{0}) \approx \frac{4 A(h) - A(2h)}{3},

        which has error :math:`O(h^4)`.

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
        # Handle vectorized evaluation
        if t.ndim > 1:
            return np.asanyarray(
                [self._spapprox_cdf(xx, tt, **solver_kwargs) for xx, tt in zip(x, t)]
            )
        if t.ndim != 1 or x.ndim != 1 or len(t) != 2 or len(x) != 2:
            raise ValueError("2 dimension vector expected in bivariate case")
        h = self._cdf_singular_width / np.sqrt(np.diag(self.cgf.d2K0))
        near = np.abs(t) < h
        if not near.any():
            return self._bivariate_formula(x, t, **solver_kwargs)
        nodes, weights = [], []
        for tj, hj, nj in zip(t, h, near):
            if nj:
                nodes.append(np.array([-hj, 0.0, hj]))
                # Quadratic Lagrange basis polynomials for the nodes -h, 0, h
                weights.append(
                    np.array([tj * (tj - hj) / 2, hj**2 - tj**2, tj * (tj + hj) / 2]) / hj**2
                )
            else:
                nodes.append(np.array([tj]))
                weights.append(np.ones(1))
        vals = np.array(
            [
                [
                    self._bivariate_formula_at(np.array([a, b]), h, **solver_kwargs)
                    for b in nodes[1]
                ]
                for a in nodes[0]
            ]
        )
        return weights[0] @ vals @ weights[1]

    def _bivariate_formula_at(self, t, h, **solver_kwargs):
        if np.all(t == 0):
            # No closed form for the double singularity: extrapolate the averages over
            # the neighbouring points on the axes, A(h) = F(0) + c h^2 + O(h^4).
            def axis_mean(hh):
                pts = [[hh[0], 0], [-hh[0], 0], [0, hh[1]], [0, -hh[1]]]
                return np.mean([self._bivariate_formula_at(np.array(p), h) for p in pts])

            return (4 * axis_mean(h) - axis_mean(2 * h)) / 3
        return self._bivariate_formula(self.cgf.dK(t), t, **solver_kwargs)

    def _bivariate_formula(self, x, t, **solver_kwargs):
        """
        Direct evaluation of the formula documented in :meth:`_spapprox_cdf` for a
        single point, using the exact limits when a component of ``t`` is exactly 0.
        """
        # Initialize
        t0 = t.copy()
        t0[..., 0] = 0
        s0 = t.copy()
        s0[..., 1] = 0
        if t[0] != 0:
            # Note, slicing a component of a cgf sets the other variables to zero
            tt = self.cgf[1].dK_inv(x.T[1], **solver_kwargs)
            tt0 = np.vstack((np.zeros(np.shape(tt)), tt)).T.squeeze()
            # Calculate components
            tx = np.sign(tt) * np.sqrt(2 * ((tt0 * x).sum(axis=-1).squeeze() - self.cgf.K(tt0)))
            tw = np.sign(t.T[1]) * np.sqrt(
                2 * (self.cgf.K(s0) - self.cgf.K(t) + (t0 * x).sum(axis=-1).squeeze())
            )
            w = np.sign(t.T[0]) * np.sqrt(
                2 * (((t - tt0) * x).sum(axis=-1).squeeze() + self.cgf.K(tt0) - self.cgf.K(t))
            )
            b = (tw - tx) / w
            ty = (w - b * tx) / np.sqrt(1 + np.square(b))
            tx = np.vstack((tx, ty)).T.squeeze()
            rho = -b / np.sqrt(1 + np.square(b))
            d2Kt11 = self.cgf.d2K(t)[..., 1, 1]
            u = t.T[0] * np.sqrt(np.linalg.det(self.cgf.d2K(t)) / d2Kt11)
            tu = t.T[1] * np.sqrt(d2Kt11)
            n = sps.norm.pdf(w) * (1 / w - 1 / u)
            if (t != 0).all():
                tn = sps.norm.pdf(tx.T[0]) * (1 / tw - 1 / tu)
            else:
                if not np.isclose(tu, 0) and not np.isclose(tw, 0):
                    raise AssertionError("Invalid singular case")
                d3K111s0 = self.cgf.d3K(s0)[..., 1, 1, 1]
                d2K11s0 = self.cgf.d2K(s0)[..., 1, 1]
                tn = sps.norm.pdf(tx.T[0]) / 6 * (d3K111s0 / d2K11s0 ** (3 / 2))
                assert np.isfinite(n) and not np.isnan(n), "Something is wrong"
        else:
            # Note, slicing a component of a cgf sets the other variables to zero
            ts = self.cgf[0].dK_inv(x.T[0], **solver_kwargs)
            ts0 = np.vstack((ts, np.zeros(np.shape(ts)))).T.squeeze()
            # Calculate components
            ty = np.sign(ts) * np.sqrt(2 * ((ts0 * x).sum(axis=-1).squeeze() - self.cgf.K(ts0)))
            tw = np.sign(t.T[0]) * np.sqrt(
                2 * (self.cgf.K(t0) - self.cgf.K(t) + (s0 * x).sum(axis=-1).squeeze())
            )
            w = np.sign(t.T[1]) * np.sqrt(
                2 * (((t - ts0) * x).sum(axis=-1).squeeze() + self.cgf.K(ts0) - self.cgf.K(t))
            )
            b = (tw - ty) / w
            tx = (w - b * ty) / np.sqrt(1 + np.square(b))
            tx = np.vstack((tx, ty)).T.squeeze()
            rho = -b / np.sqrt(1 + np.square(b))
            d2Kt00 = self.cgf.d2K(t)[..., 0, 0]
            u = t.T[1] * np.sqrt(np.linalg.det(self.cgf.d2K(t)) / d2Kt00)
            tu = t.T[0] * np.sqrt(d2Kt00)
            n = sps.norm.pdf(w) * (1 / w - 1 / u)
            if (t != 0).all():
                tn = sps.norm.pdf(tx.T[1]) * (1 / tw - 1 / tu)
            else:
                if not np.isclose(tu, 0) and not np.isclose(tw, 0):
                    raise AssertionError("Invalid singular case")
                d3K000t0 = self.cgf.d3K(t0)[..., 0, 0, 0]
                d2K00t0 = self.cgf.d2K(t0)[..., 0, 0]
                tn = sps.norm.pdf(tx.T[1]) / 6 * (d3K000t0 / d2K00t0 ** (3 / 2))
                assert np.isfinite(n) and not np.isnan(n), "Something is wrong"
        # Put everything together
        if _has_fastnorm:
            if np.ndim(rho) == 0:
                retval = fastnorm.bivar_norm_cdf(tx, rho)
            else:
                retval = np.array(
                    [fastnorm.bivar_norm_cdf(_tx, _rho) for _tx, _rho in zip(tx, rho)]
                )
        else:
            if np.ndim(rho) == 0:
                retval = sps.multivariate_normal([0, 0], np.array([[1, rho], [rho, 1]])).cdf(tx)
            else:
                retval = np.array(
                    [
                        sps.multivariate_normal([0, 0], np.array([[1, _rho], [_rho, 1]])).cdf(_tx)
                        for _tx, _rho in zip(tx, rho)
                    ]
                )
        retval += sps.norm.cdf(tw) * n + sps.norm.cdf(w) * tn + n * tn
        return retval

    def _dK_inv(self, x, **solver_kwargs):
        # TODO: implement multivariate interpolation with radial basis functions or something like that
        return super()._dK_inv(x, **solver_kwargs)
