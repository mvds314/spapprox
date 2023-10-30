#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
from scipy.integrate import quad


from .cgfs import CumulantGeneratingFunction
from .util import type_wrapper, fib
from statsmodels.tools.validation import PandasWrapper


class SaddlePointApprox:
    """
    Given the cumulant generating function of a random variable, this class
    provides the saddle point approximation of the probability density function
    and the cumulative distribution function.

    Parameters
    ----------
    cgf : CumulantGeneratingFunction
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
        assert isinstance(cgf, CumulantGeneratingFunction)
        self.cgf = cgf
        self._pdf_normalization_cache = pdf_normalization

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
        w = np.sign(t) * np.sqrt(2 * (t * x - self.cgf.K(t)))
        u = t * np.sqrt(self.cgf.d2K(t))
        with np.errstate(divide="ignore", invalid="ignore"):
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
        w = np.sign(t) * np.sqrt(2 * (t * x - self.cgf.K(t)))
        u = t * np.sqrt(self.cgf.d2K(t))
        with np.errstate(divide="ignore", invalid="ignore"):
            retval = sps.norm.cdf(w + np.log(u / w) / w)
        retval = np.where(
            ~np.isclose(t, 0),
            retval,
            1 / 2 + self.cgf.d3K0 / 6 / np.sqrt(2 * np.pi) / np.power(self.cgf.d2K0, 3 / 2),
        )
        return np.where(np.isnan(retval), fillna, retval)

    @property
    def _pdf_normalization(self):
        if not hasattr(self, "_pdf_normalization_cache") or self._pdf_normalization_cache is None:
            self._pdf_normalization_cache = quad(
                lambda t: self.pdf(t=t, normalize_pdf=False, fillna=0) * self.cgf.d2K(t, fillna=0),
                a=-np.inf,
                b=np.inf,
            )[0]
        return self._pdf_normalization_cache

    def pdf(self, x=None, t=None, normalize_pdf=True, fillna=np.nan, **solver_kwargs):
        r"""
        Saddle point approximation of the probability density function.
        Given by

        .. math::
            f(x) \approx \frac{1}{\sqrt{2\pi K''(t)}} \exp\left(K(t) - tx\right)

        where :math:`t` is the solution of the saddle point equation.

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
        return y.tolist() if len(y.shape) == 0 else wrapper.wrap(y)

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
        return y.tolist() if len(y.shape) == 0 else wrapper.wrap(y)

    @type_wrapper(xloc=1)
    def ppf(self, q, fillna=np.nan, t0=None, **kwargs):
        r"""
        Percent point function computed as the inverse of the saddle point
        approximation of the cumulative distribution function.

        Parameters
        ----------
        q : array_like
            The values at which to evaluate the inverse cumulative distribution function.
        fillna : float, optional
            The value to replace NaNs with.
        """
        assert (0 <= q <= 1).all()
        if hasattr(self, "_cdf_cache") and hasattr(self, "_x_cache"):
            t = np.interp(q, self._cdf_cache, self._t_cache)
        else:
            if len(q.shape) == 0:  # Then it is a scalar "array"
                q = q.tolist()
                kwargs["x0"] = 0 if t0 is None else t0
                bracket_methods = ["bisect", "brentq", "brenth", "ridder", "toms748"]
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
                            1 * 0.9**i
                            for i in range(10000)
                            if ~np.isnan(self.cgf.dK(1 * 0.9**i))
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
                    res = spo.root_scalar(lambda t: self.cdf(t=t) - q, **kwargs)
                    if res.converged:
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
                    res = spo.root(lambda t, q=q: self.cdf(t=t) - q, **kwargs)
                    if res.success:
                        t = np.asanyarray(res.x)
                        break
                else:
                    t = np.asanyarray(
                        [self.ppf(qq, t0=None if t0 is None else t0[i]) for i, qq in enumerate(q)]
                    )
        return self.cgf.dK(t)

    def fit_saddle_point_eqn(self, t_range=None, atol=1e-4, rtol=1e-4, num=1000, **solver_kwargs):
        """
        Evaluate the saddle point equation to the given range of values.
        And use linear interpolation to solve the saddle point equation, form now onwards.

        If t_range is not provided, it will be assumed that the cumulant generating
        function is defined in a neighborhood of zero. The range will be computed
        by scaling iteratively, until we little mass for higher or lower values of x.
        """
        if t_range is None:
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
            t_range = [lb, ub]
        # Solve saddle point equation
        x_range = np.linspace(*self.cgf.dK(t_range), num=num)
        self._x_cache = x_range
        self._t_cache = self.cgf._dK_inv(x_range, **solver_kwargs)

    def fit_ppf(self, t_range=None, atol=1e-4, rtol=1e-4, num=1000, **solver_kwargs):
        """
        Fit the inverse of the cumulative distribution function using linear interpolation.
        """
        if not hasattr(self, "_x_cache") or not hasattr(self, "_t_cache"):
            self.fit_saddle_point_eqn(
                t_range=t_range, atol=atol, rtol=rtol, num=num, **solver_kwargs
            )
        self._cdf_cache = self.cdf(t=self._t_cache)

    @type_wrapper(xloc=1)
    def _dK_inv(self, x, **solver_kwargs):
        """ """
        if hasattr(self, "_x_cache") and hasattr(self, "_t_cache"):
            return np.interp(x, self._x_cache, self._t_cache)
        else:
            return self.cgf.dK_inv(x, **solver_kwargs)


# TODO: implement cdf interpolation

# TODO: then continue with the applications: sum of random variables, approximation of the mean, bootstrap in the transformed domain, etc.
