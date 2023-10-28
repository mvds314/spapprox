#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as sps
from scipy.integrate import quad

from .cgfs import CumulantGeneratingFunction
from .util import type_wrapper
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
    def _spappox_pdf(self, x, t, fillna=np.nan):
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
    def _spappox_cdf(self, x, t, fillna=np.nan):
        t = np.asanyarray(t)
        w = np.sign(t) * np.sqrt(2 * (t * x - self.cgf.K(t)))
        u = t * np.sqrt(self.cgf.d2K(t))
        retval = sps.norm.cdf(w) + sps.norm.pdf(w) * (1 / w - 1 / u)
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

    def pdf(self, x=None, t=None, normalize_pdf=True, fillna=np.nan):
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
            raise NotImplementedError()
        wrapper = PandasWrapper(x)
        y = np.asanyarray(self._spappox_pdf(np.asanyarray(x), np.asanyarray(t)))
        if normalize_pdf:
            y *= 1 / self._pdf_normalization
        y = np.where(np.isnan(y), fillna, y)
        return y.tolist() if len(y.shape) == 0 else wrapper.wrap(y)

    def cdf(self, x=None, t=None, fillna=np.nan):
        r"""
        Saddle point approximation of the cumulative distribution function.
        Given by

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
        assert x is not None or t is not None
        if x is None:
            x = self.cgf.dK(t)
        elif t is None:
            raise NotImplementedError()
        wrapper = PandasWrapper(x)
        y = np.asanyarray(self._spappox_cdf(np.asanyarray(x), np.asanyarray(t)))
        y = np.where(np.isnan(y), fillna, y)
        return y.tolist() if len(y.shape) == 0 else wrapper.wrap(y)


# TODO: implement the inversion
