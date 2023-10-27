#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as sps
from scipy.integrate import quad

from .cgfs import CumulantGeneratingFunction


class SaddlePointApprox:
    """
    https://en.wikipedia.org/wiki/Saddlepoint_approximation_method
    """

    def __init__(self, cgf, pdf_normalization=None):
        assert isinstance(cgf, CumulantGeneratingFunction)
        self.cgf = cgf
        self._pdf_normalization_cache = pdf_normalization

    def _spappox_pdf(self, x, t, fillna=np.nan):
        d2Kt = self.cgf.d2K(t, fillna=fillna)
        with np.errstate(divide="ignore"):
            return np.where(
                ~np.isclose(d2Kt, 0) & ~np.isnan(d2Kt),
                np.exp(self.cgf.K(t, fillna=fillna) - t * x)
                * np.sqrt(np.divide(1, 2 * np.pi * d2Kt)),
                fillna,
            )

    def _spappox_cdf(self, x, t, fillna=np.nan):
        retscalar = np.isscalar(t)
        t = np.asanyarray(t)
        x = np.asanyarray(x)
        w = np.sign(t) * np.sqrt(2 * (t * x - self.cgf.K(t, fillna=fillna)))
        u = t * np.sqrt(self.cgf.d2K(t, fillna=fillna))
        retval = sps.norm.cdf(w) + sps.norm.pdf(w) * (1 / w - 1 / u)
        retval = np.where(
            ~np.isclose(t, 0),
            retval,
            1 / 2 + self.cgf.d3K0 / 6 / np.sqrt(2 * np.pi) / np.power(self.cgf.d2K0, 3 / 2),
        )
        return retval.item() if retscalar else retval

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
        assert x is not None or t is not None
        if x is None:
            x = self.cgf.dK(t)
        elif t is None:
            raise NotImplementedError()
        retval = self._spappox_pdf(x, t, fillna=fillna)
        return retval / self._pdf_normalization if normalize_pdf else retval

    def cdf(self, x=None, t=None, fillna=np.nan):
        assert x is not None or t is not None
        if x is None:
            x = self.cgf.dK(t)
        elif t is None:
            raise NotImplementedError()
        return self._spappox_cdf(x, t, fillna=fillna)


# TODO: implement the inversion
