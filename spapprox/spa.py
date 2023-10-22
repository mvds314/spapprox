#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as sps

from .cgfs import CumulantGeneratingFunction


class SaddlePointApprox:
    """
    https://en.wikipedia.org/wiki/Saddlepoint_approximation_method
    """

    def __init__(self, cgf):
        assert isinstance(cgf, CumulantGeneratingFunction)
        self.cgf = cgf

    def _spappox_pdf(self, x, t):
        return np.exp(self.cgf.K(t) - t * x) * np.sqrt(1 / (2 * np.pi * self.cgf.d2K(t)))

    def _spappox_cdf(self, x, t):
        w = np.sign(t) * np.sqrt(2 * (t * x - self.cgf.K(t)))
        u = t * np.sqrt(self.cgf.d2K(t))
        retval = sps.norm.cdf(w) + sps.norm.pdf(w) * (1 / w - 1 / u)
        return np.where(
            ~np.isclose(t, 0),
            retval,
            1 / 2 + self.cgf.d3K0 / 6 / np.sqrt(2 * np.pi) / np.power(self.cgf.d2K0, 3 / 2),
        )

    def pdf(self, x=None, t=None):
        assert x is not None or t is not None
        if x is None:
            x = self.cgf.dK(t)
        elif t is None:
            raise NotImplementedError()
        return self._spappox_pdf(x, t)

    def cdf(self, x=None, t=None):
        assert x is not None or t is not None
        if x is None:
            x = self.cgf.dK(t)
        elif t is None:
            raise NotImplementedError()
        return self._spappox_cdf(x, t)
