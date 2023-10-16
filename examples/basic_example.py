#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as sps

import matplotlib.pyplot as plt

plt.close("all")

# https://en.wikipedia.org/wiki/Cumulant


# Cumulant genrating function of the normal distribution
def K(t, mu=0, sigma=1):
    return mu * t + sigma**2 * t**2 / 2


def dK(t, mu=0, sigma=1):
    return mu + sigma**2 * t


def d2K(t, mu=0, sigma=1):
    return sigma**2 + 0 * t


def d3K(t, mu=0, sigma=1):
    return 0 * t


class SaddlePointApprox:
    """
    https://en.wikipedia.org/wiki/Saddlepoint_approximation_method
    """

    def __init__(self, K, dK=None, d2K=None, d3K=None):
        self.K = K
        self.dK = dK
        self.d2K = d2K
        self.d3K = d3K

    @property
    def _dK0(self):
        if not hasattr(self, "_dK0_cache"):
            self._dK0_cache = self.dK(0)
        return self._dK0_cache

    @property
    def _d2K0(self):
        if not hasattr(self, "_d2K0_cache"):
            self._d2K0_cache = self.d2K(0)
        return self._d2K0_cache

    @property
    def _d3K0(self):
        if not hasattr(self, "_d3K0_cache"):
            self._d3K0_cache = self.d3K(0)
        return self._d3K0_cache

    def _spappox_pdf(self, x, t):
        return np.exp(self.K(t) - t * x) * np.sqrt(1 / (2 * np.pi * self.d2K(t)))

    def _spappox_cdf(self, x, t):
        w = np.sign(t) * np.sqrt(2 * (t * x - self.K(t)))
        u = t * np.sqrt(self.d2K(t))
        retval = sps.norm.cdf(w) + sps.norm.pdf(w) * (1 / w + 1 / u)
        return np.where(
            ~np.isclose(t, 0),
            retval,
            1 / 2 + self._d3K0 / 6 / np.sqrt(2 * np.pi) / np.power(self._d2K0, 3 / 2),
        )
        # if np.isclose(t, 0):
        #     1 / 2 + self.d3K0 / 6 / np.sqrt(2 * np.pi) / np.power(self.d2K0, 3 / 2)
        # else:
        #     w = np.sign(t) * np.sqrt(2 * (t * x - self.K(t)))
        #     u = t * np.sqrt(self.dK2(t))
        #     return sps.norm.cdf(w) + sps.norm.pdf(w) * (1 / w + 1 / u)

    def pdf(self, x=None, t=None):
        assert x is not None or t is not None
        if x is None:
            x = self.dK(t)
        elif t is None:
            raise NotImplementedError()
        return self._spappox_pdf(x, t)

    def cdf(self, x=None, t=None):
        assert x is not None or t is not None
        if x is None:
            x = self.dK(t)
        elif t is None:
            raise NotImplementedError()
        return self._spappox_cdf(x, t)


spapprox_normal = SaddlePointApprox(K, dK=dK, d2K=d2K, d3K=d3K)

fig, axs = plt.subplots(1, 2, facecolor="w")
t = np.linspace(-3, 3)
x = spapprox_normal.dK(t)
spapprox_normal.pdf(t=t)
sps.norm.pdf(x)

ax = axs[0]
ax.plot(x, spapprox_normal.pdf(t=t), label="Saddle point approximation")
ax.plot(x, sps.norm.pdf(x), label="Normal pdf")
ax.legend()

ax = axs[1]
ax.plot(x, spapprox_normal.cdf(t=t), label="Saddle point approximation")
ax.plot(x, sps.norm.cdf(x), label="Normal pdf")
ax.legend()

# TODO: the cdf is not working yet it seems
