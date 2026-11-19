#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saddle point approximation of the pdf and cdf of a normal and an exponential distribution,
compared with the exact distributions from ``scipy.stats``.

Every ``pdf`` and ``cdf`` accepts either the point ``x`` or the saddle point ``t``. Sweeping
``t`` and computing ``x = cgf.dK(t)`` avoids solving the saddle point equation ``K'(t) = x``
for every point. For the normal distribution the approximation is exact. For the exponential
distribution, the cgf is finite for ``t < 1 / scale`` only, so ``t`` stays below 1.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

import spapprox as spa

plt.close("all")

cases = [
    ("Normal", spa.norm(loc=0, scale=1), sps.norm(loc=0, scale=1), np.linspace(-3, 3)),
    ("Exponential", spa.exponential(scale=1), sps.expon(scale=1), np.linspace(-3, 0.9)),
]

fig, axs = plt.subplots(len(cases), 2, facecolor="w", figsize=(10, 8))
for (name, cgf, dist, t), (ax_pdf, ax_cdf) in zip(cases, axs):
    approx = spa.UnivariateSaddlePointApprox(cgf)
    x = cgf.dK(t)
    ax_pdf.plot(x, approx.pdf(t=t), label="Saddle point approximation")
    ax_pdf.plot(x, dist.pdf(x), "--", label="Exact")
    ax_pdf.set_title(f"{name} pdf")
    ax_pdf.legend()
    ax_cdf.plot(x, approx.cdf(t=t), label="Saddle point approximation")
    ax_cdf.plot(x, dist.cdf(x), "--", label="Exact")
    ax_cdf.set_title(f"{name} cdf")
    ax_cdf.legend()
fig.tight_layout()
plt.show()
