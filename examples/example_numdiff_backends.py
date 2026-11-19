#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical derivatives of a cgf, with the two numerical differentiation backends.

A cgf only needs ``K``. Derivatives that are not supplied are computed numerically, with
``numdifftools`` (``numdiff_backend="numdifftools"``, the default) or ``findiff``
(``numdiff_backend="findiff"``). Below, the cgf of a unit exponential, ``K(t) = -log(1 - t)``
for ``t < 1``, is defined without derivatives and compared with the analytic ones, together
with the saddle point pdf based on them.

Requires the optional dependencies ``numdifftools`` and ``findiff``:
``pip install "spapprox[numdiff,findiff]"``.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

import spapprox as spa

plt.close("all")

analytic = spa.exponential()
t = np.linspace(-3, 0.9, num=200)
x = analytic.dK(t)
exact = {"dK": analytic.dK(t), "d2K": analytic.d2K(t), "d3K": analytic.d3K(t)}

fig, axs = plt.subplots(1, 2, facecolor="w", figsize=(10, 4))
pdf = spa.UnivariateSaddlePointApprox(analytic).pdf(t=t, normalize_pdf=False)
axs[0].plot(x, pdf, label="Analytic derivatives")
for backend in ["numdifftools", "findiff"]:
    cgf = spa.UnivariateCumulantGeneratingFunction(
        K=lambda t: -np.log1p(-t), domain=spa.Domain(l=1), numdiff_backend=backend
    )
    with spa.Timer(f"{backend}: dK, d2K and d3K at {len(t)} points"):
        numerical = {"dK": cgf.dK(t), "d2K": cgf.d2K(t), "d3K": cgf.d3K(t)}
    for name in numerical:
        err = np.max(np.abs(numerical[name] / exact[name] - 1))
        print(f"  {name}: maximum relative error {err:.1e}")
    approx = spa.UnivariateSaddlePointApprox(cgf)
    axs[0].plot(x, approx.pdf(t=t, normalize_pdf=False), "--", label=backend)
    axs[1].semilogy(t, np.abs(numerical["d2K"] / exact["d2K"] - 1), label=backend)

axs[0].plot(x, sps.expon.pdf(x), ":", label="Exact pdf")
axs[0].set_title("Unnormalized saddle point pdf")
axs[0].legend()
axs[1].set_title("Relative error of d2K")
axs[1].set_xlabel("t")
axs[1].legend()
fig.tight_layout()
plt.show()
