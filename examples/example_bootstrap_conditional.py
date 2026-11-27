#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional bootstrap distribution of a mean without resampling, Booth, Hall and Wood (1992),
Section 4 and Table 3.

The data are bivariate, X = (log Y, Y) with Y ~ Exp(1), and the statistic of interest is the
mean of log Y, conditional on the mean of Y. ``multivariate_empirical`` is the cgf of drawing
one observation from the sample, ``multivariate_sample_mean`` turns it into the cgf of the
bootstrap mean, and ``conditional_cdf`` approximates the conditional bootstrap distribution with
the double saddle point approximation of Skovgaard (1987), as proposed by Davison and Hinkley
(1988).

A Monte Carlo conditional bootstrap has to condition on a window around the observed mean of Y,
since the bootstrap distribution is discrete. As in Table 3, the saddle point quantiles are
evaluated under windowed bootstraps with half widths h = 0.05, 0.10 and 0.25 times the standard
error of the mean of Y. The narrow windows agree with the saddle point approximation, the wide
window, as suggested by Davison and Hinkley, has too heavy tails.

Note that this approximates the *nonpivotal* conditional bootstrap, which has error of order
n^{-1/2}; Booth, Hall and Wood show that a Studentized (pivotal) statistic is needed for error of
order n^{-1}.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import spapprox as spa

plt.close("all")

# As in Section 3, a sample of size n = 10 with mean of Y equal to one: Y = n * Dirichlet(1)
n = 10
rng = np.random.default_rng(1992)
y = n * rng.dirichlet(np.ones(n))
x = np.column_stack([np.log(y), y])
given = [x[:, 1].mean()]

# Saddle point approximation of the conditional bootstrap distribution of the mean of log Y
with spa.Timer("Saddle point, quantiles", decimals=3):
    cgf = spa.multivariate_sample_mean(spa.multivariate_empirical(x), n)
    sp = spa.MultivariateSaddlePointApprox(cgf)
    cov = cgf.cov
    mu = cgf.dK0[0] + cov[0, 1] / cov[1, 1] * (given[0] - cgf.dK0[1])
    sd = np.sqrt(cov[0, 0] - cov[0, 1] ** 2 / cov[1, 1])

    # One grid for bracketing the quantiles; beyond the convex hull of the data the saddle
    # point does not exist and the cdf is NaN (and slow to evaluate)
    grid = mu + sd * np.linspace(-5, 3, 81)
    cdf = sp.conditional_cdf(grid, given, i=0)
    grid, cdf = grid[np.isfinite(cdf)], cdf[np.isfinite(cdf)]

    def ppf(p):
        j = np.searchsorted(cdf, p)
        return brentq(lambda q: sp.conditional_cdf(q, given, i=0) - p, grid[j - 1], grid[j])

    levels = np.array([0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995])
    quantiles = np.array([ppf(p) for p in levels])

# Monte Carlo conditional bootstrap, windowed around the observed mean of Y
with spa.Timer("Monte Carlo (10^6 resamples)", decimals=3):
    sim = x[rng.integers(0, n, size=(10**6, n))].mean(axis=1)
se = x[:, 1].std() / np.sqrt(n)
windowed = {}
for c in (0.05, 0.1, 0.25):
    inside = sim[np.abs(sim[:, 1] - given[0]) < c * se, 0]
    windowed[c] = inside
    print(f"h = {c:.2f} se: {len(inside)} resamples in the window")
print(f"{'level':>7}" + "".join(f"{f'h = {c:.2f} se':>15}" for c in (0.05, 0.1, 0.25)))
for p, q in zip(levels, quantiles):
    print(f"{p:7.3f}" + "".join(f"{np.mean(windowed[c] <= q):15.5f}" for c in windowed))

# Plot the conditional cdfs
fig, ax = plt.subplots()
ax.plot(grid, cdf, "k", label="Saddle point (Skovgaard)")
for c, sample in windowed.items():
    ax.step(np.sort(sample), np.linspace(0, 1, len(sample)), label=f"Bootstrap, h = {c} se")
ax.set_xlabel("mean of log Y")
ax.set_title("Conditional bootstrap distribution given the mean of Y")
ax.legend()
plt.show()
