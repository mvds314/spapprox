#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coverage of conditional confidence intervals, Booth, Hall and Wood (1992), Section 3, Table 2.

The data are (log Y, Y) with Y ~ Exp(1). A two-sided 90% confidence interval for
theta_1 = E log Y = -0.5772 is constructed conditional on the observed mean of Y. As in BHW,
the samples are simulated conditional on the mean of Y being equal to its expectation, 1. The
intervals are

- the pivotal (Studentized) conditional bootstrap, ``ConditionalBootstrap`` with importance
  resampling and the plug-in bandwidth, for B = 200, 500 and 1000 resamples, and with a fixed
  bandwidth h = 0.5,
- the nonpivotal conditional bootstrap, ``ConditionalBootstrap(..., pivotal=False)``,
- the saddle point approximation of the nonpivotal conditional bootstrap,
  ``MultivariateSaddlePointApprox.conditional_cdf`` of the mean of ``multivariate_empirical``,
- normal theory: the conditional normal distribution, with a t distribution with n - 2 degrees
  of freedom.

Columns L (U) give the proportion of samples where theta_1 is smaller than the lower (upper)
limit, nominally 5% (95%). The pivotal bootstrap is much better on the left; the saddle point
approximation is as accurate as the nonpivotal bootstrap it approximates, and as bad. BHW use
1000 samples; increase ``n_samples`` to reproduce their table more closely.

With the plug-in bandwidth (h of about 0.13 for B = 1000), the left misses stay near 8% for
n = 30 to 50 (2000 samples), above the 5-6% BHW report; the h -> 0 limit gives the same, so this
is the O(1/n) error of the pivotal bootstrap itself. With h = 0.5, the smoothing bias of BHW
(2.14) widens the intervals, and the coverage matches BHW Table 2 (L 4.9%, U 97.7% at n = 50).
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
from scipy.optimize import brentq

import spapprox as spa

plt.close("all")

n_samples = 100
sizes = [10, 20, 50]
level = 0.9
theta1 = -np.euler_gamma
rng = np.random.default_rng(1992)
data = rng.exponential(size=(n_samples, max(sizes)))


def sp_quantile(sp, given, p, mu, sd):
    """Quantile of the conditional saddle point cdf; it is NaN outside the convex hull."""
    f = lambda q: sp.conditional_cdf(q, given, i=0) - p  # noqa: E731
    sign = 1 if f(mu) < 0 else -1
    inside, k = 0.0, 0.5
    while True:
        v = f(mu + sign * k * sd)
        if np.isnan(v):
            # bisect towards the hull boundary
            outside = k
            for _ in range(30):
                k = (inside + outside) / 2
                v = f(mu + sign * k * sd)
                if np.isnan(v):
                    outside = k
                elif sign * v > 0:
                    break
                else:
                    inside = k
            else:
                return np.nan
        if sign * v > 0:
            a, b = sorted([mu + sign * inside * sd, mu + sign * k * sd])
            return brentq(f, a, b, xtol=1e-4 * sd)
        inside, k = k, k + 0.5


def sp_ci(x):
    n = len(x)
    cgf = spa.multivariate_sample_mean(spa.multivariate_empirical(x), n)
    sp = spa.MultivariateSaddlePointApprox(cgf)
    c = cgf.cov
    given = [x[:, 1].mean()]
    mu = cgf.dK0[0] + c[0, 1] / c[1, 1] * (given[0] - cgf.dK0[1])
    sd = np.sqrt(c[0, 0] - c[0, 1] ** 2 / c[1, 1])
    hi, lo = (sp_quantile(sp, given, p, mu, sd) for p in [(1 + level) / 2, (1 - level) / 2])
    # basic interval, as for the nonpivotal bootstrap
    return 2 * x[:, 0].mean() - hi, 2 * x[:, 0].mean() - lo


def normal_ci(cb, n):
    half = sps.t.ppf((1 + level) / 2, n - 2) * cb.sigma1 * np.sqrt((1 - cb.rho**2) * n / (n - 2))
    return cb.theta1 - half, cb.theta1 + half


methods = ["B(200)", "B(500)", "B(1000)", "B(1000), h=0.5", "Nonpivotal B(1000)", "SP", "Normal"]
limits = {(m, n): [] for m in methods for n in sizes}
with spa.Timer(f"Coverage, {n_samples} samples"):
    for s, y in enumerate(data):
        for n in sizes:
            # condition on the mean of Y equal to its expectation, 1: Y / mean(Y) is uniform on
            # the simplex, independent of mean(Y)
            yn = y[:n] / y[:n].mean()
            x = np.column_stack([np.log(yn), yn])
            cb = spa.ConditionalBootstrap(x, rng=s)
            for B in [200, 500, 1000]:
                limits["B(%d)" % B, n].append(cb.ci(level, B=B))
            limits["B(1000), h=0.5", n].append(cb.ci(level, B=1000, h=0.5))
            limits["Nonpivotal B(1000)", n].append(cb.ci(level, B=1000, pivotal=False))
            limits["SP", n].append(sp_ci(x))
            limits["Normal", n].append(normal_ci(cb, n))

print(f"{'n':>20}" + "".join(f"{n:>14}" for n in sizes))
print(f"{'':>20}" + "".join(f"{'L':>7}{'U':>7}" for _ in sizes))
print(f"{'True value':>20}" + "".join(f"{'5.0%':>7}{'95.0%':>7}" for _ in sizes))
coverage = {}
for m in methods:
    row = ""
    for n in sizes:
        lo, hi = np.array(limits[m, n]).T
        coverage[m, n] = (np.nanmean(theta1 < lo), np.nanmean(theta1 < hi))
        row += "".join(f"{100 * c:6.1f}%" for c in coverage[m, n])
    print(f"{m:>20}" + row)

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for m in methods:
    for ax, k in zip(axs, [0, 1]):
        ax.plot(sizes, [100 * coverage[m, n][k] for n in sizes], "o-", label=m)
for ax, nominal, side in zip(axs, [5, 95], ["lower", "upper"]):
    ax.axhline(nominal, color="k", lw=0.5)
    ax.set_title(f"P(theta_1 < {side} limit), nominal {nominal}%")
    ax.set_xlabel("n")
axs[0].legend()
plt.show()
