#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuity corrected resampling distributions of a mean of integer data.

For integer data and integer weights W_i, the estimating equation S(tau) = sum_i (x_i - tau) W_i
of the mean lies, given sum_i W_i = m, on a lattice of span 1, so the resampled mean is
discrete. Passing ``delta=1`` to ``EstimatingEquationSaddlePointApprox``
applies the continuity correction of Skovgaard (1987), see Butler (2007) Section 4.2:

- multinomial bootstrap: Poisson(1) weights given sum W_i = n. The exact distribution of the
  resampled sum is the n-fold convolution of the empirical distribution;
- delete-d jackknife: Bernoulli(1/2) weights given sum W_i = n - d, exact by enumeration.

The correction removes most of the error for the bootstrap. For the jackknife, which samples
without replacement from a small population, it helps less.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

import spapprox as spa

x = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 8, 12.0])
n = len(x)


def psi(x, tau):
    return x - tau


# Multinomial bootstrap
with spa.Timer("Exact multinomial bootstrap (convolution)", decimals=3):
    p = np.array([1.0])
    for _ in range(n):
        p = np.convolve(p, np.bincount(x.astype(int)) / n)
    exact_boot = np.cumsum(p)
boot = {
    delta: spa.EstimatingEquationSaddlePointApprox.from_resampling(
        psi, x, spa.poisson(1), given=n, delta=delta
    )
    for delta in (None, 1.0)
}

# Delete-d jackknife
d = 5
with spa.Timer(f"Exact delete-{d} jackknife (enumeration)", decimals=3):
    sums = np.array([x[list(c)].sum() for c in combinations(range(n), n - d)])
jack = {
    delta: spa.EstimatingEquationSaddlePointApprox.from_resampling(
        psi, x, spa.binomial(1, 0.5), given=n - d, delta=delta
    )
    for delta in (None, 1.0)
}

for name, m, F_exact, approx in [
    ("multinomial bootstrap", n, lambda k: exact_boot[int(k)], boot),
    (f"delete-{d} jackknife", n - d, lambda k: np.mean(sums <= k), jack),
]:
    # Values of the resampled mean, on its lattice of span 1 / m
    tau = np.round(m * np.array([1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4])) / m
    print(f"\n{name}:")
    with spa.Timer("  Saddle point cdf", decimals=3):
        F_sp = approx[None].cdf(tau)
    with spa.Timer("  Lattice saddle point cdf", decimals=3):
        F_lattice = approx[1.0].cdf(tau)
    print(f"P(mean <= tau):\n{'tau':>7} {'exact':>8} {'SP':>8} {'lattice':>8}")
    for t, a, b in zip(tau, F_sp, F_lattice):
        print(f"{t:7.3f} {F_exact(round(m * t)):8.5f} {a:8.5f} {b:8.5f}")

tau = np.linspace(1.2, 4.5, 150)
fig, ax = plt.subplots()
ax.step(np.arange(len(exact_boot)) / n, exact_boot, where="post", label="exact")
ax.plot(tau, boot[None].cdf(tau), label="saddle point")
ax.plot(tau, boot[1.0].cdf(tau), "--", label="lattice saddle point")
ax.set_xlim(tau[0], tau[-1])
ax.set_xlabel("bootstrap mean")
ax.set_title("Multinomial bootstrap of a mean of integer data")
ax.legend()
plt.show()
