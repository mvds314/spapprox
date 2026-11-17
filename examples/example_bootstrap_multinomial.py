#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multinomial (ordinary) bootstrap of the mean, Booth and Butler (1990), Example 4 and Table 4,
which reproduces Table 1 of Davison and Hinkley (1988).

A bootstrap resample draws each observation x_i a multinomial number of times W_i, so the
bootstrap mean solves the estimating equation sum_i (x_i - tau) W_i = 0. The multinomial
counts are independent Poisson(1) variables conditioned on sum_i W_i = n. This conditioning
induces a cut, so the double saddle point (Skovgaard) approximation coincides with the single
saddle point approximation of the empirical cgf, i.e., with Davison and Hinkley (1988).
"""

import matplotlib.pyplot as plt
import numpy as np

import spapprox as spa

x = np.array([9.6, 10.4, 13.0, 15.0, 16.6, 17.2, 17.3, 21.8, 24.0, 33.8])
n = len(x)

# Estimating equation with Poisson weights, conditional on their sum
ee = spa.EstimatingEquationSaddlePointApprox.from_resampling(
    lambda x, tau: x - tau, x, spa.poisson(1), given=n, bracket=(x.min(), x.max())
)
ee.fit_cdf(num=200)

# The same, from the empirical cgf directly
mean = spa.UnivariateSaddlePointApproxMean(spa.univariate_empirical(x), n)

# Monte Carlo reference
rng = np.random.default_rng(1988)
sim = rng.choice(x, size=(10**6, n)).mean(axis=1)

# Table 4, multinomial bootstrap columns: percentage points of the bootstrap mean minus x_bar
p = np.array([1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.2, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999])
paper_exact = [-6.38, -5.85, -5.56, -4.80, -4.43, -3.32, -2.68, -1.86, 1.77, 2.84, 3.74, 5.48]
paper_exact += [6.16, 7.50]
paper_spa = [-6.31, -5.78, -5.52, -4.81, -4.43, -3.33, -2.69, -1.86, 1.80, 2.85, 3.75, 5.48]
paper_spa += [6.12, 7.46]

q = np.asarray(ee.ppf(p)) - x.mean()
print(f"{'p':>7} {'MC':>7} {'paper':>7} {'paper SP':>8} {'SP':>7}")
for row in zip(p, np.quantile(sim, p) - x.mean(), paper_exact, paper_spa, q):
    print(f"{row[0]:7.4f} " + " ".join(f"{v:7.2f}" for v in row[1:]))

# The estimating equation and the empirical cgf give the same cdf (the cut)
tau = np.linspace(13, 25, 7)
print("\nmax |difference| with the empirical cgf:", np.abs(ee.cdf(tau) - mean.cdf(x=tau)).max())
print(f"mean {ee.mean():.4f} (exact {x.mean():.4f}), var {ee.var():.4f} (exact {x.var() / n:.4f})")

tau = np.linspace(12, 26, 100)
fig, ax = plt.subplots()
ax.hist(sim, bins=200, density=True, alpha=0.4, label="Monte Carlo")
ax.plot(tau, ee.pdf(tau), label="saddle point")
ax.set_xlabel("bootstrap mean")
ax.legend()
plt.show()
