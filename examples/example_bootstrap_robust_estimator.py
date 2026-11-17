#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap of a robust estimator: Huber's M-estimator of location (Kuonen, 1998; Davison and
Hinkley, 1997, Section 9.5).

The M-estimator tau solves sum_i psi((x_i - tau) / s) = 0, with Huber's
psi(u) = max(-k, min(u, k)), k = 1.345, and the scale s fixed at the (normalized) median
absolute deviation of the data. Under the multinomial bootstrap, the resampled estimator solves
sum_i psi((x_i - tau) / s) W_i = 0 with Poisson(1) weights given sum_i W_i = n. As psi is
monotone, this is a decreasing estimating equation, so ``EstimatingEquationSaddlePointApprox``
applies without new code: P(tau* <= tau) = P(S(tau) <= 0 | sum W_i = n). For the sample mean,
psi(u) = u.

The data are Darwin's Zea mays differences (Fisher, 1935), with two outliers (-67, -48). The
saddle point approximation is compared with a Monte Carlo bootstrap.
"""

import matplotlib.pyplot as plt
import numpy as np

import spapprox as spa

x = np.array([49, -67, 8, 16, 6, 23, 28, 41, 14, 29, 56, 24, 75, 60, -48.0])
n = len(x)
s = 1.4826 * np.median(np.abs(x - np.median(x)))
k = 1.345


def psi_huber(x, tau):
    return np.clip((x - tau) / s, -k, k)


def psi_mean(x, tau):
    return x - tau


def solve(psi, w):
    """Solve sum_i psi(x_i, tau) w_i = 0 by bisection, for each row of w."""
    lo, hi = np.full(len(w), x.min()), np.full(len(w), x.max())
    for _ in range(60):
        mid = (lo + hi) / 2
        positive = (psi(x, mid[:, None]) * w).sum(axis=1) > 0
        lo, hi = np.where(positive, mid, lo), np.where(positive, hi, mid)
    return (lo + hi) / 2


rng = np.random.default_rng(0)
W = rng.multinomial(n, np.ones(n) / n, size=50000)
bracket = (x.min(), x.max())

results = {}
for name, psi in [("Huber", psi_huber), ("mean", psi_mean)]:
    estimate = solve(psi, np.ones((1, n)))[0]
    mc = solve(psi, W)
    ee = spa.EstimatingEquationSaddlePointApprox.from_resampling(
        psi, x, spa.poisson(1), given=n, bracket=bracket
    )
    ee.fit_cdf(num=60)
    results[name] = estimate, mc, ee

    print(f"\n{name} estimate {estimate:.2f}, multinomial bootstrap:")
    p = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    q = np.quantile(mc, p)
    print(f"{'p':>6} {'MC quantile':>12} {'SP cdf':>8}")
    for pp, qq, F in zip(p, q, ee.cdf(q)):
        print(f"{pp:6.2f} {qq:12.2f} {F:8.4f}")
    lo, hi = ee.ppf([0.025, 0.975])
    print(f"95% percentile interval: SP ({lo:.2f}, {hi:.2f}),", end=" ")
    print(f"MC ({np.quantile(mc, 0.025):.2f}, {np.quantile(mc, 0.975):.2f})")
    print(f"mean {ee.mean():.2f} (MC {mc.mean():.2f}), sd {np.sqrt(ee.var()):.2f}", end=" ")
    print(f"(MC {mc.std():.2f})")

tau = np.linspace(-5, 50, 80)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, (estimate, mc, ee)) in zip(axes, results.items()):
    ax.hist(mc, bins=80, range=(tau[0], tau[-1]), density=True, alpha=0.4, label="Monte Carlo")
    # normalize on the plotting grid, which saves a second grid of pdf evaluations
    f = ee.pdf(tau, normalize_pdf=False)
    ax.plot(tau, f / np.trapezoid(f, tau), label="saddle point")
    ax.axvline(estimate, color="k", lw=0.8)
    ax.set_title(f"Bootstrap distribution, {name}")
    ax.set_xlabel("tau")
    ax.legend()
plt.show()
