#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap distribution of the least squares estimate of a regression line, a vector-valued
estimating equation, with ``MultivariateEstimatingEquationSaddlePointApprox``.

The estimate (intercept, slope) solves S(tau) = sum_i z_i (y_i - z_i' tau) W_i = 0, with
z_i = (1, x_i). Case resampling gives:

- the multinomial bootstrap: Poisson(1) weights, conditional on sum W_i = n,
- the Bayesian (Dirichlet) bootstrap: exponential(1) weights, conditional on sum W_i = n.

The saddle point density is tabulated on a grid, from which the marginal distributions of the
intercept and the slope follow. They are compared to a Monte Carlo bootstrap.
"""

import matplotlib.pyplot as plt
import numpy as np

import spapprox as spa

# Small data set with skewed errors
rng = np.random.default_rng(7)
n = 20
xs = np.sort(rng.uniform(0, 10, n))
ys = 2 + 0.5 * xs + rng.exponential(1.5, n) - 1.5
Z = np.column_stack([np.ones(n), xs])
data = np.column_stack([Z, ys])


def psi(d, tau):
    z, y = d[:, :2], d[:, 2]
    return z * (y - z @ tau)[:, None]


def dpsi(d, tau):
    z = d[:, :2]
    return -z[:, :, None] * z[:, None, :]


def least_squares(W):
    # Weighted least squares for each row of weights
    A = np.einsum("bi,ij,ik->bjk", W, Z, Z)
    b = np.einsum("bi,ij,i->bj", W, Z, ys)
    return np.linalg.solve(A, b[..., None])[..., 0]


estimate = least_squares(np.ones((1, n)))[0]
sd = np.sqrt(np.diag(np.linalg.inv(Z.T @ Z)) * np.var(ys - Z @ estimate) * n / (n - 2))
bracket = np.column_stack([estimate - 6 * sd, estimate + 6 * sd])
print(f"estimate: intercept {estimate[0]:.3f}, slope {estimate[1]:.3f}")

schemes = {
    "multinomial": (spa.poisson(1), lambda: rng.multinomial(n, np.ones(n) / n, size=20000)),
    "Dirichlet": (spa.exponential(), lambda: rng.exponential(size=(20000, n))),
}
p = np.array([0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
results = {}
for name, (weights, draw) in schemes.items():
    print(f"\n{name} bootstrap")
    with spa.Timer("  Monte Carlo (20000 resamples)", decimals=3):
        mc = least_squares(draw())
    with spa.Timer("  Saddle point (fit pdf on a 51 x 51 grid)", decimals=3):
        ee = spa.MultivariateEstimatingEquationSaddlePointApprox(
            psi, data, weights, given=n, dpsi=dpsi, bracket=bracket
        )
        ee.fit_pdf(num=51)
    results[name] = mc, ee
    for j, label in enumerate(["intercept", "slope"]):
        print(f"  {label}: quantile {'MC':>8} {'SP':>8}")
        for pp, q_mc, q_sp in zip(p, np.quantile(mc[:, j], p), ee.marginal_ppf(j, p)):
            print(f"  {pp:>17.3f} {q_mc:8.3f} {q_sp:8.3f}")
    print(f"  mean MC {mc.mean(0).round(3)}, SP {ee.mean().round(3)}")
    print(f"  sd   MC {mc.std(0).round(3)}, SP {np.sqrt(np.diag(ee.cov())).round(3)}")
    corr = ee.cov()[0, 1] / np.sqrt(np.prod(np.diag(ee.cov())))
    print(f"  corr MC {np.corrcoef(mc.T)[0, 1]:.3f}, SP {corr:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (name, (mc, ee)) in zip(axes, results.items()):
    (a, b), f = ee._fitted()
    ax.hist2d(mc[:, 0], mc[:, 1], bins=60, range=bracket, cmap="Greys")
    ax.contour(a, b, f.T, levels=8, colors="C1")
    ax.set_title(f"{name} bootstrap: saddle point contours, Monte Carlo")
    ax.set_xlabel("intercept")
    ax.set_ylabel("slope")
plt.show()
