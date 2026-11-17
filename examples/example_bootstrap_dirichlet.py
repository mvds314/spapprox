#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dirichlet (Bayesian) bootstrap of the mean, Booth and Butler (1990), Example 4 and Table 4.

The Dirichlet bootstrap, Rubin (1981), replaces the multinomial resampling weights by weights
D uniformly distributed on the simplex. With independent unit exponentials W_i,
D = W / sum_i W_i, so sum_i D_i x_i solves sum_i (x_i - tau) W_i = 0, conditional on
sum_i W_i = n. Unlike for the multinomial bootstrap, the conditioning does not induce a cut,
and the double saddle point (Skovgaard) approximation is needed.
"""

import matplotlib.pyplot as plt
import numpy as np

import spapprox as spa

x = np.array([9.6, 10.4, 13.0, 15.0, 16.6, 17.2, 17.3, 21.8, 24.0, 33.8])
n = len(x)

dirichlet = spa.EstimatingEquationSaddlePointApprox.from_resampling(
    lambda x, tau: x - tau, x, spa.exponential(), given=n, bracket=(x.min(), x.max())
)
dirichlet.fit_cdf(num=200)
multinomial = spa.EstimatingEquationSaddlePointApprox.from_resampling(
    lambda x, tau: x - tau, x, spa.poisson(1), given=n, bracket=(x.min(), x.max())
)
multinomial.fit_cdf(num=200)

# Monte Carlo reference
rng = np.random.default_rng(1981)
sim = rng.dirichlet(np.ones(n), size=10**6) @ x

# Table 4, Dirichlet bootstrap columns: percentage points of sum_i D_i x_i - x_bar
p = np.array([1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.2, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999])
paper_exact = [-5.94, -5.46, -5.16, -4.42, -4.07, -3.01, -2.45, -1.72, 1.62, 2.73, 3.73, 5.75]
paper_exact += [6.50, 7.99]
paper_spa = [-6.15, -5.51, -5.21, -4.44, -4.07, -3.04, -2.47, -1.74, 1.61, 2.72, 3.72, 5.73]
paper_spa += [6.51, 8.27]

q = np.asarray(dirichlet.ppf(p)) - x.mean()
qm = np.asarray(multinomial.ppf(p)) - x.mean()
print(f"{'p':>7} {'MC':>7} {'paper':>7} {'paper SP':>8} {'SP':>7} {'multinom':>8}")
for row in zip(p, np.quantile(sim, p) - x.mean(), paper_exact, paper_spa, q, qm):
    print(f"{row[0]:7.4f} " + " ".join(f"{v:7.2f}" for v in row[1:]))

# The Dirichlet weights have mean 1/n and variance (n - 1) / (n^2 (n + 1))
print(f"\nmean {dirichlet.mean():.4f} (exact {x.mean():.4f})")
print(f"var {dirichlet.var():.4f} (exact {x.var() / (n + 1):.4f})")

tau = np.linspace(12, 28, 100)
fig, ax = plt.subplots()
ax.hist(sim, bins=200, density=True, alpha=0.4, label="Monte Carlo")
ax.plot(tau, dirichlet.pdf(tau), label="Dirichlet, saddle point")
ax.plot(tau, multinomial.pdf(tau), "--", label="multinomial, saddle point")
ax.set_xlabel("sum D_i x_i")
ax.legend()
plt.show()
