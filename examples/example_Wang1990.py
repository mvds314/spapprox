#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduces the example in Section 5 of Wang (1990), "Saddlepoint approximations for
bivariate distributions", J. Appl. Probab. 27, 586-597 (Table 1).

Let U0, U1, U2 be independent unit exponentials and put (X, Y) = (U0 + U1 - 2, U0 + U2 - 2).
The saddlepoint approximation is used for the cdf of the mean of n = 10 copies of (X, Y).
Wang's 'exact' values are Monte Carlo estimates based on 10^6 samples; we do the same.
"""

import numpy as np
import scipy.stats as sps

import spapprox as spa

n = 10
xs = np.array([-1.2, -1.0, -0.8, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
ys = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -0.8, -1.0, -1.2])

# The sum of n independent copies of (U0+U1, U0+U2) is bivariate gamma with shape (n, n, n),
# so the mean of (X, Y) is (S - 2n) / n.
cgf = spa.bivariate_gamma(a=[n, n, n], loc=-2 * n) / n
approx = spa.BivariateSaddlePointApprox(cgf)

# Normal approximation: same mean and covariance as the mean of the sample
cov = np.array([[2, 1], [1, 2]]) / n
normal = sps.multivariate_normal(mean=[0, 0], cov=cov)

# Monte Carlo reference, as in the paper
rng = np.random.default_rng(1990)
u = rng.exponential(size=(10**6, n, 3))
mean_x = u[..., [0, 1]].sum(axis=-1).mean(axis=1) - 2
mean_y = u[..., [0, 2]].sum(axis=-1).mean(axis=1) - 2
sim = np.column_stack([mean_x, mean_y])


def monte_carlo(x, y):
    return np.mean((sim[:, 0] <= x) & (sim[:, 1] <= y))


# Table 1 of Wang (1990): F(x, y) = F(y, x), so only the cells with x <= y are given.
print(f"{'y':>5} {'x':>5} {'MC':>9} {'Wang':>9} {'KL':>9} {'Kolassa':>9} {'normal':>9}")
for y in ys:
    for x in xs[xs <= y]:
        pt = np.array([x, y])
        t = cgf.dK_inv(pt)  # solve the saddlepoint equation K'(t) = (x, y)
        row = [
            monte_carlo(x, y),
            approx.cdf(t=t, backend="Wang"),
            approx.cdf(t=t, backend="KL"),
            approx.cdf(t=t, backend="Kolassa"),
            normal.cdf(pt),
        ]
        print(f"{y:5.1f} {x:5.1f} " + " ".join(f"{v:9.5f}" for v in row))
