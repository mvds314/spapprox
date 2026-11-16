#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduces the examples in Section 5 of Kolassa (2003), "Multivariate saddlepoint tail
probability approximations", Ann. Statist. 31(1), 274-286.

1. Wang's (1990) correlated example: T = Z' X, with X a vector of three independent unit
   exponentials and Z = [[1, 0], [1, 1], [0, 1]]. For the sum of ten independent copies of T,
   the lower quadrant probabilities of Kolassa's Q(t) and Wang's W(t) are compared with
   Monte Carlo (Figure 1).
2. The endometrial cancer data of Stokes, Davis and Koch (1995) (Table 2): the probability of
   the critical region {T1 >= 10 or T2 >= 10 or T3 >= 13} under the null hypothesis,
   approximated with Boole's inequality applied to univariate saddlepoint tail probabilities.
"""

import numpy as np
import scipy.stats as sps

import matplotlib.pyplot as plt

import spapprox as spa
from spapprox import Domain, MultivariateCumulantGeneratingFunction

plt.close("all")

# --- Example 1: Wang's correlated bivariate gamma example (Figure 1) ----------------------
n = 10
# K(tau) = -sum_j log(1 - z_j tau) with z_j the rows of Z, i.e. a bivariate gamma with
# shape (1, 1, 1). The sum of n independent copies has shape (n, n, n).
cgf = spa.bivariate_gamma(a=[n, n, n])
approx = spa.BivariateSaddlePointApprox(cgf)

# Monte Carlo reference for P[T <= t], one million samples as in the paper
rng = np.random.default_rng(2003)
u = rng.exponential(size=(10**6, 3, n)).sum(axis=-1)
sim = np.column_stack([u[:, 0] + u[:, 1], u[:, 1] + u[:, 2]])

sd = np.sqrt(2 * n)
grid = n * 2 + sd * np.linspace(-2.5, 2.5, 11)
ratio = np.full((grid.size, grid.size), np.nan)
for i, t2 in enumerate(grid):
    for j, t1 in enumerate(grid):
        pt = np.array([t1, t2])
        t = cgf.dK_inv(pt)
        exact = np.mean((sim[:, 0] <= t1) & (sim[:, 1] <= t2))
        err_q = abs(approx.cdf(t=t, backend="Kolassa") - exact)
        err_w = abs(approx.cdf(t=t, backend="Wang") - exact)
        ratio[i, j] = err_q / (err_q + err_w)

fig, ax = plt.subplots(facecolor="w")
cs = ax.contourf(grid, grid, ratio, levels=np.linspace(0, 1, 11), cmap="viridis")
fig.colorbar(cs, label="|err Q| / (|err Q| + |err W|)")
ax.set_xlabel("t1")
ax.set_ylabel("t2")
ax.set_title("Kolassa (2003) Figure 1: lower quadrant probabilities")

# --- Example 2: endometrial cancer data (Table 2) -----------------------------------------
# Columns of Table 2: gall bladder disease, hypertension, nonestrogen drug use, number of pairs
Z = np.array(
    [
        [-1, -1, 0],
        [-1, 0, -1],
        [-1, 1, 0],
        [0, -1, -1],
        [0, -1, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, -1, 0],
        [1, -1, 1],
        [1, 0, -1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
)
m = np.array([1, 1, 1, 2, 6, 14, 10, 12, 4, 3, 1, 1, 4, 1, 1, 1])
observed = m @ Z  # (9, 6, 13)
critical = np.array([10, 10, 13])


def sigmoid(u):
    return 1 / (1 + np.exp(-u))


# K(tau) = sum_j m_j [log(1 + exp(z_j tau)) - log(2)], the null CGF of T = sum_j m_j z_j B_j
def K(t):
    return m @ (np.logaddexp(0, Z @ np.asarray(t)) - np.log(2))


def dK(t):
    return (m * sigmoid(Z @ np.asarray(t))) @ Z


def d2K(t):
    s = sigmoid(Z @ np.asarray(t))
    return np.einsum("j,ja,jb->ab", m * s * (1 - s), Z, Z)


def d3K(t):
    s = sigmoid(Z @ np.asarray(t))
    return np.einsum("j,ja,jb,jc->abc", m * s * (1 - s) * (1 - 2 * s), Z, Z, Z)


K_vec = np.vectorize(K, signature="(3)->()")
cgf_logistic = MultivariateCumulantGeneratingFunction(
    K=K_vec,
    dK=lambda t: np.apply_along_axis(dK, -1, t),
    d2K=np.vectorize(d2K, signature="(3)->(3,3)"),
    d3K=np.vectorize(d3K, signature="(3)->(3,3,3)"),
    dim=3,
    domain=Domain(dim=3),
)


def exact_distribution():
    # T is supported on integer vectors, a convolution of independent +-z_j Bernoulli(1/2)
    pmf = {(0, 0, 0): 1.0}
    for z, mj in zip(Z, m):
        for _ in range(mj):
            new = {}
            for key, p in pmf.items():
                for step in (np.zeros(3, dtype=int), z):
                    k = tuple(np.add(key, step))
                    new[k] = new.get(k, 0) + p / 2
            pmf = new
    return pmf


def lattice_upper_tail(cgf, c):
    # Lugannani-Rice for P[T >= c] on a lattice with span one (Daniels, 1987)
    tau = float(cgf.dK_inv(c))
    w = np.sign(tau) * np.sqrt(2 * (tau * c - cgf.K(tau)))
    v = (1 - np.exp(-tau)) * np.sqrt(cgf.d2K(tau))
    return sps.norm.sf(w) + sps.norm.pdf(w) * (1 / v - 1 / w)


pmf = exact_distribution()
keys = np.array(list(pmf))
probs = np.array(list(pmf.values()))

print("Observed T:", observed)
print("Marginal tail probabilities P[T_j >= c_j] (lattice Lugannani-Rice):")
boole = 0.0
for j in range(3):
    sp = lattice_upper_tail(cgf_logistic[j], critical[j])
    exact = probs[keys[:, j] >= critical[j]].sum()
    boole += sp
    print(f"  T{j + 1} >= {critical[j]:2d}: saddlepoint {sp:.5f}, exact {exact:.5f}")
in_region = (keys >= critical).any(axis=1)
print(f"Boole's bound: {boole:.5f}")
print(f"Exact P[critical region]: {probs[in_region].sum():.5f}")
# Note: our Boole sum (about 0.01756) does not match the 0.01709 reported in the paper, while the
# exact probability (0.01713) lies within the paper's Monte Carlo interval. It is unclear how the
# paper computed its Boole figure, so the discrepancy is not resolved here.
print("Kolassa (2003) reports 0.01709 by Boole's law; Monte Carlo 95% CI (0.01702, 0.01718)")

plt.show()
