#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduces the examples in Section 5 of Kolassa (2003), "Multivariate saddlepoint tail
probability approximations", Ann. Statist. 31(1), 274-286.

1. Wang's (1990) correlated example: T = Z' X, with X a vector of three independent unit
   exponentials and Z = [[1, 0], [1, 1], [0, 1]]. For the sum of ten independent copies of T,
   the lower quadrant probabilities of Kolassa's Q(t) and Wang's W(t) are compared with
   Monte Carlo (Figure 1).
2. A hypothesis test in conditional logistic regression, on the endometrial cancer data of
   Stokes, Davis and Koch (1995) (Table 2). The data are 63 matched case-control pairs with
   three binary covariates; z_j is the case-minus-control covariate difference of a pair, with
   entries in {-1, 0, 1}. Conditioning on one case per pair removes the pair-specific
   intercepts. The model becomes a logistic regression on z with no intercept, and the
   sufficient statistic for the coefficient vector beta is T = sum_j z_j B_j over the pairs.
   Here B_j indicates which member of the pair is the case. Under H0: beta = 0 each B_j is
   Bernoulli(1/2), so the null distribution of T is a known discrete distribution on integer
   vectors.

   The test rejects H0 on the critical region {T1 >= 10 or T2 >= 10 or T3 >= 13}. Its size
   P_H0(critical region) is computed. Enumerating T exactly quickly becomes infeasible as the
   number of pairs grows, and the normal approximation is poor this far in the tail. The
   example therefore bounds the size with Boole's inequality applied to univariate saddlepoint
   tail probabilities P_H0(T_j >= c_j) at the critical values. For these data the exact
   distribution is still small enough to check against.
"""

import itertools

import numpy as np

import matplotlib.pyplot as plt

import spapprox as spa
from spapprox import MultivariateCumulantGeneratingFunction

plt.close("all")

# --- Example 1: Wang's correlated bivariate gamma example (Figure 1) ----------------------
n = 10
# K(tau) = -sum_j log(1 - z_j tau) with z_j the rows of Z, i.e. a bivariate gamma with
# shape (1, 1, 1). The sum of n independent copies has shape (n, n, n).
cgf = spa.bivariate_gamma(a=[n, n, n])
approx = spa.BivariateSaddlePointApprox(cgf)

# Monte Carlo reference for P[T <= t], one million samples as in the paper
with spa.Timer("Monte Carlo (10^6 samples)", decimals=3):
    rng = np.random.default_rng(2003)
    u = rng.exponential(size=(10**6, 3, n)).sum(axis=-1)
    sim = np.column_stack([u[:, 0] + u[:, 1], u[:, 1] + u[:, 2]])

sd = np.sqrt(2 * n)
grid = n * 2 + sd * np.linspace(-2.5, 2.5, 11)
# Points (t1, t2) on the grid, with t2 varying over the rows
pts = np.array([[t1, t2] for t2 in grid for t1 in grid])
shape = (grid.size, grid.size)
with spa.Timer(f"Monte Carlo cdf ({len(pts)} grid points)", decimals=3):
    exact = np.array([np.mean((sim[:, 0] <= t1) & (sim[:, 1] <= t2)) for t1, t2 in pts])
with spa.Timer("Solving the saddle point equations", decimals=3):
    ts = [cgf.dK_inv(pt) for pt in pts]
with spa.Timer("Saddle point cdf, Kolassa backend", decimals=3):
    F_q = np.array([approx.cdf(t=t, backend="Kolassa") for t in ts])
with spa.Timer("Saddle point cdf, Wang backend", decimals=3):
    F_w = np.array([approx.cdf(t=t, backend="Wang") for t in ts])
err_q, err_w = np.abs(F_q - exact), np.abs(F_w - exact)
ratio = (err_q / (err_q + err_w)).reshape(shape)

fig, ax = plt.subplots(facecolor="w")
cs = ax.contourf(grid, grid, ratio, levels=np.linspace(0, 1, 11), cmap="viridis")
fig.colorbar(cs, label="|err Q| / (|err Q| + |err W|)")
ax.set_xlabel("t1")
ax.set_ylabel("t2")
ax.set_title("Kolassa (2003) Figure 1: lower quadrant probabilities")

# --- Example 2: conditional logistic regression test, endometrial cancer data (Table 2) ----
# Rows of Z: the distinct case-minus-control differences in gall bladder disease,
# hypertension and nonestrogen drug use; m: the number of matched pairs with that difference
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
# Observed sufficient statistic: in the data every B_j = 1 (the case is the first member)
observed = m @ Z  # (9, 6, 13)
# Reject H0: beta = 0 if any T_j reaches its critical value
critical = np.array([10, 10, 13])


# Null CGF of T: under beta = 0, the m_j pairs with difference z_j contribute z_j N_j, with
# N_j ~ binomial(m_j, 1/2) independent, so T = Z' N
cgf_logistic = MultivariateCumulantGeneratingFunction.from_univariate(
    *[spa.binomial(n=mj, p=0.5) for mj in m]
).ldot(Z.T)


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


def lattice_upper_tail(cgf, c, correction):
    # P[T >= c] = 1 - P[T <= c - 1] on a lattice with span one, first or second continuity
    # correction (Daniels, 1987; Butler, 2007, section 1.2)
    return 1 - spa.UnivariateSaddlePointApprox(cgf).cdf(x=c - 1, delta=1, correction=correction)


with spa.Timer("\nExact null distribution (convolution)", decimals=3):
    pmf = exact_distribution()
    keys = np.array(list(pmf))
    probs = np.array(list(pmf.values()))
with spa.Timer("Saddle point tail probabilities", decimals=3):
    sp = {
        k: [lattice_upper_tail(cgf_logistic[j], critical[j], k) for j in range(3)] for k in (1, 2)
    }

print("Observed T:", observed)
print("Null tail probabilities P_H0[T_j >= c_j] (lattice Lugannani-Rice, 1st/2nd correction):")
for j in range(3):
    exact = probs[keys[:, j] >= critical[j]].sum()
    print(
        f"  T{j + 1} >= {critical[j]:2d}: saddlepoint {sp[1][j]:.5f} / {sp[2][j]:.5f}, "
        f"exact {exact:.5f}"
    )
in_region = (keys >= critical).any(axis=1)
print(f"Size of the test, Bonferroni bound: {sum(sp[1]):.5f} / {sum(sp[2]):.5f}")


def lattice_upper_orthant(A):
    # P[T_A >= c_A] = P[-T_A <= -c_A], Kolassa (2003) eq. (13) with continuity correction
    A = list(A)
    neg = spa.MultivariateSaddlePointApprox(cgf_logistic.ldot(-np.eye(3)[A]))
    return neg.cdf(x=-critical[A].astype(float), delta=1)


# Boole's law, i.e., inclusion-exclusion over the joint upper orthants
subsets = [A for r in (1, 2, 3) for A in itertools.combinations(range(3), r)]
with spa.Timer("\nSaddle point joint tail probabilities", decimals=3):
    joint = {A: lattice_upper_orthant(A) for A in subsets}
boole = 0.0
for A, p in joint.items():
    exact = probs[(keys[:, list(A)] >= critical[list(A)]).all(axis=1)].sum()
    print(f"  P[T_j >= c_j, j in {A}]: saddlepoint {p:.6f}, exact {exact:.6f}")
    boole += (-1) ** (len(A) + 1) * p
print(f"Size of the test, Boole's law (lattice Kolassa (2003)): {boole:.5f}")
print(f"Size of the test, exact P_H0[critical region]: {probs[in_region].sum():.5f}")
print("Kolassa (2003) reports 0.01709 by Boole's law; Monte Carlo 95% CI (0.01702, 0.01718)")

plt.show()
