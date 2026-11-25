#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-sample permutation test, Booth and Butler (1990), Example 2 and Table 1.

The test
--------
Two independent samples A (size m) and B, e.g. scores of two treatment groups. Is A shifted
upwards relative to B? We know which observation belongs to which sample. Suppose the group
labels have no effect. Then each observation would have the same value in either group, so
which m of the n pooled observations z_1, ..., z_n ended up in A was arbitrary: all
C(n, m) choices are equally likely. The p-value is the fraction of choices whose sum is at
least the observed sum of A. For the pain relief data, with 8 + 8 observations, this means
C(16, 8) = 12870 sums, and the count grows quickly with n.

The saddle point approximation
------------------------------
Unlike in the sign-flip test of ``example_permutation_one_sample.py``, the choices are not
independent: once m observations are in A, the rest must be in B. The trick is to start from
independent pieces anyway. Give each observation an independent fair coin
Y_i ~ Bernoulli(1/2), meaning "z_i is in A", and track two sums:

- T1 = sum_i Y_i, the size of A, and
- T2 = sum_i z_i Y_i, the sum of A.

Every set of m observations is equally likely to be the one with Y_i = 1, so conditional
on T1 = m, T2 has exactly the permutation distribution. The p-value is
P(T2 >= observed sum | T1 = m).

The pair (T1, T2) is a sum of independent vectors (1, z_i) Y_i, so its cgf follows from the
cgf algebra: ``from_univariate`` stacks n independent coins, and ``ldot`` with the rows
(1, ..., 1) and (z_1, ..., z_n) gives (T1, T2). The conditional probability needs the
conditional approximation, the double saddle point approximation of Skovgaard (1987),
``BivariateSaddlePointApprox.conditional_cdf``. It solves one saddle point equation for
(T1, T2) jointly, and one for T1 alone, and combines them. There is no shortcut: every Y_i
enters both T1 and T2, so T2 has no part independent of T1 (T1 does not induce a cut).

Confidence interval
-------------------
To estimate the size of the effect, assume A ~ B + Delta, i.e. A is B shifted by Delta. A
value Delta is plausible if the test of A - Delta against B does not reject at level 2.5% in
either direction. The 95% interval is the set of plausible values, and its endpoints are
found by root finding.

Continuity correction
---------------------
The data have one decimal, so T2 takes values on a grid with step 0.1. The paper calls the
continuity correction negligible (less than 1e-4), but for the pain relief data it moves the
one-sided p-value from 0.0486 to 0.0502, closer to the exact 0.0507: the probability of the
observed sum itself is about 0.003. The correction matters even more for rank statistics,
such as the Wilcoxon rank sum test at the end: the same test applied to the ranks
1, ..., n, a grid with step 1.

Note: our exact significance level for the analgesia data (0.044 one-sided) does not match
the 0.012 of the paper (reproduced from Robinson, 1982), probably due to a transcription
error in the data; the pain relief data do match.
"""

from itertools import combinations

import numpy as np
import scipy.optimize as spo

import spapprox as spa

data = {
    "pain relief": (
        [6.8, 3.1, 5.8, 4.5, 3.3, 4.7, 4.2, 4.9],
        [4.4, 2.5, 2.8, 2.1, 6.6, 0.0, 4.8, 2.3],
        {"p": (0.102, 0.097), "ci": ((-0.30, 3.26), (-0.31, 3.25))},
    ),
    "analgesia": (
        [17.9, 13.3, 10.6, 7.6, 5.7, 5.6, 5.4, 3.3, 3.1, 0.9],
        [7.7, 5.0, 1.7, 0.0, -3.0, -3.1, 10.5],
        {"p": (0.012, 0.011), "ci": ((1.88, 13.52), (1.92, 13.56))},
    ),
}


def saddle_point(A, B, delta=None):
    """
    Return P(T2 >= sum(A)) and P(T2 <= sum(A)) under the permutation distribution, continuity
    corrected for a lattice of span delta if given.
    """
    z = np.r_[A, B]
    n = len(z)
    Y = spa.MultivariateCumulantGeneratingFunction.from_univariate(*[spa.binomial(1, 0.5)] * n)
    approx = spa.BivariateSaddlePointApprox(Y.ldot(np.vstack([np.ones(n), z])))
    obs = np.sum(A)
    if delta is None:
        F = approx.conditional_cdf(x=obs, given=[len(A)], i=1)
        return 1 - F, F
    F = approx.conditional_cdf(x=[obs - delta, obs], given=[len(A)], i=1, delta=delta)
    return 1 - F[0], F[1]


def exact(A, B):
    z = np.r_[A, B]
    s = np.array([z[list(c)].sum() for c in combinations(range(len(z)), len(A))])
    obs = np.sum(A)
    return np.mean(s >= obs - 1e-9), np.mean(s <= obs + 1e-9)


def interval(pvalues, A, B, level=0.95):
    alpha = (1 - level) / 2
    A, B = np.asarray(A), np.asarray(B)
    est = A.mean() - B.mean()
    step = np.std(np.r_[A, B]) * np.sqrt(1 / len(A) + 1 / len(B))

    def bound(side, sign):
        # Walk away from the estimate in standard errors until the test rejects
        def f(d):
            return pvalues(A - d, B)[side] - alpha

        k = 1
        while not f(est + sign * k * step) < 0:
            k += 1
        return spo.brentq(f, est + sign * (k - 1) * step, est + sign * k * step)

    return bound(0, -1), bound(1, 1)


for name, (A, B, paper) in data.items():
    print(f"{name}:")
    with spa.Timer("  Exact significance level and interval (enumeration)", decimals=3):
        pe, ce = exact(A, B)[0], interval(exact, A, B)
    with spa.Timer("  Saddle point significance level and interval", decimals=3):
        ps, cs = saddle_point(A, B)[0], interval(saddle_point, A, B)
    print(f"  two-sided significance level: exact {2 * pe:.3f}, saddle point {2 * ps:.3f}")
    print(f"  paper (Table 1): exact {paper['p'][0]:.3f}, saddle point {paper['p'][1]:.3f}")
    print(
        f"  95% interval: exact ({ce[0]:.2f}, {ce[1]:.2f}), saddle point ({cs[0]:.2f}, {cs[1]:.2f})"
    )
    (pe0, pe1), (ps0, ps1) = paper["ci"]
    print(f"  paper (Table 1): exact ({pe0:.2f}, {pe1:.2f}), saddle point ({ps0:.2f}, {ps1:.2f})")

A, B, _ = data["pain relief"]
print("\npain relief, one-sided significance level, continuity correction for span 0.1:")
with spa.Timer("  Exact", decimals=3):
    pe = exact(A, B)[0]
with spa.Timer("  Saddle point", decimals=3):
    ps = saddle_point(A, B)[0]
with spa.Timer("  Lattice saddle point", decimals=3):
    pl = saddle_point(A, B, 0.1)[0]
print(f"  exact {pe:.5f}, saddle point {ps:.5f}, lattice {pl:.5f}")

# Wilcoxon rank sum test: the permutation test applied to the ranks, a lattice of span 1
ranks = np.argsort(np.argsort(np.r_[A, B])) + 1.0
RA, RB = ranks[: len(A)], ranks[len(A) :]
print(f"\nWilcoxon rank sum test, pain relief data, rank sum {RA.sum():.0f}:")
print(f"  one-sided significance level: exact {exact(RA, RB)[0]:.5f}")
print(f"  saddle point {saddle_point(RA, RB)[0]:.5f}, lattice {saddle_point(RA, RB, 1.0)[0]:.5f}")
