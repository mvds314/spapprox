#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-sample permutation test, Booth and Butler (1990), Example 2 and Table 1.

Pool the two samples into z_1, ..., z_n and select m of them as the first sample. With
independent Bernoulli(1/2) indicators Y_i, the permutation distribution of the sum of the
first sample, T2 = sum_i z_i Y_i, is its distribution conditional on T1 = sum_i Y_i = m. The
cgf of (T1, T2) follows from the existing algebra, and T1 does not induce a cut, so the
double saddle point approximation of Skovgaard (1987) is used.

Confidence intervals for a shift Delta (A ~ B + Delta) follow by test inversion: Delta is in
the interval if the permutation test of A - Delta against B does not reject at level 2.5% in
either direction.

The data lie on a lattice of span 0.1. The paper calls the continuity correction negligible
(less than 1e-4), but for the pain relief data it moves the one-sided significance level from
0.0486 to 0.0502, closer to the exact 0.0507: the atom at the observed sum is about 0.003. The
correction matters even more for rank statistics, such as the Wilcoxon rank sum test at the
end, on a lattice of span 1.

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
    pe, ps = exact(A, B)[0], saddle_point(A, B)[0]
    print(f"  two-sided significance level: exact {2 * pe:.3f}, saddle point {2 * ps:.3f}")
    print(f"  paper (Table 1): exact {paper['p'][0]:.3f}, saddle point {paper['p'][1]:.3f}")
    ce, cs = interval(exact, A, B), interval(saddle_point, A, B)
    print(
        f"  95% interval: exact ({ce[0]:.2f}, {ce[1]:.2f}), saddle point ({cs[0]:.2f}, {cs[1]:.2f})"
    )
    (pe0, pe1), (ps0, ps1) = paper["ci"]
    print(f"  paper (Table 1): exact ({pe0:.2f}, {pe1:.2f}), saddle point ({ps0:.2f}, {ps1:.2f})")

A, B, _ = data["pain relief"]
print("\npain relief, one-sided significance level, continuity correction for span 0.1:")
print(f"  exact {exact(A, B)[0]:.5f}")
print(f"  saddle point {saddle_point(A, B)[0]:.5f}, lattice {saddle_point(A, B, 0.1)[0]:.5f}")

# Wilcoxon rank sum test: the permutation test applied to the ranks, a lattice of span 1
ranks = np.argsort(np.argsort(np.r_[A, B])) + 1.0
RA, RB = ranks[: len(A)], ranks[len(A) :]
print(f"\nWilcoxon rank sum test, pain relief data, rank sum {RA.sum():.0f}:")
print(f"  one-sided significance level: exact {exact(RA, RB)[0]:.5f}")
print(f"  saddle point {saddle_point(RA, RB)[0]:.5f}, lattice {saddle_point(RA, RB, 1.0)[0]:.5f}")
