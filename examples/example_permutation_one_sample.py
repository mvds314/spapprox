#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-sample permutation (sign-flip) test, Booth and Butler (1990), Example 2.

Under the null hypothesis of symmetry about zero, the statistic S = sum_i d_i Z_i, with
independent signs Z_i = +-1 with probability 1/2, has cgf sum_i log cosh(t d_i). No
conditioning is involved, so this is a plain linear combination of independent variables,
built from the existing cgf algebra, and approximated with Lugannani-Rice, as in Davison and
Hinkley (1988). The data are Darwin's Zea mays differences (Fisher, 1935), for which the exact
one-sided significance level is 863 / 32768.
"""

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

import spapprox as spa

d = np.array([49, -67, 8, 16, 6, 23, 28, 41, 14, 29, 56, 24, 75, 60, -48.0])
n = len(d)

sign = 2 * spa.binomial(1, 0.5) - 1
cgf = spa.MultivariateCumulantGeneratingFunction.from_univariate(*[sign] * n).ldot(d)
approx = spa.UnivariateSaddlePointApprox(cgf)

# Exact permutation distribution, by enumeration of all 2^n sign patterns
exact = np.array(list(product([-1, 1], repeat=n))) @ d

obs = d.sum()
print(f"P(S >= observed): exact {np.mean(exact >= obs):.5f} (863/32768 = {863 / 32768:.5f})")
print(f"                  saddle point {1 - approx.cdf(x=obs - 1e-9):.5f}")
print("The statistic lies on a lattice of span 2 (all d_i are integers), the saddle point")
print("approximation is not continuity corrected.")

s = np.linspace(-300, 300, 13)
print(f"\n{'s':>6} {'exact':>8} {'SP':>8}")
for si, e, a in zip(s, [np.mean(exact <= si) for si in s], approx.cdf(x=s)):
    print(f"{si:6.0f} {e:8.5f} {a:8.5f}")

s = np.linspace(-400, 400, 200)
fig, ax = plt.subplots()
ax.step(np.sort(exact), np.arange(1, len(exact) + 1) / len(exact), label="exact")
ax.plot(s, approx.cdf(x=s), label="saddle point")
ax.set_xlabel("S")
ax.legend()
plt.show()
