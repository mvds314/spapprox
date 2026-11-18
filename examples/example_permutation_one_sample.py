#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-sample permutation (sign-flip) test, Booth and Butler (1990), Example 2.

Under the null hypothesis of symmetry about zero, the statistic S = sum_i d_i Z_i, with
independent signs Z_i = +-1 with probability 1/2, has cgf sum_i log cosh(t d_i). No
conditioning is involved, so this is a plain linear combination of independent variables,
built from the existing cgf algebra, and approximated with Lugannani-Rice, as in Davison and
Hinkley (1988). As all d_i are integers, S lies on a lattice of span 2, so the continuity
corrected version (``delta=2``, Daniels, 1987) is shown as well. The data are Darwin's Zea mays differences (Fisher, 1935), for which the exact
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

obs = d.sum()
s = np.linspace(-300, 300, 13) + obs % 2

# Exact permutation distribution, by enumeration of all 2^n sign patterns
with spa.Timer(f"Exact (enumeration of 2^{n} sign patterns)", decimals=3):
    exact = np.array(list(product([-1, 1], repeat=n))) @ d
    p_exact = np.mean(exact >= obs)
    F_exact = [np.mean(exact <= si) for si in s]
with spa.Timer("Saddle point", decimals=3):
    p_sp = 1 - approx.cdf(x=obs - 1e-9)
    F_sp = approx.cdf(x=s)
with spa.Timer("Lattice saddle point", decimals=3):
    p_lattice = 1 - approx.cdf(x=obs - 2, delta=2)
    F_lattice = approx.cdf(x=s, delta=2)

print(f"\nP(S >= observed): exact {p_exact:.5f} (863/32768 = {863 / 32768:.5f})")
print(f"                  saddle point {p_sp:.5f}")
print(f"                  lattice saddle point {p_lattice:.5f}")

print(f"\n{'s':>6} {'exact':>8} {'SP':>8} {'lattice':>8}")
for si, e, a, al in zip(s, F_exact, F_sp, F_lattice):
    print(f"{si:6.0f} {e:8.5f} {a:8.5f} {al:8.5f}")

s = np.linspace(-400, 400, 200)
fig, ax = plt.subplots()
ax.step(np.sort(exact), np.arange(1, len(exact) + 1) / len(exact), label="exact")
ax.plot(s, approx.cdf(x=s), label="saddle point")
ax.plot(s, approx.cdf(x=s, delta=2), "--", label="lattice saddle point")
ax.set_xlabel("S")
ax.legend()
plt.show()
