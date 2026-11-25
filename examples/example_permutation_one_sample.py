#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-sample permutation (sign-flip) test, Booth and Butler (1990), Example 2.

The data
--------
Darwin's Zea mays experiment (Fisher, 1935): 15 pairs of plants, each pair grown in the same
pot, one plant cross-fertilised and one self-fertilised. For each pair we know which plant is
which, and d_i = height(cross) - height(self). The observed total is sum_i d_i = 314. Is that
evidence that cross-fertilisation makes plants taller?

The test
--------
Suppose fertilisation has no effect. Then each plant would have reached the same height under
the other label, and swapping the labels within pair i merely flips the sign of d_i. So under
this null hypothesis, all 2^15 sign patterns +-d_1, ..., +-d_15 are equally likely, and the
p-value is the fraction of them whose sum is at least the observed 314. Enumerating them
gives exactly 863 / 32768 = 0.0263. With n pairs this takes 2^n sums, which is infeasible
for larger n.

The saddle point approximation
------------------------------
Under the null hypothesis, the sum is S = sum_i d_i Z_i, with independent random signs
Z_i = +-1, each with probability 1/2. So S is a sum of independent variables, and its cgf is
the sum of theirs: K(t) = sum_i log cosh(t d_i). The code builds it from the cgf algebra: a
+-1 variable is 2 * Bernoulli(1/2) - 1, ``from_univariate`` stacks 15 independent copies
into a vector, and ``ldot(d)`` takes the weighted sum. The ordinary univariate approximation
(Lugannani-Rice, as in Davison and Hinkley, 1988) then gives P(S >= 314) without
enumeration. No conditioning is needed, unlike in the two-sample test of
``example_permutation_two_sample.py``.

S is discrete: all d_i are integers, and flipping a sign changes S by 2 d_i, so S takes
values on a grid with step 2. The plain approximation treats S as continuous and slightly
underestimates the tail. Passing the step, ``delta=2``, applies the continuity correction
(Daniels, 1987), which removes most of that error.

The output compares exact, saddle point and continuity corrected p-values and cdf values,
and plots the three cdfs.
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
