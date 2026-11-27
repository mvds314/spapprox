#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduces examples of Barndorff-Nielsen and Cox (1979), "Edgeworth and saddle-point
approximations with statistical applications", J. R. Statist. Soc. B 41(3), 279-312.

1. Inverse Gaussian (Section 3.4). The saddle point density of the inverse Gaussian
   distribution is exact up to its normalizing constant (Tweedie, 1957), so the normalized
   approximation coincides with the exact density.
2. Binomial-logistic bioassay (Section 5.5, Table 2). At each of the doses d = -1, 0, 1,
   n = 16 individuals are treated, and a_d of them respond, with a_d ~ binomial(n, p_d) and
   logistic response probability p_d = 1 / (1 + exp(lambda + psi d)). The sufficient
   statistics are r = a_{-1} + a_0 + a_1, the total number responding, and
   s = a_1 - a_{-1}. Inference on the slope psi is done conditionally on r, which removes the
   intercept lambda. The example compares the exact conditional distribution of s given r,
   for psi = 1, with the double saddle point approximation (Butler, 2007, Section 4.1), with
   and without renormalization. The paper's s runs in the opposite direction of the dose, so
   it is -(a_1 - a_{-1}) in our construction. On the boundary of the support, s = 16, the
   approximation is undefined and returns NaN.
3. Conditional likelihood for Fisher's twins data (Section 6.2, Table 3, Figure 1). Of 17
   dizygotic twins of criminals, 2 were convicted; of 13 monozygotic twins, 10 were. The
   parameter of interest is the log odds ratio psi of conviction, monozygotic versus
   dizygotic. Conditioning on the total number convicted, 12, removes the nuisance
   parameter. The exact conditional likelihood is Fisher's noncentral hypergeometric
   distribution. Its double saddle point approximation follows from two independent
   binomials, as the conditional density of the dizygotic count given the total.
"""

import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
from scipy.special import expit

import matplotlib.pyplot as plt

import spapprox as spa
from spapprox import MultivariateCumulantGeneratingFunction

plt.close("all")

# --- Example 1: the inverse Gaussian saddle point density is exact ----------------------------
# The standard form pi^(-1/2) u^(-3/2) exp(2 - u - 1/u) of the paper, (3.21)
cgf = spa.inverse_gaussian(mu=0.5, scale=2)
approx = spa.UnivariateSaddlePointApprox(cgf)
t = np.linspace(-20, 0.24, 100)
x = cgf.dK(t)
exact = sps.invgauss(0.5, scale=2).pdf(x)
print(
    "Inverse Gaussian: max relative error of the normalized saddle point pdf "
    f"{np.max(np.abs(approx.pdf(t=t) / exact - 1)):.1e}"
)
fig, ax = plt.subplots(facecolor="w")
ax.plot(x, exact, label="Exact")
ax.plot(x, approx.pdf(t=t, normalize_pdf=False), ":", label="Saddle point")
ax.plot(x, approx.pdf(t=t), "--", label="Saddle point, normalized")
ax.set_xlim(0, 6)
ax.set_xlabel("u")
ax.set_title("Inverse Gaussian density (3.21)")
ax.legend()

# --- Example 2: binomial-logistic bioassay (Table 2) -----------------------------------------
n, psi = 16, 1.0
doses = np.array([-1, 0, 1])
p = 1 / (1 + np.exp(psi * doses))  # lambda = 0, it drops out by conditioning on r
# (r, a_1 - a_{-1}) as linear combination of the independent counts (a_{-1}, a_0, a_1)
cgf_bioassay = MultivariateCumulantGeneratingFunction.from_univariate(
    *[spa.binomial(n, pd) for pd in p]
).ldot([[1, 1, 1], [-1, 0, 1]])
approx_bioassay = spa.BivariateSaddlePointApprox(cgf_bioassay)

# Table 2 of the paper: the exact values of p(s; 1 | r) for s = 0, ..., 16, and the saddle
# point values for s = 0, ..., 15
paper = {
    24: [
        [0.002742, 0.007025, 0.015973, 0.032198, 0.057411, 0.090251, 0.12452, 0.14990],
        [0.15623, 0.13953, 0.10537, 0.066061, 0.033525, 0.013263, 0.003852, 0.000735, 0.000070],
        [0.002797, 0.007166, 0.016299, 0.032870, 0.058645, 0.092268, 0.12745, 0.15363],
        [0.16040, 0.14358, 0.10873, 0.068435, 0.034915, 0.013925, 0.004103, 0.000814],
    ],
    32: [
        [0.004555, 0.011588, 0.025804, 0.050189, 0.084967, 0.12454, 0.15687, 0.16808],
        [0.15106, 0.11171, 0.066127, 0.030086, 0.009873, 0.002099, 0.000243, 0.000012, 0.0],
        [0.004655, 0.011844, 0.026382, 0.051342, 0.086992, 0.12765, 0.16101, 0.17282],
        [0.15565, 0.11537, 0.068461, 0.031221, 0.010271, 0.002203, 0.000262, 0.000013],
    ],
}


def exact_conditional(r):
    # The joint pmf of (a_{-1}, a_0, a_1) on the grid, collected by s in the paper's sign
    pmf = np.zeros(2 * n + 1)
    a = np.arange(n + 1)
    f = [sps.binom(n, pd).pmf(a) for pd in p]
    for am1 in a:
        for a1 in a:
            a0 = r - am1 - a1
            if 0 <= a0 <= n:
                pmf[am1 - a1 + n] += f[0][am1] * f[1][a0] * f[2][a1]
    return pmf / pmf.sum()


s = np.arange(-n, n + 1)
for r, values in paper.items():
    exact = exact_conditional(r)
    with spa.Timer(f"\nDouble saddle point for r = {r}", decimals=3):
        sp = approx_bioassay.conditional_pdf(-s.astype(float), [r], i=1, normalize_pdf=False)
    renormalized = sp / np.nansum(sp)
    paper_exact = np.concatenate(values[:2])
    paper_sp = np.concatenate(values[2:])
    print(f"Conditional distribution of s given r = {r}")
    print("   s     exact  (paper)    saddle  (paper)  renormalized")
    for k in range(n + 1):
        j = k + n  # index of s = k
        ref = (paper_exact[k], paper_sp[k] if k < n else np.nan)
        print(
            f"  {k:2d}  {exact[j]:.6f} ({ref[0]:.6f})  {sp[j]:.6f} ({ref[1]:.6f})  "
            f"{renormalized[j]:.6f}"
        )
    valid = (s >= 0) & (s < n)
    print(
        "  Largest relative error, s = 0..14: "
        f"{np.max(np.abs(sp / exact - 1)[valid & (s < 15)]):.3f}, "
        f"renormalized {np.max(np.abs(renormalized / exact - 1)[valid & (s < 15)]):.3f}"
    )

# --- Example 3: conditional likelihood for Fisher's twins (Table 3, Figure 1) ----------------
convicted = 12
dizygotic, monozygotic = 17, 13
x11 = 2  # convicted dizygotic twins


def approx_loglik(psi):
    # The conditional distribution does not depend on the dizygotic probability; take 1/2
    cgf = MultivariateCumulantGeneratingFunction.from_univariate(
        spa.binomial(dizygotic, 0.5), spa.binomial(monozygotic, expit(psi))
    ).ldot([[1, 1], [1, 0]])
    pdf = spa.BivariateSaddlePointApprox(cgf).conditional_pdf(
        x11, [convicted], i=1, normalize_pdf=False
    )
    return np.log(pdf)


def exact_loglik(psi):
    # Odds ratio of the dizygotic versus the monozygotic twins
    nchg = sps.nchypergeom_fisher(dizygotic + monozygotic, dizygotic, convicted, np.exp(-psi))
    return nchg.logpmf(x11)


psis = np.linspace(-0.5, 10, 43)
with spa.Timer("\nConditional log-likelihood, double saddle point", decimals=3):
    ll_sp = np.array([approx_loglik(v) for v in psis])
ll_exact = exact_loglik(psis)
bounds = dict(bounds=(0, 8), method="bounded")
mle_exact = spo.minimize_scalar(lambda v: -exact_loglik(v), **bounds).x
mle_sp = spo.minimize_scalar(lambda v: -approx_loglik(v), **bounds).x
print(f"Conditional MLE of psi: exact {mle_exact:.3f}, double saddle point {mle_sp:.3f}")
print(f"Unconditional MLE log(150 / 6) = {np.log(150 / 6):.3f}")
print("The paper reports 3.06 and 3.22")

fig, ax = plt.subplots(facecolor="w")
ax.plot(psis, ll_exact - exact_loglik(mle_exact), label="Exact")
ax.plot(psis, ll_sp - approx_loglik(mle_sp), "--", label="Double saddle point")
ax.set_ylim(-5, 0.5)
ax.set_xlabel("psi")
ax.set_ylabel("Conditional log-likelihood")
ax.set_title("Barndorff-Nielsen and Cox (1979) Figure 1: Fisher's twins")
ax.legend()

plt.show()
