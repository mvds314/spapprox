#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduces examples of Barndorff-Nielsen and Cox (1979), "Edgeworth and saddle-point
approximations with statistical applications", J. R. Statist. Soc. B 41(3), 279-312.

1. Inverse Gaussian (Section 3.4). The saddle point density of the inverse Gaussian
   distribution is exact up to its normalizing constant (Tweedie, 1957), so the normalized
   approximation coincides with the exact density.
2. Major freezes of Lake Constance (Section 5.2, Table 1). The 37 freezes after the first one,
   in A.D. 875-1974, are treated as a Poisson process with rate exp(alpha + beta t + phi t^2),
   with time t rescaled to [0, 1]. Given the number of freezes, the times are iid with density
   proportional to exp(beta t + phi t^2). Is the trend monotone (phi = 0), or does the rate
   peak in between (phi < 0)? Conditioning on r = sum t_i removes beta, and the p-value is
   P(S <= s | R = r) for s = sum t_i^2. The paper uses the normal leading term of a single
   saddle point expansion, with one and two correction terms. The example uses the double
   saddle point conditional cdf, with the cgf of (t, t^2) computed by Gauss-Legendre
   quadrature, and compares it with an exact Monte Carlo estimate.
3. Binomial-logistic bioassay (Section 5.5, Table 2). At each of the doses d = -1, 0, 1,
   n = 16 individuals are treated, and a_d of them respond, with a_d ~ binomial(n, p_d) and
   logistic response probability p_d = 1 / (1 + exp(lambda + psi d)). The sufficient
   statistics are r = a_{-1} + a_0 + a_1, the total number responding, and
   s = a_1 - a_{-1}. Inference on the slope psi is done conditionally on r, which removes the
   intercept lambda. The example compares the exact conditional distribution of s given r,
   for psi = 1, with the double saddle point approximation (Butler, 2007, Section 4.1), with
   and without renormalization. The paper's s runs in the opposite direction of the dose, so
   it is -(a_1 - a_{-1}) in our construction. On the boundary of the support, s = 16, the
   approximation is undefined and returns NaN.
4. Conditional likelihood for Fisher's twins data (Section 6.2, Table 3, Figure 1). Of 17
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

# --- Example 2: major freezes of Lake Constance (Table 1) -----------------------------------
years = np.array(
    [875, 895, 928, 1074, 1076, 1108, 1217, 1227, 1277, 1323, 1325, 1378, 1379, 1383, 1409, 1431]
    + [1435, 1460, 1465, 1470, 1479, 1497, 1512, 1553, 1560, 1564, 1565, 1571, 1573, 1684, 1695]
    + [1763, 1776, 1788, 1796, 1830, 1880, 1963]
)
# The paper drops the first freeze, which fixes the origin, and takes 1974 - 875 as time unit
times = (years[1:] - 875) / 1099
n = len(times)
r, s_obs = times.sum(), (times**2).sum()

# The cgf of (t, t^2) for t uniform on [0, 1]: by sufficiency, the conditional distribution
# given r does not depend on beta, so take beta = 0. The integrals have no closed form; the
# Gauss-Legendre nodes give them, and the moments of the tilted density give the derivatives
nodes, weights = np.polynomial.legendre.leggauss(100)
nodes, weights = (nodes + 1) / 2, weights / 2
powers = np.stack([nodes, nodes**2], axis=-1)


def tilted(tau):
    # Probabilities of the nodes under the tilted density, and the log of its normalization
    e = np.asanyarray(tau, dtype=float) @ powers.T
    m = e.max(axis=-1, keepdims=True)
    w = weights * np.exp(e - m)
    total = w.sum(axis=-1, keepdims=True)
    return w / total, np.log(total[..., 0]) + m[..., 0]


def cumulant(tau, order):
    # The cumulants of order 2 and 3 are the central moments of the tilted density
    p = tilted(tau)[0]
    c = powers - (p @ powers)[..., None, :]
    if order == 2:
        return np.einsum("...m,...mi,...mj->...ij", p, c, c)
    return np.einsum("...m,...mi,...mj,...mk->...ijk", p, c, c, c)


# The sum of n independent copies: n times the cgf of one
cgf_lake = MultivariateCumulantGeneratingFunction(
    K=lambda tau: n * tilted(tau)[1],
    dK=lambda tau: n * tilted(tau)[0] @ powers,
    d2K=lambda tau: n * cumulant(tau, 2),
    d3K=lambda tau: n * cumulant(tau, 3),
    dim=2,
)
approx_lake = spa.BivariateSaddlePointApprox(cgf_lake)


def normal_term(r, s):
    # The leading term (3.16) of the paper: S given R normal, at the restricted MLE of beta
    def moment(k, beta):
        return tilted([beta, 0])[0] @ nodes**k

    beta0 = spo.brentq(lambda b: moment(1, b) - r / n, -20, 20)
    m1, m2, m3, m4 = (moment(k, beta0) for k in (1, 2, 3, 4))
    var = n * ((m4 - m2**2) - (m3 - m1 * m2) ** 2 / (m2 - m1**2))
    return sps.norm.cdf((s - n * m2) / np.sqrt(var))


# Exact Monte Carlo: given R = r, uniform times are uniform on the slice of the cube, so draw
# n - 1 of them and accept when the last one, r minus their sum, lies in [0, 1]
rng = np.random.default_rng(0)
mc_s = []
for _ in range(10):
    u = rng.random((500_000, n - 1))
    last = r - u.sum(axis=1)
    ok = (last > 0) & (last < 1)
    mc_s.append((u[ok] ** 2).sum(axis=1) + last[ok] ** 2)
mc_s = np.sort(np.concatenate(mc_s))
p_mc = np.mean(mc_s <= s_obs)

# The paper works with the rounded summaries r / n = 0.5394, s / n = 0.3443
rounded = (n * 0.3443, n * 0.5394)
print(f"\nLake Constance: n = {n}, r / n = {r / n:.4f}, s / n = {s_obs / n:.4f}")
print(f"  {'P(S <= s | R = r)':<30}  {'data':>8}  {'rounded':>8}  {'paper':>8}")
for label, f, paper_value in [
    ("Normal term (3.16)", normal_term, 0.008594),
    ("Double saddle point", lambda r, s: approx_lake.conditional_cdf(s, [r], i=1), np.nan),
    (
        "Double saddle point, Kolassa",
        lambda r, s: approx_lake.conditional_cdf(s, [r], i=1, backend="Kolassa"),
        np.nan,
    ),
    ("Paper, one correction term", lambda r, s: np.nan, 0.008370),
    ("Paper, two correction terms", lambda r, s: np.nan, 0.008180),
    ("Monte Carlo", lambda r, s: p_mc if s == s_obs else np.nan, np.nan),
]:
    values = (f(r, s_obs), f(rounded[1], rounded[0]), paper_value)
    print(f"  {label:<30}" + "".join(f"  {v:8.6f}" if v == v else " " * 10 for v in values))
print(
    f"  Monte Carlo: {len(mc_s)} draws, standard error "
    f"{np.sqrt(p_mc * (1 - p_mc) / len(mc_s)):.6f}"
)

grid = np.linspace(np.quantile(mc_s, 0.0005), np.quantile(mc_s, 0.05), 25)
fig, ax = plt.subplots(facecolor="w")
ax.semilogy(grid, np.searchsorted(mc_s, grid) / len(mc_s), label="Monte Carlo")
ax.semilogy(grid, approx_lake.conditional_cdf(grid, [r], i=1), "--", label="Double saddle point")
ax.semilogy(grid, [normal_term(r, v) for v in grid], ":", label="Normal term (3.16)")
ax.axvline(s_obs, color="grey", lw=0.5)
ax.set_xlabel("s")
ax.set_ylabel("P(S <= s | R = r)")
ax.set_title("Lake Constance freezes: lower tail given r")
ax.legend()

# --- Example 3: binomial-logistic bioassay (Table 2) -----------------------------------------
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

# --- Example 4: conditional likelihood for Fisher's twins (Table 3, Figure 1) ----------------
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
