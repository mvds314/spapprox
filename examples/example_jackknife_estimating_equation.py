#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delete-m jackknife of an estimating equation, Booth and Butler (1990), Example 3 and Table 3.

The ratio of means tau = E(V) / E(U) solves sum_i (v_i - tau u_i) = 0. A delete-m jackknife
subsample keeps each pair with an indicator Y_i, and the jackknife estimate tau* solves
sum_i (v_i - tau u_i) Y_i = 0. With independent Bernoulli(1/2) indicators conditioned on
sum_i Y_i = n - m, this is a randomly weighted estimating equation, which is decreasing in tau,
so P(tau* <= tau) = P(sum_i (v_i - tau u_i) Y_i <= 0 | sum_i Y_i = n - m). The data are those
of Hinkley and Wei (1984), with n = 25 and m = 10.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

import spapprox as spa

u = [155, 68, 28, 25, 190, 82, 92, 196, 164, 68, 82, 36, 195, 92, 185, 61, 62, 71, 207, 185]
u = np.array(u + [475, 155, 29, 96, 699.0])
v = [1546, 505, 280, 410, 1450, 517, 738, 2225, 1660, 505, 680, 145, 224, 733, 1957, 287]
v = np.array(v + [473, 473, 1260, 1958, 4375, 1499, 245, 828, 6361.0])
n, m = len(u), 10
tau_hat = v.sum() / u.sum()

ee = spa.EstimatingEquationSaddlePointApprox.from_resampling(
    lambda x, tau: x[:, 1] - tau * x[:, 0],
    np.c_[u, v],
    spa.binomial(1, 0.5),
    given=n - m,
    bracket=(6.0, 11.0),
)
ee.fit_cdf(num=200)


def normal(tau):
    # Normal approximation of Booth and Butler (1990), with the variance of the mean of a
    # sample of n - m drawn without replacement, up to a factor n / (n - 1)
    s = np.sqrt(np.mean((v - v.mean() - tau * (u - u.mean())) ** 2))
    return sps.norm.cdf(-np.sqrt((n - m) * n / m) * (v.mean() - tau * u.mean()) / s)


# Monte Carlo reference
rng = np.random.default_rng(1984)
keep = np.argsort(rng.random((10**6, n)), axis=1)[:, : n - m]
sim = v[keep].sum(axis=1) / u[keep].sum(axis=1)

# Table 3
tau = np.round(np.arange(6.6, 9.61, 0.2), 1)
paper_exact = [0.0001, 0.0003, 0.0013, 0.0041, 0.0112, 0.0310, 0.0726, 0.1578, 0.2979, 0.4661]
paper_exact += [0.6157, 0.7453, 0.8717, 0.9610, 0.9953, 0.9999]
paper_spa = [0.0001, 0.0004, 0.0015, 0.0043, 0.0118, 0.0303, 0.0725, 0.1550, 0.2844, 0.4476]
paper_spa += [0.6173, 0.7680, 0.8833, 0.9610, 0.9947, 0.9998]
print(f"{'tau':>5} {'MC':>7} {'paper':>7} {'paper SP':>8} {'SP':>7} {'normal':>7}")
for row in zip(tau, [np.mean(sim <= t) for t in tau], paper_exact, paper_spa, ee.cdf(tau)):
    print(f"{row[0]:5.1f} " + " ".join(f"{r:7.4f}" for r in row[1:]) + f" {normal(row[0]):7.4f}")

# Jackknife mean and variance estimate (n - m) / m * E(tau* - tau_hat)^2
factor = (n - m) / m
mse = ee.var() + (ee.mean() - tau_hat) ** 2
print(f"\njackknife mean: MC {sim.mean():.4f}, paper 8.454 / 8.451, saddle point {ee.mean():.4f}")
mc = factor * np.mean((sim - tau_hat) ** 2)
print(f"variance:       MC {mc:.4f}, paper 0.3074 / 0.3004, saddle point {factor * mse:.4f}")

t = np.linspace(6.5, 10.5, 100)
fig, ax = plt.subplots()
ax.hist(sim, bins=200, density=True, alpha=0.4, label="Monte Carlo")
ax.plot(t, ee.pdf(t), label="saddle point")
ax.set_xlabel("jackknife estimate of the ratio")
ax.legend()
plt.show()
