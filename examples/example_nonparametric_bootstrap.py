# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import spapprox as spa
from statsmodels.distributions import ECDF

plt.close("all")

# Consider the bootstrapped sample mean
# Can we approximate its distribution?

sample = np.random.normal(0, 1, 1000)
cgf = spa.empirical(sample)
spa_mean = spa.SaddlePointApproxMean(cgf, 100)


with spa.util.Timer("Initializing saddle point approximation"):
    cgf = spa.empirical(sample)
    spa_mean = spa.SaddlePointApproxMean(cgf, 100)
    # Initial evaluation takes most time
    spa_mean.pdf(t=0)
    spa_mean.cdf(t=0)


with spa.util.Timer("Bootstrapping with saddle point approximation"):
    t = np.linspace(-50, 50, num=100)
    x = spa_mean.cgf.dK(t)
    ypdf = spa_mean.pdf(t=t, x=x, normalize_pdf=True)
    ycdf = spa_mean.cdf(t=t, x=x)

with spa.util.Timer("Bootstrapping"):
    sample_means = np.array(
        [np.random.choice(sample, replace=True, size=100).mean() for i in range(100000)]
    )
    yecdf = ECDF(sample_means)(x)

fig, axs = plt.subplots(1, 2, facecolor="w")
ax = axs[0]
ax.plot(x, ypdf, label="Saddle point approximation")
ax.hist(sample_means, bins=int(len(sample) / 10), density=True, label="histogram")
ax.legend()

ax = axs[1]
ax.plot(x, ycdf, label="Saddle point approximation")
ax.plot(x, yecdf, label="Empirical cdf")
ax.legend()
