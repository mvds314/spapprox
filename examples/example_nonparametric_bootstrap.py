# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import spapprox as spa
from statsmodels.distributions import ECDF

plt.close("all")

# Consider the bootstrapped sample mean
# Can we approximate its distribution?

# Toggle one of the examples below
if False:
    sample = np.random.normal(0, 1, 1000)
    t = np.linspace(-50, 50, num=100)
    normalize_pdf = True
    subsample_size = 100
elif False:
    sample = np.random.gamma(3, 2, 1000)
    t = np.linspace(-25, 10, num=100)
    normalize_pdf = True  # This one is not really implemented well
    subsample_size = 100
elif True:
    sample = np.random.laplace(3, 2, 100000)
    t = np.linspace(-150, 150, num=100)
    normalize_pdf = True  # This one is not really implemented well
    subsample_size = 10000


with spa.util.Timer("Initializing saddle point approximation"):
    cgf = spa.empirical(sample)
    spa_mean = spa.SaddlePointApproxMean(cgf, subsample_size)
    # Initial evaluation takes most time
    spa_mean.pdf(t=0, normalize_pdf=normalize_pdf)
    spa_mean.cdf(t=0)


with spa.util.Timer("Bootstrapping with saddle point approximation"):
    x = spa_mean.cgf.dK(t)
    ypdf = spa_mean.pdf(t=t, x=x, normalize_pdf=normalize_pdf)
    ycdf = spa_mean.cdf(t=t, x=x)

with spa.util.Timer("Bootstrapping"):
    sample_means = np.array(
        [np.random.choice(sample, replace=True, size=subsample_size).mean() for i in range(100000)]
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
