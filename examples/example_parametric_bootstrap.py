#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution of the mean of ``n`` independent draws from a known distribution, with
``UnivariateSaddlePointApproxMean``, compared with the exact distribution of the mean.
"""

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import spapprox as spa

plt.close("all")

# Cases where the exact distribution of the mean is known

# Toggle one of the case below
n = 25
if False:
    cgf = spa.norm(loc=1, scale=1)
    sum_dist = sps.norm(loc=1, scale=1 / np.sqrt(n))
elif True:
    cgf = spa.gamma(a=3, scale=1) + 3
    sum_dist = sps.gamma(loc=3, a=n * 3, scale=1 / n)

spa_mean = spa.UnivariateSaddlePointApproxMean(cgf, n)

fig, axs = plt.subplots(1, 2, facecolor="w")

t = np.linspace(-3, 3, num=1000)
x = spa_mean.cgf.dK(t)
ax = axs[0]
ax.plot(x, spa_mean.pdf(t=t), label="Saddle point approximation")
ax.plot(x, sum_dist.pdf(x), label="pdf")
ax.legend()

ax = axs[1]
ax.plot(x, spa_mean.cdf(t=t), label="Saddle point approximation")
ax.plot(x, sum_dist.cdf(x), label="cdf")
ax.legend()
plt.show()
