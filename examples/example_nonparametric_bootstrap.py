# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import spapprox as spa

plt.close("all")

# TODO: fix this example

# Consider the bootstrapped sample mean
# Can we approximate its distribution?

x = np.random.randint(3, size=25)
cgf = spa.empirical(x)
spa_mean = spa.SaddlePointApprox(cgf)

fig, axs = plt.subplots(1, 2, facecolor="w")

# TODO: fix the return type, maybe add to unittest
t = np.linspace(cgf.dK_inv(x.min() + 1e-6)[0], cgf.dK_inv(x.max() - 1e-6)[0], num=1000)
x = spa_mean.cgf.dK(t)
ax = axs[0]
ax.plot(x, spa_mean.pdf(t=t), label="Saddle point approximation")
# ax.plot(x, sum_dist.pdf(x), label="pdf")
ax.legend()

ax = axs[1]
ax.plot(x, spa_mean.cdf(t=t), label="Saddle point approximation")
# ax.plot(x, sum_dist.cdf(x), label="pdf")
ax.legend()
