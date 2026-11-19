#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution of the sum of independent random variables: the cgf of a sum is the sum of the
cgfs, ``cgfX1 + cgfX2``. Compared with cases where the exact distribution of the sum is known.
"""

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import spapprox as spa

plt.close("all")

# Toggle one of the case below
if False:
    cgfX1 = spa.norm(loc=0, scale=1)
    cgfX2 = spa.norm(loc=0, scale=1)
    # we know the sum of normal dists should be normal again
    sum_dist = sps.norm(0, np.sqrt(2))
    cgf = cgfX1 + cgfX2
elif False:
    # The sum of gamma distributions with the same scale is gamma: Gamma(a=1 + 2, scale=1)
    cgfX1 = spa.gamma(a=1, scale=1)
    cgfX2 = spa.gamma(a=2, scale=1)
    sum_dist = sps.gamma(a=3, scale=1)
    cgf = cgfX1 + cgfX2
elif True:
    # As above, shifted by 3
    cgfX1 = spa.gamma(a=1, scale=1)
    cgfX2 = spa.gamma(a=2, scale=1)
    sum_dist = sps.gamma(a=3, scale=1, loc=3)
    cgf = cgfX1 + cgfX2 + 3

approx = spa.UnivariateSaddlePointApprox(cgf)

fig, axs = plt.subplots(1, 2, facecolor="w")

t = np.linspace(-3, 0.5, num=1000)
x = cgf.dK(t)
ax = axs[0]
ax.plot(x, approx.pdf(t=t), label="Saddle point approximation")
ax.plot(x, sum_dist.pdf(x), label="pdf")
ax.legend()

ax = axs[1]
ax.plot(x, approx.cdf(t=t), label="Saddle point approximation")
ax.plot(x, sum_dist.cdf(x), label="cdf")
ax.legend()
plt.show()
