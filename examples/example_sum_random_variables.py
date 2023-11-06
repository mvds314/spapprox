# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import spapprox as spa

plt.close("all")

# Consider the sum X1+X2
# Let's check what we get with a saddle point approx of the sum of two rvs

# https://en.wikipedia.org/wiki/Cumulant

# Toggle one of the case below
if False:
    cgfX1 = spa.norm(loc=0, scale=1)
    cgfX2 = spa.norm(loc=0, scale=1)
    # we know the sum of normal dists should be normal again
    sum_dist = sps.norm(0, np.sqrt(2))
    cgf = cgfX1 + cgfX2
elif False:
    # Sum of two gamma distributions with same scale param should be gamma as well
    cgfX1 = spa.gamma(a=1, scale=1)
    cgfX2 = spa.gamma(a=2, scale=1)
    # In this case, we know that the result should be Gamma(a=6,scale=1)
    sum_dist = sps.gamma(a=3, scale=1)
    cgf = cgfX1 + cgfX2
elif True:
    # Sum of two gamma distributions with same scale param should be gamma as well
    cgfX1 = spa.gamma(a=1, scale=1)
    cgfX2 = spa.gamma(a=2, scale=1)
    # In this case, we know that the result should be Gamma(a=6,scale=1)
    sum_dist = sps.gamma(a=3, scale=1, loc=3)
    cgf = cgfX1 + cgfX2 + 3

approx = spa.SaddlePointApprox(cgf)

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
