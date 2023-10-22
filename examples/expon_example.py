# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sps

import matplotlib.pyplot as plt

import spapprox as spa


plt.close("all")

# https://en.wikipedia.org/wiki/Cumulant
cgf = spa.exponential(scale=1)
spa = spa.SaddlePointApprox(cgf)

fig, axs = plt.subplots(1, 2, facecolor="w")
t = np.linspace(-3, 3)
x = cgf.dK(t)
spa.pdf(t=t)
sps.expon.pdf(x)

ax = axs[0]
ax.plot(x, spa.pdf(t=t), label="Saddle point approximation")
ax.plot(x, sps.expon.pdf(x), label="pdf")
ax.legend()

ax = axs[1]
ax.plot(x, spa.cdf(t=t), label="Saddle point approximation")
ax.plot(x, sps.expon.cdf(x), label="pdf")
ax.legend()
