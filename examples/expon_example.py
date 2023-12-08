# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sps

import matplotlib.pyplot as plt

import spapprox as spa


plt.close("all")

# https://en.wikipedia.org/wiki/Cumulant
cgf = spa.exponential(scale=1)
approx = spa.UnivariateSaddlePointApprox(cgf)

fig, axs = plt.subplots(1, 2, facecolor="w")
t = 0.8
approx.pdf(t=t)
sps.expon.pdf(cgf.dK(t))

t = np.linspace(-3, 0.9)
x = cgf.dK(t)
ax = axs[0]
ax.plot(x, approx.pdf(t=t), label="Saddle point approximation")
ax.plot(x, sps.expon.pdf(x), label="pdf")
ax.legend()

ax = axs[1]
ax.plot(x, approx.cdf(t=t), label="Saddle point approximation")
ax.plot(x, sps.expon.cdf(x), label="cdf")
ax.legend()
