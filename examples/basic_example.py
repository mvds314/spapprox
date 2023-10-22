#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sps

import matplotlib.pyplot as plt

import spapprox as spa


plt.close("all")

# https://en.wikipedia.org/wiki/Cumulant
cgf_normal = spa.norm(loc=0, scale=1)
spa_normal = spa.SaddlePointApprox(cgf_normal)

fig, axs = plt.subplots(1, 2, facecolor="w")
t = np.linspace(-3, 3)
x = cgf_normal.dK(t)
spa_normal.pdf(t=t)
sps.norm.pdf(x)

ax = axs[0]
ax.plot(x, spa_normal.pdf(t=t), label="Saddle point approximation")
ax.plot(x, sps.norm.pdf(x), label="Normal pdf")
ax.legend()

ax = axs[1]
ax.plot(x, spa_normal.cdf(t=t), label="Saddle point approximation")
ax.plot(x, sps.norm.cdf(x), label="Normal pdf")
ax.legend()
