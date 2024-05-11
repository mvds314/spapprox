#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:30:09 2024

@author: martins
"""
import numpy as np
from spapprox import Timer
import numdifftools as ndt
import findiff as fd

K = lambda t: np.log(1 / (1 - t))
dK = lambda t: 1 / (1 - t)
dK_inv = lambda x: 1 - 1 / x
d2K = lambda t: 1 / (1 - t) ** 2
d3K = lambda t: 2 / (1 - t) ** 3

ts = np.linspace(0, 0.99, 10000)

with Timer("explicit", decimals=6):
    dK(ts)


with Timer("numdifftools", decimals=6):
    dKndt = ndt.Derivative(K, n=1)
    dKndt(ts)


with Timer("findiff", decimals=6):
    dt = ts[1] - ts[0]
    d_dt = fd.FinDiff(0, dt, 1)
    d_dt(K(ts))


# TODO: let's just use the schema with accuracy equal to 2
# TODO: given points where we want to know the derivative
# TODO: extract the coefficients and the offsets
# TODO: this gives points where we want to evaluate K
# TODO: evaluate K
# TODO: do a couple of function evaluations

# TODO: work this out in an example
