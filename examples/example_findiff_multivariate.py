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

ts = np.linspace(0, 0.9, 10000)
h = ts[1] - ts[0]

# ts = ts[0]


# This covers the univariate case


# First order derivative
d_dt = fd.FinDiff(0, h, 1)


def dKfd(t):
    global d_dt
    # use central differences by default
    retval = d_dt(K(np.array([t - h, t, t + h])))[1]
    sel = np.isnan(retval)
    if not sel.any():
        return retval
    # fall back to left
    tsel = t[sel]
    retval[sel] = d_dt(K(np.array([tsel, tsel + h, tsel + 2 * h])))[0]
    sel = np.isnan(retval)
    if not sel.any():
        return retval
    # or to right
    tsel = t[sel]
    retval[sel] = d_dt(K(np.array([t[sel] - 2 * h, t[sel] - h, tsel])))[-1]
    return retval


# Second order derivative
d2_dt2 = fd.FinDiff(0, h, 2)


def d2Kfd(t):
    global d2_dt2
    # use central differences by default
    Ks = K(np.array([t - h, t, t + h, t + 2 * h]))
    retval = d2_dt2(Ks)[1]
    sel = np.isnan(retval)
    if not sel.any():
        return retval
    # fall back to left
    tsel = t[sel]
    retval[sel] = d2_dt2(K(np.array([tsel, tsel + h, tsel + 2 * h + tsel + 3 * h])))[0]
    sel = np.isnan(retval)
    if not sel.any():
        return retval
    # or to right
    tsel = t[sel]
    retval[sel] = d2_dt2(K(np.array([t[sel] - 3 * h, t[sel] - 2 * h, t[sel] - h, tsel])))[-1]
    return retval


# Third order derivative
d3_dt3 = fd.FinDiff(0, h, 3)


def d3Kfd(t):
    global d3_dt3
    # use central differences by default
    retval = d3_dt3(K(np.array([t - 3 * h, t - 2 * h, t - h, t, t + h, t + 2 * h])))[3]
    sel = np.isnan(retval)
    if not sel.any():
        return retval
    # fall back to left
    tsel = t[sel]
    retval[sel] = d3_dt3(
        K(np.array([tsel, tsel + h, tsel + 2 * h, tsel + 3 * h, tsel + 4 * h, tsel + 5 * h]))
    )[0]
    retval[sel] = d_dt(K(np.repeat(tsel, 6) + h * np.arange(6)))[0]
    sel = np.isnan(retval)
    if not sel.any():
        return retval
    # or to right
    tsel = t[sel]
    retval[sel] = d3_dt3(
        K(np.array([tsel - 5 * h, tsel - 4 * h, tsel - 3 * h, tsel - 2 * h, tsel - h, tsel]))
    )[-1]
    return retval


with Timer("explicit 1", decimals=6):
    dK(ts)
dKndt = ndt.Derivative(K, n=1)
with Timer("numdifftools 1", decimals=6):
    dKndt(ts)
with Timer("findiff 1", decimals=6):
    dKfd(ts)

with Timer("explicit 2", decimals=6):
    d2K(ts)
d2Kndt = ndt.Derivative(K, n=2)
with Timer("numdifftools 2", decimals=6):
    d2Kndt(ts)
with Timer("findiff 2", decimals=6):
    d2Kfd(ts)

with Timer("explicit 3", decimals=6):
    d3K(ts)
d3Kndt = ndt.Derivative(K, n=3)
with Timer("numdifftools 3", decimals=6):
    d2Kndt(ts)
with Timer("findiff 3", decimals=6):
    d3Kfd(ts)


# Now try the multivariate case, especially with the mixed derivatives

# TODO: start simple with the gradient

# TODO: then go to the jacobian

# TODO: then go to the third order derivative, can we make a differential operator?


# TODO: then built in this more advanced derivative framework

# fd.coefficients(3, acc=2)

# What would be the best way?

# Use the stencil?

# d_dt =


# myts = ts[500:503]
# d_dt.stencil(myts.shape)
# d_dt(K(myts))


# dK(myts[1])

# TODO: let's just use the schema with accuracy equal to 2
# TODO: given points where we want to know the derivative
# TODO: extract the coefficients and the offsets
# TODO: this gives points where we want to evaluate K
# TODO: evaluate K
# TODO: do a couple of function evaluations

# TODO: work this out in an example
