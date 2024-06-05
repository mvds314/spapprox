#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:30:09 2024

@author: martins
"""

import numpy as np
import spapprox as spr
from spapprox import Timer
import numdifftools as ndt
import findiff as fd


cgf = spr.multivariate_norm(dim=3)
cgf.domain = spr.Domain(dim=3, l=[10, 1 + 5e-7, 10])

K = cgf.K
dK = cgf.dK
d2K = cgf.d2K
d3K = cgf.d3K

ts = np.random.randn(1000, 3)
ts = ts[0]
ts = np.array([1, 1, 1]).astype(np.float64)

h = 1e-6

# Gradient
grad = fd.Gradient(h=[h, h, h])

# TODO: continue here, why doesn't this work yet, 1+1e-6 should be outside of the domain


def gradf(t):
    global grad
    # use central differences by default
    x = np.array([t - h, t, t + h]).T
    Xis = (
        np.array(np.meshgrid(x[0], x[1], x[2], indexing="ij"))
        .reshape((cgf.dim, cgf.dim**cgf.dim))
        .T
    )
    retval = K(Xis).reshape(tuple([cgf.dim] * cgf.dim))
    retval = grad(retval)
    # select central differences value
    
    retval = retval[*([1] * cgf.dim)]
    # TODO: contine here, what to select if there are nans?
    import pdb

    pdb.set_trace()
    sel = np.isnan(retval)
    if not sel.any():
        return retval
    else:
        import pdb

        pdb.set_trace()
        raise NotImplementedError("Dealing with nan values not implemented")


# TODO: how to deal with the edge cases
# TODO: specify domain


# TODO: how to deal with higher order derivatives


# # Second order derivative
# d2_dt2 = fd.FinDiff(0, h, 2)


# def d2Kfd(t):
#     global d2_dt2
#     # use central differences by default
#     Ks = K(np.array([t - h, t, t + h, t + 2 * h]))
#     retval = d2_dt2(Ks)[1]
#     sel = np.isnan(retval)
#     if not sel.any():
#         return retval
#     # fall back to left
#     tsel = t[sel]
#     retval[sel] = d2_dt2(K(np.array([tsel, tsel + h, tsel + 2 * h + tsel + 3 * h])))[0]
#     sel = np.isnan(retval)
#     if not sel.any():
#         return retval
#     # or to right
#     tsel = t[sel]
#     retval[sel] = d2_dt2(K(np.array([t[sel] - 3 * h, t[sel] - 2 * h, t[sel] - h, tsel])))[-1]
#     return retval


# # Third order derivative
# d3_dt3 = fd.FinDiff(0, h, 3)


# def d3Kfd(t):
#     global d3_dt3
#     # use central differences by default
#     retval = d3_dt3(K(np.array([t - 3 * h, t - 2 * h, t - h, t, t + h, t + 2 * h])))[3]
#     sel = np.isnan(retval)
#     if not sel.any():
#         return retval
#     # fall back to left
#     tsel = t[sel]
#     retval[sel] = d3_dt3(
#         K(np.array([tsel, tsel + h, tsel + 2 * h, tsel + 3 * h, tsel + 4 * h, tsel + 5 * h]))
#     )[0]
#     retval[sel] = d_dt(K(np.repeat(tsel, 6) + h * np.arange(6)))[0]
#     sel = np.isnan(retval)
#     if not sel.any():
#         return retval
#     # or to right
#     tsel = t[sel]
#     retval[sel] = d3_dt3(
#         K(np.array([tsel - 5 * h, tsel - 4 * h, tsel - 3 * h, tsel - 2 * h, tsel - h, tsel]))
#     )[-1]
#     return retval


with Timer("explicit 1", decimals=6):
    dK(ts)
dKndt = ndt.Gradient(K, n=1)
with Timer("numdifftools 1", decimals=6):
    if ts.ndim == 2:
        np.asanyarray([dKndt(t.T) for t in ts])
    elif ts.ndim == 1:
        dKndt(ts.T)
with Timer("findiff 1", decimals=6):
    gradf(ts)

# with Timer("explicit 2", decimals=6):
#     d2K(ts)
# d2Kndt = ndt.Derivative(K, n=2)
# with Timer("numdifftools 2", decimals=6):
#     d2Kndt(ts)
# with Timer("findiff 2", decimals=6):
#     d2Kfd(ts)

# with Timer("explicit 3", decimals=6):
#     d3K(ts)
# d3Kndt = ndt.Derivative(K, n=3)
# with Timer("numdifftools 3", decimals=6):
#     d2Kndt(ts)
# with Timer("findiff 3", decimals=6):
#     d3Kfd(ts)
