#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:09:21 2024

@author: martins
"""
import numpy as np
from findiff import FinDiff, coefficients, Coefficient

x, y, z = [np.linspace(0, 10, 100)] * 3
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
f = np.sin(X) * np.cos(Y) * np.sin(Z)


orders = (1, 0, 0)
dim = len(orders)
h = dx
linear_op = FinDiff(*[(i, h, orders[i]) for i in range(dim) if orders[i] > 0])
# linear_op = FinDiff((0, 0.10101010101010101, 1))
# linear_op = FinDiff(0, dx, 2) + 2 * FinDiff((0, dx), (1, dy)) + FinDiff(1, dy, 2)
result = linear_op(f)

# TODO: how does this compare to our case?

# TODO: slow change things to our case

# TODO: it is something weid with this notation
# TODO: what is wrong?
