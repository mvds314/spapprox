#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 08:28:52 2025

@author: martins
"""

import numpy as np

A = np.random.rand(2, 3)
B = np.random.rand(3, 4)

C = np.einsum("ij,kl->ik", A, B)
print(A)
print(B)
print(C)

sum([A[0, j] * B[0, l] for j in range(3) for l in range(4)])

C00 = A[0, 1] * B[0, 0]
print(C00)
