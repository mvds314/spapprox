#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import warnings
import numpy as np
import pandas as pd
import scipy.optimize as spo
import statsmodels.api as sm

from .util import type_wrapper


# TODO: how can we extend this to a multidimensional setting?
class Domain:
    """
    Represents the domain of a function.
    For a value to be in the domain, it should satisfy all the specied contraints:
    less, greater, less-equal, and greater-equal.
    """

    def __init__(self, dim=1, l=None, g=None, le=None, ge=None):
        self.dim = dim
        self.l = l
        self.g = g
        self.le = le
        self.ge = ge

    def __contains__(self, t):
        return all(self.is_in_domain(t))

    def __add__(self, other):
        assert pd.api.types.is_number(other), "Can only add a scalar"
        return Domain(
            l=self.l + other if self.l is not None else None,
            g=self.g + other if self.g is not None else None,
            le=self.le + other if self.le is not None else None,
            ge=self.ge + other if self.ge is not None else None,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        assert pd.api.types.is_number(other), "Can only multiply with a scalar"
        return Domain(
            l=self.l * other if self.l is not None else None,
            g=self.g * other if self.g is not None else None,
            le=self.le * other if self.le is not None else None,
            ge=self.ge * other if self.ge is not None else None,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def intersect(self, other):
        """
        Intersect with another domain
        """
        assert isinstance(other, Domain), "Can only intersect with another Domain"
        if self.l is not None and other.l is not None:
            l = max(self.l, other.l)
        elif self.l is not None:
            l = self.l
        elif other.l is not None:
            l = other.l
        else:
            l = None
        if self.g is not None and other.g is not None:
            g = min(self.g, other.g)
        elif self.g is not None:
            g = self.g
        elif other.g is not None:
            g = other.g
        else:
            g = None
        if self.le is not None and other.le is not None:
            le = max(self.le, other.le)
        elif self.le is not None:
            le = self.le
        elif other.le is not None:
            le = other.le
        else:
            le = None
        if self.ge is not None and other.ge is not None:
            ge = min(self.ge, other.ge)
        elif self.ge is not None:
            ge = self.ge
        elif other.ge is not None:
            ge = other.ge
        else:
            ge = None
        return Domain(l=l, g=g, le=le, ge=ge)

    @type_wrapper(xloc=1)
    def is_in_domain(self, t):
        val = np.full(t.shape, True)
        if self.l is not None:
            val &= t < self.l
        if self.g is not None:
            val &= t > self.g
        if self.le is not None:
            val &= t <= self.le
        if self.ge is not None:
            val &= t >= self.ge
        return val
