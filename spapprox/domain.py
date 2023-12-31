#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .util import type_wrapper


class Domain:
    r"""
    Represents the domain of a function.
    For a value to be in the domain, it should satisfy all the specified upper
    and lower bound contraints: less, greater, less-equal, and greater-equal.
    The specified bounds should satisfy: :math:`g>ge>le>l`.

    Also, it should satisfy the linear inequality:

    .. math::
        Ax \leq a,
        Bx < b,
    where :math:`A` and :math:`B` are matrices.
    """

    def __init__(self, dim=1, l=None, g=None, le=None, ge=None, A=None, a=None, B=None, b=None):
        assert pd.api.types.is_integer(dim) and dim > 0, "Dimension should be a positive integer"
        self.dim = dim
        # Validate bound constraints
        if dim == 1:
            assert all(
                x is None or np.isscalar(x) for x in [l, le, g, ge]
            ), "Bounds should be scalars in dim 1."
            specified_bounds = [x for x in [l, le, g, ge] if x is not None]
            if len(specified_bounds) > 1:
                assert all(
                    specified_bounds[i] > specified_bounds[i + 1]
                    for i in range(len(specified_bounds) - 1)
                ), "Bounds should satisfy: g>ge>le>l"
        else:
            assert all(
                x is None or np.isscalar(x) or len(x) == dim for x in [l, le, g, ge]
            ), "Bounds should be scalars or have matching dim"
            specified_bounds = [np.asanyarray(x) for x in [l, le, g, ge] if x is not None]
            if len(specified_bounds) > 1:
                assert all(
                    np.all(specified_bounds[i] > specified_bounds[i + 1])
                    for i in range(len(specified_bounds) - 1)
                ), "Bounds should satisfy: g>ge>le>l"
                l = np.asanyarray(l) if l is not None and not np.isscalar(l) else l
                g = np.asanyarray(g) if g is not None and not np.isscalar(g) else g
                le = np.asanyarray(le) if le is not None and not np.isscalar(le) else le
                ge = np.asanyarray(ge) if ge is not None and not np.isscalar(ge) else ge
        self.l = l
        self.g = g
        self.le = le
        self.ge = ge
        # Validate inequality constraints
        if A is not None:
            assert a is not None, "A and a should both be specified"
            A = np.asanyarray(A)
            a = np.asanyarray(a)
            assert len(A.shape) == 2, "A should be a matrix"
            assert len(a.shape) == 1, "a should be a vector"
            assert A.shape == (len(a), self.dim), "Shape of A should match dimension and bound"
        else:
            assert a is None, "A and a should both be specified"
        self.A = A
        self.a = a
        if B is not None:
            assert b is not None, "B and b should both be specified"
            B = np.asanyarray(B)
            b = np.asanyarray(b)
            assert len(B.shape) == 2, "B should be a matrix"
            assert len(b.shape) == 1, "b should be a vector"
            assert B.shape == (len(b), self.dim), "Shape of B should match dimension and bound"
        else:
            assert b is None, "B and b should both be specified"
        self.B = B
        self.b = b

    def __contains__(self, t):
        return np.all(self.is_in_domain(t))

    def add(self, other):
        if other is not None and pd.api.types.is_number(other):
            a = None if self.a is None else self.a + self.A.dot(np.full(self.dim, other))
            b = None if self.b is None else self.b + self.B.dot(np.full(self.dim, other))
        elif other is not None and pd.api.types.is_array_like(other) and len(other) == self.dim:
            a = None if self.a is None else self.a + self.A.dot(other)
            b = None if self.b is None else self.b + self.B.dot(other)
        else:
            raise ValueError("Can only add a scalar or a vector of length dim")
        return Domain(
            l=self.l + other if self.l is not None else None,
            g=self.g + other if self.g is not None else None,
            le=self.le + other if self.le is not None else None,
            ge=self.ge + other if self.ge is not None else None,
            A=self.A,
            a=a,
            B=self.B,
            b=b,
            dim=self.dim,
        )

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.__add__(other)

    def mul(self, other):
        """
        Left side multiplication
        """
        if other is not None and pd.api.types.is_number(other):
            A = self.A
            B = self.B
            a = None if self.a is None else self.a * other
            b = None if self.b is None else self.b * other
            l = self.l * other if self.l is not None else None
            g = self.g * other if self.g is not None else None
            le = self.le * other if self.le is not None else None
            ge = self.ge * other if self.ge is not None else None
        elif other is not None and self.dim > 1 and pd.api.types.is_array_like(other):
            other = np.asanyarray(other)
            assert len(other.shape) == 1 and len(other) == self.dim, "Invalid shape"
            # Divide each row of A by other
            A = None if self.A is None else np.divide(self.A, other)
            B = None if self.B is None else np.divide(self.B, other)
            a = self.a
            b = self.b
            l = self.l * other if self.l is not None else None
            g = self.g * other if self.g is not None else None
            le = self.le * other if self.le is not None else None
            ge = self.ge * other if self.ge is not None else None
        else:
            raise ValueError(
                "Can only multiply with a scalar or a vector of length dim when dim>1"
            )
        return Domain(l=l, g=g, le=le, ge=ge, A=A, a=a, B=B, b=b, dim=self.dim)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def ldot(self, other):
        """
        Left side dot product.

        Note that this might expand the domain, as several bounds are combined into one.

        Note sure how useful this is.
        """
        assert self.dim > 1, "dot product only supporeted for dim>1"
        other = np.asanyarray(other)
        if len(other.shape) == 1 and len(other) == self.dim:
            return self.ldot(np.expand_dims(other, axis=0))
        elif len(other.shape) == 2 and other.shape[1] == self.dim:
            assert self.A is None and self.a is None, "Cannot transform with a matrix twice"
            assert self.B is None and self.b is None, "Cannot transform with a matrix twice"
            if self.le is not None or self.ge is not None:
                A = np.full((0, self.dim), 0)
                a = np.full(0, 0)
                if self.le is not None:
                    A = np.vstack((A, other))
                    a = np.append(
                        a,
                        other.dot(np.full(self.dim, self.le) if np.isscalar(self.le) else self.le),
                    )
                if self.ge is not None:
                    A = np.vstack((A, -other))
                    a = np.append(
                        a,
                        -other.dot(
                            np.full(self.dim, self.ge) if np.isscalar(self.ge) else self.ge
                        ),
                    )
            if self.l is not None or self.g is not None:
                B = np.full((0, self.dim), 0)
                b = np.full(0, 0)
                if self.l is not None:
                    B = np.vstack((B, other))
                    b = np.append(
                        b, other.dot(np.full(self.dim, self.l) if np.isscalar(self.l) else self.l)
                    )
                if self.g is not None:
                    B = np.vstack((B, -other))
                    b = np.append(
                        b, -other.dot(np.full(self.dim, self.g) if np.isscalar(self.g) else self.g)
                    )
            l = None
            g = None
            le = None
            ge = None
        else:
            raise ValueError("Invalid shape")
        return Domain(l=l, g=g, le=le, ge=ge, A=A, a=a, B=B, b=b, dim=self.dim)

    def intersect(self, other):
        """
        Intersect with another domain, i.e., the result should be in both.
        """
        assert isinstance(other, Domain), "Can only intersect with another Domain"
        assert self.dim == other.dim, "Dimensions should match"
        if self.l is not None and other.l is not None:
            if self.dim == 1 or (
                pd.api.types.is_number(self.l) and pd.api.types.is_number(other.l)
            ):
                l = max(self.l, other.l)
            else:
                l = np.max([self.l * np.ones(self.dim), other.l * np.ones(self.dim)], axis=0)
        elif self.l is not None:
            l = self.l
        elif other.l is not None:
            l = other.l
        else:
            l = None
        if self.g is not None and other.g is not None:
            if self.dim == 1 or (
                pd.api.types.is_number(self.g) and pd.api.types.is_number(other.g)
            ):
                g = min(self.g, other.g)
            else:
                g = np.min([self.g * np.ones(self.dim), other.g * np.ones(self.dim)], axis=0)
        elif self.g is not None:
            g = self.g
        elif other.g is not None:
            g = other.g
        else:
            g = None
        if self.le is not None and other.le is not None:
            if self.dim == 1 or (
                pd.api.types.is_number(self.le) and pd.api.types.is_number(other.le)
            ):
                le = max(self.le, other.le)
            else:
                le = np.max([self.le * np.ones(self.dim), other.le * np.ones(self.dim)], axis=0)
        elif self.le is not None:
            le = self.le
        elif other.le is not None:
            le = other.le
        else:
            le = None
        if self.ge is not None and other.ge is not None:
            if self.dim == 1 or (
                pd.api.types.is_number(self.ge) and pd.api.types.is_number(other.ge)
            ):
                ge = min(self.ge, other.ge)
            else:
                ge = np.min([self.ge * np.ones(self.dim), other.ge * np.ones(self.dim)], axis=0)
        elif self.ge is not None:
            ge = self.ge
        elif other.ge is not None:
            ge = other.ge
        else:
            ge = None
        if self.A is not None and other.A is not None:
            A = np.vstack(self.A, other.A)
            a = np.append(self.a, other.a)
        elif self.A is not None:
            A = self.A
            a = self.a
        elif other.A is not None:
            A = other.A
            a = other.a
        else:
            A = None
            a = None
        if self.B is not None and other.B is not None:
            B = np.vstack(self.B, other.B)
            b = np.append(self.b, other.b)
        elif self.B is not None:
            B = self.B
            b = self.b
        elif other.B is not None:
            B = other.B
            b = other.b
        else:
            B = None
            b = None
        return Domain(l=l, g=g, le=le, ge=ge, A=A, a=a, B=B, b=b, dim=self.dim)

    @type_wrapper(xloc=1)
    def is_in_domain(self, t):
        if self.dim == 1:
            val = np.full(t.shape, True)
            t = np.expand_dims(t, axis=len(t.shape))
        else:
            assert len(t.shape) in [1, 2], "Only vectors or lists of vectors are supported"
            assert t.shape[-1] == self.dim, "Shape does not match with dimension"
            if len(t.shape) == 1:
                val = True
            else:
                val = np.full(t.shape[0], True)
        if self.l is not None:
            val &= np.all(t < self.l, axis=len(t.shape) - 1)
        if self.g is not None:
            val &= np.all(t > self.g, axis=len(t.shape) - 1)
        if self.le is not None:
            val &= np.all(t <= self.le, axis=len(t.shape) - 1)
        if self.ge is not None:
            val &= np.all(t >= self.ge, axis=len(t.shape) - 1)
        if self.A is not None:
            val &= np.all((self.A.dot(t.T) <= self.a).T, len(t.shape) - 1)
        if self.B is not None:
            val &= np.all((self.B.dot(t.T) < self.b).T, axis=len(t.shape) - 1)
        return val
