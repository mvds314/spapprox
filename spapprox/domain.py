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
                x is None or np.isscalar(x) for x in [l, le, ge, g]
            ), "Bounds should be scalars in dim 1."
            specified_bounds = [x for x in [l, le, ge, g] if x is not None]
            if len(specified_bounds) > 1:
                assert all(
                    specified_bounds[i] > specified_bounds[i + 1]
                    for i in range(len(specified_bounds) - 1)
                ), "Bounds should satisfy: g>ge>le>l"
        else:
            assert all(
                x is None or np.isscalar(x) or len(x) == dim for x in [l, le, ge, g]
            ), "Bounds should be scalars or have matching dim"
            specified_bounds = [np.asanyarray(x) for x in [l, le, ge, g] if x is not None]
            if len(specified_bounds) > 1:
                df = pd.DataFrame(
                    data=np.nan, index=range(self.dim), columns=["l", "le", "ge", "g"]
                )
                df["l"] = l
                df["le"] = le
                df["ge"] = ge
                df["g"] = g
                for _, sr in df.iterrows():
                    assert sr.dropna().is_monotonic_decreasing, "Bounds should satisfy: g>ge>le>l"
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

    @property
    def has_lower_bounds(self):
        return self.g is not None or self.ge is not None

    @property
    def has_upper_bounds(self):
        return self.l is not None or self.le is not None

    @property
    def has_inclusive_bounds(self):
        return self.le is not None or self.ge is not None

    @property
    def has_strict_bounds(self):
        return self.l is not None or self.g is not None

    @property
    def has_bounds(self):
        return self.has_inclusive_bounds or self.has_strict_bounds

    @property
    def has_ineq_constraints(self):
        return self.A is not None or self.B is not None

    @property
    def l_vect(self):
        if not hasattr(self, "_l_vect_cache"):
            if self.l is None:
                self._l_vect_cache = np.full(self.dim, np.nan)
            elif np.isscalar(self.l):
                self._l_vect_cache = np.full(self.dim, self.l)
            elif isinstance(self.l, np.ndarray) and len(self.l.shape) == 0:
                self._l_vect_cache = np.full(self.dim, self.l.tolist())
            elif isinstance(self.l, np.ndarray) and len(self.l.shape) == 1:
                self._l_vect_cache = self.l
            else:
                raise ValueError("Unexpected value encountered")
        return self._l_vect_cache

    @property
    def g_vect(self):
        if not hasattr(self, "_g_vect_cache"):
            if self.g is None:
                self._g_vect_cache = np.full(self.dim, np.nan)
            elif np.isscalar(self.g):
                self._g_vect_cache = np.full(self.dim, self.g)
            elif isinstance(self.g, np.ndarray) and len(self.g.shape) == 0:
                self._g_vect_cache = np.full(self.dim, self.g.tolist())
            elif isinstance(self.g, np.ndarray) and len(self.g.shape) == 1:
                self._g_vect_cache = self.g
            else:
                raise ValueError("Unexpected value encountered")
        return self._g_vect_cache

    @property
    def le_vect(self):
        if not hasattr(self, "_le_vect_cache"):
            if self.le is None:
                self._le_vect_cache = np.full(self.dim, np.nan)
            elif np.isscalar(self.le):
                self._le_vect_cache = np.full(self.dim, self.le)
            elif isinstance(self.le, np.ndarray) and len(self.le.shape) == 0:
                self._le_vect_cache = np.full(self.dim, self.le.tolist())
            elif isinstance(self.le, np.ndarray) and len(self.le.shape) == 1:
                self._le_vect_cache = self.le
            else:
                raise ValueError("Unexpected value encountered")
        return self._le_vect_cache

    @property
    def ge_vect(self):
        if not hasattr(self, "_ge_vect_cache"):
            if self.ge is None:
                self._ge_vect_cache = np.full(self.dim, np.nan)
            elif np.isscalar(self.ge):
                self._ge_vect_cache = np.full(self.dim, self.ge)
            elif isinstance(self.ge, np.ndarray) and len(self.ge.shape) == 0:
                self._ge_vect_cache = np.full(self.dim, self.ge.tolist())
            elif isinstance(self.ge, np.ndarray) and len(self.ge.shape) == 1:
                self._ge_vect_cache = self.ge
            else:
                raise ValueError("Unexpected value encountered")
        return self._ge_vect_cache

    def add(self, other):
        r"""
        Translation of the domain by :math:`\beta`.

        If :math:`x` is in the domain of :math:`\tilde x`, where :math:`\tilde x=x+\beta`.
        """
        assert other is not None
        if pd.api.types.is_number(other):
            a = None if self.a is None else self.a + self.A.dot(np.full(self.dim, other))
            b = None if self.b is None else self.b + self.B.dot(np.full(self.dim, other))
        elif isinstance(other, np.ndarray) and len(other.shape) == 0:
            return self.add(other.tolist())
        elif pd.api.types.is_array_like(other) and len(other) == self.dim:
            a = None if self.a is None else self.a + self.A.dot(other)
            b = None if self.b is None else self.b + self.B.dot(other)
        else:
            raise ValueError("Can only add a scalar or a vector (of length dim)")
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

    def __sub__(self, other):
        return self.add(-1 * other)

    def __rsub__(self, other):
        return (-1 * self).__add__(other)

    def mul(self, other):
        r"""
        Left side multiplication by a factor :math:`\alpha`.
        It stretches the domain by a factor :math:`\alpha`.

        If :math:`x` is in the domain of :math:`\tilde x`, where :math:`\tilde x= \alpha  x`.

        In the multivariate case, multiplication by a vector is implemented as component wise
        multiplication.
        """
        assert other is not None
        if pd.api.types.is_number(other):
            assert not np.isclose(other, 0), "Cannot multiply with zero"
            A = None if self.A is None else self.A * np.sign(other)
            B = None if self.B is None else self.B * np.sign(other)
            a = None if self.a is None else self.a * np.abs(other)
            b = None if self.b is None else self.b * np.abs(other)
            if other > 0:
                l = self.l * other if self.l is not None else None
                g = self.g * other if self.g is not None else None
                le = self.le * other if self.le is not None else None
                ge = self.ge * other if self.ge is not None else None
            else:
                g = self.l * other if self.l is not None else None
                l = self.g * other if self.g is not None else None
                ge = self.le * other if self.le is not None else None
                le = self.ge * other if self.ge is not None else None
        elif isinstance(other, np.ndarray) and len(other.shape) == 0:
            return self.mul(other.tolist())
        elif self.dim > 1 and pd.api.types.is_array_like(other):
            other = np.asanyarray(other)
            assert not np.any(np.isclose(other, 0)), "Cannot multiply with zero"
            assert len(other.shape) == 1 and len(other) == self.dim, "Invalid shape"
            # Divide each row of A by other
            A = None if self.A is None else np.divide(self.A, other)
            B = None if self.B is None else np.divide(self.B, other)
            a = self.a
            b = self.b
            sgncond = np.sign(other) > 0
            if np.all(sgncond):  # all positive
                l = self.l * other if self.l is not None else None
                g = self.g * other if self.g is not None else None
                le = self.le * other if self.le is not None else None
                ge = self.ge * other if self.ge is not None else None
            elif not np.any(sgncond):  # all negative
                g = self.l * other if self.l is not None else None
                l = self.g * other if self.g is not None else None
                ge = self.le * other if self.le is not None else None
                le = self.ge * other if self.ge is not None else None
            else:
                if self.has_strict_bounds:
                    l = np.full(self.dim, np.nan) if self.l is None else self.l
                    g = np.full(self.dim, np.nan) if self.g is None else self.g
                    l, g = (
                        np.where(sgncond, l * other, g * other),
                        np.where(sgncond, g * other, l * other),
                    )
                else:
                    g = None
                    l = None
                if self.has_inclusive_bounds:
                    le = np.full(self.dim, np.nan) if self.le is None else self.le
                    ge = np.full(self.dim, np.nan) if self.ge is None else self.ge
                    le, ge = (
                        np.where(sgncond, le * other, ge * other),
                        np.where(sgncond, ge * other, le * other),
                    )
                else:
                    ge = None
                    le = None
        else:
            raise ValueError(
                "Can only multiply with a scalar or a vector of length dim (when dim>1)."
            )
        return Domain(l=l, g=g, le=le, ge=ge, A=A, a=a, B=B, b=b, dim=self.dim)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def ldot(self, other):
        r"""
        Left side dot product. Leads to the domain of :math:`y=Cx`.
        This extends the multiplication (scaling) functionality to generic linear transformations.

        As :math:`A` is not necessarily invertible, sometimes this transformation is not possible.

        Also, note that this might expand the domain, as several bounds are combined into one.

        Not completely sure how useful this is.
        """
        assert other is not None
        other = np.asanyarray(other)
        if self.has_ineq_constraints:
            try:
                inv = np.linalg.inv(other)
            except np.linalg.LinAlgError:
                raise Exception(
                    "Provided transformation must be invertible for domains with ineq constraints"
                )
            return self.ldotinv(inv)
        elif len(other.shape) == 1 and len(other) == self.dim:
            return self.ldot(np.atleast_2d(other))
        elif len(other.shape) == 2 and other.shape[1] == self.dim:
            if self.l is None:
                l = None
            elif np.isscalar(self.l):
                l = other.dot(np.full(self.dim, self.l))
            else:
                l = other.dot(self.l)
            if self.g is None:
                g = None
            elif np.isscalar(self.g):
                g = other.dot(np.full(self.dim, self.g))
            else:
                g = other.dot(self.g)
            if self.le is None:
                le = None
            elif np.isscalar(self.le):
                le = other.dot(np.full(self.dim, self.le))
            else:
                le = other.dot(self.le)
            if self.ge is None:
                ge = None
            elif np.isscalar(self.ge):
                ge = other.dot(np.full(self.dim, self.ge))
            else:
                ge = other.dot(self.ge)
            dim = other.shape[0] if len(other.shape) == 2 else 1
            if dim == 1:
                l = l[0] if l is not None else None
                g = g[0] if g is not None else None
                le = le[0] if le is not None else None
                ge = ge[0] if ge is not None else None
            return Domain(l=l, g=g, le=le, ge=ge, A=None, a=None, B=None, b=None, dim=dim)
        else:
            raise ValueError("Invalid shape")

    def ldotinv(self, other):
        """
        Left side inverse dot product.
        This leads to the domain :math:`y`, where :math:`Cy=x`.
        """
        # Initialize
        assert other is not None
        other = np.asanyarray(other)
        if len(other.shape) == 1 and len(other) == self.dim:
            if not self.has_bounds and not self.has_ineq_constraints:
                return Domain(dim=1)
            else:
                return self.ldot(np.expand_dims(other, axis=0))
        elif len(other.shape) == 2 and other.shape[0] == self.dim:
            if not self.has_bounds and not self.has_ineq_constraints:
                return Domain(dim=other.shape[1])
            # Define A and a
            if self.has_inclusive_bounds:
                A = np.full((0, self.dim), 0) if self.A is None else self.A
                a = np.full(0, 0) if self.a is None else self.a
                if self.le is not None:
                    A = np.vstack((A, np.eye(self.dim)))
                    a = np.append(
                        a,
                        np.full(self.dim, self.le) if np.isscalar(self.le) else self.le,
                    )
                if self.ge is not None:
                    A = np.vstack((A, -np.eye(self.dim)))
                    a = np.append(
                        a,
                        np.full(self.dim, self.ge) if np.isscalar(self.ge) else self.ge,
                    )
                sel = ~np.isnan(a)
                A = A[sel]
                a = a[sel]
            else:
                A = self.A
                a = self.a
            # Define B and b
            if self.has_strict_bounds:
                B = np.full((0, self.dim), 0) if self.B is None else self.B
                b = np.full(0, 0) if self.b is None else self.b
                if self.l is not None:
                    B = np.vstack((B, np.eye(self.dim)))
                    b = np.append(b, np.full(self.dim, self.l) if np.isscalar(self.l) else self.l)
                if self.g is not None:
                    B = np.vstack((B, -np.eye(self.dim)))
                    b = np.append(b, np.full(self.dim, self.g) if np.isscalar(self.g) else self.g)
                sel = ~np.isnan(b)
                B = B[sel]
                b = b[sel]
            else:
                B = self.B
                b = self.b
        else:
            raise ValueError("Invalid shape")
        return Domain(
            l=None,
            g=None,
            le=None,
            ge=None,
            A=A if A is None else A.dot(other),
            a=a,
            B=B if B is None else B.dot(other),
            b=b,
            dim=other.shape[1],
        )

    def intersect(self, other):
        """
        Intersect with another domain, i.e., the result should be in both.
        """
        assert other is not None and isinstance(
            other, Domain
        ), "Can only intersect with another Domain"
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

    def stack(self, *domains):
        dim = sum(d.dim for d in domains)
        l = (
            np.hstack(tuple([d.l_vect for d in domains]))
            if any(d.l is not None for d in domains)
            else None
        )
        g = (
            np.hstack(tuple([d.g_vect for d in domains]))
            if any(d.g is not None for d in domains)
            else None
        )
        le = (
            np.hstack(tuple([d.le_vect for d in domains]))
            if any(d.le is not None for d in domains)
            else None
        )
        ge = (
            np.hstack(tuple([d.ge_vect for d in domains]))
            if any(d.ge is not None for d in domains)
            else None
        )
        if any(d.A is not None for d in domains):
            A = None
            a = None
        else:
            A = None
            a = None
        if any(d.B is not None for d in domains):
            B = None
            b = None
        else:
            B = None
            b = None
        return Domain(dim=dim, l=l, g=g, le=le, ge=ge, A=A, a=a, B=B, b=b)

    @type_wrapper(xloc=1)
    def is_in_domain(self, t):
        if self.dim == 1:
            val = np.full(t.shape, pd.notnull(t))
            t = np.expand_dims(t, axis=len(t.shape))
        else:
            assert len(t.shape) in [1, 2], "Only vectors or lists of vectors are supported"
            assert t.shape[-1] == self.dim, "Shape does not match with dimension"
            if len(t.shape) == 1:
                val = pd.notnull(t).all()
            else:
                val = pd.notnull(t).all(axis=1)
        if self.l is not None:
            val &= np.all((t < self.l) | np.isnan(self.l), axis=len(t.shape) - 1)
        if self.g is not None:
            val &= np.all((t > self.g) | np.isnan(self.g), axis=len(t.shape) - 1)
        if self.le is not None:
            val &= np.all((t <= self.le) | np.isnan(self.le), axis=len(t.shape) - 1)
        if self.ge is not None:
            val &= np.all((t >= self.ge) | np.isnan(self.ge), axis=len(t.shape) - 1)
        if self.A is not None:
            val &= np.all((self.A.dot(t.T) <= self.a).T, len(t.shape) - 1)
        if self.B is not None:
            val &= np.all((self.B.dot(t.T) < self.b).T, axis=len(t.shape) - 1)
        return val
