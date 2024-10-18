#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pytest
from spapprox.diff import Gradient, PartialDerivative, TensorDerivative, _has_findiff


@pytest.mark.skipif(not _has_findiff, reason="findiff not installed")
@pytest.mark.parametrize(
    "f, gradf, dim, h, points, error",
    [
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
        ),
        (
            lambda x: np.sum(np.square(x) * np.array([1, 2]), axis=-1),
            lambda x: 2 * x * np.array([1, 2]),
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
        ),
        (
            lambda x: np.where(np.all(x > 0, axis=-1), np.sum(np.square(x), axis=-1), np.nan),
            lambda x: np.where(
                np.tile(np.all(x > 0, axis=-1), (2, 1)).T, 2 * x, np.nan * np.ones_like(x)
            ).squeeze(),
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
        ),
        (
            lambda x: np.where(np.all(x >= 0, axis=-1), np.sum(np.square(x), axis=-1), np.nan),
            lambda x: np.where(
                np.tile(np.all(x >= 0, axis=-1), (2, 1)).T, 2 * x, np.nan * np.ones_like(x)
            ).squeeze(),
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
        ),
        (
            lambda x: np.where(np.all(x <= 0, axis=-1), np.sum(np.square(x), axis=-1), np.nan),
            lambda x: np.where(
                np.tile(np.all(x <= 0, axis=-1), (2, 1)).T, 2 * x, np.nan * np.ones_like(x)
            ).squeeze(),
            2,
            1e-6,
            np.array([np.linspace(-1, 0, 10), np.linspace(-1, 0, 10)]).T,
            None,
        ),
        (
            lambda x: np.square(x),
            lambda x: 2 * x,
            1,
            1e-6,
            np.atleast_2d(np.linspace(0, 1, 10)).T,
            None,
        ),
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            1,
            1e-6,
            np.linspace(0, 1, 10),
            ValueError,
        ),
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            0,
            1e-6,
            np.linspace(0, 1, 10),
            ValueError,
        ),
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            1,
            -1e-6,
            np.linspace(0, 1, 10),
            ValueError,
        ),
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10)] * 3).T,
            ValueError,
        ),
    ],
)
def test_grad(f, gradf, dim, h, points, error):
    if error is None:
        grad = Gradient(f, dim, h=h)
        for p in points:
            gp = grad(p)
            gradfp = gradf(p)
            assert gp.ndim == gradfp.ndim == 1
            assert len(gp) == len(gradfp) == dim
            assert np.allclose(gp, gradfp, atol=1e-6, equal_nan=True)
            # Test versus partial
            grad_from_partial = np.array(
                [
                    PartialDerivative(f, *np.eye(dim, dtype=int)[i].tolist(), h=grad._h_vect[i])(p)
                    for i in range(dim)
                ]
            )
            assert np.allclose(grad_from_partial, gp, atol=1e-6, equal_nan=True)
            # Test versus tensor derivative
            tdp = TensorDerivative(f, dim, 1, h=h, acc=2)(p)
            assert np.allclose(tdp, gp, atol=1e-6, equal_nan=True)
        assert np.allclose(grad(points), gradf(points), equal_nan=True)
    else:
        for p in points:
            with pytest.raises(error):
                grad = Gradient(f, dim, h=h)
                grad(p)
        with pytest.raises(error):
            grad = Gradient(f, dim, h=h)
            grad(points)


@pytest.mark.skipif(not _has_findiff, reason="findiff not installed")
@pytest.mark.parametrize(
    "f, gradf, ndim, h, points, error",
    [
        # Tests for the vector case
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
        ),
        (
            lambda x: np.sum(np.square(x) * np.array([1, 2]), axis=-1),
            lambda x: 2 * x * np.array([1, 2]),
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
        ),
        # Vector case with different domain constraints
        (
            lambda x: np.where(np.all(x > 0, axis=-1), np.sum(np.square(x), axis=-1), np.nan),
            lambda x: np.where(
                np.tile(np.all(x > 0, axis=-1), (2, 1)).T, 2 * x, np.nan * np.ones_like(x)
            ).squeeze(),
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
        ),
        (
            lambda x: np.where(np.all(x >= 0, axis=-1), np.sum(np.square(x), axis=-1), np.nan),
            lambda x: np.where(
                np.tile(np.all(x >= 0, axis=-1), (2, 1)).T, 2 * x, np.nan * np.ones_like(x)
            ).squeeze(),
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
        ),
        (
            lambda x: np.where(np.all(x <= 0, axis=-1), np.sum(np.square(x), axis=-1), np.nan),
            lambda x: np.where(
                np.tile(np.all(x <= 0, axis=-1), (2, 1)).T, 2 * x, np.nan * np.ones_like(x)
            ).squeeze(),
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
        ),
        # Tests for the scalar case
        (
            lambda x: np.square(x),
            lambda x: 2 * x,
            1,
            1e-6,
            np.linspace(0, 1, 10),
            None,
        ),
        (
            lambda x: np.where(np.asanyarray(x) > 0, np.square(x), np.nan),
            lambda x: np.where(
                np.asanyarray(x) > 0, 2 * np.asanyarray(x), np.nan * np.ones_like(x)
            ),
            1,
            1e-6,
            np.linspace(0, 1, 10),
            None,
        ),
        (
            lambda x: np.where(np.asanyarray(x) >= 0, np.square(x), np.nan),
            lambda x: np.where(
                np.asanyarray(x) >= 0, 2 * np.asanyarray(x), np.nan * np.ones_like(x)
            ),
            1,
            1e-6,
            np.linspace(0, 1, 10),
            None,
        ),
        (
            lambda x: np.where(np.asanyarray(x) <= 0, np.square(x), np.nan),
            lambda x: np.where(
                np.asanyarray(x) <= 0, 2 * np.asanyarray(x), np.nan * np.ones_like(x)
            ),
            1,
            1e-6,
            np.linspace(-1, 0, 10),
            None,
        ),
        # Those should raise an error
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            2,
            1e-6,
            np.linspace(-1, 0, 10),
            ValueError,
        ),
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            2,
            -1e-6,
            np.linspace(-1, 0, 10),
            ValueError,
        ),
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            1,
            [1e-6, 1e-6],
            np.linspace(-1, 0, 10),
            ValueError,
        ),
    ],
)
def test_first_order_partial_derivatives(f, gradf, ndim, h, points, error):
    if error is None:
        assert ndim >= 1, "Invalid test"
        for i in range(ndim):
            # Note we only test first order derivatives here
            if ndim == 1:
                orders = 1
                pdi = PartialDerivative(f, orders, h=h)
            else:
                orders = np.eye(ndim, dtype=int)[i].tolist()
                pdi = PartialDerivative(f, *orders, h=h)
            for p in points:
                gradfp = gradf(p)
                gradfpi = gradfp if np.isscalar(gradfp) or gradfp.ndim == 0 else gradfp[i]
                pdip = pdi(p)
                assert np.asanyarray(p).ndim <= 1, "Invalid test case"
                assert np.isscalar(pdip)
                assert np.allclose(pdip, gradfpi, atol=1e-6, equal_nan=True)
            dpipoints = pdi(points)
            assert dpipoints.ndim == 1, "A vector is expected as return value"
            if ndim <= 1:
                assert np.allclose(dpipoints, gradf(points), equal_nan=True)
            else:
                assert np.allclose(dpipoints, gradf(points)[:, i], equal_nan=True)
    else:
        for i in range(ndim):
            orders = np.eye(ndim, dtype=int)[i].tolist()
            for p in points:
                with pytest.raises(error):
                    PartialDerivative(f, *orders, h=h)(p)
            with pytest.raises(error):
                PartialDerivative(f, *orders, h=h)(points)


@pytest.mark.skipif(not _has_findiff, reason="findiff not installed")
@pytest.mark.parametrize(
    "f, df, orders, h, points, error",
    [
        # Scalar higher order derivatives
        pytest.param(
            lambda x: np.square(x),
            lambda x: 2 * np.ones_like(x),
            [2],
            None,
            np.linspace(0, 1, 10),
            None,
            id="Scalar square second order",
        ),
        pytest.param(
            lambda x: np.power(x, 3),
            lambda x: 3 * 2 * x,
            [2],
            None,
            np.linspace(0, 1, 10),
            None,
            id="Scalar cube second order",
        ),
        pytest.param(
            lambda x: np.power(x, 3),
            lambda x: 3 * 2 * np.ones_like(x),
            [3],
            None,
            np.linspace(0, 1, 2),
            None,
            id="Scalar cube third order",
        ),
        pytest.param(
            lambda x: np.power(x, 4),
            lambda x: 4 * 3 * 2 * np.ones_like(x),
            [4],
            1e-2,
            np.linspace(0, 1, 10),
            None,
            id="Scalar fourth order",
        ),
        # Vector case with higher order derivatives
        pytest.param(
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * np.ones_like(x.take(0, axis=-1)),
            [2, 0],
            None,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
            id="Vector square second order",
        ),
        pytest.param(
            lambda x: np.sum(np.power(x, 3), axis=-1),
            lambda x: 3 * 2 * x.take(1, axis=-1),
            [0, 2],
            None,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
            id="Vector cube second order",
        ),
        pytest.param(
            lambda x: np.sum(np.power(x, 3), axis=-1),
            lambda x: 3 * 2 * np.ones_like(x.take(0, axis=-1)),
            [3, 0],
            None,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
            id="Vector cube third order",
        ),
        # Higher order mixed derivates
        pytest.param(
            lambda x: np.sum(np.power(x, 2), axis=-1),
            lambda x: np.sum(np.zeros_like(x), axis=-1),
            [1, 1],
            None,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
            id="Vector square second order mixed",
        ),
        pytest.param(
            lambda x: np.prod(np.power(x, 2), axis=-1),
            lambda x: 4 * np.prod(x, axis=-1),
            [1, 1],
            None,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
            id="Vector product second order mixed",
        ),
        # Should raise Errors
        pytest.param(
            lambda x: np.sum(np.power(x, 3), axis=-1),
            lambda x: 3 * 2 * x,
            [2, 2],
            1e-6,
            np.linspace(0, 1, 10),
            ValueError,
            id="ValueError, vector case with scalar input",
        ),
    ],
)
def test_higher_order_partial_derivatives(f, df, orders, h, points, error):
    if error is None:
        pd = PartialDerivative(f, *orders, h=h, acc=2)
        for p in points:
            assert np.asanyarray(p).ndim <= 1, "Invalid test case"
            pdp = pd(p)
            assert np.isscalar(pdp)
            assert np.allclose(pdp, df(p), atol=1e-3, equal_nan=True)
        dppoints = pd(points)
        assert dppoints.ndim == 1, "A vector is expected as return value"
        assert np.allclose(dppoints, df(points), atol=1e-3, equal_nan=True)
    else:
        for p in points:
            with pytest.raises(error):
                PartialDerivative(f, *orders, h=h, acc=2)(p)
        with pytest.raises(error):
            PartialDerivative(f, *orders, h=h)(points)


@pytest.mark.parametrize(
    "f, df, dim, order, h, points, error",
    [
        pytest.param(
            lambda x: np.sum(np.square(x), axis=-1),
            np.vectorize(lambda x: 2 * x, signature="(2)->(2)"),
            2,
            1,
            None,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
            id="Tensor derivate 1D gradient",
        ),
        pytest.param(
            lambda x: np.sum(np.square(x), axis=-1),
            np.vectorize(lambda x: 2 * np.eye(2), signature="(2)->(2,2)"),
            2,
            2,
            None,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
            id="Simple 2D square, Hessian",
        ),
        pytest.param(
            lambda x: np.sum(1 / 6 * np.power(x, 3), axis=-1) + np.prod(x, axis=-1),
            np.vectorize(
                lambda x: np.diag(x) + np.ones((2, 2)) - np.eye(2), signature="(2)->(2,2)"
            ),
            2,
            2,
            None,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
            id="2D polynomial, Hessian",
        ),
        pytest.param(
            lambda x: np.sum(1 / 6 * np.power(x, 3), axis=-1) + np.prod(x, axis=-1),
            np.vectorize(
                lambda x: np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]),
                signature="(2)->(2,2,2)",
            ),
            2,
            3,
            None,
            np.array([np.linspace(0, 1, 10)] * 2).T,
            None,
            id="2D polynomial, Tressian",
            marks=pytest.mark.tofix,
        ),
        # TODO: continue here and add these tests
        # TODO: test the tressian case
        # TODO: do a more involved tressian case
        # TODO: test Hessian and Tressian explicitly
    ],
)
def test_tensor_derivative(f, df, dim, order, h, points, error):
    td = TensorDerivative(f, dim, order, h=h, acc=2)
    assert td.order == order
    assert td.dim == dim
    assert td.shape == (dim,) * order
    assert td.f is f, "f should be unique and not be copied"
    assert len(td) == dim
    # TODO: fix this part of tressian test
    if dim > 1 and td.order > 1:
        assert (
            td[tuple(np.eye(td.dim, dtype=int)[0])] is td[tuple(np.eye(td.dim, dtype=int)[1])]
        ), "Those two partials should be equal"
    for p in points:
        assert np.asanyarray(p).ndim <= 1, "Invalid test case"
        tdp = td(p)
        assert tdp.shape == td.shape
        assert np.allclose(tdp, df(p), atol=1e-3, equal_nan=True)
    tdpoints = td(points)
    assert tdpoints.ndim == td.order + 1, "A tensor is expected as return value"
    assert np.allclose(tdpoints, df(points), atol=5e-3)


# TODO: test third order tensor derivative!

# TODO: integrate those in multivariate cgfs, one by one
# TODO: test that stuff
# TODO: resolve remaining xfails


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # str(Path(__file__)) + "::test_grad",
                # str(Path(__file__)) + "::test_first_order_partial_derivatives",
                # str(Path(__file__)) + "::test_higher_order_partial_derivatives",
                # "-k",
                # "test_partial_derivative",
                # "--tb=auto",
                # "--pdb",
                "-m tofix",
                "-s",
            ]
        )
