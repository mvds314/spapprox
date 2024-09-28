#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pytest
from spapprox.diff import Gradient, PartialDerivative


@pytest.mark.parametrize(
    "f, df, dim, h, points, error",
    [
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T,
            None,
        ),
        (
            lambda x: np.where(np.all(x > 0, axis=-1), np.sum(np.square(x), axis=-1), np.nan),
            lambda x: np.where(
                np.tile(np.all(x > 0, axis=-1), (2, 1)).T, 2 * x, np.nan * np.ones_like(x)
            ).squeeze(),
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T,
            None,
        ),
        (
            lambda x: np.where(np.all(x >= 0, axis=-1), np.sum(np.square(x), axis=-1), np.nan),
            lambda x: np.where(
                np.tile(np.all(x >= 0, axis=-1), (2, 1)).T, 2 * x, np.nan * np.ones_like(x)
            ).squeeze(),
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T,
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
            lambda x: np.sum(np.square(x), axis=-1),
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
    ],
)
def test_grad(f, df, dim, h, points, error):
    if error is None:
        grad = Gradient(f, dim, h=h)
        for p in points:
            gp = grad(p)
            dfp = df(p)
            assert gp.ndim == dfp.ndim == 1
            assert len(gp) == len(dfp) == dim
            assert np.allclose(gp, dfp, atol=1e-6, equal_nan=True)
        assert np.allclose(grad(points), df(points), equal_nan=True)
    else:
        for p in points:
            with pytest.raises(error):
                grad = Gradient(f, dim, h=h)
                grad(p)
        with pytest.raises(error):
            grad = Gradient(f, dim, h=h)
            grad(points)


# TODO: something seems inconsistent, grad does not accept scalar input, but does provide the result of scalar output!


@pytest.mark.parametrize(
    "f, df, dim, h, points",
    [
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T,
        ),
        (
            lambda x: np.where(np.all(x > 0, axis=-1), np.sum(np.square(x), axis=-1), np.nan),
            lambda x: np.where(
                np.tile(np.all(x > 0, axis=-1), (2, 1)).T, 2 * x, np.nan * np.ones_like(x)
            ).squeeze(),
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T,
        ),
        (
            lambda x: np.where(np.all(x >= 0, axis=-1), np.sum(np.square(x), axis=-1), np.nan),
            lambda x: np.where(
                np.tile(np.all(x >= 0, axis=-1), (2, 1)).T, 2 * x, np.nan * np.ones_like(x)
            ).squeeze(),
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T,
        ),
        (
            lambda x: np.where(np.all(x <= 0, axis=-1), np.sum(np.square(x), axis=-1), np.nan),
            lambda x: np.where(
                np.tile(np.all(x <= 0, axis=-1), (2, 1)).T, 2 * x, np.nan * np.ones_like(x)
            ).squeeze(),
            2,
            1e-6,
            np.array([np.linspace(-1, 0, 10), np.linspace(-1, 0, 10)]).T,
        ),
    ],
)
def test_partial_derivative(f, df, dim, h, points):
    for i in range(dim):
        orders = [0] * dim
        orders[i] = 1
        pdi = PartialDerivative(f, *orders, h=h)
        for p in points:
            assert np.allclose(pdi(p), df(p)[i], atol=1e-6, equal_nan=True)
        assert np.allclose(pdi(points), df(points)[:, i], equal_nan=True)


# TODO: test gradient in 1 dim case -> does it get squeezed, and do we want that?

# TODO: test equivalance partial derivative and the gradient

# TODO: build and test higher order derivatives


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # str(Path(__file__)) + "::test_grad",
                # "-k",
                # "test_partial_derivative",
                # "--tb=auto",
                "--pdb",
                "-s",
            ]
        )
