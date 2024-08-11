#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pytest
from spapprox.diff import Gradient, PartialDerivative


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
def test_grad(f, df, dim, h, points):
    grad = Gradient(f, dim, h=h)
    for p in points:
        equal = np.allclose(grad(p), df(p), atol=1e-6)
        allnan = np.isnan(df(p)).all() and np.isnan(grad(p)).all()
        assert equal or allnan
    assert np.allclose(grad(points), df(points), equal_nan=True)


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
def test_partial_derivative():
    pass
    if True:
        pass
    # TODO: first test equivalance partial derivative and the gradient
    # TODO: test higher order derivatives


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                "-k",
                "test_grad",
                # "--tb=auto",
                # "--pdb",
                "-s",
            ]
        )
