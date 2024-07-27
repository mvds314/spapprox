#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pytest
from spapprox.diff import Gradient


@pytest.mark.parametrize(
    "f, df, dim, h, points",
    [
        (
            lambda x: np.sum(np.square(x), axis=-1),
            lambda x: 2 * x,
            2,
            1e-6,
            np.array([np.linspace(0, 1, 10), np.linspace(0, 1, 10)]).T,
        )
        # TODO: what other cases to test?
    ],
)
def test_grad(f, df, dim, h, points):
    grad = Gradient(f, dim, h=h)
    for p in points:
        assert np.allclose(grad(p), df(p), atol=1e-6)
    assert np.allclose(grad(points), df(points))
    # TODO: what else to test?


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                "-k",
                "test_grad",
                "--tb=auto",
                # "--pdb",
            ]
        )
