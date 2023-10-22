#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pytest
import scipy.stats as sps

from spapprox import (
    norm,
    exponential,
    gamma,
    chi2,
    laplace,
    poisson,
    binomial,
    SaddlePointApprox,
)


@pytest.mark.parametrize(
    "cgf,dist,trange",
    [
        (norm(loc=0.5, scale=3), sps.norm(loc=0.5, scale=3), [-3, 3]),
        (norm(loc=0, scale=1), sps.norm(loc=0, scale=1), [-3, 3]),
    ],
)
def test_norm_spa(cgf, dist, trange):
    spa = SaddlePointApprox(cgf)
    t = np.linspace(trange[0], trange[1], 1000)
    x = spa.cgf.dK(t)
    # These ones should be exact
    assert np.allclose(spa.pdf(t=t), dist.pdf(x))


@pytest.mark.parametrize(
    "cgf,dist,trange",
    [
        (
            exponential(scale=3),
            sps.expon(scale=3),
            [0, 3],
        ),
    ],
)
def test_norm_spa(cgf, dist, trange):
    spa = SaddlePointApprox(cgf)
    t = np.linspace(trange[0], trange[1], 1000)
    x = spa.cgf.dK(t)
    # assert np.allclose(spa.pdf(t=t), dist.pdf(x))


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # "-k",
                # "test_cgf",
                "--tb=auto",
                "--pdb",
            ]
        )
