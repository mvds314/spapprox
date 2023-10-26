#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pytest
import scipy.stats as sps

from scipy.integrate import quad

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
        (norm(loc=0.5, scale=3), sps.norm(loc=0.5, scale=3), [-10, 10]),
        (norm(loc=0, scale=1), sps.norm(loc=0, scale=1), [-5, 5]),
    ],
)
def test_norm_spa(cgf, dist, trange):
    spa = SaddlePointApprox(cgf)
    t = np.linspace(*trange, 1000)
    x = spa.cgf.dK(t)
    # These ones should be exact
    assert np.allclose(spa.pdf(t=t), dist.pdf(x))
    assert np.isclose(quad(lambda t: spa.pdf(t=t) * cgf.d2K(t), a=trange[0], b=trange[1])[0], 1)
    assert np.isclose(spa.cdf(t=trange[0]), 0, atol=1e-6)
    assert np.isclose(spa.cdf(t=trange[1]), 1, atol=1e-6)


@pytest.mark.parametrize(
    "cgf,trange",
    [
        (norm(loc=0.5, scale=3), [-10, 10]),
        (norm(loc=0, scale=1), [-5, 5]),
        (exponential(scale=10), [-1e5, 1 / 10 * 0.99999999]),
    ],
)
def test_normalization(cgf, trange):
    spa = SaddlePointApprox(cgf)
    assert np.isclose(spa.cdf(t=trange[0]), 0, atol=1e-5)
    assert np.isclose(spa.cdf(t=trange[1]), 1, atol=1e-5)
    assert np.isclose(
        quad(lambda t: spa.pdf(t=t, fillna=0) * cgf.d2K(t), a=trange[0], b=trange[1])[0], 1
    )
    # TODO: do we need to test more?


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
def test_expon_spa(cgf, dist, trange):
    spa = SaddlePointApprox(cgf)
    t = np.linspace(*trange, 1000)
    x = spa.cgf.dK(t)
    # assert np.allclose(spa.pdf(t=t), dist.pdf(x))
    # TODO: how are we going to test this?
    # TODO: maybe renormalize first!


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                "-k",
                "test_normalization",
                "--tb=auto",
                "--pdb",
            ]
        )
