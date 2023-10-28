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
    assert np.allclose(spa.pdf(t=t, normalize_pdf=False), dist.pdf(x))
    assert np.isclose(
        quad(lambda t: spa.pdf(t=t, normalize_pdf=False) * cgf.d2K(t), a=trange[0], b=trange[1])[
            0
        ],
        1,
    )
    assert np.isclose(spa.cdf(t=trange[0]), 0, atol=1e-6)
    assert np.isclose(spa.cdf(t=trange[1]), 1, atol=1e-6)


@pytest.mark.parametrize(
    "cgf,trange",
    [
        (norm(loc=0.5, scale=3), [-10, 10]),
        (norm(loc=0, scale=1), [-5, 5]),
        (exponential(scale=10), [-1e5, 1 / 10 * 0.99999999]),
        (gamma(a=2, scale=3), [-1e3, 1 / 3 * 0.99999999]),
        (chi2(df=3), [-1e4, 1 / 2 * 0.99999999]),
        (laplace(loc=1, scale=3), [-1 / 3 * 0.99999999, 1 / 3 * 0.99999999]),
    ],
)
def test_normalization(cgf, trange):
    spa = SaddlePointApprox(cgf)
    assert np.isclose(spa.cdf(t=trange[0]), 0, atol=1e-5)
    assert np.isclose(spa.cdf(t=trange[1]), 1, atol=1e-5)
    assert np.isclose(
        quad(
            lambda t: spa.pdf(t=t, fillna=0, normalize_pdf=True) * cgf.d2K(t, fillna=0),
            a=-np.inf,
            b=np.inf,
        )[0],
        1,
    )
    assert np.isclose(
        quad(
            lambda t: spa.pdf(t=t, fillna=0, normalize_pdf=True) * cgf.d2K(t, fillna=0),
            a=trange[0],
            b=trange[1],
        )[0],
        1,
    )


@pytest.mark.parametrize(
    "cgf,dist,trange",
    [
        (
            exponential(scale=3),
            sps.expon(scale=3),
            [0, 1 / 3],
        ),
    ],
)
def test_expon_spa(cgf, dist, trange):
    spa = SaddlePointApprox(cgf)
    for f in [
        spa.pdf,
        lambda t=None, fillna=np.nan, backend="LR": spa.cdf(t=t, fillna=fillna, backend=backend),
        lambda t=None, fillna=np.nan, backend="BN": spa.cdf(t=t, fillna=fillna, backend=backend),
    ]:
        assert np.isscalar(f(t=0))
        assert ~np.isscalar(f(t=[0, 0]))
        assert np.isscalar(f(t=1 / 3)) and np.isnan(f(t=1 / 3))
        assert ~np.isscalar(f(t=[1 / 3, 1])) and np.isnan(f(t=[1, 1 / 3])).all()
        assert ~np.isscalar(f(t=[1 / 3, 1], fillna=0)) and np.allclose(
            f(t=[1, 1 / 3], fillna=10), 10
        )
        assert ~np.isscalar(spa.pdf(t=[1 / 3, 0])) and ~np.isnan(spa.pdf(t=[1, 0])).all()
    # approximation accuracy test
    t = np.linspace(*trange, 1000)[:-1]
    x = spa.cgf.dK(t)
    assert np.allclose(spa.pdf(t=t), dist.pdf(x)), "This should approx be equal"
    assert np.allclose(
        spa.cdf(t=t, backend="LR"), dist.cdf(x), atol=5e-3
    ), "This should approx be equal"
    assert np.allclose(
        spa.cdf(t=t, backend="BN"), dist.cdf(x), atol=5e-3
    ), "This should approx be equal"
    assert not np.allclose(
        spa.cdf(t=t, backend="BN"), spa.cdf(t=t, backend="LR")
    ), "the approximation should not be exactly equal"


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # "-k",
                # "test_expon_spa",
                "--tb=auto",
                "--pdb",
            ]
        )