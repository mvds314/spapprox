#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import numdifftools as nd
import numpy as np
import pandas as pd
import pytest
import scipy as sp
from scipy.integrate import quad
import scipy.stats as sps

from spapprox import cumulant_generating_function, norm, exponential


@pytest.mark.parametrize(
    "cgf_to_test,pdf,a,b",
    [
        (norm(mu=0, sigma=1), sps.norm.pdf, -10, 10),
        (norm(mu=1, sigma=0.5), sps.norm(loc=1, scale=0.5).pdf, -10, 10),
        (
            cumulant_generating_function(
                K=lambda t, mu=0, sigma=1: mu * t + sigma**2 * t**2 / 2
            ),
            sps.norm.pdf,
            -10,
            10,
        ),
        (exponential(lam=1), sps.expon.pdf, 0, 100),
        (exponential(lam=2), sps.expon(scale=1 / 2).pdf, 0, 100),
        (
            cumulant_generating_function(K=lambda t, lam=1: np.log(lam / (lam - t))),
            sps.expon.pdf,
            0,
            100,
        ),
    ],
)
def test_norm_cgf(cgf_to_test, pdf, a, b):
    assert isinstance(cgf_to_test, cumulant_generating_function)
    cgf = lambda t: np.log(quad(lambda x: pdf(x) * np.exp(t * x), a=a, b=b)[0])
    assert np.isclose(cgf(0.2), cgf_to_test.K(0.2))
    dcgf = nd.Derivative(cgf, n=1)
    assert np.isclose(dcgf(0.5), cgf_to_test.dK(0.5))
    d2cgf = nd.Derivative(cgf, n=2)
    assert np.isclose(d2cgf(0.55), cgf_to_test.d2K(0.55))
    d3cgf = nd.Derivative(cgf, n=3)
    assert np.isclose(d3cgf(0.22), cgf_to_test.d3K(0.22))


def test_domain():
    cgf = cumulant_generating_function(
        K=lambda t: t**4,
        domain=lambda t: t < 1,
    )
    assert np.isclose(cgf.K(0.5), 0.0625)
    assert np.isnan(cgf.K(1.5))
    assert np.isclose(cgf.dK(0.5), 0.5)
    assert np.isnan(cgf.dK(1.5))
    assert np.isclose(cgf.d2K(0.5), 3)
    assert np.isnan(cgf.d2K(1.5))
    assert np.isclose(cgf.d3K(0.5), 12)
    assert np.isnan(cgf.d3K(1.5))
    cgf = cumulant_generating_function(
        K=lambda t: t**4,
        domain=(1, 2),
    )
    assert np.isnan(cgf.K(0.5))
    assert ~np.isnan(cgf.K(1.5))
    assert np.isnan(cgf.dK(0.5))
    assert ~np.isnan(cgf.dK(1.5))
    assert np.isnan(cgf.d2K(0.5))
    assert ~np.isnan(cgf.d2K(1.5))
    assert np.isnan(cgf.d3K(0.5))
    assert ~np.isnan(cgf.d3K(1.5))
    cgf = cumulant_generating_function(
        K=lambda t: t**4,
        domain=(0, 2),
    )
    val = cgf.K([-1, 1])
    assert np.isnan(val[0])
    assert ~np.isnan(val[1])


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                "-k",
                "test_domain",
                "--tb=auto",
                "--pdb",
            ]
        )
