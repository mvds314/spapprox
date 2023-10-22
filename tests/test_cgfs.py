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

from spapprox import (
    cumulant_generating_function,
    norm,
    exponential,
    poisson,
    gamma,
    chi2,
)


@pytest.mark.parametrize(
    "cgf_to_test,cgf,ts",
    [
        (
            norm(loc=0, scale=1),
            lambda t, pdf=sps.norm.pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
            ),
            [0.2, 0.55],
        ),
        (
            norm(loc=1, scale=0.5),
            lambda t: np.log(
                quad(
                    lambda x, pdf=sps.norm(loc=1, scale=0.5).pdf: pdf(x) * np.exp(t * x), a=-5, b=5
                )[0]
            ),
            [0.2, 0.55],
        ),
        (
            cumulant_generating_function(
                K=lambda t, loc=0, scale=1: loc * t + scale**2 * t**2 / 2
            ),
            lambda t, pdf=sps.norm.pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
            ),
            [0.2, 0.55],
        ),
        (
            exponential(scale=1),
            lambda t, pdf=sps.expon.pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
            ),
            [0.2, 0.55],
        ),
        (
            exponential(scale=0.5),
            lambda t, pdf=sps.expon(scale=0.5).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
            ),
            [0.2, 0.55],
        ),
        (
            cumulant_generating_function(K=lambda t: np.log(1 / (1 - t))),
            lambda t, pdf=sps.expon.pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
            ),
            [0.2, 0.55],
        ),
        (
            poisson(mu=2),
            lambda t, pmf=sps.poisson(mu=2).pmf: np.log(
                np.sum([np.exp(t * x) * pmf(x) for x in range(100)])
            ),
            [0.2, 0.55],
        ),
        (
            gamma(a=2, scale=0.5),
            lambda t, pdf=sps.gamma(a=2, scale=0.5).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
            ),
            [0.2, 0.55],
        ),
        (
            chi2(df=3),
            lambda t, pdf=sps.chi2(df=3).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
            ),
            [0.2, 0.25],
        ),
    ],
)
def test_cgf(cgf_to_test, cgf, ts):
    assert isinstance(cgf_to_test, cumulant_generating_function)
    for t in ts:
        assert np.isclose(cgf(t), cgf_to_test.K(t))
        dcgf = nd.Derivative(cgf, n=1)
        assert np.isclose(dcgf(t), cgf_to_test.dK(t))
        d2cgf = nd.Derivative(cgf, n=2)
        assert np.isclose(d2cgf(t), cgf_to_test.d2K(t))
        d3cgf = nd.Derivative(cgf, n=3)
        assert np.isclose(d3cgf(t), cgf_to_test.d3K(t))


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
                "test_cgf",
                "--tb=auto",
                "--pdb",
            ]
        )
