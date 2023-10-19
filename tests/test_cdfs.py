#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

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
    [(norm(mu=0, sigma=1), sps.norm.pdf, -10, 10), (exponential(lam=1), sps.expon.pdf, 0, 100)],
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


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # "-k",
                # "test_example",
                "--tb=auto",
                "--pdb",
            ]
        )
