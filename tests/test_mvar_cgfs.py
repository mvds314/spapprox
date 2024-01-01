#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

import pytest
from scipy.integrate import dblquad, quad
import scipy.stats as sps

from spapprox import (
    # UnivariateCumulantGeneratingFunction,
    MultivariateCumulantGeneratingFunction,
    # Domain,
    norm,
    multivariate_norm,
    # exponential,
    # gamma,
    # chi2,
    # laplace,
    # poisson,
    # binomial,
    # univariate_sample_mean,
    # univariate_empirical,
)
from spapprox.util import type_wrapper


def test_2d_from_uniform():
    cgf1 = norm()
    cgf2 = norm()
    # From univariate
    mcgf_from_univ = MultivariateCumulantGeneratingFunction.from_univariate(cgf1, cgf2)
    for t in [[1, 2]]:
        assert np.allclose(mcgf_from_univ.K(t), np.sum([cgf1.K(t[0]), cgf2.K(t[1])]))
    assert np.isscalar(mcgf_from_univ.K([1, 2]))
    # The default implementation
    mcgf = multivariate_norm(loc=np.zeros(2), scale=1)
    assert np.isscalar(mcgf.K([1, 2]))
    # Manual by integration
    # K = lambda t, pdf=sps.multivariate_normal(mean=np.zeros(2), cov=np.eye(2)).pdf: np.log(
    #     dblquad(lambda x, y: pdf([x, y]) * np.exp(np.dot([x, y], t)), -6, 6, -6, 6)[0]
    # ) #Takes too long
    K = lambda t, pdfx=sps.norm().pdf, pdfy=sps.norm().pdf: np.log(
        quad(lambda x: pdfx(x) * np.exp(x * t[0]), -6, 6)[0]
        * quad(lambda y: pdfy(y) * np.exp(y * t[1]), -6, 6)[0]
    )
    K = type_wrapper()(np.vectorize(K, signature="(n)->()"))
    mcgf_int = MultivariateCumulantGeneratingFunction(K, dim=2)
    assert np.isscalar(mcgf_int.K([1, 2]))
    # Compare the implementations
    ts = [[1, 2], [0, 0], [1, 0], [0, 1]]
    for t in ts:
        val = mcgf.K(t)
        assert np.isscalar(val)
        assert np.allclose(val, mcgf_from_univ.K(t))
        assert np.allclose(val, mcgf_int.K(t), atol=1e-3)
    val = [mcgf.K(t) for t in ts]
    assert np.allclose(mcgf.K(ts), val)
    assert np.allclose(mcgf_from_univ.K(ts), val)
    assert np.allclose(mcgf_int.K(ts), val, atol=1e-3)
    # Test the derivatives
    for t in ts:
        val = mcgf.dK(t)
        assert pd.api.types.is_array_like(val) and len(val.shape) == 1 and len(val) == 2
        assert np.allclose(val, mcgf_from_univ.dK(t))
        assert np.allclose(val, mcgf_int.dK(t), atol=1e-3)
    val = [mcgf.dK(t) for t in ts]
    assert np.allclose(mcgf.dK(ts), val)
    # assert np.allclose(mcgf_from_univ.dK(ts), val)
    # assert np.allclose(mcgf_int.dK(ts), val)
    # Test the second derivatives
    for t in ts:
        val = mcgf.d2K(t)
        assert pd.api.types.is_array_like(val) and len(val.shape) == 2 and val.shape == (2, 2)
        assert np.allclose(val, mcgf_from_univ.d2K(t))
        assert np.allclose(val, mcgf_int.d2K(t), atol=1e-3)
    val = np.array([mcgf.d2K(t) for t in ts])
    assert np.allclose(mcgf.d2K(ts), val)
    # assert np.allclose(mcgf_from_univ.d2K(ts), val)
    # assert np.allclose(mcgf_int.d2K(ts), val)

    # A bit more vectorized tests
    # What to do with the third derivative?
    # Do we need to test other things?
    # Seperate test for transformations


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                "-k",
                "test_2d_from_uniform",
                "--tb=auto",
                "--pdb",
            ]
        )
