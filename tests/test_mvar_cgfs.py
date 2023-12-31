#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import numdifftools as nd
import numpy as np
import pytest
from scipy.integrate import dblquad
import scipy.stats as sps

from spapprox import (
    UnivariateCumulantGeneratingFunction,
    MultivariateCumulantGeneratingFunction,
    Domain,
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


def test_2d_from_uniform():
    cgf1 = norm()
    cgf2 = norm()
    # Create from univariate
    mcgf_from_univ = MultivariateCumulantGeneratingFunction.from_univariate(cgf1, cgf2)
    for t in [[1, 2]]:
        assert np.allclose(mcgf_from_univ.K(t), np.sum([cgf1.K(t[0]), cgf2.K(t[1])]))
    assert np.isscalar(mcgf_from_univ.K([1, 2]))
    # The default implementation
    mcgf = multivariate_norm(loc=np.zeros(2), scale=1)
    assert np.isscalar(mcgf.K([1, 2]))
    # Manual by integration
    K = lambda t, pdf=sps.multivariate_normal.pdf: np.log(
        dblquad(lambda x, y: pdf(x, y) * np.exp(np.dot([x, y], t)), -10, 10, -10, 10)[0]
    )
    mcgf_int = MultivariateCumulantGeneratingFunction(K, dim=2)
    assert np.isscalar(mcgf_int.K([1, 2]))
    # TODO: is in domain does not work as expected
    # TODO: there is something in the wrapping logic that makes the result into an array

    # TODO: continue to do some testing from here
    # TODO: what kind of tests do we exactly want to do?
    # Maybe just implement multivariate normal in several ways
    # They all should yield the same result


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
