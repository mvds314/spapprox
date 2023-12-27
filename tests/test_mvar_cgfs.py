#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import numdifftools as nd
import numpy as np
import pytest
from scipy.integrate import quad
import scipy.stats as sps

from spapprox import (
    UnivariateCumulantGeneratingFunction,
    MultivariateCumulantGeneratingFunction,
    Domain,
    norm,
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
    mcgf = MultivariateCumulantGeneratingFunction.from_univariate(cgf1, cgf2)
    for t in [[1, 2]]:
        assert np.allclose(mcgf.K(t), np.sum([cgf1.K(t[0]), cgf2.K(t[1])]))
    # TODO: continue to do some testing from here
    pass


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
