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


def test_spa():
    cgf = norm()
    spa = SaddlePointApprox(cgf)
    t = np.linspace(-3, 3)
    x = spa.cgf.dK(t)
    spa.pdf(t=t)
    sps.norm.pdf(x)
    assert np.allclose(spa.pdf(t=t), sps.norm.pdf(x))


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
