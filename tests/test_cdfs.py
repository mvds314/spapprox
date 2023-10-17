#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as sps
from scipy.integrate import quad

import pytest
import warnings
from pathlib import Path
from spapprox import cumulant_generating_function, norm


def test_norm_cgf():
    ncgf = norm(mu=0, sigma=1)
    cgf = lambda t: np.log(quad(lambda x: sps.norm.pdf(x) * np.exp(t * x), a=-10, b=10)[0])
    assert isinstance(ncgf, cumulant_generating_function)
    assert np.isclose(cgf(1), ncgf.K(1))


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
