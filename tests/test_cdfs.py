#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import pytest
import warnings
from pathlib import Path
from spapprox import cumulant_generating_function, norm


def test_norm_cgf():
    ncgf = norm(mu=0, sigma=1)
    assert isinstance(ncgf, cumulant_generating_function)
    assert True


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
