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

from spapprox import Domain


def test_domain():
    domain = Domain()
    assert 1 in domain
    import pdb

    pdb.set_trace()
    1 in domain
    domain._is_in_domain([1, 2])

    domain


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # "-k",
                # "test_return_type",
                "--tb=auto",
                "--pdb",
            ]
        )
