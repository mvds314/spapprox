#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from spapprox.diff import Gradient


def test_grad():
    grad = Gradient(lambda x: x**2)
    pass


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                "-k",
                "test_grad",
                "--tb=auto",
                "--pdb",
            ]
        )
