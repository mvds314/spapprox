#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import pytest
import warnings
from pathlib import Path

def test_example():
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
