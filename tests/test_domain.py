#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pytest

from spapprox import Domain


def test_domain_1D():
    # Basic example
    dom = Domain()
    assert 1 in dom
    res = dom.is_in_domain([1, 2])
    assert not np.isscalar(res) and len(res) == 2 and all(res)
    # Should raise Errors
    with pytest.raises(AssertionError):
        Domain(l=1, g=2)
    with pytest.raises(AssertionError):
        Domain(le=1, g=2)
    with pytest.raises(AssertionError):
        Domain(le=np.array(1))
    with pytest.raises(AssertionError):
        Domain(dim=0)
    # Some tests with actual bounds
    dom = Domain(l=3, g=-1, le=2)
    assert 3 not in dom
    assert [-1, 3] not in dom
    assert 2 in dom
    assert all(dom.is_in_domain([2, 2]))


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
