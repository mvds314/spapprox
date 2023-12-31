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
    # Test At <= a and Bt<b bounds
    dom = Domain(A=np.array([[1]]), a=np.array([2]))
    assert 1 in dom
    assert len(dom.is_in_domain([1, 1])) == 2 and all(dom.is_in_domain([2, 2]))
    assert len(dom.is_in_domain([3, 3])) == 2 and not any(dom.is_in_domain([3, 3]))
    dom = Domain(A=np.array([[1]]), a=np.array([2]), B=np.array([[2]]), b=np.array([1]))
    assert 1 not in dom
    assert len(dom.is_in_domain([0.1, 0.1])) == 2 and all(dom.is_in_domain([0.2, 0.2]))
    assert len(dom.is_in_domain([0.5, 0.5])) == 2 and not any(dom.is_in_domain([0.5, 0.5]))


def test_trans_domain_1D():
    # Addtion
    dom = Domain(l=3, g=-1, le=2)
    assert 3 not in dom
    assert -0.9 in dom
    dom = dom + 1
    assert 3 in dom
    assert -0.9 not in dom
    assert 4 not in dom
    # Test transformation with A, B
    dom = Domain(l=3, g=-1, le=2, A=[[1]], a=[1])
    assert 2 not in dom
    dom = dom + 1
    assert 2 in dom
    with pytest.raises(ValueError):
        dom.mul(2 * np.ones(1))
    # Scaling
    dom = 2 * Domain(l=3, g=-1, le=2)
    assert 4 in dom
    assert 4.1 not in dom
    assert -2 not in dom
    assert -2.01 not in dom


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # "-k",
                # "test_trans_domain_1D",
                "--tb=auto",
                "--pdb",
            ]
        )
