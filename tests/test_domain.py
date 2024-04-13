#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
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
    # Test At <= a
    dom = Domain(A=np.array([[1]]), a=np.array([2]))
    assert 1 in dom
    assert len(dom.is_in_domain([1, 1])) == 2 and all(dom.is_in_domain([2, 2]))
    assert len(dom.is_in_domain([3, 3])) == 2 and not any(dom.is_in_domain([3, 3]))
    # Test At <= a and Bt<b bounds
    dom = Domain(A=np.array([[1]]), a=np.array([2]), B=np.array([[2]]), b=np.array([1]))
    assert 1 not in dom
    assert len(dom.is_in_domain([0.1, 0.1])) == 2 and all(dom.is_in_domain([0.2, 0.2]))
    assert len(dom.is_in_domain([0.5, 0.5])) == 2 and not any(dom.is_in_domain([0.5, 0.5]))
    # Test combined bounds
    dom = Domain(l=2, A=np.array([[1]]), a=np.array([2]))
    assert 1 in dom
    assert 2 not in dom
    res = dom.is_in_domain([1, 2])
    assert pd.api.types.is_array_like(res) and len(res) == 2 and all(res == [True, False])


def test_domain_nD():
    # Basic example
    dom = Domain(dim=3)
    with pytest.raises(AssertionError):
        1 in dom
    assert [1, 1, 1] in dom
    assert np.isscalar(dom.is_in_domain([1, 1, 1]))
    res = dom.is_in_domain([[1, 1, 1], [2, 2, 2]])
    assert not np.isscalar(res) and len(res) == 2 and all(res)
    # Should raise Errors
    with pytest.raises(AssertionError):
        Domain(l=1, g=[0, 0, 2], dim=3)
    with pytest.raises(AssertionError):
        Domain(g=[0, 0], dim=3)
    with pytest.raises(AssertionError):
        Domain(le=[1, 3, 3], g=2, dim=3)
    # Some tests with actual bounds
    dom = Domain(l=3, g=-1, le=2, dim=3)
    assert [3, 3, 3] not in dom
    assert [[3, 0, 0], [0, 0, 0]] not in dom
    assert [[0, 0, 0], [0, 0, 0]] in dom
    res = dom.is_in_domain([0, 0, 0])
    assert np.isscalar(res) and res
    res = dom.is_in_domain([[3, 0, 0], [0, 0, 0]])
    assert pd.api.types.is_array_like(res) and all(res == [False, True])
    # Another test with actual bounds
    dom = Domain(l=3, g=-1, le=2, dim=1)
    assert [3] not in dom
    assert [[0]] in dom
    res = dom.is_in_domain([0])
    assert np.isscalar(res) and res
    res = dom.is_in_domain([[3], [0]])
    assert pd.api.types.is_array_like(res) and all(res == [False, True])
    # Test At <= a
    with pytest.raises(AssertionError):
        Domain(A=np.array([[1]]), a=np.array([2]), dim=3)
    dom = Domain(A=np.array([[1, 2, 3]]), a=np.array([2]), dim=3)
    assert np.zeros(3) in dom
    assert np.zeros((2, 3)) in dom
    res = dom.is_in_domain([0.5, 0, 0.5])
    assert np.isscalar(res) and res
    res = dom.is_in_domain([[0.5, 0, 0.5], [0.5, 1e-4, 0.5]])
    assert pd.api.types.is_array_like(res) and len(res) == 2 and all(res == [True, False])
    # Test Bt<b bounds
    with pytest.raises(AssertionError):
        Domain(B=np.array([[1]]), b=np.array([2]), dim=3)
    dom = Domain(B=np.array([[1, 2, 3]]), b=np.array([2]), dim=3)
    assert np.zeros(3) in dom
    assert np.zeros((2, 3)) in dom
    res = dom.is_in_domain([0.5, 0, 0.5])
    assert np.isscalar(res) and not res
    res = dom.is_in_domain([[0.5, 0, 0.5], [0.5, -1e-4, 0.5]])
    assert pd.api.types.is_array_like(res) and len(res) == 2 and all(res == [False, True])
    # Test combined bounds
    dom = Domain(l=0.5, A=np.array([[1, 2, 3]]), a=np.array([2]), dim=3)
    assert [0.5, 0, 0.5] not in dom
    assert [0.4, 0, 0.4] in dom
    res = dom.is_in_domain([[0.5, 0, 0.5], [0.4, 0, 0.4]])
    assert pd.api.types.is_array_like(res) and len(res) == 2 and all(res == [False, True])


def test_trans_domain_1D():
    # Addtion
    dom = Domain(l=3, g=-1, le=2)
    assert 3 not in dom
    assert -0.9 in dom
    dom = dom + 1
    assert 3 in dom
    assert -0.9 not in dom
    assert 4 not in dom
    # Scaling
    dom = 2 * Domain(l=3, g=-1, le=2)
    assert 4 in dom
    assert 4.1 not in dom
    assert -2 not in dom
    assert -2.01 not in dom
    # Negative scaling
    dom = -2 * Domain(l=3, g=-1, le=2)
    assert -4 in dom
    assert -4.1 not in dom
    assert 2 not in dom
    assert 2.01 not in dom
    # Test transformation with A, B
    dom = Domain(l=3, g=-1, le=2, A=[[1]], a=[1])
    assert 2 not in dom
    dom = dom + 1
    assert 2 in dom
    with pytest.raises(ValueError):
        dom.mul(2 * np.ones(1))
    assert 4 not in dom
    assert 4 in dom.mul(2)
    assert -4 not in dom.mul(-1)
    assert -4 in dom.mul(-2)


def test_trans_domain_nD():
    # Addtion
    dom = Domain(l=3, g=-1, le=2, dim=3)
    assert [3, 2, 1] not in dom
    assert -0.9 * np.ones(3) in dom
    dom = dom + 1
    assert [3, 2, 1] in dom
    assert -0.9 * np.ones(3) not in dom
    # Scaling
    with pytest.raises(AssertionError):
        Domain(l=3, g=-1, le=2, dim=3).mul(0)
    dom = 2 * Domain(l=3, g=-1, le=2, dim=3)
    assert [4, 0, 0] in dom
    assert [4.1, 0, 0] not in dom
    assert [0, -2, 0] not in dom
    assert [0, -2.01, 0] not in dom
    # Scaling with vector
    with pytest.raises(AssertionError):
        Domain(l=3, g=-1, le=2, dim=3).mul(2 * np.ones(1))
    with pytest.raises(AssertionError):
        Domain(l=3, g=-1, le=2, dim=3).mul(np.zeros(3))
    dom = Domain(l=3, g=-1, le=2, dim=3).mul(2 * np.ones(3))
    assert [4, 0, 0] in dom
    assert [4.1, 0, 0] not in dom
    assert [0, -2, 0] not in dom
    assert [0, -2.01, 0] not in dom
    dom = Domain(l=3, g=-1, le=2, dim=3).mul(np.array([-2, 2, 2]))
    assert [4, 0, 0] not in dom
    assert [-4, 0, 0] in dom
    assert [2, 0, 0] not in dom
    assert [-4.1, 0, 0] not in dom
    assert [0, -2, 0] not in dom
    assert [0, -2.01, 0] not in dom
    # Test transformation with A, B
    dom = Domain(l=3, ge=-1, le=2, A=[[1, 2, 3]], a=[1], dim=3)
    assert [2, 0, 0] not in dom
    dom = dom + 1
    assert [2, 0, 0] in dom
    assert [0, 1, 0] in dom
    # Test multiplication ldot
    with pytest.raises(Exception):
        Domain(l=3, ge=-1, le=2, A=[[1, 2, 3]], a=[1], dim=3).ldot(np.ones(3))
    with pytest.raises(ValueError):
        Domain(l=3, ge=-1, le=2, dim=3).ldot(np.ones(2))
    dom = Domain(l=3, ge=-1, le=2, dim=3)
    assert [2, 2, 2] in dom
    assert [-1, -1, -1] in dom
    assert [2.1, 0, 0] not in dom
    dom = dom.ldot(np.ones(3))
    assert dom.dim == 1
    assert 6 in dom
    assert 6.1 not in dom
    assert -3 in dom
    assert -3.1 not in dom
    # Test multiplication ldotinv
    with pytest.raises(ValueError):
        Domain(l=3, ge=-1, le=2, dim=1).ldotinv(np.ones(2))
    with pytest.raises(ValueError):
        Domain(l=3, ge=-1, le=2, dim=1).ldotinv(np.ones((2, 1)))
    dom = Domain(l=3, ge=-1, le=2, dim=3)
    assert [2, 2, 2] in dom
    assert [-1, -1, -1] in dom
    assert [2.1, 0, 0] not in dom
    dom = dom.ldotinv(np.ones((3, 4)))
    assert dom.dim == 4
    assert [1, 1, 0, 0] in dom
    assert [1, 1, 1, 0] not in dom
    dom = Domain(l=3, ge=-1, le=2, dim=3).ldotinv(np.ones((3, 1)))
    assert 2 in dom
    assert 3 not in dom
    # Test ldotinv in 1 dim
    dom = dom.ldotinv(np.ones((1, 2)))
    assert dom.dim == 2
    assert [1, 1] in dom
    assert [-1, -1] not in dom
    dom = Domain(l=3, ge=-1, le=2, dim=1).ldotinv(np.ones((1, 2)))
    assert dom.dim == 2
    # Test invertible matrix case
    dom = Domain(l=3, ge=-1, le=2, dim=2).ldot(np.eye(2))
    assert dom.dim == 2
    assert [3, 3] not in dom
    assert [2, 2] in dom


def test_from_domains():
    # Initialize
    dom1 = Domain(l=3, g=-1, le=2, dim=3)
    dom2 = Domain(l=3, g=-1, le=2, dim=2) + 4
    dom = Domain.from_domains(dom1, dom2)
    # Tests
    assert dom.dim == dom1.dim + dom2.dim
    for x in [[1, 2, 1, 5, 6], [0, 0, 0, 0, 0]]:
        if x[: dom1.dim] in dom1 and x[dom1.dim :] in dom2:
            assert x in dom
        else:
            assert x not in dom


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                "-k",
                "test_domain_nD",
                "--tb=auto",
                "--pdb",
            ]
        )
