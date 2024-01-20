#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

import pytest
from scipy.integrate import quad  # , dblquad
import scipy.stats as sps
from statsmodels.stats.moment_helpers import cov2corr

from spapprox import (
    UnivariateCumulantGeneratingFunction,
    MultivariateCumulantGeneratingFunction,
    # Domain,
    norm,
    multivariate_norm,
    # exponential,
    # gamma,
    # chi2,
    # laplace,
    # poisson,
    # binomial,
    # univariate_sample_mean,
    # univariate_empirical,
)
from spapprox.util import type_wrapper


def test_2d_from_uniform():
    cgf1 = norm()
    cgf2 = norm()
    # From univariate
    mcgf_from_univ = MultivariateCumulantGeneratingFunction.from_univariate(cgf1, cgf2)
    for t in [[1, 2]]:
        assert np.allclose(mcgf_from_univ.K(t), np.sum([cgf1.K(t[0]), cgf2.K(t[1])]))
    assert np.isscalar(mcgf_from_univ.K([1, 2]))
    # The default implementation
    mcgf = multivariate_norm(loc=np.zeros(2), scale=1)
    assert np.isscalar(mcgf.K([1, 2]))
    # Manual by integration
    # K = lambda t, pdf=sps.multivariate_normal(mean=np.zeros(2), cov=np.eye(2)).pdf: np.log(
    #     dblquad(lambda x, y: pdf([x, y]) * np.exp(np.dot([x, y], t)), -6, 6, -6, 6)[0]
    # ) #Takes too long
    K = lambda t, pdfx=sps.norm().pdf, pdfy=sps.norm().pdf: np.log(
        quad(lambda x: pdfx(x) * np.exp(x * t[0]), -6, 6)[0]
        * quad(lambda y: pdfy(y) * np.exp(y * t[1]), -6, 6)[0]
    )
    K = type_wrapper()(np.vectorize(K, signature="(n)->()"))
    mcgf_int = MultivariateCumulantGeneratingFunction(K, dim=2)
    assert np.isscalar(mcgf_int.K([1, 2]))
    # Compare the implementations
    ts = [[1, 2], [0, 0], [1, 0], [0, 1]]
    for t in ts:
        val = mcgf.K(t)
        assert np.isscalar(val)
        assert np.allclose(val, mcgf_from_univ.K(t))
        assert np.allclose(val, mcgf_int.K(t), atol=1e-3)
    val = np.array([mcgf.K(t) for t in ts])
    assert np.allclose(mcgf.K(ts), val)
    assert np.allclose(mcgf_from_univ.K(ts), val)
    assert np.allclose(mcgf_int.K(ts), val, atol=1e-3)
    # Test the derivatives
    for t in ts:
        val = mcgf.dK(t)
        assert pd.api.types.is_array_like(val) and len(val.shape) == 1 and len(val) == 2
        assert np.allclose(val, mcgf_from_univ.dK(t))
        assert np.allclose(val, mcgf_int.dK(t), atol=1e-3)
    val = np.array([mcgf.dK(t) for t in ts])
    assert np.allclose(mcgf.dK(ts), val)
    assert np.allclose(mcgf_from_univ.dK(ts), val)
    assert np.allclose(mcgf_int.dK(ts), val, atol=1e-3)
    # Test the second derivatives
    for t in ts:
        val = mcgf.d2K(t)
        assert pd.api.types.is_array_like(val) and len(val.shape) == 2 and val.shape == (2, 2)
        assert np.allclose(val, mcgf_from_univ.d2K(t))
        assert np.allclose(val, mcgf_int.d2K(t), atol=1e-3)
    val = np.array([mcgf.d2K(t) for t in ts])
    assert np.allclose(mcgf.d2K(ts), val)
    assert np.allclose(mcgf_from_univ.d2K(ts), val)
    assert np.allclose(mcgf_int.d2K(ts), val, atol=1e-3)


def test_statistics():
    # Standard multivariate normal
    mcgf = multivariate_norm()
    assert np.allclose(0, mcgf.mean)
    assert np.allclose(np.eye(mcgf.dim), mcgf.cov)
    # Multivariate normal with covmat specification
    loc = [1, 2]
    cov = np.array([[2, 1], [1, 2]])
    mcgf = multivariate_norm(loc=loc, cov=cov)
    assert np.allclose(loc, mcgf.mean)
    assert np.allclose(cov, mcgf.cov)
    assert np.allclose(np.sqrt(2), mcgf.std)
    # Multivariate normal with scale specification
    scale = np.array([[1, 2], [3, 4]])
    loc = [1, 2]
    mcgf = multivariate_norm(loc=loc, scale=scale)
    assert np.allclose(loc, mcgf.mean)
    assert np.allclose(scale.dot(scale.T), mcgf.cov)
    assert np.allclose(np.sqrt(np.diag(scale.dot(scale.T))), mcgf.std)


def test_addition():
    # add vector
    mcgf1 = MultivariateCumulantGeneratingFunction.from_univariate(norm() + 1, norm() + 2)
    mcgf2 = multivariate_norm(loc=np.zeros(2), scale=1) + np.array([1, 2])
    assert mcgf1.dim == mcgf2.dim == 2
    ts = [[1, 2], [0, 0], [1, 0], [0, 1]]
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    # add a vector in a different way
    mcgf2 = multivariate_norm(loc=np.zeros(2), scale=1)
    mcgf2.add(np.array([1, 2]), inplace=True)
    assert mcgf1.dim == mcgf2.dim == 2
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    # and when specified as loc
    mcgf2 = multivariate_norm(loc=np.array([1, 2]), scale=1)
    assert mcgf1.dim == mcgf2.dim == 2
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    # Add a constant
    mcgf2 = multivariate_norm(loc=np.array([0, 1]), scale=1) + 1
    assert mcgf1.dim == mcgf2.dim == 2
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    # add scalar
    mcgf1 = multivariate_norm(loc=np.zeros(2), scale=1) + 1
    mcgf2 = multivariate_norm(loc=np.ones(2), scale=1)
    assert mcgf1.dim == mcgf2.dim == 2
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    # Add multivariate cumulant genering function
    mcgf1 = multivariate_norm(loc=np.ones(2), scale=1) + multivariate_norm(
        loc=np.zeros(2), scale=1
    )
    mcgf2 = multivariate_norm(loc=np.ones(2), scale=np.sqrt(2))
    assert mcgf1.dim == mcgf2.dim == 2
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    # Add multivariate cumulant genering function
    mcgf1 = multivariate_norm(loc=np.ones(2), scale=1) + norm(loc=0, scale=1)
    mcgf2 = multivariate_norm(loc=np.ones(2), cov=np.array([[2, 1], [1, 2]]))
    assert mcgf1.dim == mcgf2.dim == 2
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)


def test_multiplication():
    # multiply vector in several equivalent ways
    mcgf1 = MultivariateCumulantGeneratingFunction.from_univariate(norm(), norm() * 2)
    mcgf2 = multivariate_norm(loc=np.zeros(2), scale=1) * np.array([1, 2])
    assert mcgf1.dim == mcgf2.dim == 2
    ts = [[1, 2], [0, 0], [1, 0], [0, 1]]
    for t in ts:
        assert np.isclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    mcgf2 = multivariate_norm(loc=np.zeros(2), scale=1)
    mcgf2.mul(np.array([1, 2]), inplace=True)
    assert mcgf1.dim == mcgf2.dim == 2
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    mcgf2 = multivariate_norm(loc=0, scale=np.array([1, 2]))
    assert mcgf1.dim == mcgf2.dim == 2
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    mcgf2 = multivariate_norm(loc=0, scale=np.diag(np.array([1, 2])))
    assert mcgf1.dim == mcgf2.dim == 2
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    # Multiply by a constant
    mcgf1 = multivariate_norm(loc=np.array([0, 3]), scale=3)
    mcgf2 = multivariate_norm(loc=np.array([0, 1]), scale=1) * 3
    assert mcgf1.dim == mcgf2.dim == 2
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)


# TODO: test with transformations
def test_ldot():
    # Multiplication with identity
    mcgf1 = multivariate_norm(loc=np.zeros(2), scale=1).ldot(2 * np.eye(2))
    mcgf2 = multivariate_norm(loc=np.zeros(2), scale=2)
    assert isinstance(mcgf1, MultivariateCumulantGeneratingFunction)
    assert mcgf1.dim == mcgf2.dim == 2
    ts = [[1, 2], [0, 0], [1, 0], [0, 1]]
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    # Correlate through matrix multiplication
    cov = np.array([[2, 1], [1, 3]])
    cor, std = cov2corr(cov, return_std=True)
    mcgf1 = multivariate_norm(loc=np.zeros(2), scale=1).ldot(
        np.linalg.cholesky(cor)
    ) * std + np.array([1, 2])
    mcgf2 = multivariate_norm(loc=[1, 2], cov=cov)
    assert isinstance(mcgf1, MultivariateCumulantGeneratingFunction)
    assert mcgf1.dim == mcgf2.dim == 2
    assert np.allclose(mcgf1.cov, cov)
    assert np.allclose(mcgf2.cov, cov)
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    # Projection vs addition
    mcgf1 = multivariate_norm(loc=0, scale=1, dim=3).ldot(np.array([[1, 0, 1], [0, 1, 1]]))
    mcgf2 = multivariate_norm(loc=0, scale=1, dim=2) + norm()
    assert mcgf1.dim == mcgf2.dim == 2
    assert mcgf1.domain.dim == 3
    assert mcgf2.domain.dim == 2
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
    assert np.allclose(mcgf2.cov, mcgf1.cov)

    # TODO: continue here: test vectorized evaluation of mcgf! also with addtition and multiplication

    # TODO: test the projection on the 1dim case, but implement slicing first
    # mcgf1 = multivariate_norm(loc=np.zeros(2), scale=1).ldot(np.ones(2))
    # assert isinstance(mcgf1, UnivariateCumulantGeneratingFunction)
    # mcgf2 = norm(loc=0, scale=2)


# TODO: implement slicing and test that
def test_stack_and_slicing():
    pass


# TODO: implement dKinv
def test_dKinv():
    pass


# TODO: from univariate does not stack domains -> test this properly
def test_univariate():
    pass


# TODO: test vectorized evaluation of the derivatives
# TODO: Can we do things more efficient?
# TODO: For example, do not use loc_vect, or scale mat unnecisarily, better work it out with if statements?
# TODO: decide on third derivative
# TODO: double check if domain is handled properly everywhere

if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                "-k",
                "test_multiplication",
                "--tb=auto",
                "--pdb",
            ]
        )
