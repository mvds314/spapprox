from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy as sp
import scipy.stats as sps
from scipy.integrate import quad  # , dblquad
from statsmodels.stats.moment_helpers import cov2corr

from spapprox import (
    MultivariateCumulantGeneratingFunction,
    UnivariateCumulantGeneratingFunction,
    bivariate_gamma,
    exponential,
    gamma,
    multivariate_norm,
    norm,
)
from spapprox.diff import block_diag_3d
from spapprox.util import type_wrapper


@pytest.mark.slow
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
    def K(t, pdfx=sps.norm().pdf, pdfy=sps.norm().pdf):
        return np.log(
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
    # Test inverse of the derivatives
    for t in ts:
        assert np.allclose(mcgf.dK_inv(mcgf.dK(t)), t)
        assert np.allclose(mcgf_from_univ.dK_inv(mcgf_from_univ.dK(t)), t)
        assert np.allclose(mcgf_int.dK_inv(mcgf_int.dK(t)), t, atol=1e-3)
    val = np.array([mcgf.dK(t) for t in ts])
    assert np.allclose(mcgf.dK_inv(val), ts)
    assert np.allclose(mcgf_from_univ.dK_inv(val), ts)
    assert np.allclose(mcgf_int.dK_inv(val), ts, atol=1e-3)
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
    # TODO: test third order derivatives


@pytest.mark.parametrize(
    "mcgf,mean,cov",
    [
        # Standard multivariate normal
        (multivariate_norm(), 0, np.eye(2)),
        # Multivariate normal with covmat specification
        (
            multivariate_norm(loc=[1, 2], cov=np.array([[2, 1], [1, 2]])),
            [1, 2],
            np.array([[2, 1], [1, 2]]),
        ),
        # Multivariate normal with scale specification
        (
            multivariate_norm(loc=[1, 2], scale=np.arange(1, 5).reshape((2, 2))),
            [1, 2],
            np.arange(1, 5).reshape((2, 2)).dot(np.arange(1, 5).reshape((2, 2)).T),
        ),
        (bivariate_gamma([1, 2, 3]), [1 + 2, 1 + 3], np.array([[1 + 2, 1], [1, 1 + 3]])),
    ],
)
def test_statistics(mcgf, mean, cov):
    assert np.allclose(mean, mcgf.mean)
    assert np.allclose(np.sqrt(np.diag(cov)), mcgf.std)
    assert np.allclose(cov, mcgf.cov)


@pytest.mark.parametrize(
    "mcgf1,mcgf2,ts,dim",
    [
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(norm() + 1, norm() + 2),
            multivariate_norm(loc=np.zeros(2), scale=1) + np.array([1, 2]),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Add vector",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                norm() + 1, norm() + 2, dK0=[1, 2], d2K0=np.eye(2), d3K0=np.zeros((2, 2, 2))
            ),
            multivariate_norm(loc=np.zeros(2), scale=1) + np.array([1, 2]),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Add vector and test derivatives at zero logic",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(norm() + 1, norm() + 2),
            multivariate_norm(loc=np.zeros(2), scale=1).add(np.array([1, 2])),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Add vector in a different way",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(norm() + 1, norm() + 2),
            multivariate_norm(loc=np.array([1, 2]), scale=1),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Add and compare using loc",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(norm() + 1, norm() + 2),
            multivariate_norm(loc=np.array([0, 1]), scale=1) + 1,
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Add a constant",
        ),
        pytest.param(
            multivariate_norm(loc=np.zeros(2), scale=1) + 1,
            multivariate_norm(loc=np.ones(2), scale=1),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Add scalar",
        ),
        pytest.param(
            multivariate_norm(loc=np.ones(2), scale=1)
            + multivariate_norm(loc=np.zeros(2), scale=1),
            multivariate_norm(loc=np.ones(2), scale=np.sqrt(2)),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Add multivariate cumulant generating function",
        ),
        pytest.param(
            multivariate_norm(loc=np.ones(2), scale=1) + norm(loc=0, scale=1),
            multivariate_norm(loc=np.ones(2), cov=np.array([[2, 1], [1, 2]])),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Addition with covariance structure",
        ),
        pytest.param(
            multivariate_norm(loc=np.ones(2), cov=np.array([[2, 1], [1, 2]]))
            + multivariate_norm(loc=np.ones(2), cov=np.array([[2, 1], [1, 2]])),
            np.sqrt(2)
            * multivariate_norm(loc=np.sqrt(2) * np.ones(2), cov=np.array([[2, 1], [1, 2]])),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Another test with a covariance structure",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(gamma(1.2), gamma(1.3))
            + gamma(1.1),
            bivariate_gamma([1.1, 1.2, 1.3]),
            [[-1, -2], [0, 0], [0.9, 0], [0, 0.9]],
            2,
            id="Test with nonzero higher order derivatives through bivariate gamma",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                gamma(1.2),
                gamma(1.3),
                dK0=[gamma(1.2).dK0, gamma(1.3).dK0],
                d2K0=np.diag([gamma(1.2).d2K0, gamma(1.3).d2K0]),
                d3K0=np.array([[[gamma(1.2).d3K0, 0], [0, 0]], [[0, 0], [0, gamma(1.3).d3K0]]]),
            ),
            bivariate_gamma([0, 1.2, 1.3]),
            [[-1, -2], [0, 0], [0.9, 0], [0, 0.9]],
            2,
            id="Test with nonzero higher order derivatives specified at 0 through bivariate gamma",
        ),
    ],
)
def test_addition(mcgf1, mcgf2, ts, dim):
    assert mcgf1.dim == mcgf2.dim == dim
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
        assert np.allclose(mcgf1.d3K(t), mcgf2.d3K(t))
        assert np.allclose(mcgf1.dK_inv(mcgf1.dK(t)), t)
        assert np.allclose(mcgf2.dK_inv(mcgf2.dK(t)), t)
    for f in ["K", "dK", "d2K", "d3K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    assert np.allclose(mcgf1.dK_inv(mcgf1.dK(ts)), ts)
    assert np.allclose(mcgf2.dK_inv(mcgf2.dK(ts)), ts)
    assert np.allclose(mcgf1.dK0, mcgf2.dK0)
    assert np.allclose(mcgf1.d2K0, mcgf2.d2K0)
    assert np.allclose(mcgf1.d3K0, mcgf2.d3K0)


@pytest.mark.parametrize(
    "mcgf1,mcgf2,ts,dim",
    [
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(norm(), norm() * 2),
            multivariate_norm(loc=np.zeros(2), scale=1) * np.array([1, 2]),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Multiply normal by vector",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(gamma(1.2), gamma(1.3) * 2),
            bivariate_gamma([0, 1.2, 1.3]) * np.array([1, 2]),
            [[0.1, 0.2], [0, 0], [0.1, 0], [0, 0.1]],
            2,
            id="Multiply gamma by vector",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(norm(), norm() * 2),
            multivariate_norm(loc=np.zeros(2), scale=1) / np.array([1, 0.5]),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Divide by vector",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(gamma(1.2), gamma(1.3) * 0.5)
            + MultivariateCumulantGeneratingFunction.from_univariate(gamma(1.1)).ldot(
                [[1], [0.5]]
            ),
            bivariate_gamma([1.1, 1.2, 1.3]) / np.array([1, 2]),
            [[0.1, 0.2], [0, 0], [0.1, 0], [0, 0.1]],
            2,
            marks=pytest.mark.tofix,
            id="Divide gamma by vector",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(norm(), norm() * 2),
            multivariate_norm(loc=np.zeros(2), scale=1).mul(np.array([1, 2]), inplace=True),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Multiply by vector inplace",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(norm(), norm() * 2),
            multivariate_norm(loc=0, scale=np.array([1, 2])),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Multiply through scaling with vector",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(norm(), norm() * 2),
            multivariate_norm(loc=0, scale=np.diag(np.array([1, 2]))),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Multiply through scaling with diagonal matrix",
        ),
        pytest.param(
            multivariate_norm(loc=np.array([0, 3]), scale=3),
            multivariate_norm(loc=np.array([0, 1]), scale=1) * 3,
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Multiply by constant",
        ),
        pytest.param(
            multivariate_norm(loc=np.array([0, 3]), scale=3),
            multivariate_norm(loc=np.array([0, 1]), scale=1) / (1 / 3),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
            id="Divide by constant",
        ),
        # TODO: add bivariate gamma
    ],
)
def test_multiplication_and_division(mcgf1, mcgf2, ts, dim):
    assert mcgf1.dim == mcgf2.dim == dim
    for t in ts:
        assert np.isclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.dK_inv(mcgf1.dK(t)), t)
        assert np.allclose(mcgf2.dK_inv(mcgf2.dK(t)), t)
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
        assert np.allclose(mcgf1.d3K(t), mcgf2.d3K(t))
    for f in ["K", "dK", "d2K", "d3K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    assert np.allclose(mcgf1.dK_inv(mcgf1.dK(ts)), ts)
    assert np.allclose(mcgf2.dK_inv(mcgf2.dK(ts)), ts)
    assert np.allclose(mcgf1.dK0, mcgf2.dK0)
    assert np.allclose(mcgf1.d2K0, mcgf2.d2K0)
    assert np.allclose(mcgf1.d3K0, mcgf2.d3K0)


@pytest.mark.parametrize(
    "mcgf1,mcgf2,ts,dim",
    [
        # Multiplication with identity
        (
            multivariate_norm(loc=np.zeros(2), scale=1).ldot(2 * np.eye(2)),
            multivariate_norm(loc=np.zeros(2), scale=2),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
        ),
        (
            multivariate_norm(loc=np.zeros(2), scale=1).ldot(
                np.linalg.cholesky(cov2corr(np.array([[2, 1], [1, 3]]), return_std=False))
            )
            * np.sqrt(np.array([2, 3]))
            + np.array([1, 2]),
            multivariate_norm(loc=[1, 2], cov=np.array([[2, 1], [1, 3]])),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
        ),
        (
            multivariate_norm(loc=0, scale=1, dim=3).ldot(np.array([[1, 0, 1], [0, 1, 1]])),
            multivariate_norm(loc=0, scale=1, dim=2) + norm(),
            [[1, 2], [0, 0], [1, 0], [0, 1]],
            2,
        ),
        (
            multivariate_norm(loc=np.zeros(2), scale=1).ldot(np.ones(2)),
            norm(loc=0, scale=np.sqrt(2)),
            [-1, -2, 0, 2, 4],
            None,
        ),
        (
            multivariate_norm(loc=np.zeros(2), scale=1).ldot(np.atleast_2d(np.ones(2)))[[0]],
            multivariate_norm(loc=0, scale=np.sqrt(2), dim=1),
            [[-1], [-2], [0], [2], [4]],
            None,
        ),
        (
            multivariate_norm(loc=np.zeros(2), scale=1).ldot(np.atleast_2d(np.ones(2))),
            multivariate_norm(loc=0, scale=np.sqrt(2), dim=1),
            [[-1], [-2], [0], [2], [4]],
            None,
        ),
        # TODO: add bivariate gamma
    ],
)
def test_ldot(mcgf1, mcgf2, ts, dim):
    assert isinstance(mcgf1, mcgf2.__class__)
    if dim is not None:
        assert mcgf1.dim == mcgf2.dim == dim
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
        assert np.allclose(mcgf1.dK_inv(mcgf1.dK(t)), t)
        assert np.allclose(mcgf2.dK_inv(mcgf2.dK(t)), t)
    # TODO: add d3K
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    assert np.allclose(mcgf1.dK_inv(mcgf1.dK(ts)), ts)
    assert np.allclose(mcgf2.dK_inv(mcgf2.dK(ts)), ts)
    if not isinstance(mcgf1, UnivariateCumulantGeneratingFunction):
        assert np.allclose(mcgf2.cov, mcgf1.cov)


@pytest.mark.parametrize(
    "mcgf1,mcgf2,ts,dim",
    [
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_cgfs(
                multivariate_norm(loc=np.zeros(2), scale=1),
                multivariate_norm(loc=np.zeros(2), scale=2),
            ),
            multivariate_norm(loc=np.zeros(4), scale=np.array([1, 1, 2, 2])),
            [[1, 2, 3, 4], [0, 0, 0, 0], [1, 0, 2, 3], [0, 1, 0, 1]],
            4,
            id="Stack mulvariate normals with difference scale",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_cgfs(
                multivariate_norm(loc=[1, 2], cov=np.array([[2, 1], [1, 3]])),
                multivariate_norm(loc=[3, 4], cov=np.array([[2, 0.5], [0.5, 1]])),
            ),
            multivariate_norm(
                loc=[1, 2, 3, 4],
                cov=sp.linalg.block_diag(
                    np.array([[2, 1], [1, 3]]), np.array([[2, 0.5], [0.5, 1]])
                ),
            ),
            [[1, 2, 3, 4], [0, 0, 0, 0], [1, 0, 2, 3], [0, 1, 0, 1]],
            4,
            id="Stack different multivariate normals",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_cgfs(
                multivariate_norm(loc=[1, 2], cov=np.array([[2, 1], [1, 3]])),
                norm(loc=3, scale=2),
                norm(loc=4, scale=1),
            ),
            multivariate_norm(
                loc=[1, 2, 3, 4],
                cov=sp.linalg.block_diag(np.array([[2, 1], [1, 3]]), np.array([[4, 0], [0, 1]])),
            ),
            [[1, 2, 3, 4], [0, 0, 0, 0], [1, 0, 2, 3], [0, 1, 0, 1]],
            4,
            id="Stack different multivariate normals and univariates",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_cgfs(
                bivariate_gamma([1.1, 1.2, 1.3]),
                norm(loc=3, scale=2),
            ),
            MultivariateCumulantGeneratingFunction.from_univariate(
                gamma(1.2), gamma(1.3), norm(loc=3, scale=2)
            )
            + MultivariateCumulantGeneratingFunction.from_univariate(gamma(1.1)).ldot(
                [[1], [1], [0]]
            ),
            [[0.1, 0.2, 0.3], [0, 0, 0], [0.1, 0, 0.2], [0, 0.1, 0]],
            3,
            id="Stack different gammas and univariates",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_cgfs(
                bivariate_gamma([1.1, 1.2, 1.3]),
                norm(loc=3, scale=2),
                dK0=np.append(bivariate_gamma([1.1, 1.2, 1.3]).dK0, norm(loc=3, scale=2).dK0),
                d2K0=sp.linalg.block_diag(
                    bivariate_gamma([1.1, 1.2, 1.3]).d2K0, norm(loc=3, scale=2).d2K0
                ),
                d3K0=block_diag_3d(
                    bivariate_gamma([1.1, 1.2, 1.3]).d3K0, np.atleast_3d(norm(loc=3, scale=2).d3K0)
                ),
            ),
            MultivariateCumulantGeneratingFunction.from_univariate(
                gamma(1.2), gamma(1.3), norm(loc=3, scale=2)
            )
            + MultivariateCumulantGeneratingFunction.from_univariate(gamma(1.1)).ldot(
                [[1], [1], [0]]
            ),
            [[0.1, 0.2, 0.3], [0, 0, 0], [0.1, 0, 0.2], [0, 0.1, 0]],
            3,
            id="Stack different gammas and univariates with derivatives at zero specified",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_cgfs(
                bivariate_gamma([1.1, 1.2, 1.3]),
                dK0=bivariate_gamma([1.1, 1.2, 1.3]).dK0,
                d2K0=bivariate_gamma([1.1, 1.2, 1.3]).d2K0,
                d3K0=bivariate_gamma([1.1, 1.2, 1.3]).d3K0,
            ),
            bivariate_gamma([1.1, 1.2, 1.3]),
            [[0.1, 0.2], [0, 0], [0.1, 0], [0, 0.1]],
            2,
            id="Stack different gammas and univariates with derivatives at zero specified",
        ),
    ],
)
def test_stack(mcgf1, mcgf2, ts, dim):
    assert isinstance(mcgf1, MultivariateCumulantGeneratingFunction)
    assert mcgf1.dim == mcgf2.dim == dim
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.dK_inv(mcgf1.dK(t)), t)
        assert np.allclose(mcgf2.dK_inv(mcgf2.dK(t)), t)
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
        assert np.allclose(mcgf1.d3K(t), mcgf2.d3K(t))
    for f in ["K", "dK", "d2K", "d3K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    assert np.allclose(mcgf1.dK_inv(mcgf1.dK(ts)), ts)
    assert np.allclose(mcgf2.dK_inv(mcgf2.dK(ts)), ts)
    assert np.allclose(mcgf1.dK0, mcgf2.dK0)
    assert np.allclose(mcgf1.d2K0, mcgf2.d2K0)
    assert np.allclose(mcgf1.d3K0, mcgf2.d3K0)


@pytest.mark.parametrize(
    "mcgf1,mcgf2,ts,dim",
    [
        (
            multivariate_norm(loc=np.zeros(4), scale=np.array([1, 1, 2, 2]))[0],
            norm(loc=0, scale=1),
            [-2, -1, 0, 1, 2],
            None,
        ),
        (
            multivariate_norm(loc=[1, 2, 3, 4], scale=np.array([1, 1, 2, 2]))[-1],
            norm(loc=4, scale=2),
            [-2, -1, 0, 1, 2],
            None,
        ),
        (
            MultivariateCumulantGeneratingFunction.from_cgfs(
                multivariate_norm(loc=np.zeros(2), scale=1),
                multivariate_norm(loc=np.zeros(2), scale=2),
            )[0],
            norm(loc=0, scale=1),
            [-2, -1, 0, 1, 2],
            None,
        ),
        (
            multivariate_norm(loc=[1, 2, 3, 4], scale=np.array([1, 1, 2, 2]))[[1, 2]],
            multivariate_norm(loc=[2, 3], scale=np.array([1, 2])),
            [[-2, -1], [0, 1], [2, 0]],
            2,
        ),
        (
            multivariate_norm(loc=[1, 2, 3], scale=np.diag([1, 2, 3]))[[0, 2]],
            multivariate_norm(loc=[1, 3], scale=np.diag([1, 3])),
            [[-2, -1], [0, 1], [2, 0]],
            2,
        ),
        (
            multivariate_norm(
                loc=[1, 2, 3],
                cov=np.array([[1, 0.5, 0.1], [0.5, 2, 0.2], [0.1, 0.2, 3]]),
            )[[0, 2]],
            multivariate_norm(loc=[1, 3], cov=np.array([[1, 0.1], [0.1, 3]])),
            [[-2, -1], [0, 1], [2, 0]],
            2,
        ),
        # TODO: add bivariate gamma
    ],
)
def test_slicing(mcgf1, mcgf2, ts, dim):
    assert isinstance(mcgf1, mcgf2.__class__)
    if dim is not None:
        assert mcgf1.dim == mcgf2.dim == dim
    for t in ts:
        assert np.allclose(mcgf1.K(t), mcgf2.K(t))
        assert np.allclose(mcgf1.dK(t), mcgf2.dK(t))
        assert np.allclose(mcgf1.d2K(t), mcgf2.d2K(t))
        assert np.allclose(mcgf1.dK_inv(mcgf1.dK(t)), t)
        assert np.allclose(mcgf2.dK_inv(mcgf2.dK(t)), t)
    # TODO: add d3K
    for f in ["K", "dK", "d2K"]:
        val = np.array([getattr(mcgf1, f)(t) for t in ts])
        assert np.allclose(getattr(mcgf1, f)(ts), val)
        assert np.allclose(getattr(mcgf2, f)(ts), val)
    assert np.allclose(mcgf1.dK_inv(mcgf1.dK(ts)), ts)
    assert np.allclose(mcgf2.dK_inv(mcgf2.dK(ts)), ts)


@pytest.mark.parametrize(
    "mcgf,ts",
    [
        # multivariate normal with inverse defined
        (
            multivariate_norm(dim=2),
            [[-2, -1], [0, 1], [2, 0]],
        ),
        (
            multivariate_norm(dim=2, scale=[1, 2], loc=[1, 2]),
            [[-2, -1], [0, 1], [2, 0]],
        ),
        # multivariate normal without inverse defined
        (
            MultivariateCumulantGeneratingFunction.from_univariate(norm(), norm()),
            [[-2, -1], [0, 1], [2, 0]],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(norm(), norm()).ldot(
                np.array([[1, 2], [2, 1]])
            ),
            [[-2, -1], [0, 1], [2, 0]],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(
                norm(loc=2, scale=3), norm(loc=1, scale=2)
            ).ldot(np.array([[1, 2], [2, 1], [3, 2]])),
            [[-2, -1, 0], [0, 1, 1], [2, 0, 0]],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(
                norm(loc=2, scale=3), norm(loc=1, scale=2)
            ).ldot(np.array([[1, 2], [2, 1], [3, 2]])),
            [[-2, -1, 0]],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(
                norm(loc=2, scale=3), exponential(loc=1, scale=2)
            ).ldot(np.array([[1, 2], [2, 1], [3, 2]])),
            [[-2, -1, 0]],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(
                exponential(loc=5),
                norm(loc=3),
                norm(loc=2),
            ).ldot(
                np.array(
                    [
                        [1, 2, 2],
                        [2, 1, 3],
                    ]
                )
            ),
            [[-2, -1], [0, -5], [-2, 0]],
        ),
        # TODO: test also with bivariate gamma
    ],
)
def test_dKinv(mcgf, ts):
    assert isinstance(mcgf, MultivariateCumulantGeneratingFunction)
    for t in ts:
        x = mcgf.dK(t)
        tt = mcgf.dK_inv(x)
        xx = mcgf.dK(tt)
        assert np.allclose(xx, x)
        if mcgf.domain.dim == mcgf.dim:
            assert np.allclose(tt, t)
    # Test the same thing vectorized
    xs = mcgf.dK(ts)
    tts = mcgf.dK_inv(xs)
    xxs = mcgf.dK(tts)
    assert np.allclose(xxs, xs)
    if mcgf.domain.dim == mcgf.dim:
        assert np.allclose(tts, ts)


def test_from_univariate():
    cgf1 = exponential()
    cgf2 = exponential(scale=2)
    mcgf = MultivariateCumulantGeneratingFunction.from_univariate(cgf1, cgf2)
    assert isinstance(mcgf, MultivariateCumulantGeneratingFunction)
    assert mcgf.dim == 2
    ts = [[1, 2], [0, 0], [1, 0], [0, 1]]
    for t in ts:
        if cgf1.domain.is_in_domain(t[0]) and cgf2.domain.is_in_domain(t[1]):
            assert mcgf.domain.is_in_domain(t)
            assert np.allclose(mcgf.K(t), cgf1.K(t[0]) + cgf2.K(t[1]))
        else:
            assert not mcgf.domain.is_in_domain(t)


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # str(Path(__file__)) + "::test_statistics",
                # "-k",
                # "test_dKinv",
                "--tb=no",
                "--durations=10",
                # "--pdb",
                "-s",
                "-v",
                # "-m 'not slow'",
                # "-m tofix",
            ]
        )
