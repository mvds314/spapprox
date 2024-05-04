#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import itertools
import pytest
import scipy.stats as sps
from scipy.integrate import quad
from spapprox import (
    # poisson,
    # binomial,
    UnivariateSaddlePointApprox,
    MultivariateSaddlePointApprox,
    MultivariateCumulantGeneratingFunction,
    BivariateSaddlePointApprox,
    chi2,
    exponential,
    gamma,
    laplace,
    norm,
    multivariate_norm,
)


@pytest.mark.parametrize(
    "cgf,dist,trange",
    [
        (norm(loc=0.5, scale=3), sps.norm(loc=0.5, scale=3), [-10, 10]),
        (norm(loc=0, scale=1), sps.norm(loc=0, scale=1), [-5, 5]),
    ],
)
def test_norm_spa(cgf, dist, trange):
    spa = UnivariateSaddlePointApprox(cgf)
    assert spa.dim == 1, "Univariate approximations are supposed to be 1 dimensional"
    t = np.linspace(*trange, 1000)
    x = spa.cgf.dK(t)
    # These ones should be exact
    assert np.allclose(spa.pdf(t=t, normalize_pdf=False), dist.pdf(x))
    assert np.isclose(
        quad(
            lambda t: spa.pdf(t=t, normalize_pdf=False) * cgf.d2K(t),
            a=trange[0],
            b=trange[1],
        )[0],
        1,
    )
    assert np.isclose(spa.cdf(t=trange[0]), 0, atol=1e-6)
    assert np.isclose(spa.cdf(t=trange[1]), 1, atol=1e-6)
    # Test inversion saddle point
    spa.fit_saddle_point_eqn(num=10000)
    for t in [-2, -1, 1 / 6]:
        x = spa.cgf.dK(t)
        assert np.isclose(spa.cgf.dK(spa._dK_inv(x)), x, atol=1e-3)
    # Test clear cache
    assert hasattr(spa, "_x_cache") and hasattr(spa, "_t_cache")
    spa.clear_cache()
    assert not hasattr(spa, "_x_cache") and not hasattr(spa, "_t_cache")
    # Test cdf
    qs = [0.05, 0.1, 0.3, 0.5, 0.9, 0.95]
    for q in qs:
        # Note: It's better compare x than p
        assert np.isclose(dist.ppf(spa.cdf(x=dist.ppf(q))), dist.ppf(q), atol=5e-2)
    for q in qs:
        assert np.isclose(spa.cdf(x=spa.ppf(q)), q, atol=1e-6)
    assert np.allclose(spa.cdf(x=spa.ppf(qs)), qs, atol=1e-4)
    # Same tests but then with ppf fitted
    spa.fit_ppf()
    for q in qs:
        assert np.isclose(spa.cdf(x=spa.ppf(q)), q, atol=1e-3)
    assert np.allclose(spa.cdf(x=spa.ppf(qs)), qs, atol=1e-3)


@pytest.mark.parametrize(
    "cgf,trange",
    [
        (norm(loc=0.5, scale=3), [-10, 10]),
        (norm(loc=0, scale=1), [-5, 5]),
        (exponential(scale=10), [-1e5, 1 / 10 * 0.99999999]),
        (gamma(a=2, scale=3), [-1e3, 1 / 3 * 0.99999999]),
        (chi2(df=3), [-1e4, 1 / 2 * 0.99999999]),
        (laplace(loc=1, scale=3), [-1 / 3 * 0.99999999, 1 / 3 * 0.99999999]),
    ],
)
def test_normalization(cgf, trange):
    spa = UnivariateSaddlePointApprox(cgf)
    assert spa.dim == 1, "Univariate approximations are supposed to be 1 dimensional"
    assert np.isclose(spa.cdf(t=trange[0]), 0, atol=1e-5)
    assert np.isclose(spa.cdf(t=trange[1]), 1, atol=1e-5)
    assert np.isclose(
        quad(
            lambda t: spa.pdf(t=t, fillna=0, normalize_pdf=True) * cgf.d2K(t, fillna=0),
            a=-np.inf,
            b=np.inf,
        )[0],
        1,
        atol=5e-4,
    )
    assert np.isclose(
        quad(
            lambda t: spa.pdf(t=t, fillna=0, normalize_pdf=True) * cgf.d2K(t, fillna=0),
            a=trange[0],
            b=trange[1],
        )[0],
        1,
        atol=5e-4,
    )


@pytest.mark.parametrize(
    "cgf,dist,trange",
    [
        (
            exponential(scale=3),
            sps.expon(scale=3),
            [0, 1 / 3],
        ),
    ],
)
def test_expon_spa(cgf, dist, trange):
    spa = UnivariateSaddlePointApprox(cgf)
    assert spa.dim == 1, "Univariate approximations are supposed to be 1 dimensional"
    for f in [
        spa.pdf,
        lambda t=None, fillna=np.nan, backend="LR": spa.cdf(t=t, fillna=fillna, backend=backend),
        lambda t=None, fillna=np.nan, backend="BN": spa.cdf(t=t, fillna=fillna, backend=backend),
    ]:
        assert np.isscalar(f(t=0))
        assert not np.isscalar(f(t=[0, 0]))
        assert np.isscalar(f(t=1 / 3)) and np.isnan(f(t=1 / 3))
        assert not np.isscalar(f(t=[1 / 3, 1])) and np.isnan(f(t=[1, 1 / 3])).all()
        assert not np.isscalar(f(t=[1 / 3, 1], fillna=0)) and np.allclose(
            f(t=[1, 1 / 3], fillna=10), 10
        )
        assert not np.isscalar(spa.pdf(t=[1 / 3, 0])) and not np.isnan(spa.pdf(t=[1, 0])).all()
    # approximation accuracy test
    t = np.linspace(*trange, 1000)[:-1]
    x = spa.cgf.dK(t)
    assert np.allclose(spa.pdf(t=t), dist.pdf(x), atol=5e-5), "This should approx be equal"
    assert np.allclose(
        spa.cdf(t=t, backend="LR"), dist.cdf(x), atol=5e-3
    ), "This should approx be equal"
    assert np.allclose(
        spa.cdf(t=t, backend="BN"), dist.cdf(x), atol=5e-3
    ), "This should approx be equal"
    assert not np.allclose(
        spa.cdf(t=t, backend="BN"), spa.cdf(t=t, backend="LR")
    ), "the approximation should not be exactly equal"
    # Test investion saddle point
    spa.fit_saddle_point_eqn(num=10000)
    for t in [-2, -1, 1 / 6]:
        x = spa.cgf.dK(t)
        assert np.isclose(spa.cgf.dK(spa._dK_inv(x)), x, atol=1e-3)
    # Test clear cache
    assert hasattr(spa, "_x_cache") and hasattr(spa, "_t_cache")
    spa.clear_cache()
    assert not hasattr(spa, "_x_cache") and not hasattr(spa, "_t_cache")
    # Test cdf
    qs = [0.05, 0.1, 0.3, 0.5, 0.9, 0.95]
    for q in qs:
        # Note: It's better compare x than p
        assert np.isclose(dist.ppf(spa.cdf(x=dist.ppf(q))), dist.ppf(q), atol=5e-2)
    for q in qs:
        assert np.isclose(spa.cdf(x=spa.ppf(q)), q, atol=1e-6)
    assert np.allclose(spa.cdf(x=spa.ppf(qs)), qs, atol=1e-4)
    # Same tests but then with ppf fitted
    spa.fit_ppf()
    for q in qs:
        assert np.isclose(spa.cdf(x=spa.ppf(q)), q, atol=1e-3)
    assert np.allclose(spa.cdf(x=spa.ppf(qs)), qs, atol=1e-3)


@pytest.mark.parametrize(
    "cgf, dist, ts, dim",
    [
        # Basic test with uncorrelated variables
        (
            multivariate_norm(loc=0.5, scale=3),
            sps.multivariate_normal(mean=[0.5, 0.5], cov=9),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 10), 2)),
            2,
        ),
        # Test with correlated variables
        (
            multivariate_norm(loc=[0.5, 0.2], cov=[[3, 1], [1, 3]]),
            sps.multivariate_normal(mean=[0.5, 0.2], cov=[[3, 1], [1, 3]]),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 10), 2)),
            2,
        ),
        # 3 dim test
        (
            multivariate_norm(loc=0.5, scale=3, dim=3),
            sps.multivariate_normal(mean=[0.5, 0.5, 0.5], cov=9),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 10), 3)),
            3,
        ),
        # Other distribution
        (
            MultivariateCumulantGeneratingFunction.from_univariate(
                norm(loc=0.5, scale=3), norm(loc=0.5, scale=3)
            ),
            sps.multivariate_normal(mean=[0.5, 0.5], cov=9),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 10), 2)),
            2,
        ),
        # TODO: test non-normal distribution?
        # TODO: also test the bivariate implementation explicitly
    ],
)
def test_mvar_spa(cgf, dist, ts, dim):
    spa = MultivariateSaddlePointApprox(cgf)
    assert spa.dim == dim
    for t in ts:
        x = spa.cgf.dK(t)
        assert np.allclose(spa.pdf(t=t, normalize_pdf=False), dist.pdf(x))
    assert np.allclose(spa.pdf(t=ts, normalize_pdf=False), dist.pdf(spa.cgf.dK(ts)))
    if dim <= 2:
        # TODO: these tests takes very long in 3D, what is the alternative?
        assert np.allclose(spa.pdf(t=ts, normalize_pdf=True), dist.pdf(spa.cgf.dK(ts)))
        assert np.isclose(spa._pdf_normalization_cache, 1, atol=5e-4)
    spa.fit_saddle_point_eqn(num=10)
    x = spa._x_cache
    x = np.vstack([xi.ravel() for xi in np.meshgrid(*x)]).T
    for xx in x:
        cgf.dK(cgf.dK_inv(xx))
        spa._dK_inv(xx)
        assert np.allclose(xx, cgf.dK(spa._dK_inv(xx)))
    np.allclose(cgf.dK_inv(x), spa._dK_inv(x))
    # Same logic, but now with cache already provided
    spa.fit_saddle_point_eqn(num=10)
    x = spa._x_cache
    x = np.vstack([xi.ravel() for xi in np.meshgrid(*x)]).T
    for xx in x:
        assert np.allclose(cgf.dK_inv(xx), spa._dK_inv(xx))
    np.allclose(cgf.dK_inv(x), spa._dK_inv(x))


# TODO: continue with testing the bivariate spa
@pytest.mark.parametrize(
    "cgf, dist, ts, dim",
    [
        # Basic test with uncorrelated variables
        (
            multivariate_norm(loc=0.5, scale=3),
            sps.multivariate_normal(mean=[0.5, 0.5], cov=9),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 10), 2)),
            2,
        ),
    ],
)
def test_bvar_spa(cgf, dist, ts, dim):
    spa = BivariateSaddlePointApprox(cgf)
    assert spa.dim == 2
    # test pdf
    for t in ts:
        x = spa.cgf.dK(t)
        assert np.allclose(spa.pdf(t=t, normalize_pdf=False), dist.pdf(x))
    assert np.allclose(spa.pdf(t=ts, normalize_pdf=False), dist.pdf(spa.cgf.dK(ts)))
    # test cdf
    # TODO: continue here and fix this test
    for t in ts[5:6]:
        x = spa.cgf.dK(t)
        print(dist.cdf(x))
        assert np.allclose(spa.cdf(t=t), dist.cdf(x))
    # import pdb

    # pdb.set_trace()
    # assert np.allclose(spa.cdf(t=ts), dist.cdf(spa.cgf.dK(ts)))
    # # TODO: what to include from this?
    # spa.fit_saddle_point_eqn(num=10)
    # x = spa._x_cache
    # x = np.vstack([xi.ravel() for xi in np.meshgrid(*x)]).T
    # for xx in x:
    #     cgf.dK(cgf.dK_inv(xx))
    #     spa._dK_inv(xx)
    #     assert np.allclose(xx, cgf.dK(spa._dK_inv(xx)))
    # np.allclose(cgf.dK_inv(x), spa._dK_inv(x))
    # # Same logic, but now with cache already provided
    # spa.fit_saddle_point_eqn(num=10)
    # x = spa._x_cache
    # x = np.vstack([xi.ravel() for xi in np.meshgrid(*x)]).T
    # for xx in x:
    #     assert np.allclose(cgf.dK_inv(xx), spa._dK_inv(xx))
    # np.allclose(cgf.dK_inv(x), spa._dK_inv(x))


# TODO: contintue with the conditional Skovgaard approximation

if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                "-k",
                "test_bvar_spa",
                "--tb=auto",
                "--pdb",
            ]
        )
