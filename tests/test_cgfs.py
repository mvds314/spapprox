#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from spapprox.cgf_base import has_numdifftools
from spapprox.diff import _has_findiff as has_findiff

if has_numdifftools:
    from spapprox.cgf_base import nd

import numpy as np
import pytest
import scipy.stats as sps
from scipy.integrate import quad
from spapprox import (
    Domain,
    MultivariateCumulantGeneratingFunction,
    UnivariateCumulantGeneratingFunction,
    binomial,
    chi2,
    exponential,
    gamma,
    laplace,
    multivariate_norm,
    norm,
    poisson,
    univariate_empirical,
    univariate_sample_mean,
)
from spapprox.diff import PartialDerivative


@pytest.mark.parametrize(
    "cgf_to_test, cgf, ts, dist, backend",
    [
        # Test cases are set up for a particular logic, and then one more based on the multivariate implementation
        # Case 1: Univariate normal distribution
        pytest.param(
            norm(loc=0, scale=1),
            np.vectorize(
                lambda t, pdf=sps.norm.pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
                )
            ),
            [0.2, 0.55],
            sps.norm(loc=0, scale=1),
            "numdifftools",
            marks=[
                pytest.mark.skipif(not has_numdifftools, reason="No numdifftools"),
                pytest.mark.slow,
            ],
            id="univariate normal numdifftools",
        ),
        pytest.param(
            norm(loc=0, scale=1),
            np.vectorize(
                lambda t, pdf=sps.norm.pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
                )
            ),
            [0.2, 0.55],
            sps.norm(loc=0, scale=1),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.slow],
            id="univariate normal findiff",
        ),
        pytest.param(
            multivariate_norm(loc=[0, 2], scale=[1, 3])[0],
            norm(loc=0, scale=1).K,
            [0.2, 0.55],
            sps.norm(loc=0, scale=1),
            "findiff",
            marks=pytest.mark.skipif(not has_findiff, reason="No findiff"),
            id="univariate normal from multivariate findiff",
        ),
        pytest.param(
            multivariate_norm(loc=[0, 2], scale=[1, 3])[0],
            norm(loc=0, scale=1).K,
            [0.2, 0.55],
            sps.norm(loc=0, scale=1),
            "numdifftools",
            marks=pytest.mark.skipif(not has_numdifftools, reason="No numdifftools"),
            id="univariate normal from multivariate numdifftools",
        ),
        # Case 2: Sum of two univariate normal distributions
        pytest.param(
            norm(loc=0, scale=1) + norm(loc=0, scale=1),
            np.vectorize(
                lambda t, pdf=sps.norm(0, np.sqrt(2)).pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
                )
            ),
            [0.2, 0.55],
            sps.norm(loc=0, scale=np.sqrt(2)),
            "findiff",
            marks=pytest.mark.skipif(not has_findiff, reason="No findiff"),
            id="sum of two univariate normal distributions",
        ),
        pytest.param(
            (multivariate_norm(loc=0, scale=1) + multivariate_norm(loc=0, scale=1))[0],
            (norm(loc=0, scale=1) + norm(loc=0, scale=1)).K,
            [0.2, 0.55],
            sps.norm(loc=0, scale=np.sqrt(2)),
            "findiff",
            marks=pytest.mark.skipif(not has_findiff, reason="No findiff"),
            id="sum of two univariate normal distributions from multivariate",
        ),
        # Case 3: Sum of two univariate normal distributions with different means
        pytest.param(
            norm(loc=1, scale=1) + norm(loc=2, scale=1),
            np.vectorize(
                lambda t, pdf=sps.norm(3, np.sqrt(2)).pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=-12, b=12)[0]
                )
            ),
            [0.2, 0.55],
            sps.norm(loc=3, scale=np.sqrt(2)),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.slow],
            id="sum of two univariate normal distributions with different means",
        ),
        pytest.param(
            multivariate_norm(loc=[1, 2], scale=[1, 1])[0]
            + multivariate_norm(loc=[1, 2], scale=[1, 1])[1],
            (norm(loc=1, scale=1) + norm(loc=2, scale=1)).K,
            [0.2, 0.55],
            sps.norm(loc=3, scale=np.sqrt(2)),
            "findiff",
            marks=pytest.mark.skipif(not has_findiff, reason="No findiff"),
            id="sum of two univariate normal distributions with different means from multivariate",
        ),
        # Case 4: Univariate normal distribution scaled by a constant
        pytest.param(
            1.1 * norm(loc=0, scale=1),
            np.vectorize(
                lambda t, pdf=sps.norm(0, 1.1).pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
                )
            ),
            [0.2, 0.55],
            sps.norm(loc=0, scale=1.1),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.slow],
            id="univariate normal distribution scaled by a constant",
        ),
        pytest.param(
            1.1 * multivariate_norm(loc=0, scale=1)[0],
            (1.1 * norm(loc=0, scale=1)).K,
            [0.2, 0.55],
            sps.norm(loc=0, scale=1.1),
            "findiff",
            marks=pytest.mark.skipif(not has_findiff, reason="No findiff"),
            id="univariate normal distribution scaled by a constant from multivariate",
        ),
        # Case 5: Sum of two univariate normal distributions scaled by a constant and shifted
        pytest.param(
            1.1 * (norm(loc=0, scale=1) + norm(loc=1, scale=1)) - 0.3,
            np.vectorize(
                lambda t, pdf=sps.norm(0.8, 1.1 * np.sqrt(2)).pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
                )
            ),
            [0.2, 0.55],
            sps.norm(loc=0.8, scale=1.1 * np.sqrt(2)),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.slow],
            id="sum of two univariate normal distributions scaled by a constant and shifted",
        ),
        pytest.param(
            1.1 * (multivariate_norm(loc=0, scale=1)[0] + multivariate_norm(loc=1, scale=1)[0])
            - 0.3,
            (1.1 * (norm(loc=0, scale=1) + norm(loc=1, scale=1)) - 0.3).K,
            [0.2, 0.55],
            sps.norm(loc=0.8, scale=1.1 * np.sqrt(2)),
            "findiff",
            marks=pytest.mark.skipif(not has_findiff, reason="No findiff"),
            id="sum of two univariate normal distributions scaled by a constant and shifted from multivariate",
        ),
        # Case 6: Univariate normal distribution with loc and scale
        pytest.param(
            norm(loc=1, scale=0.5),
            np.vectorize(
                lambda t: np.log(
                    quad(
                        lambda x, pdf=sps.norm(loc=1, scale=0.5).pdf: pdf(x) * np.exp(t * x),
                        a=-5,
                        b=5,
                    )[0]
                )
            ),
            [0.2, 0.55],
            sps.norm(loc=1, scale=0.5),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.slow],
            id="univariate normal distribution with loc and scale",
        ),
        pytest.param(
            multivariate_norm(loc=1, scale=0.5, dim=2)[0],
            norm(loc=1, scale=0.5).K,
            [0.2, 0.55],
            sps.norm(loc=1, scale=0.5),
            "findiff",
            marks=pytest.mark.skipif(not has_findiff, reason="No findiff"),
            id="univariate normal distribution with loc and scale from multivariate",
        ),
        # Case 7: Univariate normal manually specified
        pytest.param(
            UnivariateCumulantGeneratingFunction(
                K=lambda t, loc=0, scale=1: loc * t + scale**2 * t**2 / 2
            ),
            np.vectorize(
                lambda t, pdf=sps.norm.pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
                )
            ),
            [0.2, 0.55],
            sps.norm(loc=0, scale=1),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.slow],
            id="univariate normal manually specified",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                UnivariateCumulantGeneratingFunction(
                    K=lambda t, loc=0, scale=1: loc * t + scale**2 * t**2 / 2
                ),
            )[0],
            UnivariateCumulantGeneratingFunction(
                K=lambda t, loc=0, scale=1: loc * t + scale**2 * t**2 / 2
            ).K,
            [0.2, 0.55],
            sps.norm(loc=0, scale=1),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.xfail],
            id="univariate normal manually specified from multivariate",
        ),
        # Case 8: Univariate exponential
        pytest.param(
            exponential(scale=1),
            np.vectorize(
                lambda t, pdf=sps.expon.pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
                )
            ),
            [0.2, 0.55],
            sps.expon(scale=1),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.slow],
            id="univariate exponential",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                norm(0, 1), 2 * exponential(scale=1) / 2
            )[1],
            exponential(scale=1).K,
            [0.2, 0.55],
            sps.expon(scale=1),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.xfail],
            id="univariate exponential from multivariate",
        ),
        # Case 9: Univariate exponential with scale
        pytest.param(
            exponential(scale=0.5),
            np.vectorize(
                lambda t, pdf=sps.expon(scale=0.5).pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
                )
            ),
            [0.2, 0.55],
            sps.expon(scale=0.5),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.slow],
            id="univariate exponential with scale",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                exponential(scale=0.5), exponential(scale=0.5)
            ).ldot([1, 0]),
            exponential(scale=0.5).K,
            [0.2, 0.55],
            sps.expon(scale=0.5),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.xfail],
            id="univariate exponential with scale from multivariate",
        ),
        # Case 10: Univariate exponential cgf manually specified
        pytest.param(
            UnivariateCumulantGeneratingFunction(K=lambda t: np.log(1 / (1 - t))),
            exponential(scale=1).K,
            [0.2, 0.55],
            sps.expon(scale=1),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.xfail],
            id="univariate exponential cgf manually specified",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                UnivariateCumulantGeneratingFunction(
                    K=lambda t: np.log(1 / (1 - t)),
                )
            )[0],
            exponential(scale=1).K,
            [0.2, 0.55],
            sps.expon(scale=1),
            "findiff",
            marks=pytest.mark.skipif(not has_findiff, reason="No findiff"),
            id="univariate exponential cgf manually specified from multivariate",
        ),
        # Case 11: Univariate gamma
        pytest.param(
            gamma(a=1.1, scale=0.9),
            np.vectorize(
                lambda t, pdf=sps.gamma(a=1.1, scale=0.9).pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
                )
            ),
            [0.2, 0.55],
            sps.gamma(a=1.1, scale=0.9),
            "findiff",
            marks=[
                pytest.mark.skipif(not has_findiff, reason="No findiff"),
                pytest.mark.slow,
                pytest.mark.xfail,
            ],
            id="univariate gamma",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                gamma(a=1.1, scale=0.9), gamma(a=2, scale=3)
            ).ldot([1, 0]),
            gamma(a=1.1, scale=0.9).K,
            [0.2, 0.55],
            sps.gamma(a=1.1, scale=0.9),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.xfail],
            id="univariate gamma from multivariate",
        ),
        # Case 12: Univariate chi2
        pytest.param(
            chi2(df=3),
            lambda t, pdf=sps.chi2(df=3).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
            ),
            [0.2, 0.25],
            sps.chi2(df=3),
            "findiff",
            marks=[
                pytest.mark.skipif(not has_findiff, reason="No findiff"),
                pytest.mark.slow,
                pytest.mark.xfail,
            ],
            id="univariate chi2",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(chi2(df=2), chi2(df=3)).ldot(
                [0, 1]
            ),
            chi2(df=3).K,
            [0.2, 0.25],
            sps.chi2(df=3),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.xfail],
            id="univariate chi2 from multivariate",
        ),
        # Case 13: Univariate laplace
        pytest.param(
            laplace(loc=0, scale=1),
            np.vectorize(
                lambda t, pdf=sps.laplace(loc=0, scale=1).pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=-50, b=50)[0]
                )
            ),
            [0.2, 0.55, -0.23],
            sps.laplace(loc=0, scale=1),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.slow],
            id="univariate laplace",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                laplace(loc=0, scale=3), laplace(loc=0, scale=1)
            ).ldot([1, 0]),
            laplace(loc=0, scale=3).K,
            [0.2, 0.3, -0.23],
            sps.laplace(loc=0, scale=3),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.xfail],
            id="univariate laplace from multivariate",
        ),
        # Case 14: Univariate poisson
        # TODO: fix this test
        pytest.param(
            poisson(mu=2),
            lambda t, pmf=sps.poisson(mu=2).pmf: np.log(
                np.sum(np.exp(t * x) * pmf(x) for x in range(100))
            ),
            [0.2, 0.55],
            sps.poisson(mu=2),
            "findiff",
            marks=pytest.mark.skipif(not has_findiff, reason="No findiff"),
            id="univariate poisson",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(poisson(mu=2), poisson(mu=2))[
                0
            ],
            lambda t, pmf=sps.poisson(mu=2).pmf: np.log(
                np.sum([np.exp(t * x) * pmf(x) for x in np.arange(100)], axis=0)
            ),
            [0.2, 0.55],
            sps.poisson(mu=2),
            "findiff",
            marks=[
                pytest.mark.skipif(not has_findiff, reason="No findiff"),
                pytest.mark.xfail(reason="Fails, change to findiff in multivariate"),
            ],
            id="univariate poisson from multivariate",
        ),
        # Case 15: Univariate binomial
        pytest.param(
            binomial(n=10, p=0.5),
            lambda t, pmf=sps.binom(n=10, p=0.5).pmf: np.log(
                np.sum([np.exp(t * x) * pmf(x) for x in range(100)], axis=0)
            ),
            [0.2, 0.55],
            sps.binom(n=10, p=0.5),
            "findiff",
            marks=pytest.mark.skipif(not has_findiff, reason="No findiff"),
            id="univariate binomial",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                binomial(n=10, p=0.5), binomial(n=10, p=0.5)
            )[0],
            binomial(n=10, p=0.5).K,
            [0.2, 0.55],
            sps.binom(n=10, p=0.5),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.xfail],
            id="univariate binomial from multivariate",
        ),
        # Case 16: Univariate sample mean
        pytest.param(
            univariate_sample_mean(norm(2, 1), 25),
            np.vectorize(
                lambda t, pdf=sps.norm(loc=2, scale=0.2).pdf: np.log(
                    quad(lambda x: pdf(x) * np.exp(t * x), a=-50, b=50)[0]
                )
            ),
            [0.2, 0.55, -0.23],
            sps.norm(loc=2, scale=0.2),
            "findiff",
            marks=[
                pytest.mark.skipif(not has_findiff, reason="No findiff"),
                pytest.mark.slow,
                pytest.mark.xfail,
            ],
            id="univariate sample mean",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                univariate_sample_mean(norm(2, 1), 25), univariate_sample_mean(norm(2, 1), 25)
            )[0],
            univariate_sample_mean(norm(2, 1), 25).K,
            [0.2, 0.55, -0.23],
            sps.norm(loc=2, scale=0.2),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.xfail],
            id="univariate sample mean from multivariate",
        ),
        # Case 17: Univariate empirical
        pytest.param(
            univariate_empirical(np.arange(10)),
            np.vectorize(
                lambda t, x=np.arange(10): np.log(np.sum(np.exp(t * x) / len(x), axis=0))
            ),
            [0.2, 0.55, -0.23],
            np.arange(10),
            "findiff",
            marks=pytest.mark.skipif(not has_findiff, reason="No findiff"),
            id="univariate empirical",
        ),
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                univariate_empirical(np.arange(10)), univariate_empirical(np.arange(10))
            )[0],
            univariate_empirical(np.arange(10)).K,
            [0.2, 0.55, -0.23],
            np.arange(10),
            "findiff",
            marks=[pytest.mark.skipif(not has_findiff, reason="No findiff"), pytest.mark.xfail],
            id="univariate empirical from multivariate",
        ),
    ],
)
def test_basic(cgf_to_test, cgf, ts, dist, backend):
    # Test function evaluations
    assert isinstance(cgf_to_test, UnivariateCumulantGeneratingFunction)
    for t in ts:
        assert np.isclose(cgf(t), cgf_to_test.K(t), atol=1e-4)
        if backend == "numdifftools":
            dcgf = nd.Derivative(cgf, n=1)
            d2cgf = nd.Derivative(cgf, n=2)
            d3cgf = nd.Derivative(cgf, n=3)
        elif backend == "findiff":
            dcgf = PartialDerivative(cgf, 1)
            d2cgf = PartialDerivative(cgf, 2)
            d3cgf = PartialDerivative(cgf, 3)
        else:
            raise ValueError(f"Backend {backend} not supported in test")
        assert np.isclose(dcgf(t), cgf_to_test.dK(t))
        assert np.isclose(d2cgf(t), cgf_to_test.d2K(t))
        assert np.isclose(d3cgf(t), cgf_to_test.d3K(t), atol=5e-3)
    # Test vectorized evaluation
    assert np.isclose(cgf_to_test.mean, dist.mean())
    assert np.isclose(cgf_to_test.variance, dist.var())
    # Test addition scalar
    for t in ts:
        # Test cumulant generating function
        assert np.isclose(cgf(t) + t, (cgf_to_test + float(1)).K(t))
        assert np.isclose(cgf(t) - 2 * t, (cgf_to_test - int(2)).K(t), atol=1e-5)
        cgf_to_test.add(3, inplace=True)
        assert np.isclose(cgf(t) + 3 * t, cgf_to_test.K(t))
        cgf_to_test.add(-3, inplace=True)
        cgf_to_test.add(3, inplace=False)
        assert np.isclose(cgf(t), cgf_to_test.K(t))
        # Test first derivative
        if backend == "numdifftools":
            dcgf = nd.Derivative(cgf, n=1)
        elif backend == "findiff":
            dcgf = PartialDerivative(cgf, 1)
        else:
            raise ValueError(f"Backend {backend} not supported in test")
        cgf_to_test.add(3, inplace=True)
        assert np.isclose(dcgf(t) + 3, cgf_to_test.dK(t), atol=1e-4)
        assert np.isclose(cgf_to_test.dK0, dcgf(0) + 3)
        cgf_to_test.add(-3, inplace=True)
        assert np.isclose(cgf_to_test.dK0, dcgf(0))
        # Test stored derivatives
        assert np.isclose(cgf_to_test.dK0, dcgf(0))
        assert np.isclose(cgf_to_test.d2K0, d2cgf(0), atol=1e-5)
        assert np.isclose(cgf_to_test.d3K0, d3cgf(0), atol=1e-3)
    # Test multiplication and division with scalar
    for t in ts:
        # Test cumulant generating function
        assert np.isclose(cgf(1.01 * t), (1.01 * cgf_to_test).K(t))
        assert np.isclose(cgf(1.01 * t), (cgf_to_test / (1 / 1.01)).K(t), atol=1e-5)
        cgf_to_test.mul(1.01, inplace=True)
        assert np.isclose(cgf(1.01 * t), cgf_to_test.K(t))
        cgf_to_test.mul(1 / 1.01, inplace=True)
        cgf_to_test.mul(2, inplace=False)
        assert np.isclose(cgf(t), cgf_to_test.K(t))
        # Initialize derivatives
        if backend == "numdifftools":
            dcgf = nd.Derivative(cgf, n=1)
            d2cgf = nd.Derivative(cgf, n=2)
        elif backend == "findiff":
            dcgf = PartialDerivative(cgf, 1)
            d2cgf = PartialDerivative(cgf, 2)
        else:
            raise ValueError(f"Backend {backend} not supported in test")
        # Test first derivative
        cgf_to_test.mul(1.01, inplace=True)
        assert np.isclose(dcgf(1.01 * t) / (1 / 1.01), cgf_to_test.dK(t), atol=1e-4)
        assert np.isclose(cgf_to_test.dK0, dcgf(0) * 1.01)
        cgf_to_test.mul(1 / 1.01, inplace=True)
        assert np.isclose(cgf_to_test.dK0, dcgf(0))
        # Test second derivative
        cgf_to_test.mul(1.01, inplace=True)
        assert np.isclose(d2cgf(1.01 * t) / (1 / 1.01) ** 2, cgf_to_test.d2K(t), atol=1e-4)
        assert np.isclose(cgf_to_test.d2K0, d2cgf(0) * 1.01**2)
        cgf_to_test.mul(1 / 1.01, inplace=True)
        assert np.isclose(cgf_to_test.d2K0, d2cgf(0))
    # Test addition other cumulant generating function
    for t in ts:
        assert np.isclose(cgf(t) + cgf(t), (cgf_to_test + cgf_to_test).K(t))
        with pytest.raises(AssertionError):
            cgf_to_test.add(cgf_to_test, inplace=True)


# @pytest.mark.xfail
@pytest.mark.tofix
def test_domain():
    import pdb

    pdb.set_trace()
    # Test simple domain
    cgf = UnivariateCumulantGeneratingFunction(
        K=lambda t: t**4,
        domain=Domain(l=1),
    )
    assert np.isclose(cgf.K(0.5), 0.0625)
    assert np.isnan(cgf.K(1.5))
    assert np.isclose(cgf.dK(0.5), 0.5)
    assert np.isnan(cgf.dK(1.5))
    assert np.isclose(cgf.d2K(0.5), 3)
    assert np.isnan(cgf.d2K(1.5))
    assert np.isclose(cgf.d3K(0.5), 12)
    assert np.isnan(cgf.d3K(1.5))
    # Test domain with bounds
    cgf = UnivariateCumulantGeneratingFunction(
        K=lambda t: t**4,
        domain=Domain(g=1, l=2),
    )
    assert np.isnan(cgf.K(0.5))
    assert not np.isnan(cgf.K(1.5))
    assert np.isnan(cgf.dK(0.5))
    assert not np.isnan(cgf.dK(1.5))
    assert np.isnan(cgf.d2K(0.5))
    assert not np.isnan(cgf.d2K(1.5))
    assert np.isnan(cgf.d3K(0.5))
    assert not np.isnan(cgf.d3K(1.5))
    # Test by indexing MultivariateCumulantGeneratingFunction
    cgf = MultivariateCumulantGeneratingFunction(
        K=lambda t: np.sum(t**4),
        domain=Domain(ge=[1, 0], l=[2, 2], dim=2),
        dim=2,
    )
    cgf = cgf[0]
    assert np.isnan(cgf.K(0.5))
    assert not np.isnan(cgf.K(1.5))
    assert np.isnan(cgf.dK(0.5))
    assert not np.isnan(cgf.dK(1.5))
    assert np.isnan(cgf.d2K(0.5))
    assert not np.isnan(cgf.d2K(1.5))
    assert np.isnan(cgf.d3K(0.5))
    # TODO: continue here and fix this one
    import pdb

    pdb.set_trace()
    assert not np.isnan(cgf.d3K(1.5))
    # Another test
    cgf = UnivariateCumulantGeneratingFunction(
        K=lambda t: t**4,
        domain=Domain(g=0, l=2),
    )
    val = cgf.K([-1, 1])
    assert np.isnan(val[0])
    assert not np.isnan(val[1])


@pytest.mark.parametrize(
    "cgf",
    [
        UnivariateCumulantGeneratingFunction(
            K=lambda t: t**4,
            domain=Domain(l=1),
        ),
        univariate_empirical(np.arange(10)),
        multivariate_norm(np.zeros(2), np.eye(2))[0],
    ],
)
def test_return_type(cgf):
    for f in [cgf.K, cgf.dK]:
        assert np.isscalar(f(10))
        assert not np.isscalar(f([10]))
        assert not np.isscalar(f([10, 10]))
        if 10 not in cgf.domain:
            assert np.isnan(f(10))
            assert np.isnan(f([10])).all()
            assert np.isnan(f([10, 10])).all()
        assert np.isscalar(f(0)) and not np.isnan(f(0))
        assert not np.isscalar(f([0, 1]))
        if 1 not in cgf.domain:
            assert np.isnan(f([0, 1])).any() and not np.isnan(f([0, 1])).all()
            assert (
                not np.isscalar(f([1, 1], fillna=0))
                and not np.isnan(f([1, 1], fillna=0)).any()
                and np.allclose(f([1, 1], fillna=0), 0)
            )


@pytest.mark.parametrize(
    "cgf,ts",
    [
        # Case 1: Univariate normal
        (
            norm(loc=0, scale=1),
            [0.2, 0.55],
        ),
        (
            multivariate_norm(np.zeros(2), np.eye(2))[0],
            [0.2, 0.55],
        ),
        # Case 2: Univariate normal with loc and scale
        (
            norm(loc=1, scale=0.5),
            [0.2, 0.55],
        ),
        (
            multivariate_norm(np.ones(2), np.eye(2))[0],
            [0.2, 0.55],
        ),
        # Case 3: Univariate manually specified
        (
            UnivariateCumulantGeneratingFunction(
                K=lambda t, loc=0, scale=1: loc * t + scale**2 * t**2 / 2
            ),
            [0.2, 0.55],
        ),
        # Case 4: Univariate exponential
        (
            exponential(scale=1),
            [0.2, 0.55, 0.95],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(exponential(scale=1))[0],
            [0.2, 0.55, 0.95],
        ),
        # Case 5: Univariate exponential with scale
        (
            exponential(scale=0.5),
            [0.2, 0.55, 0.95],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(exponential(scale=0.5))[0],
            [0.2, 0.55, 0.95],
        ),
        # Case 6: Univariate exponential manually specified
        (
            UnivariateCumulantGeneratingFunction(K=lambda t: np.log(1 / (1 - t))),
            [0.2, 0.55, 0.95],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(
                UnivariateCumulantGeneratingFunction(K=lambda t: np.log(1 / (1 - t)))
            )[0],
            [0.2, 0.55, 0.95],
        ),
        # Case 7: Univariate gamma
        (
            gamma(a=2, scale=0.5),
            [0.2, 0.55],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(gamma(a=2, scale=0.5))[0],
            [0.2, 0.55],
        ),
        # Case 8: Univariate chi2
        (
            chi2(df=3),
            [0.2, 0.25],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(chi2(df=3))[0],
            [0.2, 0.25],
        ),
        # Case 9: Univariate laplace
        (
            laplace(loc=0, scale=1),
            [0.2, 0.55, -0.23],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(laplace(loc=0, scale=1))[0],
            [0.2, 0.55, -0.23],
        ),
        # Case 10: Univariate poisson
        (
            poisson(mu=2),
            [0.2, 0.55],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(poisson(mu=2))[0],
            [0.2, 0.55],
        ),
        # Case 11: Univariate binomial
        (
            binomial(n=10, p=0.5),
            [0.2, 0.55],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(binomial(n=10, p=0.5))[0],
            [0.2, 0.55],
        ),
        # Case 12: Univariate beta
        (
            univariate_sample_mean(norm(2, 1), 25),
            [0.2, 0.55, -0.23],
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(
                univariate_sample_mean(norm(2, 1), 25)
            )[0],
            [0.2, 0.55, -0.23],
        ),
    ],
)
def test_dKinv(cgf, ts):
    for t in ts:
        assert np.isclose(cgf.dK_inv(cgf.dK(t)), t)
    assert np.allclose(cgf.dK_inv(cgf.dK(ts[:1])), ts[:1])
    assert np.allclose(cgf.dK_inv(cgf.dK(ts)), ts)
    assert np.allclose(cgf.dK_inv(cgf.dK(ts)), [cgf.dK_inv(cgf.dK(t)) for t in ts])


# TODO: integrate diff in multivariate cgfs, one by one
# TODO: test that stuff
# TODO: resolve remaining xfails in this file


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                str(Path(__file__)),
                # str(Path(__file__)) + "::test_basic",
                # "-k",
                # "test_basic",
                "--durations=10",
                # "--tb=auto",
                "--tb=no",
                # "--pdb",
                "-s",
                # "-m 'not slow'",
                "-m tofix",
            ]
        )
