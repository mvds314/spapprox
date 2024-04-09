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

from spapprox import (
    UnivariateCumulantGeneratingFunction,
    MultivariateCumulantGeneratingFunction,
    Domain,
    norm,
    multivariate_norm,
    exponential,
    gamma,
    chi2,
    laplace,
    poisson,
    binomial,
    univariate_sample_mean,
    univariate_empirical,
)


@pytest.mark.parametrize(
    "cgf_to_test,cgf,ts,dist",
    [
        # Test cases are set up for a particular logic, and then one more based on the multivariate implementation
        # Case 1: Univariate normal distribution
        (
            norm(loc=0, scale=1),
            lambda t, pdf=sps.norm.pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
            ),
            [0.2, 0.55],
            sps.norm(loc=0, scale=1),
        ),
        (
            multivariate_norm(loc=[0, 2], scale=[1, 3])[0],
            norm(loc=0, scale=1).K,
            [0.2, 0.55],
            sps.norm(loc=0, scale=1),
        ),
        # Case 2: Sum of two univariate normal distributions
        (
            norm(loc=0, scale=1) + norm(loc=0, scale=1),
            lambda t, pdf=sps.norm(0, np.sqrt(2)).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
            ),
            [0.2, 0.55],
            sps.norm(loc=0, scale=np.sqrt(2)),
        ),
        (
            (multivariate_norm(loc=0, scale=1) + multivariate_norm(loc=0, scale=1))[0],
            (norm(loc=0, scale=1) + norm(loc=0, scale=1)).K,
            [0.2, 0.55],
            sps.norm(loc=0, scale=np.sqrt(2)),
        ),
        # Case 3: Sum of two univariate normal distributions with different means
        (
            norm(loc=1, scale=1) + norm(loc=2, scale=1),
            lambda t, pdf=sps.norm(3, np.sqrt(2)).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
            ),
            [0.2, 0.55],
            sps.norm(loc=3, scale=np.sqrt(2)),
        ),
        (
            multivariate_norm(loc=[1, 2], scale=[1, 1])[0]
            + multivariate_norm(loc=[1, 2], scale=[1, 1])[1],
            (norm(loc=1, scale=1) + norm(loc=2, scale=1)).K,
            [0.2, 0.55],
            sps.norm(loc=3, scale=np.sqrt(2)),
        ),
        # Case 4: Univariate normal distribution scaled by a constant
        (
            1.1 * norm(loc=0, scale=1),
            lambda t, pdf=sps.norm(0, 1.1).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
            ),
            [0.2, 0.55],
            sps.norm(loc=0, scale=1.1),
        ),
        (
            1.1 * multivariate_norm(loc=0, scale=1)[0],
            (1.1 * norm(loc=0, scale=1)).K,
            [0.2, 0.55],
            sps.norm(loc=0, scale=1.1),
        ),
        # Case 5: Sum of two univariate normal distributions scaled by a constant and shifted
        (
            1.1 * (norm(loc=0, scale=1) + norm(loc=1, scale=1)) - 0.3,
            lambda t, pdf=sps.norm(0.8, 1.1 * np.sqrt(2)).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
            ),
            [0.2, 0.55],
            sps.norm(loc=0.8, scale=1.1 * np.sqrt(2)),
        ),
        (
            1.1 * (multivariate_norm(loc=0, scale=1)[0] + multivariate_norm(loc=1, scale=1)[0])
            - 0.3,
            (1.1 * (norm(loc=0, scale=1) + norm(loc=1, scale=1)) - 0.3).K,
            [0.2, 0.55],
            sps.norm(loc=0.8, scale=1.1 * np.sqrt(2)),
        ),
        # Case 6: Univariate normal distribution with loc and scale
        (
            norm(loc=1, scale=0.5),
            lambda t: np.log(
                quad(
                    lambda x, pdf=sps.norm(loc=1, scale=0.5).pdf: pdf(x) * np.exp(t * x),
                    a=-5,
                    b=5,
                )[0]
            ),
            [0.2, 0.55],
            sps.norm(loc=1, scale=0.5),
        ),
        (
            multivariate_norm(loc=1, scale=0.5, dim=2)[0],
            norm(loc=1, scale=0.5).K,
            [0.2, 0.55],
            sps.norm(loc=1, scale=0.5),
        ),
        # Case 7: Univariate normal manually specified
        (
            UnivariateCumulantGeneratingFunction(
                K=lambda t, loc=0, scale=1: loc * t + scale**2 * t**2 / 2
            ),
            lambda t, pdf=sps.norm.pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=-10, b=10)[0]
            ),
            [0.2, 0.55],
            sps.norm(loc=0, scale=1),
        ),
        (
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
        ),
        # Case 8: Univariate exponential
        (
            exponential(scale=1),
            lambda t, pdf=sps.expon.pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
            ),
            [0.2, 0.55],
            sps.expon(scale=1),
        ),
        # TODO: fix this test
        # (
        #     MultivariateCumulantGeneratingFunction.from_univariate(
        #         norm(0, 1), 2 * exponential(scale=1) / 2
        #     )[1],
        #     exponential(scale=1).K,
        #     [0.2, 0.55],
        #     sps.expon(scale=1),
        # ),
        # Case 9: Univariate exponential with scale
        (
            exponential(scale=0.5),
            lambda t, pdf=sps.expon(scale=0.5).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
            ),
            [0.2, 0.55],
            sps.expon(scale=0.5),
        ),
        # (
        #     MultivariateCumulantGeneratingFunction.from_univariate(
        #         exponential(scale=0.5), exponential(scale=0.5)
        #     ).ldot([1, 0]),
        #     exponential(scale=0.5).K,
        #     [0.2, 0.55],
        #     sps.expon(scale=0.5),
        # ),
        # Case 10: Univariate exponential cgf manually specified
        (
            UnivariateCumulantGeneratingFunction(K=lambda t: np.log(1 / (1 - t))),
            exponential(scale=1).K,
            [0.2, 0.55],
            sps.expon(scale=1),
        ),
        (
            MultivariateCumulantGeneratingFunction.from_univariate(
                UnivariateCumulantGeneratingFunction(
                    K=lambda t: np.log(1 / (1 - t)),
                )
            )[0],
            exponential(scale=1).K,
            [0.2, 0.55],
            sps.expon(scale=1),
        ),
        # Case 11: Univariate gamma
        (
            gamma(a=2, scale=0.5),
            lambda t, pdf=sps.gamma(a=2, scale=0.5).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
            ),
            [0.2, 0.55],
            sps.gamma(a=2, scale=0.5),
        ),
        # TODO: fix this test
        # (
        #     MultivariateCumulantGeneratingFunction.from_univariate(
        #         gamma(a=2, scale=0.5), gamma(a=2, scale=0.5)
        #     ).ldot([1, 0])[0],
        #     gamma(a=2, scale=0.5),
        #     [0.2, 0.55],
        #     sps.gamma(a=2, scale=0.5),
        # ),
        # Case 12: Univariate chi2
        (
            chi2(df=3),
            lambda t, pdf=sps.chi2(df=3).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=0, b=100)[0]
            ),
            [0.2, 0.25],
            sps.chi2(df=3),
        ),
        # TODO: fix this test
        # (
        #     MultivariateCumulantGeneratingFunction.from_univariate(chi2(df=2), chi2(df=3)).ldot(
        #         [1, 0]
        #     )[1],
        #     chi2(df=3),
        #     [0.2, 0.25],
        #     sps.chi2(df=3),
        # ),
        # Case 13: Univariate laplace
        (
            laplace(loc=0, scale=1),
            lambda t, pdf=sps.laplace(loc=0, scale=1).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=-50, b=50)[0]
            ),
            [0.2, 0.55, -0.23],
            sps.laplace(loc=0, scale=1),
        ),
        # TODO: fix this test
        # (
        #     MultivariateCumulantGeneratingFunction.from_univariate(
        #         laplace(loc=0, scale=1), laplace(loc=0, scale=1)
        #     ).ldot([1, 0])[0],
        #     laplace(loc=0, scale=1),
        #     [0.2, 0.55, -0.23],
        #     sps.laplace(loc=0, scale=1),
        # ),
        # Case 14: Univariate poisson
        (
            poisson(mu=2),
            lambda t, pmf=sps.poisson(mu=2).pmf: np.log(
                np.sum([np.exp(t * x) * pmf(x) for x in range(100)])
            ),
            [0.2, 0.55],
            sps.poisson(mu=2),
        ),
        # TODO: fix this test
        # (
        #     MultivariateCumulantGeneratingFunction.from_univariate(
        #         poisson(mu=2), poisson(mu=2)
        #     )[0],
        #     poisson(mu=2),
        #     [0.2, 0.55],
        #     sps.poisson(mu=2),
        # ),
        # Case 15: Univariate binomial
        (
            binomial(n=10, p=0.5),
            lambda t, pmf=sps.binom(n=10, p=0.5).pmf: np.log(
                np.sum([np.exp(t * x) * pmf(x) for x in range(100)])
            ),
            [0.2, 0.55],
            sps.binom(n=10, p=0.5),
        ),
        # TODO: fix this test
        # (
        #     MultivariateCumulantGeneratingFunction.from_univariate(
        #         binomial(n=10, p=0.5), binomial(n=10, p=0.5)
        #     )[0],
        #     binomial(n=10, p=0.5),
        #     [0.2, 0.55],
        #     sps.binom(n=10, p=0.5),
        # ),
        # Case 16: Univariate sample mean
        (
            univariate_sample_mean(norm(2, 1), 25),
            lambda t, pdf=sps.norm(loc=2, scale=0.2).pdf: np.log(
                quad(lambda x: pdf(x) * np.exp(t * x), a=-50, b=50)[0]
            ),
            [0.2, 0.55, -0.23],
            sps.norm(loc=2, scale=0.2),
        ),
        # TODO: fix this test
        # (
        #     MultivariateCumulantGeneratingFunction.from_univariate(
        #         univariate_sample_mean(norm(2, 1), 25), univariate_sample_mean(norm(2, 1), 25)
        #     )[0],
        #     univariate_sample_mean(norm(2, 1), 25),
        #     [0.2, 0.55, -0.23],
        #     sps.norm(loc=2, scale=0.2),
        # ),
        # Case 17: Univariate empirical
        (
            univariate_empirical(np.arange(10)),
            lambda t, x=np.arange(10): np.log(np.sum([np.exp(t * x) / len(x)])),
            [0.2, 0.55, -0.23],
            np.arange(10),
        ),
        # TODO: fix this test
        # (
        #     MultivariateCumulantGeneratingFunction.from_univariate(
        #         univariate_empirical(np.arange(10)), univariate_empirical(np.arange(10))
        #     )[0],
        #     univariate_empirical(np.arange(10)),
        #     [0.2, 0.55, -0.23],
        #     np.arange(10),
        # ),
    ],
)
def test_cgf(cgf_to_test, cgf, ts, dist):
    # Test function evaluations
    assert isinstance(cgf_to_test, UnivariateCumulantGeneratingFunction)
    for t in ts:
        assert np.isclose(cgf(t), cgf_to_test.K(t), atol=1e-4)
        dcgf = nd.Derivative(cgf_to_test.K, n=1)
        assert np.isclose(dcgf(t), cgf_to_test.dK(t))
        d2cgf = nd.Derivative(cgf_to_test.K, n=2)
        assert np.isclose(d2cgf(t), cgf_to_test.d2K(t))
        d3cgf = nd.Derivative(cgf_to_test.K, n=3)
        assert np.isclose(d3cgf(t), cgf_to_test.d3K(t))
    # Test vectorized evaluation
    assert np.isclose(cgf_to_test.mean, dist.mean())
    assert np.isclose(cgf_to_test.variance, dist.var())
    # Test addition scalar
    for t in ts:
        assert np.isclose(cgf(t) + t, (cgf_to_test + float(1)).K(t))
        dcgf = nd.Derivative(cgf_to_test.K, n=1)
        assert np.isclose(cgf(t) - 2 * t, (cgf_to_test - int(2)).K(t), atol=1e-5)
        cgf_to_test.add(3, inplace=True)
        assert np.isclose(cgf(t) + 3 * t, cgf_to_test.K(t))
        cgf_to_test.add(-3, inplace=True)
        cgf_to_test.add(3, inplace=False)
        assert np.isclose(cgf(t), cgf_to_test.K(t))
        dcgf = nd.Derivative(cgf, n=1)
        cgf_to_test.add(3, inplace=True)
        assert np.isclose(dcgf(t) + 3, cgf_to_test.dK(t), atol=1e-4)
        assert np.isclose(cgf_to_test.dK0, dcgf(0) + 3)
        cgf_to_test.add(-3, inplace=True)
        assert np.isclose(cgf_to_test.dK0, dcgf(0))
    # Test addition other cumulant generating function
    for t in ts:
        assert np.isclose(cgf(t) + cgf(t), (cgf_to_test + cgf_to_test).K(t))
        with pytest.raises(AssertionError):
            cgf_to_test.add(cgf_to_test, inplace=True)


def test_domain():
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
    # TODO: fix this test
    # cgf = MultivariateCumulantGeneratingFunction(
    #     K=lambda t: t**4,
    #     domain=Domain(g=0, l=2),
    #     dim=2,
    # )[0]
    # assert np.isnan(cgf.K(0.5))
    # assert not np.isnan(cgf.K(1.5))
    # assert np.isnan(cgf.dK(0.5))
    # assert not np.isnan(cgf.dK(1.5))
    # assert np.isnan(cgf.d2K(0.5))
    # assert not np.isnan(cgf.d2K(1.5))
    # assert np.isnan(cgf.d3K(0.5))
    # assert not np.isnan(cgf.d3K(1.5))
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
        # TODO: fix this test
        # (
        # MultivariateCumulantGeneratingFunction.from_univariate(exponential(scale=1))[0],
        # [0.2, 0.55, 0.95],
        # ),
        # TODO: continue here
        # Case 5: Univariate exponential with scale
        (
            exponential(scale=0.5),
            [0.2, 0.55, 0.95],
        ),
        # TODO: fix this test
        # (
        #     MultivariateCumulantGeneratingFunction.from_univariate(exponential(scale=0.5))[0],
        #     [0.2, 0.55, 0.95],
        # )
        # Case 6: Univariate exponential manually specified
        (
            UnivariateCumulantGeneratingFunction(K=lambda t: np.log(1 / (1 - t))),
            [0.2, 0.55, 0.95],
        ),
        # TODO: add multivariate case
        # Case 7: Univariate gamma
        (
            gamma(a=2, scale=0.5),
            [0.2, 0.55],
        ),
        # TODO: add multivariate case
        # Case 8: Univariate chi2
        (
            chi2(df=3),
            [0.2, 0.25],
        ),
        # TODO: add multivariate case
        # Case 9: Univariate laplace
        (
            laplace(loc=0, scale=1),
            [0.2, 0.55, -0.23],
        ),
        # TODO: add multivariate case
        # Case 10: Univariate poisson
        (
            poisson(mu=2),
            [0.2, 0.55],
        ),
        # TODO: add multivariate case
        # Case 11: Univariate binomial
        (
            binomial(n=10, p=0.5),
            [0.2, 0.55],
        ),
        # TODO: add multivariate case
        # Case 12: Univariate beta
        (
            univariate_sample_mean(norm(2, 1), 25),
            [0.2, 0.55, -0.23],
        ),
        # TODO: add multivariate case
    ],
)
def test_dKinv(cgf, ts):
    for t in ts:
        assert np.isclose(cgf.dK_inv(cgf.dK(t)), t)
    assert np.allclose(cgf.dK_inv(cgf.dK(ts[:1])), ts[:1])
    assert np.allclose(cgf.dK_inv(cgf.dK(ts)), ts)
    assert np.allclose(cgf.dK_inv(cgf.dK(ts)), [cgf.dK_inv(cgf.dK(t)) for t in ts])


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
