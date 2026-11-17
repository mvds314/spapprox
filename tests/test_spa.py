import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats as sps
from scipy.integrate import quad

from spapprox import (
    BivariateSaddlePointApprox,
    EstimatingEquationSaddlePointApprox,
    MultivariateCumulantGeneratingFunction,
    MultivariateSaddlePointApprox,
    # poisson,
    # binomial,
    UnivariateCumulantGeneratingFunction,
    UnivariateSaddlePointApprox,
    UnivariateSaddlePointApproxMean,
    binomial,
    bivariate_gamma,
    chi2,
    exponential,
    gamma,
    laplace,
    multivariate_norm,
    norm,
    poisson,
    univariate_empirical,
)
from spapprox.domain import Domain
from spapprox.mvn import _mvn_logtail, _mvn_tail_derivatives
from spapprox.spa import _bvn_tail_derivatives


@pytest.mark.parametrize(
    "cgf,dist,trange",
    [
        pytest.param(
            norm(loc=0, scale=1), sps.norm(loc=0, scale=1), [-5, 5], id="Standard normal"
        ),
        pytest.param(
            norm(loc=0.5, scale=3), sps.norm(loc=0.5, scale=3), [-10, 10], id="Transformed normal"
        ),
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
        pytest.param(norm(loc=0, scale=1), [-5, 5], id="Standard normal"),
        pytest.param(norm(loc=0.5, scale=3), [-10, 10], id="Transformed normal"),
        pytest.param(exponential(scale=10), [-1e5, 1 / 10 * 0.99999999], id="Exponential"),
        pytest.param(gamma(a=2, scale=3), [-1e3, 1 / 3 * 0.99999999], id="Gamma"),
        pytest.param(chi2(df=3), [-1e4, 1 / 2 * 0.99999999], id="Chi2"),
        pytest.param(
            laplace(loc=1, scale=3), [-1 / 3 * 0.99999999, 1 / 3 * 0.99999999], id="Laplace"
        ),
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
            points=[0],  # prevent quad from missing the mass around zero on wide ranges
        )[0],
        1,
        atol=5e-4,
    )


@pytest.mark.parametrize(
    "cgf,dist,trange",
    [
        pytest.param(exponential(scale=3), sps.expon(scale=3), [0, 1 / 3], id="Exponential"),
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
    assert np.allclose(spa.cdf(t=t, backend="LR"), dist.cdf(x), atol=5e-3), (
        "This should approx be equal"
    )
    assert np.allclose(spa.cdf(t=t, backend="BN"), dist.cdf(x), atol=5e-3), (
        "This should approx be equal"
    )
    assert not np.allclose(spa.cdf(t=t, backend="BN"), spa.cdf(t=t, backend="LR")), (
        "the approximation should not be exactly equal"
    )
    # Test investigation saddle point
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
        pytest.param(
            multivariate_norm(loc=0.5, scale=3),
            sps.multivariate_normal(mean=[0.5, 0.5], cov=9),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 10), 2)),
            2,
            id="Multivariate normal uncorrelated",
        ),
        # Test with correlated variables
        pytest.param(
            multivariate_norm(loc=[0.5, 0.2], cov=[[3, 1], [1, 3]]),
            sps.multivariate_normal(mean=[0.5, 0.2], cov=[[3, 1], [1, 3]]),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 10), 2)),
            2,
            id="Multivariate normal correlated",
            marks=pytest.mark.slow,
        ),
        # 3 dim test
        pytest.param(
            multivariate_norm(loc=0.5, scale=3, dim=3),
            sps.multivariate_normal(mean=[0.5, 0.5, 0.5], cov=9),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 10), 3)),
            3,
            id="Multivariate normal 3 dim",
            marks=pytest.mark.slow,
        ),
        # Other distribution
        pytest.param(
            MultivariateCumulantGeneratingFunction.from_univariate(
                norm(loc=0.5, scale=3), norm(loc=0.5, scale=3)
            ),
            sps.multivariate_normal(mean=[0.5, 0.5], cov=9),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 10), 2)),
            2,
            id="Multivariate normal from univariate",
            marks=pytest.mark.slow,
        ),
        # TODO: test non-normal distribution?
        # TODO: add multivariate gamma
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


@pytest.mark.parametrize(
    "cgf, dist, ts, dim",
    [
        # Basic test with uncorrelated variables
        pytest.param(
            multivariate_norm(loc=0, scale=1),
            sps.multivariate_normal(mean=[0, 0], cov=1),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 13), 2)),
            2,
            id="Bivariate normal uncorrelated",
        ),
        pytest.param(
            multivariate_norm(loc=0, scale=np.sqrt(2)),
            sps.multivariate_normal(mean=[0, 0], cov=2),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 13), 2)),
            2,
            id="Bivariate normal uncorrelated, scaled, using scale",
        ),
        pytest.param(
            multivariate_norm(loc=0, cov=[[2, 0], [0, 2]]),
            sps.multivariate_normal(mean=[0, 0], cov=2),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 13), 2)),
            2,
            id="Bivariate normal uncorrelated, scaled using cov",
        ),
        pytest.param(
            multivariate_norm(loc=0, cov=[[2, 1], [1, 2]]),
            sps.multivariate_normal(mean=[0, 0], cov=[[2, 1], [1, 2]]),
            list(itertools.combinations_with_replacement(np.linspace(-10, 10, 13), 2)),
            2,
            id="Bivariate normal correlated",
        ),
        # TODO: maybe add more tests later
        # TODO: maybe look for book examples
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
    for i, t in enumerate(ts):
        x = spa.cgf.dK(t)
        assert np.allclose(spa.cdf(t=t), dist.cdf(x), atol=1e-6)
    assert np.allclose(spa.cdf(t=ts), dist.cdf(spa.cgf.dK(ts)), atol=1e-6)
    assert np.allclose(spa.cdf(t=ts, backend="Kolassa"), dist.cdf(spa.cgf.dK(ts)), atol=1e-6)


def _gamma_conditional(a, x2):
    """Exact cdf and pdf of X1 given X2=x2 for bivariate_gamma(a): X1 = Y0 + Y1, X2 = Y0 + Y2."""
    a0, a1, a2 = a
    y0 = sps.beta(a0, a2)
    g1 = sps.gamma(a1)

    def pdf(x1):
        return quad(lambda u: y0.pdf(u / x2) / x2 * g1.pdf(x1 - u), 0, min(x1, x2))[0]

    def cdf(x1):
        return quad(lambda u: y0.pdf(u / x2) / x2 * g1.cdf(x1 - u), 0, min(x1, x2))[0]

    return np.vectorize(cdf), np.vectorize(pdf)


@pytest.mark.parametrize(
    "cov,given",
    [
        pytest.param([[2, 1], [1, 2]], [0.7], id="Bivariate"),
        pytest.param(
            [[2, 0.5, -0.3], [0.5, 1, 0.4], [-0.3, 0.4, 1.5]], [0.7, -0.4], id="Trivariate"
        ),
    ],
)
@pytest.mark.parametrize("i", [0, -1])
def test_conditional_normal(cov, given, i):
    cov = np.asarray(cov, dtype=float)
    d = len(cov)
    i = i % d
    J = [j for j in range(d) if j != i]
    spa = MultivariateSaddlePointApprox(multivariate_norm(loc=[0.3] * d, cov=cov))
    # Exact conditional of a normal
    mu = 0.3 + cov[i, J] @ np.linalg.solve(cov[np.ix_(J, J)], np.asarray(given) - 0.3)
    var = cov[i, i] - cov[i, J] @ np.linalg.solve(cov[np.ix_(J, J)], cov[J, i])
    dist = sps.norm(loc=mu, scale=np.sqrt(var))
    x = np.linspace(mu - 4 * np.sqrt(var), mu + 4 * np.sqrt(var), 41)
    assert np.allclose(spa.conditional_cdf(x, given, i=i), dist.cdf(x), atol=1e-8)
    assert np.allclose(spa.conditional_pdf(x, given, i=i), dist.pdf(x), atol=1e-8)
    # Scalar input, and the point at the singularity of the formula
    assert np.isclose(spa.conditional_cdf(mu, given, i=i), 0.5, atol=1e-6)
    assert np.isclose(spa.conditional_pdf(mu, given, i=i), dist.pdf(mu), atol=1e-8)


def test_conditional_independent():
    spa = MultivariateSaddlePointApprox(multivariate_norm(loc=0, cov=[[1, 0], [0, 4]]))
    x = np.linspace(-3, 3, 7)
    for y in [-1.0, 0.0, 2.0]:
        assert np.allclose(spa.conditional_cdf(x, [y], i=0), sps.norm.cdf(x), atol=1e-8)


@pytest.mark.parametrize("a", [(1, 2, 3), (3, 2, 4)])
def test_conditional_bivariate_gamma(a):
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=a))
    x2 = 5.0
    cdf, pdf = _gamma_conditional(a, x2)
    x = np.linspace(0.5, 12, 24)
    got = spa.conditional_cdf(x, [x2], i=0)
    assert np.allclose(got, cdf(x), atol=6e-3)
    got = spa.conditional_pdf(x, [x2], i=0)
    assert np.allclose(got, pdf(x), rtol=0.1, atol=1e-3)


def test_conditional_bivariate_gamma_convergence():
    # The relative error is O(1/n) for the sum of n iid vectors, here a -> k * a
    errors = []
    for k in [1, 8]:
        a = np.array([1, 2, 3]) * k
        spa = BivariateSaddlePointApprox(bivariate_gamma(a=a))
        x2 = 5.0 * k
        cdf, _ = _gamma_conditional(a, x2)
        x = np.array([0.5, 1.0, 1.5]) * 2 * k
        errors.append(np.max(np.abs(spa.conditional_cdf(x, [x2], i=0) / cdf(x) - 1)))
    assert errors[1] < errors[0] / 5


def _trivariate_gamma(a):
    """Common gamma Y0 plus independent Y_j: X_j = Y0 + Y_j, j=1..3, a=(a0,a1,a2,a3)."""
    a0, a1, a2, a3 = a
    c = np.array([a1, a2, a3], dtype=float)

    def K(t):
        t = np.asanyarray(t)
        return -(c * np.log(1 - t)).sum(-1) - a0 * np.log(1 - t.sum(-1))

    def dK(t):
        t = np.asanyarray(t)
        return c / (1 - t) + a0 / (1 - t.sum(-1, keepdims=True))

    def d2K(t):
        t = np.asarray(t, float)
        return np.diag(c / (1 - t) ** 2) + a0 / (1 - t.sum()) ** 2

    def d3K(t):
        t = np.asarray(t, float)
        T = np.full((3, 3, 3), 2 * a0 / (1 - t.sum()) ** 3)
        for j in range(3):
            T[j, j, j] += 2 * c[j] / (1 - t[j]) ** 3
        return T

    return MultivariateCumulantGeneratingFunction(
        K=K,
        dK=dK,
        d2K=np.vectorize(d2K, signature="(3)->(3,3)"),
        d3K=np.vectorize(d3K, signature="(3)->(3,3,3)"),
        dim=3,
        domain=Domain(l=np.ones(3), B=np.ones((1, 3)), b=np.ones(1), dim=3),
    )


def _trivariate_gamma_conditional(a, x2, x3):
    """Exact cdf of X1 given X2=x2, X3=x3: Y0|X2,X3 has density ~ y^(a0-1)(x2-y)^(a2-1)(x3-y)^(a3-1)e^y."""
    a0, a1, a2, a3 = a
    m = min(x2, x3)
    w = lambda y: y ** (a0 - 1) * (x2 - y) ** (a2 - 1) * (x3 - y) ** (a3 - 1) * np.exp(y)
    Z = quad(w, 0, m)[0]
    g = sps.gamma(a1)
    return np.vectorize(lambda x: quad(lambda y: w(y) * g.cdf(x - y), 0, m)[0] / Z)


@pytest.mark.parametrize("i", [0, 1])
def test_conditional_trivariate_gamma(i):
    # Second order beats first order and converges faster in n, also for a middle index
    errors = {"Skovgaard": [], "Kolassa": []}
    for k in [1, 4]:
        a = np.array([1, 2, 3, 2]) * k
        x2, x3 = 5.0 * k, 4.0 * k
        cdf = _trivariate_gamma_conditional(a, x2, x3)
        # Variable i takes the role of X1, relabel the others to keep the given values in order
        spa = MultivariateSaddlePointApprox(_trivariate_gamma(a if i == 0 else a[[0, 2, 1, 3]]))
        x = np.array([0.8, 1.0, 1.3]) * 2 * k
        for backend in errors:
            got = spa.conditional_cdf(x, [x2, x3], i=i, backend=backend)
            errors[backend].append(np.max(np.abs(got - cdf(x))))
    assert errors["Kolassa"][0] < errors["Skovgaard"][0] / 4
    assert errors["Kolassa"][1] < errors["Skovgaard"][1] / 8
    assert errors["Kolassa"][1] < errors["Kolassa"][0] / 15


@pytest.mark.parametrize("dim", [2, 3])
def test_conditional_kolassa_normal(dim):
    cov = np.array([[2.0, 0.6, 0.3], [0.6, 1.0, -0.4], [0.3, -0.4, 1.5]])[:dim, :dim]
    mu = np.arange(dim) + 1.0
    spa = MultivariateSaddlePointApprox(multivariate_norm(loc=mu, cov=cov))
    given = np.array([1.5, 2.8])[: dim - 1]
    x = np.linspace(-1, 5, 9)
    m = mu[0] + cov[0, 1:] @ np.linalg.solve(cov[1:, 1:], given - mu[1:])
    s = np.sqrt(cov[0, 0] - cov[0, 1:] @ np.linalg.solve(cov[1:, 1:], cov[1:, 0]))
    got = spa.conditional_cdf(x, given, backend="Kolassa")
    assert np.allclose(got, sps.norm.cdf(x, m, s), atol=1e-6)


def test_conditional_kolassa_bivariate_gamma():
    # Second order is more accurate than first order and converges faster in n
    errors = {"Skovgaard": [], "Kolassa": []}
    for k in [1, 4]:
        a = np.array([1, 2, 3]) * k
        spa = BivariateSaddlePointApprox(bivariate_gamma(a=a))
        x2 = 5.0 * k
        cdf, _ = _gamma_conditional(a, x2)
        x = np.array([0.7, 1.0, 1.5]) * 2 * k
        for backend in errors:
            got = spa.conditional_cdf(x, [x2], backend=backend)
            errors[backend].append(np.max(np.abs(got - cdf(x))))
    assert errors["Kolassa"][0] < errors["Skovgaard"][0] / 5
    assert errors["Kolassa"][1] < errors["Skovgaard"][1] / 50
    assert errors["Kolassa"][1] < errors["Kolassa"][0] / 20


def test_conditional_kolassa_continuous_at_singularity():
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=[3, 4, 5]))
    center = float(spa.cgf.dK(spa._conditional_denominator_saddle(0, [1], [6.0]))[0])
    x = center + np.array([-1e-1, -2e-2, -1e-3, 0, 1e-3, 2e-2, 1e-1])
    got = np.array(spa.conditional_cdf(x, [6.0], backend="Kolassa"))
    assert np.all(np.isfinite(got)) and np.all(np.diff(got) > 0)


def test_conditional_unknown_backend():
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=[1, 2, 3]))
    with pytest.raises(ValueError):
        spa.conditional_cdf(1.0, [2.0], backend="nope")
    with pytest.raises(ValueError):
        spa.conditional_cdf(1.0, [2.0], delta=0)


def _poisson_sum(lams, A):
    """Cgf of X = A^T N with N_m independent Poisson(lams_m): a vector on a lattice."""
    lams = np.asarray(lams, float)
    A = np.asarray(A, float)
    d = A.shape[1]
    K = lambda t: (lams * (np.exp(np.asanyarray(t) @ A.T) - 1)).sum(-1)
    dK = lambda t: (lams * np.exp(np.asanyarray(t) @ A.T)) @ A

    def d2K(t):
        e = lams * np.exp(A @ np.asarray(t, float))
        return np.einsum("m,mi,mj->ij", e, A, A)

    def d3K(t):
        e = lams * np.exp(A @ np.asarray(t, float))
        return np.einsum("m,mi,mj,mk->ijk", e, A, A, A)

    return MultivariateCumulantGeneratingFunction(
        K=K,
        dK=dK,
        d2K=np.vectorize(d2K, signature=f"({d})->({d},{d})"),
        d3K=np.vectorize(d3K, signature=f"({d})->({d},{d},{d})"),
        dim=d,
        domain=Domain(dim=d),
    )


def test_conditional_lattice_binomial():
    # N1 given N1 + N2 = s is binomial; the lattice corrections are accurate and converge in n
    errors = {"Skovgaard": [], "Kolassa": []}
    for k in [1, 4]:
        spa = MultivariateSaddlePointApprox(_poisson_sum([2.0 * k, 3.0 * k], [[1, 1], [0, 1]]))
        s = 5 * k
        n = np.arange(1, s)
        for backend in errors:
            got = spa.conditional_cdf(n, [s], backend=backend, delta=1.0)
            errors[backend].append(np.max(np.abs(got - sps.binom.cdf(n, s, 0.4))))
    assert errors["Skovgaard"][1] < 5e-4
    assert errors["Kolassa"][0] < errors["Skovgaard"][0] / 3
    assert errors["Kolassa"][1] < errors["Kolassa"][0] / 20
    # Without the lattice correction the approximation is bad
    assert np.max(np.abs(spa.conditional_cdf(n, [s]) - sps.binom.cdf(n, s, 0.4))) > 0.05


def _nb_pair(r1, r2, p):
    """Cgf of (N1, N1 + N2) for independent negative binomials with common success prob p."""
    q = 1 - p
    c = -np.log(q)
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    r = np.array([r1, r2])

    def parts(t):
        s = A @ np.asarray(t, float)
        e = q * np.exp(s)
        z = e / (1 - e)
        return z

    K = lambda t: (
        (r * np.log(p)).sum() - (r * np.log(1 - q * np.exp(np.asanyarray(t) @ A.T))).sum(-1)
    )

    def dK(t):
        t = np.asanyarray(t)
        z = q * np.exp(t @ A.T)
        z = z / (1 - z)
        return (r * z) @ A

    def d2K(t):
        z = parts(t)
        return np.einsum("m,mi,mj->ij", r * z * (1 + z), A, A)

    def d3K(t):
        z = parts(t)
        return np.einsum("m,mi,mj,mk->ijk", r * z * (1 + z) * (1 + 2 * z), A, A, A)

    return MultivariateCumulantGeneratingFunction(
        K=K,
        dK=dK,
        d2K=np.vectorize(d2K, signature="(2)->(2,2)"),
        d3K=np.vectorize(d3K, signature="(2)->(2,2,2)"),
        dim=2,
        domain=Domain(B=A, b=np.array([c, c]), dim=2),
    )


def test_conditional_lattice_polya():
    # N1 given N1 + N2 = s is beta-binomial (Polya) for negative binomials with a common p
    errors = {"Skovgaard": [], "Kolassa": []}
    for k in [1, 4, 16]:
        r1, r2, s = 2.0 * k, 3.0 * k, 6 * k
        spa = MultivariateSaddlePointApprox(_nb_pair(r1, r2, 0.4))
        n = np.arange(1, s)
        for backend in errors:
            got = spa.conditional_cdf(n, [s], backend=backend, delta=1.0)
            errors[backend].append(np.max(np.abs(got - sps.betabinom.cdf(n, s, r1, r2))))
    assert errors["Kolassa"][0] < errors["Skovgaard"][0] / 2
    assert errors["Skovgaard"][2] < errors["Skovgaard"][1] / 5
    assert errors["Kolassa"][2] < 1e-6
    assert errors["Kolassa"][2] < errors["Skovgaard"][2] / 50


@pytest.mark.parametrize("backend", ["Skovgaard", "Kolassa"])
def test_conditional_lattice_scale_invariant(backend):
    n = np.arange(1, 5)
    unit = MultivariateSaddlePointApprox(_poisson_sum([2.0, 3.0], [[1, 1], [0, 1]]))
    double = MultivariateSaddlePointApprox(_poisson_sum([2.0, 3.0], [[2, 1], [0, 1]]))
    assert np.allclose(
        unit.conditional_cdf(n, [5], backend=backend, delta=1.0),
        double.conditional_cdf(2 * n, [5], backend=backend, delta=2.0),
    )


def test_conditional_continuous_at_singularity():
    # tau_hat[i] = 0 at x_i = E[X_i | X_J] under the saddlepoint, where w = u = 0
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=(1, 2, 3)))
    x2 = 5.0
    x0 = spa.cgf[1].dK_inv(x2)
    center = float(spa.cgf.dK([0.0, x0])[0])
    xs = center + np.linspace(-0.05, 0.05, 41)
    vals = spa.conditional_cdf(xs, [x2], i=0)
    assert np.isfinite(vals).all()
    assert np.all(np.diff(vals) > 0)
    assert np.abs(np.diff(vals, 2)).max() < 1e-5


def test_conditional_input_shapes():
    spa = MultivariateSaddlePointApprox(multivariate_norm(loc=0, cov=[[2, 1], [1, 2]]))
    xs = np.array([-1.0, 0.0, 1.0])
    ys = np.array([0.5, -0.5, 1.0])
    # One conditioning value per point
    got = spa.conditional_cdf(xs, ys[:, None], i=0)
    expected = [spa.conditional_cdf(x, [y], i=0) for x, y in zip(xs, ys)]
    assert np.allclose(got, expected)
    with pytest.raises(ValueError):
        spa.conditional_cdf(xs, [0.0, 1.0], i=0)


def test_ppf_with_t0():
    spa = UnivariateSaddlePointApprox(norm())
    q = np.array([0.2, 0.7])
    assert np.allclose(spa.ppf(q, t0=np.zeros(2)), sps.norm.ppf(q), atol=1e-4)


def test_fit_ppf_after_fit_saddle_point_eqn():
    qs = np.array([0.025, 0.3, 0.5, 0.975])
    spa = UnivariateSaddlePointApprox(norm())
    spa.fit_saddle_point_eqn()
    spa.fit_ppf()
    assert np.allclose(spa.ppf(qs), sps.norm.ppf(qs), atol=1e-3)


def test_fit_ppf_and_saddle_point_eqn_caches_are_independent():
    qs = np.array([0.025, 0.3, 0.5, 0.975])
    spa = UnivariateSaddlePointApprox(norm())
    spa.fit_ppf()
    spa.fit_saddle_point_eqn(num=200)
    assert np.allclose(spa.ppf(qs), sps.norm.ppf(qs), atol=1e-3)
    assert np.allclose(spa.pdf(x=[-1.0, 0.5]), sps.norm.pdf([-1.0, 0.5]), atol=1e-3)
    spa.fit_ppf(num=50)
    assert len(spa._x_cache) == len(spa._t_cache) == 200
    assert np.allclose(spa.pdf(x=[-1.0, 0.5]), sps.norm.pdf([-1.0, 0.5]), atol=1e-3)


def test_fit_ppf_honours_t_range():
    spa = UnivariateSaddlePointApprox(gamma(a=3))
    spa.fit_ppf(t_range=[-2, 0.8], num=50)
    assert len(spa._ppf_t_cache) == 50
    assert np.allclose(spa._ppf_t_cache[[0, -1]], [-2, 0.8])
    assert np.isclose(spa.ppf(0.5), sps.gamma(a=3).ppf(0.5), atol=1e-2)


@pytest.mark.parametrize("backend", ["LR", "BN"])
def test_cdf_at_zero_without_analytic_derivatives(backend):
    spa = UnivariateSaddlePointApprox(UnivariateCumulantGeneratingFunction(K=lambda t: t**2 / 2))
    assert np.isclose(spa.cdf(t=0.0, backend=backend), 0.5)


@pytest.mark.parametrize("backend", ["LR", "BN"])
def test_cdf_near_zero_no_cancellation(backend):
    a = 3
    spa = UnivariateSaddlePointApprox(gamma(a=a))
    rho3 = 2 / np.sqrt(a)
    limit = 0.5 + rho3 / 6 / np.sqrt(2 * np.pi) if backend == "LR" else sps.norm.cdf(rho3 / 6)
    assert np.isclose(spa.cdf(t=0.0, backend=backend), limit)
    st = np.array([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 9.99e-3, 1.01e-2, 2e-2])
    t = np.concatenate([-st[::-1], [0.0], st]) / np.sqrt(a)
    cdf = np.asarray(spa.cdf(t=t, backend=backend))
    assert np.all(np.diff(cdf) > 0)
    assert np.allclose(cdf, sps.gamma(a).cdf(spa.cgf.dK(t)), atol=2e-3)
    # Points very close to zero are continuous with the limit
    assert np.allclose(cdf[np.abs(t) < 1e-5], limit, atol=1e-5)
    for tt in [1e-7, -1e-6]:
        assert np.isclose(spa.cdf(t=tt, backend=backend), limit, atol=1e-5)


def test_cdf_at_mean_after_fit_saddle_point_eqn():
    spa = UnivariateSaddlePointApprox(gamma(a=3))
    spa.fit_saddle_point_eqn()
    assert np.isclose(spa.cdf(x=3.0), 0.5 + 2 / np.sqrt(3) / 6 / np.sqrt(2 * np.pi), atol=1e-4)


_INFER_T_RANGE_LATTICE_SCRIPT = """
import warnings
import numpy as np
from spapprox import UnivariateSaddlePointApprox, binomial, gamma, poisson

for cgf in [binomial(10, 0.3), poisson(3)]:
    spa = UnivariateSaddlePointApprox(cgf)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lb, ub = spa.infer_t_range()
    assert any("Could not expand t range" in str(wi.message) for wi in w)
    assert lb < 0 < ub
    assert 0 <= spa.cdf(t=lb) < 0.05 and 0.95 < spa.cdf(t=ub) <= 1
    val = spa.pdf(t=0.1)
    assert np.isfinite(val) and val > 0, val

# Distributions whose cdf does reach 0 and 1 are unaffected and do not warn
with warnings.catch_warnings():
    warnings.simplefilter("error", RuntimeWarning)
    assert np.allclose(UnivariateSaddlePointApprox(gamma(a=3)).infer_t_range(), [-64.0, 0.9])
"""


def test_infer_t_range_lattice_terminates():
    # Run in a subprocess so a regression to an infinite loop fails instead of hanging
    import subprocess
    import sys

    try:
        res = subprocess.run(
            [sys.executable, "-c", _INFER_T_RANGE_LATTICE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("infer_t_range did not terminate")
    assert res.returncode == 0, res.stderr


def test_bvar_cdf_near_zero():
    spa = BivariateSaddlePointApprox(multivariate_norm(dim=2))
    assert np.isclose(spa.cdf(t=np.array([1e-9, 1e-9])), 0.25)


def test_bvar_cdf_independent_reduces_to_product():
    # For independent components the bivariate approximation should factorize
    g1, g2 = gamma(a=3), gamma(a=5)
    spa = BivariateSaddlePointApprox(
        MultivariateCumulantGeneratingFunction.from_univariate(g1, g2)
    )
    u1, u2 = UnivariateSaddlePointApprox(g1), UnivariateSaddlePointApprox(g2)
    for t in [[-1.5, -1.2], [-0.5, 0.5], [0.3, -0.4], [0.6, 0.2]]:
        t = np.array(t)
        x = spa.cgf.dK(t)
        exact = sps.gamma(3).cdf(x[0]) * sps.gamma(5).cdf(x[1])
        cdf = spa.cdf(t=t, backend="Wang")
        assert np.isclose(cdf, u1.cdf(t=t[0]) * u2.cdf(t=t[1]), atol=1e-6)
        assert np.isclose(cdf, exact, atol=1e-3)
        # Kolassa and Li omit the O(1/n) product term, so it does not factorize exactly
        assert np.isclose(spa.cdf(t=t), exact, atol=6e-3)


def test_bvar_cdf_independent_one_saddlepoint_component_zero():
    # Singular cases where one saddlepoint component is zero should also factorize
    g1, g2 = gamma(a=3), gamma(a=5)
    spa = BivariateSaddlePointApprox(
        MultivariateCumulantGeneratingFunction.from_univariate(g1, g2)
    )
    u1, u2 = UnivariateSaddlePointApprox(g1), UnivariateSaddlePointApprox(g2)
    for t in [[0.5, 0], [-0.8, 0], [0, 0.4], [0, -1.0]]:
        t = np.array(t, dtype=float)
        x = spa.cgf.dK(t)
        exact = sps.gamma(3).cdf(x[0]) * sps.gamma(5).cdf(x[1])
        cdf = spa.cdf(t=t, backend="Wang")
        assert np.isclose(cdf, u1.cdf(t=t[0]) * u2.cdf(t=t[1]), atol=1e-6)
        assert np.isclose(cdf, exact, atol=1e-3)
        assert np.isclose(spa.cdf(t=t), exact, atol=6e-3)


def test_bvar_cdf_near_axes_no_cancellation():
    g1, g2 = gamma(a=3), gamma(a=5)
    spa = BivariateSaddlePointApprox(
        MultivariateCumulantGeneratingFunction.from_univariate(g1, g2)
    )
    u1, u2 = UnivariateSaddlePointApprox(g1), UnivariateSaddlePointApprox(g2)
    ts = [
        [0.3, 1e-6],
        [0.3, -2e-5],
        [1e-6, -0.5],
        [1e-7, 1e-7],
        [0, 0],
        [-1e-5, 2e-6],
        [-0.6, 2e-3],
    ]
    for t in ts:
        t = np.array(t, dtype=float)
        expected = u1.cdf(t=t[0]) * u2.cdf(t=t[1])
        assert np.isclose(spa.cdf(t=t, backend="Wang"), expected, atol=1e-6)
    cov = np.array([[1, 0.6], [0.6, 2.0]])
    spa = BivariateSaddlePointApprox(multivariate_norm(loc=np.zeros(2), cov=cov))
    for t in [[0.3, 1e-6], [1e-7, 1e-7], [1e-6, -0.4], [0, 0]]:
        t = np.array(t, dtype=float)
        expected = sps.multivariate_normal(np.zeros(2), cov).cdf(spa.cgf.dK(t))
        for backend in ["KL", "Wang"]:
            assert np.isclose(spa.cdf(t=t, backend=backend), expected, atol=1e-6)


def test_bvar_cdf_kl_continuous_near_singularities():
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=[2, 1, 3]))
    sd = np.sqrt(np.diag(spa.cgf.d2K0))
    eps = 1e-9
    for z in [-0.8, -0.2, 0.3, 0.9]:
        for p, q in [([z, eps], [z, -eps]), ([eps, z], [-eps, z])]:
            assert np.isclose(spa.cdf(t=np.array(p) / sd), spa.cdf(t=np.array(q) / sd), atol=1e-6)
    for z in [-0.6, 0.5]:
        for p in [[z, z], [z, -z]]:
            p = np.array(p) / sd
            assert np.isclose(spa.cdf(t=p * [1, 1 + eps]), spa.cdf(t=p * [1 + eps, 1]), atol=1e-6)
    # The marginal saddlepoint of the second component is zero at its mean
    vals = [spa.cdf(x=[4.0, spa.cgf.dK0[1] + d]) for d in [-1e-6, 0, 1e-6]]
    assert np.allclose(vals, vals[1], atol=1e-6)


def _bivariate_gamma_cdf(x, a):
    # (Z0 + Z1, Z0 + Z2) with independent Zi ~ Gamma(a[i])
    integrand = lambda z: (
        sps.gamma(a[0]).pdf(z) * sps.gamma(a[1]).cdf(x[0] - z) * sps.gamma(a[2]).cdf(x[1] - z)
    )
    return quad(integrand, 0, min(x), limit=200, epsabs=1e-12, epsrel=1e-10)[0]


@pytest.mark.parametrize(
    "dx,dy,expected",
    [
        (-1.2, 2.0, 0.00027),
        (-1.0, 2.0, 0.0036),
        (-0.8, 2.0, 0.022),
        (-0.5, 2.0, 0.129),
        (0.0, 2.0, 0.539),
        (0.5, 2.0, 0.871),
        (1.0, 2.0, 0.979),
        (-1.0, 1.5, 0.0036),
        (-0.5, 1.5, 0.128),
        (0.0, 1.5, 0.537),
        (0.5, 1.5, 0.869),
        (-0.5, 1.0, 0.127),
        (0.0, 1.0, 0.532),
    ],
)
def test_bvar_cdf_wang_1990_table_1(dx, dy, expected):
    # Saddlepoint column of Table 1 of Wang (1990): sample means of n=10 bivariate
    # gamma variables, evaluated at deviations (dx, dy) from the mean 2
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=[10, 10, 10]))
    x = 10 * (np.array([dx, dy]) + 2)
    assert np.isclose(spa.cdf(x=x, backend="Wang"), expected, rtol=5e-3, atol=6e-4)


@pytest.mark.parametrize(
    "backend,k,max_err",
    [
        # With Wang's order-of-integration rule; without it the errors are 0.245 and 0.033
        ("Wang", 1, 0.04),
        ("Wang", 16, 0.003),
        # Relative error O(1/n)
        ("KL", 1, 0.006),
        ("KL", 16, 4e-4),
        ("KL", 64, 1e-4),
    ],
)
def test_bvar_cdf_dependent_accuracy(backend, k, max_err):
    a = np.array([2.0, 1.0, 3.0]) * k
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=a))
    errs = []
    for zs, zt in itertools.product([-1.5, -0.8, -0.3, 0.4, 1.0, 1.8], [-1.5, -0.7, 0.35, 1.7]):
        t = np.array([zs, zt]) / np.sqrt(k) / 2.2
        if t.sum() >= 0.97:
            continue
        x = spa.cgf.dK(t)
        errs.append(abs(spa.cdf(t=t, backend=backend) - _bivariate_gamma_cdf(x, a)))
    assert max(errs) < max_err


@pytest.mark.parametrize("backend", ["KL", "Wang"])
def test_bvar_cdf_invariant_to_relabeling(backend):
    a = np.array([2.0, 1.0, 3.0])
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=a))
    spa_swapped = BivariateSaddlePointApprox(bivariate_gamma(a=a[[0, 2, 1]]))
    for t in [[-0.6, 0.3], [0.4, -0.7], [0.2, 0.3], [-0.5, -0.2]]:
        t = np.array(t)
        expected = spa_swapped.cdf(t=t[::-1], backend=backend)
        assert np.isclose(spa.cdf(t=t, backend=backend), expected, atol=1e-10)


def test_bvar_cdf_invalid_backend():
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=[2, 1, 3]))
    with pytest.raises(ValueError):
        spa.cdf(t=np.array([0.1, 0.2]), backend="foo")


def test_marginal_bivariate_gamma():
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=[2, 1, 3]))
    for i, shape in [(0, 3), (1, 5), (-1, 5), (np.int64(0), 3)]:
        marginal = spa[i]
        assert isinstance(marginal, UnivariateSaddlePointApprox)
        direct = UnivariateSaddlePointApprox(spa.cgf[int(i)])
        x = np.array([0.5, 2.0, 4.0, 9.0])
        assert np.allclose(marginal.cdf(x=x), direct.cdf(x=x))
        assert np.allclose(marginal.pdf(x=x), direct.pdf(x=x))
        assert np.allclose(marginal.cdf(x=x), sps.gamma(shape).cdf(x), atol=5e-3)
        assert np.allclose(marginal.pdf(x=x), sps.gamma(shape).pdf(x), rtol=2e-2)


def test_marginal_normal():
    mean = np.array([0.5, -1.0, 2.0])
    cov = np.array([[2.0, 0.5, 0.3], [0.5, 1.0, -0.4], [0.3, -0.4, 3.0]])
    spa = MultivariateSaddlePointApprox(multivariate_norm(loc=mean, cov=cov))
    for i in range(3):
        x = mean[i] + np.array([-2.0, -0.3, 0.7, 2.5])
        ref = sps.norm(mean[i], np.sqrt(cov[i, i]))
        assert np.allclose(spa[i].cdf(x=x), ref.cdf(x), atol=1e-8)
        assert np.allclose(spa[i].pdf(x=x), ref.pdf(x), atol=1e-8)
    idx = [2, 0]
    sub = spa[idx]
    assert isinstance(sub, BivariateSaddlePointApprox)
    ref = sps.multivariate_normal(mean[idx], cov[np.ix_(idx, idx)])
    for x in [[1.0, 0.0], [3.5, -0.5], [2.0, 1.5]]:
        x = np.array(x)
        assert np.isclose(sub.cdf(x=x), ref.cdf(x), atol=1e-6)
    assert isinstance(spa[[1]], MultivariateSaddlePointApprox)
    assert not isinstance(spa[[1]], BivariateSaddlePointApprox)
    assert spa[[1]].dim == 1
    assert spa[0:3].dim == 3
    assert isinstance(spa[(0, 1)], BivariateSaddlePointApprox)


def test_marginal_indexing_and_cache():
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=[2, 1, 3]))
    assert spa[0] is spa[0]
    assert spa[1] is spa[-1]
    assert spa[[1, 0]] is spa[(1, 0)]
    first = spa[0]
    spa.clear_cache()
    assert spa[0] is not first
    for item in [2, -3, [0, 2]]:
        with pytest.raises(IndexError):
            spa[item]
    with pytest.raises(IndexError):
        spa[[0, 0]]
    with pytest.raises(IndexError):
        spa[[]]
    with pytest.raises(TypeError):
        spa[0.5]


# TODO: then continue with a bootstrapping application

# TODO: then go to other applications, esaddle?, the empirical saddlepoint approximation
# TODO: then go to other applications, estimating equations,
# TODO: then go to other applications, discrete correction


if __name__ == "__main__":
    if True:
        pytest.main(
            [
                # str(Path(__file__)),
                str(Path(__file__)) + "::test_bvar_spa",
                # "-k",
                # "test_bvar_spa",
                "--durations=10",
                "--tb=auto",
                "--pdb",
                "-v",
                "-s",
                # "-m 'not slow'",
                "-m tofix",
            ]
        )


def test_clear_cache_does_not_evaluate_properties():
    class Approx(UnivariateSaddlePointApprox):
        @property
        def _pdf_normalization(self):
            raise AssertionError("pdf normalization should not be evaluated")

    spa = Approx(norm())
    spa.fit_saddle_point_eqn(num=100)
    spa.clear_cache()
    assert not hasattr(spa, "_x_cache") and not hasattr(spa, "_t_cache")


def test_clear_cache_keeps_user_pdf_normalization():
    spa = UnivariateSaddlePointApprox(norm(), pdf_normalization=2.0)
    spa.fit_saddle_point_eqn(num=100)
    spa.clear_cache()
    assert spa._pdf_normalization == 2.0
    assert np.isclose(spa.pdf(t=0), sps.norm.pdf(0) / 2)
    # Computed normalizations are cleared
    spa = UnivariateSaddlePointApprox(norm())
    spa.pdf(t=0)
    assert spa._pdf_normalization_cache is not None
    spa.clear_cache()
    assert not hasattr(spa, "_pdf_normalization_cache")


def test_fitted_dK_inv_outside_grid_uses_solver():
    spa = UnivariateSaddlePointApprox(gamma(a=3))
    x = np.array([2.0, 40.0, 1e-3])
    expected = spa.cgf.dK_inv(x)
    spa.fit_saddle_point_eqn()
    assert spa._x_cache[0] > x[2] and spa._x_cache[-1] < x[1]
    t = spa._dK_inv(x)
    assert np.allclose(t[1:], expected[1:])
    assert np.isclose(t[0], expected[0], atol=1e-3)
    assert np.isclose(spa._dK_inv(40.0), expected[1])
    assert np.isclose(spa.pdf(x=40.0), sps.gamma(a=3).pdf(40), rtol=0.1)


@pytest.mark.parametrize(
    "cgf,dist",
    [
        pytest.param(norm(loc=1, scale=2), sps.norm(loc=1, scale=2), id="Normal"),
        pytest.param(norm(scale=1e-5), sps.norm(scale=1e-5), id="Normal small scale"),
        pytest.param(exponential(scale=2), sps.expon(scale=2), id="Exponential"),
        pytest.param(gamma(a=3, scale=0.5), sps.gamma(a=3, scale=0.5), id="Gamma"),
    ],
)
def test_normalized_pdf_exact(cgf, dist):
    # The normalized saddle point pdf is exact for these distributions
    spa = UnivariateSaddlePointApprox(cgf)
    x = dist.ppf([0.01, 0.3, 0.5, 0.9])
    assert np.allclose(spa.pdf(x=x), dist.pdf(x), rtol=1e-6)


def test_mvn_logtail_relative_accuracy_in_tails():
    # Equicorrelated: Z_i = sqrt(r) U + sqrt(1 - r) E_i gives a 1-D reference integral
    for d, r in [(2, 0.5), (3, 0.3), (4, 0.6)]:
        S = (1 - r) * np.eye(d) + r
        for scale in [0.0, 1.0, 4.0, 8.0]:
            b = S @ np.full(d, scale)
            integrand = lambda u: (
                sps.norm.pdf(u) * np.prod(sps.norm.sf((b - np.sqrt(r) * u) / np.sqrt(1 - r)))
            )
            ref = quad(integrand, -12, 12 + b.max(), epsabs=0, epsrel=1e-13, limit=500)[0]
            assert np.isclose(np.exp(_mvn_logtail(b, S)), ref, rtol=1e-9, atol=0)


def test_mvn_tail_derivatives():
    c = 0.7
    b = np.array([0.4, -0.3])
    ms = [(), (0,), (1,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1)]
    D = _mvn_tail_derivatives(b, np.array([[1 + c**2, c], [c, 1.0]]), ms)
    ref = _bvn_tail_derivatives(b, c)
    # Without fastnorm, the reference orthant probability is quasi-Monte Carlo
    assert np.isclose(D[()], ref[0], atol=2e-5)
    assert np.allclose([D[R] for R in ms[1:]], ref[1:], atol=1e-12)
    # Finite differences in 3-D, and the log-scale
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 3))
    S = X @ X.T + 0.5 * np.eye(3)
    b = S @ np.array([0.3, 0.1, 0.5])
    ms = [()] + [
        R for o in (1, 2, 3) for R in itertools.combinations_with_replacement(range(3), o)
    ]
    D = _mvn_tail_derivatives(b, S, ms)
    h = 1e-3

    def fd(R, bb):
        if not R:
            return np.exp(_mvn_logtail(bb, S))
        e = np.zeros(3)
        e[R[0]] = h
        return (fd(R[1:], bb + e) - fd(R[1:], bb - e)) / (2 * h)

    for R in ms:
        assert np.isclose(D[R], fd(R, b), atol=1e-6)
    D2 = _mvn_tail_derivatives(b, S, ms, logscale=3.0)
    assert np.allclose([D2[R] * np.exp(-3.0) for R in ms], [D[R] for R in ms], atol=1e-15)


@pytest.mark.parametrize("d", [3, pytest.param(4, marks=pytest.mark.slow)])
def test_mvar_cdf_kolassa_normal_exact(d):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(d, d))
    cov = X @ X.T + 0.5 * np.eye(d)
    mean = rng.normal(size=d)
    spa = MultivariateSaddlePointApprox(multivariate_norm(loc=mean, cov=cov))
    xs = mean + rng.normal(size=(4, d)) * np.sqrt(np.diag(cov)) * 1.5
    ref = [
        sps.multivariate_normal.cdf(
            x, mean, cov, abseps=1e-9, maxpts=10**7, rng=np.random.default_rng(0)
        )
        for x in xs
    ]
    assert np.allclose(spa.cdf(x=xs), ref, atol=1e-8)


def _common_gamma(a):
    # (Z0 + Z1, Z0 + Z2, Z0 + Z3) with independent Zi ~ Gamma(a[i])
    A = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1.0]])
    return MultivariateCumulantGeneratingFunction.from_univariate(*[gamma(a=ai) for ai in a]).ldot(
        A
    )


def _common_gamma_cdf(x, a):
    integrand = lambda z: (
        sps.gamma(a[0]).pdf(z) * np.prod([sps.gamma(ai).cdf(xi - z) for ai, xi in zip(a[1:], x)])
    )
    return quad(integrand, 0, min(x), limit=200, epsabs=1e-13, epsrel=1e-10)[0]


@pytest.mark.parametrize(
    "k,max_err",
    [
        (1, 0.015),
        pytest.param(4, 4e-3, marks=pytest.mark.slow),
        pytest.param(16, 1e-3, marks=pytest.mark.slow),
    ],
)
def test_mvar_cdf_kolassa_gamma_accuracy(k, max_err):
    # Error O(1/n), with n proportional to k
    a = np.array([2.0, 1.0, 3.0, 1.5]) * k
    spa = MultivariateSaddlePointApprox(_common_gamma(a))
    errs = []
    for z in itertools.product([-1.4, 0.5, 1.6], repeat=3):
        t = np.array(z) / np.sqrt(k) / 3
        x = spa.cgf.dK(t)
        if np.isnan(x).any():
            # Outside the domain
            assert np.isnan(spa.cdf(t=t)) and spa.cdf(t=t, fillna=0) == 0
            continue
        errs.append(abs(spa.cdf(t=t) - _common_gamma_cdf(x, a)))
    assert max(errs) < max_err


@pytest.mark.parametrize("k,max_err", [(1, 0.008), pytest.param(16, 5e-4, marks=pytest.mark.slow)])
def test_bvar_cdf_kolassa_accuracy(k, max_err):
    a = np.array([2.0, 1.0, 3.0]) * k
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=a))
    errs = []
    for zs, zt in itertools.product([-1.5, -0.3, 0.4, 1.8], [-1.5, 0.35, 1.7]):
        t = np.array([zs, zt]) / np.sqrt(k) / 2.2
        if t.sum() >= 0.97:
            continue
        x = spa.cgf.dK(t)
        errs.append(abs(spa.cdf(t=t, backend="Kolassa") - _bivariate_gamma_cdf(x, a)))
    assert max(errs) < max_err


def test_mvar_cdf_kolassa_continuous_and_symmetric():
    a = np.array([2.0, 1.0, 3.0, 1.5])
    spa = MultivariateSaddlePointApprox(_common_gamma(a))
    eps = 1e-9
    for base in [[0.2, -0.3, 0.1], [-0.3, -0.2, -0.1]]:
        for idx in [[0], [1], [2], [0, 2]]:
            p, q = np.array(base), np.array(base)
            p[idx], q[idx] = eps, -eps
            assert np.isclose(spa.cdf(t=p), spa.cdf(t=q), atol=1e-7)
    assert np.isclose(spa.cdf(t=np.zeros(3)), spa.cdf(t=np.full(3, 1e-9)), atol=1e-7)
    # Relabeling the variables permutes the result
    perm = [2, 0, 1]
    A = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1.0]])
    cgf = MultivariateCumulantGeneratingFunction.from_univariate(*[gamma(a=ai) for ai in a])
    spa_perm = MultivariateSaddlePointApprox(cgf.ldot(A[perm]))
    for t in [[0.2, -0.3, 0.1], [0.1, 0.2, 0.15]]:
        t = np.array(t)
        assert np.isclose(spa.cdf(t=t), spa_perm.cdf(t=t[perm]), atol=1e-12)
    # P(X <= x) + P(X1 > x1, X2 <= x2, X3 <= x3) = P(X2 <= x2, X3 <= x3)
    x = spa.cgf.dK(np.array([0.2, -0.3, 0.1]))
    D = np.diag([-1.0, 1, 1])
    spa_refl = MultivariateSaddlePointApprox(spa.cgf.ldot(D))
    lhs = spa.cdf(x=x) + spa_refl.cdf(x=D @ x)
    assert np.isclose(lhs, spa[[1, 2]].cdf(x=x[1:], backend="Kolassa"), atol=1e-10)


def test_mvar_cdf_vectorized_and_invalid_backend():
    cov = np.array([[2.0, 0.5, 0.3], [0.5, 1.0, -0.4], [0.3, -0.4, 3.0]])
    spa = MultivariateSaddlePointApprox(multivariate_norm(loc=np.zeros(3), cov=cov))
    xs = np.array([[0.1, 0.2, -0.3], [1.0, -0.5, 2.0]])
    vals = spa.cdf(x=xs)
    assert np.allclose(vals, [spa.cdf(x=x) for x in xs])
    assert np.allclose(spa.cdf(x=pd.DataFrame(xs)).values, vals)
    assert np.isclose(spa[[1]].cdf(x=[0.3]), sps.norm.cdf(0.3))
    with pytest.raises(ValueError):
        spa.cdf(x=xs[0], backend="KL")
    with pytest.raises(ValueError):
        spa.cdf(x=xs[0, :2])


# Booth and Butler (1990), Tables 3 and 4
_BB_U = [155, 68, 28, 25, 190, 82, 92, 196, 164, 68, 82, 36, 195, 92, 185, 61, 62, 71, 207, 185]
_BB_U = np.array(_BB_U + [475, 155, 29, 96, 699.0])
_BB_V = [1546, 505, 280, 410, 1450, 517, 738, 2225, 1660, 505, 680, 145, 224, 733, 1957, 287]
_BB_V = np.array(_BB_V + [473, 473, 1260, 1958, 4375, 1499, 245, 828, 6361.0])
_BB_X = np.array([9.6, 10.4, 13.0, 15.0, 16.6, 17.2, 17.3, 21.8, 24.0, 33.8])


def test_estimating_equation_iid_mean():
    # S(tau) = sum_i (X_i - tau) has root the sample mean
    n, cgf = 5, gamma(a=2)
    total = MultivariateCumulantGeneratingFunction.from_univariate(*[cgf] * n).ldot(np.ones(n))
    ee = EstimatingEquationSaddlePointApprox(lambda tau: total - n * tau)
    ref = UnivariateSaddlePointApproxMean(cgf, n)
    tau = np.array([1.0, 1.7, 2.0, 2.5, 3.5])
    assert np.allclose(ee.cdf(tau), ref.cdf(x=tau), atol=1e-8)
    assert np.allclose(ee.pdf(tau, normalize_pdf=False), ref.pdf(x=tau, normalize_pdf=False))
    with pytest.raises(ValueError):
        ee.pdf(2.0)


def test_estimating_equation_cut():
    # Poisson weights given their sum: the double saddle point approximation equals the
    # single one of the multinomial bootstrap, as T_1 induces a cut, Booth and Butler (1990)
    n = len(_BB_X)
    ee = EstimatingEquationSaddlePointApprox.from_resampling(
        lambda x, t: x - t, _BB_X, poisson(1), given=n
    )
    ref = UnivariateSaddlePointApproxMean(univariate_empirical(_BB_X), n)
    tau = np.array([14.0, 16.0, 17.88, 20.0, 24.0])
    assert np.allclose(ee.cdf(tau), ref.cdf(x=tau), atol=1e-6)
    assert np.allclose(
        ee.pdf(tau, normalize_pdf=False), ref.pdf(x=tau, normalize_pdf=False), rtol=1e-4
    )


@pytest.mark.parametrize("decreasing", [True, False])
def test_estimating_equation_jackknife(decreasing):
    # Delete-10 jackknife of a ratio of means, saddle point column of Table 3
    sign = 1 if decreasing else -1
    ee = EstimatingEquationSaddlePointApprox.from_resampling(
        lambda x, t: sign * (x[:, 1] - t * x[:, 0]),
        np.c_[_BB_U, _BB_V],
        binomial(1, 0.5),
        given=15,
        decreasing=decreasing,
    )
    tau = [7.0, 7.6, 8.0, 8.4, 9.0, 9.2]
    expected = [0.0015, 0.0303, 0.1550, 0.4476, 0.8833, 0.9610]
    assert np.allclose(ee.cdf(tau), expected, atol=3e-4)


def test_estimating_equation_dirichlet():
    # Dirichlet bootstrap of the mean, saddle point column of Table 4, and the exact mean and
    # variance of sum_i D_i x_i
    x, n = _BB_X, len(_BB_X)
    ee = EstimatingEquationSaddlePointApprox.from_resampling(
        lambda x, t: x - t, x, exponential(), given=n, bracket=(x.min(), x.max())
    )
    ee.fit_cdf(num=60)
    q = ee.ppf([0.01, 0.05, 0.95, 0.99]) - x.mean()
    assert np.allclose(q, [-4.07, -3.04, 3.72, 5.73], atol=0.015)
    # The mean is off by about 0.01, a saddle point error of 0.6% of the standard deviation
    assert np.isclose(ee.mean(), x.mean(), atol=0.02)
    assert np.isclose(ee.var(), x.var() / (n + 1), rtol=0.02)
