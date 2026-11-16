import itertools
from pathlib import Path

import numpy as np
import pytest
import scipy.stats as sps
from scipy.integrate import quad

from spapprox import (
    BivariateSaddlePointApprox,
    MultivariateCumulantGeneratingFunction,
    MultivariateSaddlePointApprox,
    # poisson,
    # binomial,
    UnivariateCumulantGeneratingFunction,
    UnivariateSaddlePointApprox,
    bivariate_gamma,
    chi2,
    exponential,
    gamma,
    laplace,
    multivariate_norm,
    norm,
)


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
            marks=pytest.mark.tofix,
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


# TODO: continue here with the conditional Skovgaard approximation


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
        cdf = spa.cdf(t=t)
        assert np.isclose(cdf, u1.cdf(t=t[0]) * u2.cdf(t=t[1]), atol=1e-6)
        assert np.isclose(cdf, sps.gamma(3).cdf(x[0]) * sps.gamma(5).cdf(x[1]), atol=1e-3)


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
        cdf = spa.cdf(t=t)
        assert np.isclose(cdf, u1.cdf(t=t[0]) * u2.cdf(t=t[1]), atol=1e-6)
        assert np.isclose(cdf, sps.gamma(3).cdf(x[0]) * sps.gamma(5).cdf(x[1]), atol=1e-3)


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
        assert np.isclose(spa.cdf(t=t), u1.cdf(t=t[0]) * u2.cdf(t=t[1]), atol=1e-6)
    cov = np.array([[1, 0.6], [0.6, 2.0]])
    spa = BivariateSaddlePointApprox(multivariate_norm(loc=np.zeros(2), cov=cov))
    for t in [[0.3, 1e-6], [1e-7, 1e-7], [1e-6, -0.4]]:
        t = np.array(t)
        expected = sps.multivariate_normal(np.zeros(2), cov).cdf(spa.cgf.dK(t))
        assert np.isclose(spa.cdf(t=t), expected, atol=1e-6)


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
    assert np.isclose(spa.cdf(x=x), expected, rtol=5e-3, atol=6e-4)


@pytest.mark.parametrize("k,max_err", [(1, 0.04), (16, 0.003)])
def test_bvar_cdf_dependent_accuracy(k, max_err):
    # Wang's order-of-integration rule; without it the errors are 0.245 and 0.033
    a = np.array([2.0, 1.0, 3.0]) * k
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=a))
    errs = []
    for zs, zt in itertools.product([-1.5, -0.8, -0.3, 0.4, 1.0, 1.8], [-1.5, -0.7, 0.35, 1.7]):
        t = np.array([zs, zt]) / np.sqrt(k) / 2.2
        if t.sum() >= 0.97:
            continue
        x = spa.cgf.dK(t)
        errs.append(abs(spa.cdf(t=t) - _bivariate_gamma_cdf(x, a)))
    assert max(errs) < max_err


def test_bvar_cdf_invariant_to_relabeling():
    a = np.array([2.0, 1.0, 3.0])
    spa = BivariateSaddlePointApprox(bivariate_gamma(a=a))
    spa_swapped = BivariateSaddlePointApprox(bivariate_gamma(a=a[[0, 2, 1]]))
    for t in [[-0.6, 0.3], [0.4, -0.7], [0.2, 0.3], [-0.5, -0.2]]:
        t = np.array(t)
        assert np.isclose(spa.cdf(t=t), spa_swapped.cdf(t=t[::-1]), atol=1e-10)


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
