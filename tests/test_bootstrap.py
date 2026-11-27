import numpy as np
import pytest
import scipy.stats as sps
from scipy.integrate import quad

from spapprox import (
    ConditionalBootstrap,
    MultivariateSaddlePointApprox,
    conditional_bootstrap_bandwidth,
    multivariate_empirical,
    multivariate_sample_mean,
    optimal_tilt,
)
from spapprox.bootstrap import _KERNELS


def _sample(n=20, seed=3):
    # Booth, Hall and Wood (1992), Section 3: (log Y, Y) with Y ~ Exp(1)
    y = np.random.default_rng(seed).exponential(size=n)
    return np.column_stack([np.log(y), y])


@pytest.mark.parametrize(
    "u, t",
    [(0.1, -0.06), (0.5, -0.31), (1.0, -0.60), (1.5, -0.86), (2.0, -1.09), (3.0, -1.51)],
)
def test_optimal_tilt_table1(u, t):
    assert round(optimal_tilt(u), 2) == t
    assert round(optimal_tilt(-u), 2) == -t


def test_optimal_tilt_efficiency():
    t, e = optimal_tilt([0.0, 1.645, 1.96], return_efficiency=True)
    assert np.allclose(t, [0.0, -0.93, -1.07], atol=0.005)
    assert np.allclose(e, [1.0, 2.85, 4.34], atol=0.01)


@pytest.mark.parametrize("kernel", sorted(_KERNELS))
def test_kernel_constants(kernel):
    K, kappa1, kappa2 = _KERNELS[kernel]
    for f, expected in [(K, 1.0), (lambda y: y**2 * K(y) / 2, kappa1), (lambda y: K(y) ** 2, kappa2)]:
        assert np.isclose(quad(f, -10, 10, points=[-1, 1])[0], expected)


def test_bandwidth():
    h1 = conditional_bootstrap_bandwidth(1.645, 0.6, 1000)
    h2 = conditional_bootstrap_bandwidth(1.645, 0.6, 32000)
    assert np.isclose(h2 / h1, 32**-0.2)
    # The bias vanishes at u = 0 and for rho = 0: capped
    h = conditional_bootstrap_bandwidth([0.0, 0.25, 0.5], 0.6, 1000)
    assert np.all(np.isfinite(h)) and np.all(h <= 1)
    assert conditional_bootstrap_bandwidth(1.0, 0.0, 1000) == 1.0
    # Importance resampling has a smaller variance factor, so a smaller bandwidth
    assert conditional_bootstrap_bandwidth(2.0, 0.6, 1000, importance=False) > h1
    with pytest.raises(ValueError):
        conditional_bootstrap_bandwidth(1.0, 1.0, 1000)


def test_statistics():
    x = _sample()
    cb = ConditionalBootstrap(x, rng=0)
    n = len(x)
    assert np.isclose(cb.theta1, x[:, 0].mean())
    assert np.isclose(cb.sigma2, x[:, 1].std() / np.sqrt(n))
    assert np.isclose(cb.rho, np.corrcoef(x.T)[0, 1])
    # Numerical gradients give the same as the default analytic ones
    cb2 = ConditionalBootstrap(x, g1=lambda z: z[..., 0], g2=lambda z: 2 * z[..., 1], rng=0)
    assert np.isclose(cb2.sigma1, cb.sigma1) and np.isclose(cb2.rho, cb.rho)
    with pytest.raises(ValueError):
        ConditionalBootstrap(x[:, :1])
    with pytest.raises(ValueError):
        ConditionalBootstrap(np.column_stack([x[:, 0], x[:, 0]]))


@pytest.mark.parametrize("importance", [False, True])
def test_nonpivotal_matches_saddle_point(importance):
    # The saddle point approximation of #56 is the small-window limit of the nonpivotal estimate
    x = _sample()
    cb = ConditionalBootstrap(x, rng=1)
    sp = MultivariateSaddlePointApprox(multivariate_sample_mean(multivariate_empirical(x), len(x)))
    xs = cb.theta1 + cb.sigma1 * np.array([-1.0, 0.0, 0.5])
    expected = sp.conditional_cdf(xs, [cb.theta2], i=0)
    est = cb.cdf(xs, pivotal=False, B=50000, h=0.05, importance=importance)
    assert np.allclose(est, expected, atol=0.02)


def test_importance_reduces_variance():
    cb = ConditionalBootstrap(_sample(), rng=2)
    uniform = [cb.cdf(-1.645, B=1000, h=0.3, importance=False) for _ in range(100)]
    tilted = [cb.cdf(-1.645, B=1000, h=0.3) for _ in range(100)]
    assert abs(np.mean(uniform) - np.mean(tilted)) < 0.01
    assert np.var(uniform) > 2 * np.var(tilted)


def test_pivotal_against_exact():
    # Booth, Hall and Wood (1992), Section 3: given the mean of Y, (Y_i) / mean(Y) is uniform on
    # the simplex, so the pivot given W = 0 is that of n * Dirichlet(1) data
    n = 50
    d = n * np.random.default_rng(1).dirichlet(np.ones(n), size=100000)
    lg = np.log(d)
    t1 = lg.mean(axis=1)
    rho = ((lg - t1[:, None]) * (d - 1)).mean(axis=1) / (lg.std(axis=1) * d.std(axis=1))
    u = (t1 + np.euler_gamma) / (lg.std(axis=1) / np.sqrt(n) * np.sqrt(1 - rho**2))
    levels = np.array([0.05, 0.5, 0.95])
    uq = np.quantile(u, levels)
    est = np.mean(
        [ConditionalBootstrap(_sample(n, seed), rng=seed).cdf(uq) for seed in range(10)], axis=0
    )
    assert np.allclose(est, levels, atol=0.06)
    # Better than the normal approximation in the median, which is off by 0.07 here
    assert abs(est[1] - 0.5) < abs(sps.norm.cdf(uq[1]) - 0.5)


@pytest.mark.parametrize("pivotal", [True, False])
def test_ppf_and_ci(pivotal):
    cb = ConditionalBootstrap(_sample(), rng=3)
    kw = dict(B=100000, h=0.1, importance=False, pivotal=pivotal)
    q = cb.ppf(0.1, **kw)
    assert np.isclose(cb.cdf(q, **kw), 0.1, atol=0.01)
    lo, hi = cb.ci(level=0.9, pivotal=pivotal)
    assert lo < cb.theta1 < hi
    assert cb.ci(level=0.95, sided="lower", pivotal=pivotal)[1] == np.inf
    assert cb.ci(level=0.95, sided="upper", pivotal=pivotal)[0] == -np.inf
    with pytest.raises(ValueError):
        cb.ppf(1.0)
    with pytest.raises(ValueError):
        cb.ci(sided="both")
