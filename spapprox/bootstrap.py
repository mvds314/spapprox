r"""
Monte Carlo conditional bootstrap, following Booth, Hall and Wood (1992).

The target is the conditional distribution of an estimator :math:`\hat\theta_1` given the
value of another estimator :math:`\hat\theta_2`, with :math:`\hat\theta_k=g_k(\bar X)` smooth
functions of a vector mean. The bootstrap distribution is discrete, so conditioning on
:math:`\hat\theta_2^*=y` is replaced by kernel weighting of the resamples on how close
:math:`\hat\theta_2^*` is to :math:`y`.

Two versions are provided by :class:`ConditionalBootstrap`:

- nonpivotal, :math:`P^*(\hat\theta_1^*\leq x\mid\hat\theta_2^*=y)`, error
  :math:`O_p(n^{-1/2})`. For a vector mean it is approximated without resampling by the double
  saddle point approximation, see :func:`~spapprox.multivariate_empirical` and
  :meth:`~spapprox.MultivariateSaddlePointApprox.conditional_cdf`.
- pivotal, the distribution of the Studentized and decorrelated
  :math:`U=(1-\hat\rho^2)^{-1/2}(V-\hat\rho W)` given :math:`W=w`, error :math:`O_p(n^{-1})`.

Importance resampling with the exponential tilt of Booth, Hall and Wood, Section 2.5, see
:func:`optimal_tilt`, reduces the Monte Carlo variance in the tails, and
:func:`conditional_bootstrap_bandwidth` is their plug-in bandwidth.

References
----------
[1] Booth, Hall and Wood (1992) - Bootstrap estimation of conditional distributions
"""

from functools import lru_cache

import numpy as np
import scipy.stats as sps
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, ndtr

_KERNELS = {
    # name: (kernel, kappa_1 = int y^2 K / 2, kappa_2 = int K^2)
    "gaussian": (lambda y: np.exp(-(y**2) / 2) / np.sqrt(2 * np.pi), 0.5, 0.5 / np.sqrt(np.pi)),
    "epanechnikov": (lambda y: np.where(np.abs(y) <= 1, 0.75 * (1 - y**2), 0.0), 0.1, 0.6),
    "uniform": (lambda y: np.where(np.abs(y) <= 1, 0.5, 0.0), 1 / 6, 0.5),
}


def _kernel(kernel):
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {sorted(_KERNELS)}")
    return _KERNELS[kernel]


def _b(t, u):
    Phi = ndtr(u)
    return ((1 - 2 * Phi) * ndtr(u - t) + Phi**2) * np.exp(t**2)


@lru_cache(maxsize=4096)
def _tilt_abs(a):
    """Minimizer and minimum of b(t, a) for a > 0."""
    res = minimize_scalar(
        lambda s: _b(s, a), bounds=(-a - 3, 0), method="bounded", options={"xatol": 1e-9}
    )
    return float(res.x), float(res.fun)


def optimal_tilt(u, return_efficiency=False):
    r"""
    Optimal exponential tilt for importance resampling, Booth, Hall and Wood (1992), (2.11).

    The tilt :math:`t(u)` minimizes

    .. math::
        b(t,u) = \left[\{1-2\Phi(u)\}\Phi(u-t) + \Phi(u)^2\right]e^{t^2},

    the asymptotic variance factor of the importance resampled estimate of
    :math:`P(U\leq u\mid W=0)`. It is odd, :math:`t(-u)=-t(u)`, and negative for :math:`u>0`.
    The efficiency relative to uniform resampling is
    :math:`e(u)=\Phi(u)\{1-\Phi(u)\}/\min_t b(t,u)`, (2.12).

    Parameters
    ----------
    u : array_like
        Points at which the conditional cdf of the pivot is estimated.
    return_efficiency : bool, optional
        Also return :math:`e(u)`.

    Returns
    -------
    t : float or numpy.ndarray
        The optimal tilt, same shape as ``u``.
    e : float or numpy.ndarray
        The efficiency, only if ``return_efficiency``.

    Examples
    --------
    >>> import spapprox as spa
    >>> round(float(spa.optimal_tilt(1.0)), 2)
    -0.6

    References
    ----------
    [1] Booth, Hall and Wood (1992) - Bootstrap estimation of conditional distributions,
        Section 2.5 and Table 1
    """
    u = np.asanyarray(u, dtype=float)
    t = np.zeros(u.shape)
    bmin = np.full(u.shape, 0.25)
    for idx, v in np.ndenumerate(u):
        if v != 0:
            tt, bmin[idx] = _tilt_abs(abs(float(v)))
            t[idx] = tt if v > 0 else -tt
    if u.ndim == 0:
        t, bmin = float(t), float(bmin)
    if not return_efficiency:
        return t
    Phi = sps.norm.cdf(u)
    return t, Phi * (1 - Phi) / bmin


def _beta(u, w, importance):
    if importance:
        _, e = optimal_tilt(u, return_efficiency=True)
        Phi = sps.norm.cdf(u)
        return Phi * (1 - Phi) / e * np.exp(-np.asanyarray(w) ** 2 / 2)
    Phi = sps.norm.cdf(u)
    return Phi * (1 - Phi)


def conditional_bootstrap_bandwidth(u, rho, B, w=0.0, kernel="gaussian", importance=True):
    r"""
    Plug-in bandwidth of the kernel-smoothed conditional bootstrap, Booth, Hall and Wood
    (1992), Section 2.6 and (5.5).

    Minimizing the asymptotic mean squared error
    :math:`(Bh)^{-1}\kappa_2\beta/\phi(w) + h^4\kappa_1^2\psi(u,w)^2` gives

    .. math::
        h = \left[\frac{\kappa_2\beta}{4B\phi(w)}\right]^{1/5}
        \left\{\kappa_1|\psi(u,w)|\right\}^{-2/5},\qquad
        \psi(u,w) = \left\{\frac{2\rho w}{(1-\rho^2)^{1/2}}
        - \frac{\rho^2u}{1-\rho^2}\right\}\phi(u),

    with :math:`\kappa_1=\tfrac12\int y^2K(y)dy`, :math:`\kappa_2=\int K^2`, and :math:`\beta`
    the minimized variance factor (2.9) with importance resampling, or
    :math:`\Phi(u)\{1-\Phi(u)\}` with uniform resampling. The bandwidth is on the scale of
    :math:`W`, the Studentized :math:`\hat\theta_2`; multiply it by :math:`\hat\sigma_2` for
    the scale of :math:`\hat\theta_2`.

    The bias :math:`\psi` vanishes at :math:`u=0` for :math:`w=0`, and for :math:`\rho=0`, where
    the rule breaks down. Therefore :math:`|\psi|` is bounded below by its value at
    :math:`u=1/2, w=0`, and :math:`h` is capped at one.

    Parameters
    ----------
    u : array_like
        Points at which the conditional cdf of the pivot is estimated.
    rho : float
        The (estimated) correlation of :math:`\hat\theta_1` and :math:`\hat\theta_2`.
    B : int
        Number of resamples.
    w : float, optional
        Conditioning value of :math:`W`. Default is 0.
    kernel : str, optional
        ``"gaussian"`` (default), ``"epanechnikov"`` or ``"uniform"``.
    importance : bool, optional
        Whether importance resampling is used, default ``True``.

    Returns
    -------
    float or numpy.ndarray
        The bandwidth, same shape as ``u``.

    References
    ----------
    [1] Booth, Hall and Wood (1992) - Bootstrap estimation of conditional distributions,
        (2.13), (2.14) and (5.5)
    """
    _, kappa1, kappa2 = _kernel(kernel)
    u = np.asanyarray(u, dtype=float)
    if not abs(rho) < 1:
        raise ValueError("rho must lie in (-1, 1)")
    c = rho**2 / (1 - rho**2)
    psi = (2 * rho / np.sqrt(1 - rho**2) * w - c * u) * sps.norm.pdf(u)
    psi = np.maximum(np.abs(psi), c * 0.5 * sps.norm.pdf(0.5))
    beta = _beta(u, w, importance)
    with np.errstate(divide="ignore"):
        h = (kappa2 * beta / (4 * B * sps.norm.pdf(w))) ** 0.2 * (kappa1 * psi) ** -0.4
    h = np.minimum(h, 1.0)
    return float(h) if h.ndim == 0 else h


def _numerical_gradient(g):
    def grad(z):
        z = np.asarray(z, dtype=float)
        step = 1e-6 * np.maximum(1.0, np.abs(z))
        out = np.empty(z.shape)
        for k in range(z.shape[-1]):
            zp, zm = z.copy(), z.copy()
            zp[..., k] += step[..., k]
            zm[..., k] -= step[..., k]
            out[..., k] = (g(zp) - g(zm)) / (2 * step[..., k])
        return out

    return grad


class ConditionalBootstrap:
    r"""
    Kernel-smoothed Monte Carlo conditional bootstrap of Booth, Hall and Wood (1992).

    For data :math:`X_1,\ldots,X_n\in\mathbb{R}^d` and estimators
    :math:`\hat\theta_k=g_k(\bar X)`, :math:`k=1,2`, the delta method gives the influence values
    :math:`D_{kj}=n^{-1}(X_j-\bar X)^T\nabla g_k(\bar X)`, the standard errors
    :math:`\hat\sigma_k^2=\sum_jD_{kj}^2`, and the correlation
    :math:`\hat\rho=\sum_j\epsilon_{1j}\epsilon_{2j}` with :math:`\epsilon_{kj}=D_{kj}/\hat\sigma_k`.

    On every resample, :math:`\hat\sigma_k^*` and :math:`\hat\rho^*` are recomputed in the same
    way, which gives :math:`V^*=(\hat\theta_1^*-\hat\theta_1)/\hat\sigma_1^*`,
    :math:`W^*=(\hat\theta_2^*-\hat\theta_2)/\hat\sigma_2^*` and the pivot

    .. math::
        U^* = (1-\hat\rho^{*2})^{-1/2}(V^*-\hat\rho^*W^*).

    The pivotal estimate of :math:`P(U\leq u\mid W=w)` is, (2.7),

    .. math::
        \hat p(u, w) = \frac{\sum_b I(U_b^*\leq u)K\{(w-W_b^*)/h\}\omega_b}
        {\sum_b K\{(w-W_b^*)/h\}\omega_b},

    with :math:`\omega_b=\prod_j(n\pi_j)^{-M_{bj}}` the importance weights of a resample that
    contains :math:`X_j` :math:`M_{bj}` times, drawn with probabilities :math:`\pi_j`. The
    nonpivotal estimate of :math:`P(\hat\theta_1\leq x\mid\hat\theta_2=y)`, (2.3), replaces
    :math:`U^*\leq u` by :math:`\hat\theta_1^*\leq x` and the kernel argument by
    :math:`(y-\hat\theta_2^*)/(h\hat\sigma_2)`.

    The pivotal version estimates the conditional distribution with error
    :math:`O_p\{n^{-1}+h^2+(Bh)^{-1/2}\}`, which is :math:`O_p(n^{-1})` for :math:`h\asymp
    n^{-1/2}` and :math:`B\asymp n^{5/2}`; the nonpivotal version has error
    :math:`O_p(n^{-1/2})` whatever :math:`h` and :math:`B`, since its limit depends on the
    unknown :math:`\rho`. Pivoting requires that :math:`\hat\sigma_k^*` and :math:`\hat\rho^*`
    are stable, which is not the case for, e.g., ratios with a denominator near zero.

    Parameters
    ----------
    x : array_like
        Data of shape ``(n, d)``, one observation per row.
    g1, g2 : callable, optional
        The estimators as functions of the mean, mapping arrays of shape ``(..., d)`` to
        ``(...)``. Default: the means of the first and the second column.
    grad1, grad2 : callable, optional
        Their gradients, mapping ``(..., d)`` to ``(..., d)``. Default: central differences
        (exact for the default ``g1`` and ``g2``).
    kernel : str, optional
        ``"gaussian"`` (default), ``"epanechnikov"`` or ``"uniform"``. The uniform kernel is the
        window :math:`|w-W^*|\leq h` of (2.2).
    rng : numpy.random.Generator or int, optional
        Random number generator or seed.

    Examples
    --------
    >>> import numpy as np
    >>> import spapprox as spa
    >>> y = np.random.default_rng(0).exponential(size=30)
    >>> cb = spa.ConditionalBootstrap(np.column_stack([np.log(y), y]), rng=1)
    >>> lo, hi = cb.ci(level=0.9)
    >>> bool(lo < cb.theta1 < hi)
    True

    References
    ----------
    [1] Booth, Hall and Wood (1992) - Bootstrap estimation of conditional distributions
    """

    def __init__(self, x, g1=None, g2=None, grad1=None, grad2=None, kernel="gaussian", rng=None):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1:
            raise ValueError("x must be a 2-d array with at least three rows")
        if (g1 is None or g2 is None) and x.shape[1] < 2:
            raise ValueError("the default estimators need at least two columns")
        self.x = x
        self.n, self.d = x.shape
        self._kernel, self._kappa1, self._kappa2 = _kernel(kernel)
        self.kernel = kernel
        self._g = []
        self._grad = []
        for k, (g, grad) in enumerate([(g1, grad1), (g2, grad2)]):
            if g is None:
                e = np.eye(self.d)[k]
                self._g.append(lambda z, k=k: np.asarray(z)[..., k])
                self._grad.append(lambda z, e=e: np.broadcast_to(e, np.shape(z)))
            else:
                self._g.append(g)
                self._grad.append(_numerical_gradient(g) if grad is None else grad)
        self._rng = np.random.default_rng(rng)
        xbar = x.mean(axis=0)
        self.theta1, self.theta2 = (float(g(xbar)) for g in self._g)
        D = np.stack([(x - xbar) @ np.asarray(gr(xbar), dtype=float) / self.n for gr in self._grad])
        sigma = np.sqrt((D**2).sum(axis=1))
        if np.any(sigma == 0):
            raise ValueError("an estimator has zero estimated variance")
        self.sigma1, self.sigma2 = sigma
        self.influence = D / sigma[:, None]
        self.rho = float(self.influence[0] @ self.influence[1])
        if not abs(self.rho) < 1:
            raise ValueError("the estimators are perfectly correlated")

    def _standardize(self, x, y):
        w = (y - self.theta2) / self.sigma2
        u = ((x - self.theta1) / self.sigma1 - self.rho * w) / np.sqrt(1 - self.rho**2)
        return u, w

    def _log_pi(self, u, w):
        a1 = optimal_tilt(u) / np.sqrt(1 - self.rho**2)
        a2 = -a1 * self.rho - w
        eta = -(a1 * self.influence[0] + a2 * self.influence[1])
        return eta - logsumexp(eta)

    def _resample(self, B, log_pi=None):
        """Statistics of ``B`` resamples: theta1*, theta2*, U*, W* and log importance weights."""
        n, x = self.n, self.x
        pi = None if log_pi is None else np.exp(log_pi)
        chunk = max(1, 2_000_000 // n)
        out = []
        for start in range(0, B, chunk):
            M = self._rng.multinomial(n, np.full(n, 1 / n) if pi is None else pi, min(chunk, B - start))
            xb = M @ x / n
            t1, t2 = (np.asarray(g(xb), dtype=float) for g in self._g)
            z1, z2 = (np.asarray(gr(xb), dtype=float) @ x.T for gr in self._grad)
            m1, m2 = (M * z1).sum(axis=1) / n, (M * z2).sum(axis=1) / n
            c11 = (M * z1**2).sum(axis=1) / n - m1**2
            c22 = (M * z2**2).sum(axis=1) / n - m2**2
            c12 = (M * z1 * z2).sum(axis=1) / n - m1 * m2
            with np.errstate(divide="ignore", invalid="ignore"):
                V = (t1 - self.theta1) / np.sqrt(c11 / n)
                W = (t2 - self.theta2) / np.sqrt(c22 / n)
                r = c12 / np.sqrt(c11 * c22)
                U = (V - r * W) / np.sqrt(1 - r**2)
            logw = np.zeros(len(M)) if log_pi is None else -M @ (np.log(n) + log_pi)
            out.append(np.stack([t1, t2, U, W, logw]))
        return np.concatenate(out, axis=1)

    def _setup(self, u, given, B, h, importance, pivotal):
        if given is None:
            given = 0.0 if pivotal else self.theta2
        B = self.default_B() if B is None else int(B)
        if B < 1:
            raise ValueError("B must be positive")
        if pivotal:
            us, ws = u, given
        else:
            us, ws = self._standardize(u, given)
        if h is None:
            h = conditional_bootstrap_bandwidth(
                us, self.rho, B, w=ws, kernel=self.kernel, importance=importance
            )
        log_pi = self._log_pi(us, ws) if importance else None
        t1, t2, U, W, logw = self._resample(B, log_pi)
        if pivotal:
            stat, arg = U, (given - W) / h
        else:
            stat, arg = t1, (given - t2) / (h * self.sigma2)
        weights = self._kernel(arg) * np.exp(logw - logw.max())
        ok = np.isfinite(stat) & np.isfinite(weights)
        return stat[ok], weights[ok]

    def default_B(self):
        r"""
        Default number of resamples, :math:`\lceil n^{5/2}\rceil`, at least 1000 and at most
        :math:`10^6`. With the plug-in bandwidth this gives :math:`h^4+(Bh)^{-1}=O(n^{-2})`,
        and so error :math:`O_p(n^{-1})` of the pivotal estimate.
        """
        return int(min(max(np.ceil(self.n**2.5), 1000), 10**6))

    def cdf(self, u, given=None, B=None, h=None, importance=True, pivotal=True):
        r"""
        Conditional bootstrap estimate of the cdf.

        Parameters
        ----------
        u : array_like
            With ``pivotal=True``, values :math:`u` of the pivot :math:`U`; otherwise values
            :math:`x` of :math:`\hat\theta_1`.
        given : float, optional
            With ``pivotal=True``, the value :math:`w` of :math:`W`, default 0, i.e.,
            conditional on the observed :math:`\hat\theta_2`; otherwise the value :math:`y`
            of :math:`\hat\theta_2`, default the observed one.
        B : int, optional
            Number of resamples, default :meth:`default_B`. Every value of ``u`` has its own
            resamples.
        h : float, optional
            Bandwidth on the scale of :math:`W`, i.e., in units of :math:`\hat\sigma_2`.
            Default: :func:`conditional_bootstrap_bandwidth` at the standardized ``u``.
            Booth, Hall and Wood give no bandwidth rule for the nonpivotal estimate; the
            pivotal rule is used.
        importance : bool, optional
            Importance resampling with the tilt of :func:`optimal_tilt`, default ``True``.
        pivotal : bool, optional
            Pivotal (default) or nonpivotal estimate.

        Returns
        -------
        float or numpy.ndarray
            :math:`\hat p(u, w)` or :math:`\hat q(x\mid y)`, same shape as ``u``.
        """
        u = np.asanyarray(u, dtype=float)
        res = np.empty(u.shape)
        for idx, v in np.ndenumerate(u):
            stat, weights = self._setup(v, given, B, h, importance, pivotal)
            total = weights.sum()
            res[idx] = weights[stat <= v].sum() / total if total > 0 else np.nan
        return float(res) if res.ndim == 0 else res

    def ppf(self, q, given=None, B=None, h=None, importance=True, pivotal=True):
        r"""
        Conditional bootstrap estimate of the quantile function,
        :math:`\hat u_\alpha=\inf\{u:\hat p(u,w)\geq\alpha\}`, Booth, Hall and Wood (1992),
        Section 2.7.

        The tilt and the bandwidth are set at :math:`u=\Phi^{-1}(\alpha)` (on the standardized
        scale for the nonpivotal estimate). Then :math:`\hat p` is a weighted empirical cdf
        and the quantile a weighted quantile of the resamples. The parameters are as for
        :meth:`cdf`.
        """
        q = np.asanyarray(q, dtype=float)
        if np.any((q <= 0) | (q >= 1)):
            raise ValueError("q must lie in (0, 1)")
        res = np.empty(q.shape)
        for idx, a in np.ndenumerate(q):
            u0 = sps.norm.ppf(a)
            if not pivotal:
                # x at which the standardized u equals u0, given y
                y = self.theta2 if given is None else given
                w = (y - self.theta2) / self.sigma2
                u0 = self.theta1 + self.sigma1 * (np.sqrt(1 - self.rho**2) * u0 + self.rho * w)
            stat, weights = self._setup(u0, given, B, h, importance, pivotal)
            order = np.argsort(stat)
            cum = np.cumsum(weights[order])
            if cum[-1] <= 0:
                res[idx] = np.nan
                continue
            j = min(np.searchsorted(cum, a * cum[-1], side="left"), len(cum) - 1)
            res[idx] = stat[order][j]
        return float(res) if res.ndim == 0 else res

    def ci(self, level=0.9, sided="two", B=None, h=None, importance=True, pivotal=True):
        r"""
        Confidence interval for :math:`\theta_1`, conditional on the observed
        :math:`\hat\theta_2`, Booth, Hall and Wood (1992), Section 2.7.

        Pivotal, two-sided and equal-tailed:

        .. math::
            \left[\hat\theta_1 - (1-\hat\rho^2)^{1/2}\hat\sigma_1\hat u_{(1+\alpha)/2},\;
            \hat\theta_1 - (1-\hat\rho^2)^{1/2}\hat\sigma_1\hat u_{(1-\alpha)/2}\right],

        with :math:`\alpha` the level. Nonpivotal, the basic bootstrap interval
        :math:`[2\hat\theta_1-\hat x_{(1+\alpha)/2}, 2\hat\theta_1-\hat x_{(1-\alpha)/2}]`, with
        :math:`\hat x` the conditional quantiles of :math:`\hat\theta_1^*`.

        Parameters
        ----------
        level : float, optional
            Coverage level, default 0.9.
        sided : str, optional
            ``"two"`` (default), ``"lower"`` for :math:`(\text{lower}, \infty)`, or ``"upper"``
            for :math:`(-\infty, \text{upper})`.
        B, h, importance, pivotal
            As for :meth:`cdf`.

        Returns
        -------
        tuple of float
            The lower and upper limits.
        """
        if not 0 < level < 1:
            raise ValueError("level must lie in (0, 1)")
        if sided not in ("two", "lower", "upper"):
            raise ValueError("sided must be 'two', 'lower' or 'upper'")
        kw = dict(B=B, h=h, importance=importance, pivotal=pivotal)

        def limit(a):
            if pivotal:
                return self.theta1 - np.sqrt(1 - self.rho**2) * self.sigma1 * self.ppf(a, **kw)
            return 2 * self.theta1 - self.ppf(a, **kw)

        if sided == "lower":
            return float(limit(level)), np.inf
        if sided == "upper":
            return -np.inf, float(limit(1 - level))
        return float(limit((1 + level) / 2)), float(limit((1 - level) / 2))
