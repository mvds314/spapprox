r"""
Normal orthant probabilities, their derivatives, and the multivariate tail
approximation of Kolassa (2003) that is built on them.

The orthant probabilities are evaluated in log-space with relative accuracy, also
far in the tails. That matters, as Kolassa's approximation multiplies them by
:math:`\exp(\hat\tau^T\Sigma\hat\tau/2)`, which can be huge. The quasi-Monte Carlo
method of :func:`scipy.stats.multivariate_normal.cdf` only controls the absolute
error, which is useless there.

References
----------
[1] Genz (1992) - Numerical computation of multivariate normal probabilities

[2] Kolassa (2003) - Multivariate saddlepoint tail probability approximations
"""

import itertools
import math

import numpy as np
import scipy.stats as sps
from scipy.special import logsumexp

# Total number of quadrature nodes allowed for a single orthant probability.
_max_nodes = 200_000
# Gauss-Legendre nodes per dimension, by the number of integration dimensions d-1.
_nodes_per_dim = {1: 32, 2: 24, 3: 20}
# The integration interval of every variable is truncated where the integrand has
# dropped by a factor of exp(-_cut) relative to its value at the lower limit.
_cut = 50.0


def _gauss_legendre(m):
    g, w = np.polynomial.legendre.leggauss(m)
    return (g + 1) / 2, np.log(w / 2)


def _mvn_logtail(b, S):
    r"""
    The logarithm of the orthant probability :math:`P(\mathbf{Z}\geq\mathbf{b})` with
    :math:`\mathbf{Z}\sim N(0, S)`, in any dimension.

    With the Cholesky factorization :math:`S=LL^T` and :math:`\mathbf{Z}=L\mathbf{Y}`,
    the probability is separated in the variables (see [1]):

    .. math::
        P(\mathbf{Z}\geq\mathbf{b}) = \int_{a_1}^\infty\phi(y_1)\cdots
        \int_{a_{d-1}(y_{1:d-2})}^\infty\phi(y_{d-1})
        \bar\Phi\left(a_d(y_{1:d-1})\right)dy_{d-1}\cdots dy_1,\quad
        a_i = \frac{b_i - \sum_{k<i} L_{ik}y_k}{L_{ii}}.

    The innermost integral is in closed form, and the other ones are evaluated by a
    tensor product of Gauss-Legendre rules, in log-space. The integral over
    :math:`y_i` is truncated to :math:`[a_i, a_i+s_i]`, where :math:`s_i` solves
    :math:`a_is_i+s_i^2/2 = c`, i.e., where :math:`\phi(y_i)` has dropped by a factor
    :math:`e^{-c}` relative to its value at :math:`a_i`. The integrands are analytic
    on these intervals, so the rules converge exponentially fast. The variables are
    ordered by decreasing :math:`b_i/\sqrt{S_{ii}}`.

    The truncation is accurate if the probability mass is concentrated near the
    corner :math:`\mathbf{b}`, i.e., if :math:`S^{-1}\mathbf{b}\geq 0`, which is the
    case for all orthant probabilities in Kolassa's approximation. Then, the result
    has a relative error close to machine precision for :math:`d\leq 4`. For larger
    :math:`d`, the number of nodes per dimension is reduced to keep the total below
    ``_max_nodes``, at the expense of accuracy.
    """
    b = np.asarray(b, dtype=float)
    d = len(b)
    if d == 0:
        return 0.0
    S = np.asarray(S, dtype=float).reshape(d, d)
    if d == 1:
        return sps.norm.logsf(b[0] / np.sqrt(S[0, 0]))
    order = np.argsort(-b / np.sqrt(np.diag(S)), kind="stable")
    b, S = b[order], S[np.ix_(order, order)]
    L = np.linalg.cholesky(S)
    m = _nodes_per_dim.get(d - 1, max(4, int(_max_nodes ** (1 / (d - 1)))))
    g, lgw = _gauss_legendre(m)
    y = np.zeros((1, 0))
    acc = np.zeros(1)
    for i in range(d - 1):
        a = (b[i] - y @ L[i, :i]) / L[i, i]
        s = -a + np.sqrt(a**2 + 2 * _cut)
        yi = a[:, None] + s[:, None] * g[None, :]
        acc = (acc[:, None] + np.log(s)[:, None] + lgw[None, :] + sps.norm.logpdf(yi)).ravel()
        y = np.hstack([np.repeat(y, m, axis=0), yi.reshape(-1, 1)])
    a = (b[-1] - y @ L[-1, :-1]) / L[-1, -1]
    return logsumexp(acc + sps.norm.logsf(a))


def _mvn_tail_derivatives(b, S, multisets, logscale=0.0):
    r"""
    Partial derivatives of the orthant probability
    :math:`T(\mathbf{b})=P(\mathbf{Z}\geq\mathbf{b})`, :math:`\mathbf{Z}\sim N(0, S)`,
    with respect to :math:`\mathbf{b}`, multiplied by :math:`e^{c}` with :math:`c` the
    ``logscale``.

    Every element of ``multisets`` is a tuple of indices; the empty tuple denotes
    :math:`T` itself. The result is a dict mapping the sorted tuples to the scaled
    derivatives. The scale is applied in log-space, so that it may be large while
    :math:`T` is tiny.

    Differentiating with respect to :math:`b_j` conditions on :math:`Z_j=b_j`
    (Kolassa (2003) eq. (8)):

    .. math::
        \partial_j T(\mathbf{b}) = -\phi(b_j; S_{jj})\,
        T_{-j}\left(\mathbf{b}_{-j} - \boldsymbol\beta b_j\right),\quad
        \boldsymbol\beta = S_{-j,j}/S_{jj},

    where :math:`T_{-j}` is the orthant probability of the conditional distribution,
    with covariance :math:`S_{-j,-j}-S_{-j,j}S_{j,-j}/S_{jj}`. Higher derivatives
    follow by the product and chain rules, and recursively applying the above to
    :math:`T_{-j}`. Repeated indices differentiate the density :math:`\phi` and,
    through the argument :math:`\mathbf{b}_{-j}-\boldsymbol\beta b_j`, the
    conditional probability. Derivatives up to order 3 are supported.
    """
    b = np.asarray(b, dtype=float)
    S = np.asarray(S, dtype=float)
    logtails, derivs = {}, {}

    def deriv(labels, y, Sc, R, ls):
        # exp(ls) times the derivative R of the orthant probability of the variables
        # with the original indices `labels`, at y, with covariance Sc
        key = (labels, R)
        if key in derivs:
            return derivs[key]
        if not R:
            if labels not in logtails:
                logtails[labels] = _mvn_logtail(y, Sc)
            val = np.exp(ls + logtails[labels])
        else:
            j, rest = R[0], R[1:]
            m = rest.count(j)
            keep = [k for k in range(len(y)) if k != j]
            pos = {k: p for p, k in enumerate(keep)}
            others = [pos[k] for k in rest if k != j]
            s = Sc[j, j]
            beta = Sc[keep, j] / s
            Sn = Sc[np.ix_(keep, keep)] - np.outer(Sc[keep, j], Sc[j, keep]) / s
            yn = y[keep] - beta * y[j]
            lsn = ls + sps.norm.logpdf(y[j], scale=np.sqrt(s))
            labels_n = tuple(labels[k] for k in keep)
            # Derivatives of the normal density, divided by the density itself
            poly = [1.0, -y[j] / s, y[j] ** 2 / s**2 - 1 / s]
            val = 0.0
            for i in range(m + 1):
                n = m - i
                sub = 0.0
                for ps in itertools.product(range(len(keep)), repeat=n):
                    coef = (-1) ** n * np.prod(beta[list(ps)])
                    Rn = tuple(sorted(others + list(ps)))
                    sub += coef * deriv(labels_n, yn, Sn, Rn, lsn)
                val -= math.comb(m, i) * poly[i] * sub
        derivs[key] = val
        return val

    labels = tuple(range(len(b)))
    out = {}
    for R in multisets:
        R = tuple(sorted(R))
        if len(R) > 3:
            raise ValueError("Only derivatives up to order 3 are supported")
        out[R] = deriv(labels, b, S, R, logscale)
    return out


def _multiplicity(R):
    """Number of distinct permutations of the index tuple ``R``."""
    counts = [R.count(k) for k in set(R)]
    return math.factorial(len(R)) // math.prod(math.factorial(c) for c in counts)


def _lattice_factors(tau, delta):
    r"""
    The factor :math:`\tau/\rho(\tau)` and the coefficient
    :math:`a(\tau)=1/\tau-\rho'(\tau)/\rho(\tau)=1/\tau-(\delta/2)\coth(\tau\delta/2)` of
    Kolassa (2003) eq. (13), with :math:`\rho(\tau)=2\sinh(\tau\delta/2)/\delta` for a
    lattice with span :math:`\delta`. For :math:`\delta=0`, a continuous variable, they are 1
    and 0. Series expansions are used for small :math:`\tau\delta`.
    """
    z = np.asarray(tau, dtype=float) * np.asarray(delta, dtype=float) / 2
    small = np.abs(z) < 1e-3
    zs = np.where(small, 1.0, z)
    fac = np.where(small, 1 - z**2 / 6 + 7 * z**4 / 360, zs / np.sinh(zs))
    # a = (delta / 2) (1/z - coth(z)), with 1/z - coth(z) = -z/3 + z^3/45 - ...
    g = np.where(small, -z / 3 + z**3 / 45, 1 / zs - 1 / np.tanh(zs))
    return fac, np.asarray(delta, dtype=float) / 2 * g


def _kolassa_orthant(K, k2, k3, tau, tx, delta=None):
    r"""
    Kolassa (2003) eq. (5): the approximation of :math:`P(\mathbf{X}\geq\mathbf{x})`
    for a random vector with cumulant generating function :math:`K`, when all
    components of the saddlepoint :math:`\hat\tau` are non-negative.

    With :math:`\Sigma=K''(\hat\tau)` and :math:`\mathbf{b}=\Sigma\hat\tau`, expanding
    :math:`K` around :math:`\hat\tau` up to third order in the inversion integral
    gives

    .. math::
        P(\mathbf{X}\geq\mathbf{x}) \approx \exp\left(K(\hat\tau) - \hat\tau\cdot\mathbf{x}
            + \tfrac12\hat\tau^T\Sigma\hat\tau\right)
        \left[T - \frac16 \sum_{j,k,l} K'''_{jkl}(\hat\tau)
            (\partial_j+\hat\tau_j)(\partial_k+\hat\tau_k)(\partial_l+\hat\tau_l)T\right],

    where :math:`T(\mathbf{b})=P(\mathbf{Z}\geq\mathbf{b})` with
    :math:`\mathbf{Z}\sim N(0,\Sigma)`, and the derivatives are with respect to
    :math:`\mathbf{b}`. In terms of Kolassa's integrals,
    :math:`I(\Sigma, s, 0, \mathbf{b}, \hat\tau)=T`, and a factor
    :math:`\tau_l-\hat\tau_l` in the integrand corresponds to applying
    :math:`-\partial_l-\hat\tau_l` to :math:`T`. The approximation is exact for the
    normal distribution, and is not singular when components of :math:`\hat\tau` are
    zero.

    For components on lattices with spans :math:`\delta_j`, evaluated at the continuity
    corrected point, eq. (13) of Kolassa (2003) replaces :math:`1/\tau_j` in the inversion
    integrand by :math:`1/\rho_j(\tau_j)`, with
    :math:`\rho_j(\tau)=2\sinh(\tau\delta_j/2)/\delta_j`. Expanding
    :math:`\tau_j/\rho_j(\tau_j)` to first order around :math:`\hat\tau_j` multiplies the
    approximation by :math:`\prod_j\hat\tau_j/\rho_j(\hat\tau_j)` and adds
    :math:`-\sum_j a_j(\partial_j+\hat\tau_j)T` within the brackets, with
    :math:`a_j=1/\hat\tau_j-(\delta_j/2)\coth(\hat\tau_j\delta_j/2)`, see
    :func:`_lattice_factors`. A continuous component has :math:`\delta_j=0`, factor 1 and
    :math:`a_j=0`, so that components may be mixed.

    Parameters
    ----------
    K : float
        :math:`K(\hat\tau)`.
    k2 : array_like
        :math:`K''(\hat\tau)`.
    k3 : array_like
        :math:`K'''(\hat\tau)`.
    tau : array_like
        The saddlepoint :math:`\hat\tau\geq 0`.
    tx : float
        :math:`\hat\tau\cdot\mathbf{x}`.
    delta : array_like, optional
        Lattice spans, 0 for continuous components. ``None`` means all continuous.
    """
    tau = np.asarray(tau, dtype=float)
    d = len(tau)
    k2 = np.asarray(k2, dtype=float).reshape(d, d)
    k3 = np.asarray(k3, dtype=float).reshape(d, d, d)
    b = k2 @ tau
    logscale = K - tx + tau @ b / 2
    # Expand (d + tau)^3 contracted with the symmetric k3
    M = np.einsum("jkl,j->kl", k3, tau)
    v = M @ tau
    k0 = v @ tau
    ms = [()]
    for order in (1, 2, 3):
        ms += list(itertools.combinations_with_replacement(range(d), order))
    D = _mvn_tail_derivatives(b, k2, ms, logscale=logscale)
    cubic = k0 * D[()]
    for R in ms[1:]:
        mult = _multiplicity(R)
        if len(R) == 1:
            cubic += 3 * v[R[0]] * D[R]
        elif len(R) == 2:
            cubic += 3 * mult * M[R] * D[R]
        else:
            cubic += mult * k3[R] * D[R]
    if delta is None or not np.any(delta):
        return D[()] - cubic / 6
    fac, a = _lattice_factors(tau, delta)
    linear = sum(a[j] * (D[(j,)] + tau[j] * D[()]) for j in range(d))
    return np.prod(fac) * (D[()] - linear - cubic / 6)
