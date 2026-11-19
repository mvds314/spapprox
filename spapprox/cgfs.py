import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gammaln, logsumexp, softmax

from .cgf_base import (
    MultivariateCumulantGeneratingFunction,
    UnivariateCumulantGeneratingFunction,
)
from .domain import Domain
from .util import type_wrapper


def norm(loc=0, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t: t**2 / 2,
        dK=lambda t: t,
        dK_inv=lambda x: x,
        d2K=lambda t: np.ones(np.asanyarray(t).shape),
        d3K=lambda t: np.zeros(np.asanyarray(t).shape),
        loc=loc,
        scale=scale,
    )


def multivariate_norm(loc=0, scale=None, dim=None, cov=None):
    """
    Multivariate normal distribution with mean vector `loc` and covariance matrix `scale`.
    Scale can also be a vector of standard deviations.

    Parameters
    ----------
    loc: array_like, or float
       location vector
    scale: matrix, array_like, or float
       scale matrix. Whan an array is provided, it is interpreted as a vector of standard deviations.
    dim: int
       dimension of the multivariate normal distribution. When not specified, it is either
       inferred or defaults to 2 if, loc and scale can be scalars.
    cov: matrix, array_like, or float
       can be used instead of scale to provide the covariance matrix. The scale equals
       the Cholesky decomposition of the covariance matrix.

    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications
    """
    # Initialize scale
    if cov is None and scale is None:
        scale = 1
    elif cov is not None:
        assert scale is None, "Cannot specify both scale and cov"
        scale = np.linalg.cholesky(cov)
    assert scale is not None
    scale = np.asanyarray(scale)
    # Initialize loc
    assert loc is not None
    loc = np.asanyarray(loc)
    # Infer dimension
    if dim is None:
        if loc is not None and loc.ndim > 0:
            dim = loc.shape[0]
        elif scale is not None and scale.ndim > 0:
            dim = scale.shape[0]
        else:
            dim = 2
    # Validate input
    assert loc is None or loc.ndim == 0 or (loc.ndim == 1 and loc.shape[0] == dim), (
        "loc has wrong shape"
    )
    assert (
        scale is None
        or scale.ndim == 0
        or (scale.ndim == 1 and scale.shape[0] == dim)
        or (scale.ndim == 2 and scale.shape[1] == dim)
    ), "scale has wrong shape"
    # Return
    return MultivariateCumulantGeneratingFunction(
        K=lambda t: np.sum(t * t, axis=-1) / 2,
        dim=dim,
        dK=lambda t: t,
        dK_inv=lambda x: x,
        d2K=lambda t: (
            np.tile(np.eye(dim), (np.asanyarray(t).shape[0], 1, 1))
            if np.asanyarray(t).ndim == 2
            else np.eye(dim)
        ),
        d3K=np.vectorize(
            lambda t, dim=dim: np.zeros((dim, dim, dim)), signature=f"({dim})->({dim},{dim},{dim})"
        ),
        loc=loc if loc is None or loc.ndim > 0 else loc.tolist(),
        scale=scale if scale is None or scale.ndim > 0 else scale.tolist(),
    )


def exponential(loc=0, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t: np.log(1 / (1 - t)),
        dK=lambda t: 1 / (1 - t),
        dK_inv=lambda x: 1 - 1 / x,
        d2K=lambda t: 1 / (1 - t) ** 2,
        d3K=lambda t: 2 / (1 - t) ** 3,
        domain=Domain(l=1),
        loc=loc,
        scale=scale,
    )


def gamma(a=1, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, a=a: -a * np.log(1 - t),
        dK=lambda t, a=a: a / (1 - t),
        dK_inv=lambda x, a=a: 1 - a / x,
        d2K=lambda t, a=a: a / (1 - t) ** 2,
        d3K=lambda t, a=a: 2 * a / (1 - t) ** 3,
        domain=Domain(l=1),
        scale=scale,
    )


def bivariate_gamma(a=None, loc=0, scale=1):
    r"""
    Bivariate gamma distribution with shape parameter `a` and scale parameter `scale`.

    There are many definitions of the multivariate gamma distribution [3].
    A general requirement is that the (conditional) marginals are univariate gamma distributions.
    Here, we use what is referred to as the Cheriyan and Ramabhadran's bivariate gamma distribution [3].
    Saddlepoint approximations are discussed in [2].
    The bivariate gamma distribution discussed in [1] differs and is known as the Jensen bivariate gamma distribution [3].

    Specifically, let :math:`Y_0, Y_1` and :math:`Y_2` have a gamma distribution with shape parameters :math:`a_i`.
    Then the random vector :math:`(X_1, X_2)` defined by :math:`X_1 = Y_0 + Y_1` and :math:`X_2 = Y_0 + Y_2`
    has a bivariate gamma distribution with shape parameter :math:`a = (a_0, a_1, a_2)`.

    It's density is given by:

    .. math::
        p_{X_1,X_2}(x_1,x_2) = \frac{\exp(-x_1-x_2)}{\Gamma(a_0)\Gamma(a_1)\Gamma(a_2)} \int_0^{\min(x_1,x_2)} y_0^{a_0-1} (x_1-y_0)^{a_1-1} (x_2-y_0)^{a_2-1} dy_0

    Note that :math:`X_1` and :math:`X_2` are independent when :math:`a_0 = 0`.
    More specifically, we have that :math:`\text{Cov}(X_1, X_2) = \text{Var}(Y_0) = a_0`.

    Also, :math:`a_1` and :math:`a_2` further control the shape of the marginals.
    More specifically, the marginal density of :math:`X_1` and :math:`X_2` are gamma distributions
    with shape parameters :math:`a_0+a_1` and :math:`a_0+a_2`, respectively.

    As we have :math:`\text{Var}(X_1)=a_0+a_1` and :math:`Var(X_2)=a_0+a_2`, it follows that the correlation
    between :math:`X_1` and :math:`X_2` is given by:

    .. math::
        \rho = \frac{\text{Cov}(X_1, X_2)}{\sqrt{\text{Var}(X_1)\text{Var}(X_2)}} = \frac{a_0}{\sqrt{(a_0+a_1)(a_0+a_2)}}

    Finally, the cumulant generating function is given by:

    .. math::
        K(t_1,t_2) &=\log \text{E} \left[ \exp(t_1 X_1 + t_2 X_2) \right] \\
        &=\log\left[ \frac{1}{(1-t_1)^{a_1}(1-t_2)^{a_2}(1-t_1-t_2)^{a_0}}\right]\\
        &= -a_1 \log(1-t_1) - a_2 \log(1-t_2) - a_0 \log(1-t_1-t_2)


    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications
    [2] Huzurbazar (2012) - Saddlepoint Approximations with Applications
    [3] Kotz, Balakrishnan, Johnson (2000) - Continuous Multivariate distributions. Volume 1
    """
    # Initialize a
    if a is None:
        a = np.array([0, 1, 1])
    else:
        a = np.asanyarray(a)
    if a.shape != (3,):
        raise ValueError("a must be a vector of length 3")
    if a[0] < 0:
        raise ValueError("a[0] must be non-negative")
    if np.any(a[1:] <= 0):
        raise ValueError("a[1] and a[2] must be positive")
    # Initialize loc and scale
    loc = np.asanyarray(loc)
    scale = np.asanyarray(scale)

    def K(t, a=a):
        t = np.asanyarray(t)
        val = -a[1] * np.log(1 - t.T[0]) - a[2] * np.log(1 - t.T[1])
        if a[0] > 0:
            # For a[0] = 0, skip the term, as it would evaluate to 0 * nan for t1 + t2 >= 1
            val = val - a[0] * np.log(1 - t.T[0] - t.T[1])
        return val

    def dK(t, a=a):
        t = np.asanyarray(t)
        return np.add(
            np.array([a[1] / (1 - t.T[0]), a[2] / (1 - t.T[1])]),
            a[0] / (1 - t.T[0] - t.T[1]),
        ).T

        return np.dot(-a, np.log([1 - t.T[0] - t.T[1], 1 - t.T[0], 1 - t.T[1]]))

    def d2K(t, a=a):
        t = np.asanyarray(t)
        return (
            np.diag([a[1] / (1 - t.T[0]) ** 2, a[2] / (1 - t.T[1]) ** 2])
            + a[0] / (1 - t.T[0] - t.T[1]) ** 2
        )

    def d3K(t, a=a):
        t = np.asanyarray(t)
        return (
            np.array(
                [
                    [[2 * a[1] / (1 - t[0]) ** 3, 0], [0, 0]],
                    [[0, 0], [0, 2 * a[2] / (1 - t[1]) ** 3]],
                ]
            )
            + 2 * a[0] / (1 - t[0] - t[1]) ** 3
        )

    return MultivariateCumulantGeneratingFunction(
        K=K,
        dK=dK,
        d2K=np.vectorize(d2K, signature="(2)->(2,2)"),
        d3K=np.vectorize(d3K, signature="(2)->(2,2,2)"),
        dim=2,
        domain=(
            Domain(l=np.ones(2), B=np.ones((1, 2)), b=np.ones(1), dim=2)
            if a[0] > 0
            else Domain(l=np.ones(2), dim=2)
        ),
        loc=loc,
        scale=scale,
    )


def chi2(df=1):
    return gamma(a=df / 2, scale=2)


def jensen_multivariate_gamma(a=1, sigma=None, dim=None):
    r"""
    Jensen's multivariate gamma distribution with shape :math:`a` and positive definite
    scale matrix :math:`\Sigma`, with moment generating function, see [1, Section 3.1],

    .. math::
        M(t) = |I - \Sigma T|^{-a}, \qquad T = \text{diag}(t),

    finite where :math:`\Sigma^{-1}-T` is positive definite. For :math:`2a` a positive integer,
    it is the distribution of half the diagonal of a Wishart matrix with :math:`2a` degrees of
    freedom and scale :math:`\Sigma`. The marginals are gamma with shape :math:`a` and scales
    :math:`\Sigma_{ii}`, and :math:`\text{Cov}(X_i, X_j) = a\Sigma_{ij}^2`. In two dimensions it
    is a valid distribution for every :math:`a>0` (Kibble's bivariate gamma), in :math:`d`
    dimensions in general only for :math:`2a\in\{1,\ldots,d-2\}` or :math:`2a>d-2` [3].

    With :math:`S(t) = (\Sigma^{-1}-T)^{-1}`, the derivatives are

    .. math::
        K'_i = aS_{ii}, \qquad K''_{ij} = aS_{ij}^2, \qquad K'''_{ijk} = 2aS_{ij}S_{jk}S_{ki}.

    The domain declared to the cgf is the linear outer bound :math:`t_i < 1/\Sigma_{ii}`, the
    cgf evaluates to NaN where :math:`\Sigma^{-1}-T` is not positive definite.

    Parameters
    ----------
    a: float
        positive shape parameter
    sigma: matrix
        positive definite scale matrix, defaults to the identity
    dim: int
        dimension, only used when `sigma` is not given, defaults to 2

    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications
    [2] Jensen (1970) - The joint distribution of quadratic forms and related distributions
    [3] Kotz, Balakrishnan, Johnson (2000) - Continuous Multivariate distributions. Volume 1
    """
    if not a > 0:
        raise ValueError("a must be positive")
    sigma = np.eye(2 if dim is None else dim) if sigma is None else np.asanyarray(sigma, float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("sigma must be a square matrix")
    if not np.allclose(sigma, sigma.T) or np.linalg.eigvalsh(sigma).min() <= 0:
        raise ValueError("sigma must be symmetric positive definite")
    dim = sigma.shape[0]
    sigma_inv = np.linalg.inv(sigma)
    logdet_sigma = np.linalg.slogdet(sigma)[1]

    def S(t):
        # (Sigma^{-1} - T)^{-1}, NaN where Sigma^{-1} - T is not positive definite
        t = np.asanyarray(t, float)
        A = sigma_inv - t[..., None, :] * np.eye(dim)
        pd = np.linalg.eigvalsh(A).min(axis=-1) > 0
        A = np.where(pd[..., None, None], A, np.eye(dim))
        return np.where(pd[..., None, None], np.linalg.inv(A), np.nan), A, pd

    def K(t):
        _, A, pd = S(t)
        return np.where(pd, -a * (logdet_sigma + np.linalg.slogdet(A)[1]), np.nan)

    return MultivariateCumulantGeneratingFunction(
        K=K,
        dK=lambda t: a * np.diagonal(S(t)[0], axis1=-2, axis2=-1),
        d2K=lambda t: a * S(t)[0] ** 2,
        d3K=lambda t: 2 * a * np.einsum("...ij,...jk,...ki->...ijk", *[S(t)[0]] * 3),
        dim=dim,
        domain=Domain(l=1 / np.diag(sigma), dim=dim),
    )


def multinomial(n=1, p=None):
    r"""
    Multinomial distribution with :math:`n` trials and cell probabilities
    :math:`p=(p_1,\ldots,p_k)`, as ``scipy.stats.multinomial(n, p)``.

    The counts sum to :math:`n`, so the cgf describes the first :math:`d=k-1` counts, whose
    covariance is non-singular, see [1, Section 3.1]:

    .. math::
        K(t) = n\log\left(p_k + \sum_{i=1}^{k-1} p_ie^{t_i}\right).

    With :math:`w_i = p_ie^{t_i}/(p_k + \sum_j p_je^{t_j})`, :math:`K'_i=nw_i`,
    :math:`K''_{ij}=n(\delta_{ij}w_i-w_iw_j)`, and the saddle point equation is solved by
    :math:`t_i=\log(x_ip_k/(x_kp_i))` with :math:`x_k=n-\sum_i x_i`. The distribution is a
    lattice distribution with span 1 in every coordinate.

    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications
    """
    p = np.full(2, 0.5) if p is None else np.asanyarray(p, float)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("p must be a vector of at least two probabilities")
    if np.any(p <= 0) or not np.isclose(p.sum(), 1):
        raise ValueError("p must be positive and sum to one")
    if not n > 0:
        raise ValueError("n must be positive")
    dim = len(p) - 1
    logp = np.log(p)
    eye = np.eye(dim)

    def logS(t):
        t = np.asanyarray(t, float)
        z = np.concatenate([logp[:-1] + t, np.broadcast_to(logp[-1], t[..., :1].shape)], -1)
        return logsumexp(z, axis=-1)

    def w(t):
        t = np.asanyarray(t, float)
        return np.exp(logp[:-1] + t - logS(t)[..., None])

    def d2K(t):
        ww = w(t)
        return n * (ww[..., None, :] * eye - ww[..., :, None] * ww[..., None, :])

    def d3K(t):
        ww = w(t)
        d3 = np.einsum("...i,ij,ik->...ijk", ww, eye, eye)
        d2w = np.einsum("...i,...k,ij->...ijk", ww, ww, eye)
        return n * (
            d3
            - d2w
            - np.swapaxes(d2w, -1, -2)
            - np.einsum("...ijk->...kij", d2w)
            + 2 * np.einsum("...i,...j,...k->...ijk", ww, ww, ww)
        )

    def dK_inv(x):
        x = np.asanyarray(x, float)
        xk = n - x.sum(axis=-1, keepdims=True)
        return np.log(x / xk) + logp[-1] - logp[:-1]

    return MultivariateCumulantGeneratingFunction(
        K=lambda t: n * logS(t),
        dK=lambda t: n * w(t),
        dK_inv=dK_inv,
        d2K=d2K,
        d3K=d3K,
        dim=dim,
    )


def bivariate_poisson(mu=None):
    r"""
    Bivariate Poisson distribution of :math:`(X_1, X_2)=(Y_0+Y_1, Y_0+Y_2)` for independent
    Poisson :math:`Y_i` with means :math:`\mu=(\mu_0,\mu_1,\mu_2)`, see [1, Section 3.1]:

    .. math::
        K(t) = \mu_1(e^{t_1}-1) + \mu_2(e^{t_2}-1) + \mu_0(e^{t_1+t_2}-1).

    The marginals are Poisson with means :math:`\mu_0+\mu_1` and :math:`\mu_0+\mu_2`, and
    :math:`\text{Cov}(X_1,X_2)=\mu_0`. With :math:`a=e^{t_1}` and :math:`b=e^{t_2}`, the saddle
    point equation reduces to the quadratic
    :math:`\mu_0\mu_2b^2+(\mu_1\mu_2+\mu_0(x_1-x_2))b-\mu_1x_2=0` and
    :math:`a=x_1/(\mu_1+\mu_0b)`. The distribution is a lattice distribution with span 1 in
    both coordinates.

    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications
    [2] Kocherlakota, Kocherlakota (1992) - Bivariate Discrete Distributions
    """
    mu = np.array([0, 1, 1], float) if mu is None else np.asanyarray(mu, float)
    if mu.shape != (3,):
        raise ValueError("mu must be a vector of length 3")
    if mu[0] < 0:
        raise ValueError("mu[0] must be non-negative")
    if np.any(mu[1:] <= 0):
        raise ValueError("mu[1] and mu[2] must be positive")
    m0, m1, m2 = mu

    def terms(t):
        # The three Poisson terms e^{t1}, e^{t2} and e^{t1 + t2}, weighted by their means
        t = np.asanyarray(t, float)
        return m1 * np.exp(t[..., 0]), m2 * np.exp(t[..., 1]), m0 * np.exp(t[..., 0] + t[..., 1])

    def K(t):
        e1, e2, e0 = terms(t)
        return e1 + e2 + e0 - mu.sum()

    def dK(t):
        e1, e2, e0 = terms(t)
        return np.stack([e1 + e0, e2 + e0], axis=-1)

    def d2K(t):
        e1, e2, e0 = terms(t)
        return np.stack([np.stack([e1, 0 * e1], -1), np.stack([0 * e2, e2], -1)], -2) + e0[
            ..., None, None
        ] * np.ones((2, 2))

    def d3K(t):
        e1, e2, e0 = terms(t)
        val = e0[..., None, None, None] * np.ones((2, 2, 2))
        val[..., 0, 0, 0] += e1
        val[..., 1, 1, 1] += e2
        return val

    def dK_inv(x):
        x = np.asanyarray(x, float)
        x1, x2 = x[..., 0], x[..., 1]
        B = m1 * m2 + m0 * (x1 - x2)
        b = 2 * m1 * x2 / (B + np.sqrt(B**2 + 4 * m0 * m2 * m1 * x2))
        return np.log(np.stack([x1 / (m1 + m0 * b), b], axis=-1))

    return MultivariateCumulantGeneratingFunction(
        K=K, dK=dK, dK_inv=dK_inv, d2K=d2K, d3K=d3K, dim=2
    )


def laplace(loc=0, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t: -np.log(1 - t**2),
        dK=lambda t: 2 * t / (1 - t**2),
        dK_inv=lambda x: x / (1 + np.sqrt(1 + x**2)),
        d2K=lambda t: 2 * (1 + t**2) / (1 - t**2) ** 2,
        d3K=lambda t: 4 * t * (3 + t**2) / (1 - t**2) ** 3,
        domain=Domain(g=-1, l=1),
        loc=loc,
        scale=scale,
    )


def asymmetric_laplace(kappa=1, loc=0, scale=1):
    r"""
    Asymmetric Laplace distribution, parametrized as ``scipy.stats.laplace_asymmetric``.

    The standardized density is :math:`e^{-\kappa x}/(\kappa+1/\kappa)` for :math:`x\geq 0`
    and :math:`e^{x/\kappa}/(\kappa+1/\kappa)` for :math:`x<0`, i.e., the distribution of
    :math:`E_1/\kappa-\kappa E_2` for independent standard exponentials :math:`E_1, E_2`. Hence

    .. math::
        K(t) = -\log(1-t/\kappa) - \log(1+\kappa t), \qquad -1/\kappa < t < \kappa.

    For :math:`\kappa=1` it reduces to :func:`laplace`.

    References
    ----------
    [1] Kotz, Kozubowski, Podgorski (2001) - The Laplace Distribution and Generalizations
    """
    if not kappa > 0:
        raise ValueError("kappa must be positive")

    def dK_inv(x, k=kappa):
        # The root in (-1/k, k) of the quadratic obtained by multiplying x = dK(t) by
        # (k - t)(1 + k t); this rationalized form has no removable singularities
        x = np.asanyarray(x)
        A = x * k
        B = 2 * k - x * (k**2 - 1)
        C = -(x * k + k**2 - 1)
        return -2 * C / (B + np.sqrt(B**2 - 4 * A * C))

    return UnivariateCumulantGeneratingFunction(
        K=lambda t, k=kappa: -np.log(1 - t / k) - np.log(1 + k * t),
        dK=lambda t, k=kappa: 1 / (k - t) - k / (1 + k * t),
        dK_inv=dK_inv,
        d2K=lambda t, k=kappa: 1 / (k - t) ** 2 + k**2 / (1 + k * t) ** 2,
        d3K=lambda t, k=kappa: 2 / (k - t) ** 3 - 2 * k**3 / (1 + k * t) ** 3,
        domain=Domain(g=-1 / kappa, l=kappa),
        loc=loc,
        scale=scale,
    )


def generalized_normal(beta=2, loc=0, scale=1):
    r"""
    Generalized normal distribution, parametrized as ``scipy.stats.gennorm``.

    The standardized density is :math:`\beta/(2\Gamma(1/\beta))\exp(-|x|^\beta)`. The moment
    generating function is finite for all :math:`t` when :math:`\beta>1`, but has no closed
    form, so it is evaluated by quadrature of the exponentially tilted density. Its
    derivatives are the mean, variance and third central moment of the tilted density:

    .. math::
        K(t) = \log\frac{\beta}{2\Gamma(1/\beta)}\int e^{tx-|x|^\beta}\,dx.

    For :math:`\beta=1` this is the Laplace distribution, see :func:`laplace`, and for
    :math:`\beta<1` the moment generating function is infinite for all :math:`t\neq 0`, so no
    saddle point approximation exists.

    References
    ----------
    [1] Nadarajah (2005) - A generalized normal distribution
    """
    if beta == 1:
        return laplace(loc=loc, scale=scale)
    if not beta > 1:
        raise ValueError("beta must be at least 1, otherwise the cgf is infinite for t != 0")
    lognorm = np.log(beta / 2) - gammaln(1 / beta)

    def tilted(t, b=beta):
        # Integrate in u = x - mode of the tilted density, with exponent max subtracted
        mode = np.sign(t) * (np.abs(t) / b) ** (1 / (b - 1))
        c = t * mode - np.abs(mode) ** b

        def f(u):
            x = mode + u
            return np.exp(t * x - np.abs(x) ** b - c) * u ** np.arange(4)

        mom = sum(
            quad_vec(f, lo, hi, epsabs=0, epsrel=1e-12)[0]
            for lo, hi in [(-np.inf, 0), (0, np.inf)]
        )
        m1, m2, m3 = mom[1:] / mom[0]
        return lognorm + c + np.log(mom[0]), mode + m1, m2 - m1**2, m3 - 3 * m1 * m2 + 2 * m1**3

    def moment(i):
        return np.vectorize(lambda t: tilted(t)[i], otypes=[float])

    return UnivariateCumulantGeneratingFunction(
        K=moment(0),
        dK=moment(1),
        d2K=moment(2),
        d3K=moment(3),
        dK0=0,
        d2K0=np.exp(gammaln(3 / beta) - gammaln(1 / beta)),
        d3K0=0,
        loc=loc,
        scale=scale,
    )


def univariate_sample_mean(cgf, sample_size):
    r"""
    Given the cumulant generating function of a univariate random variable, this class
    provides the saddle point approximation of the sample mean of the random variable.

    Given :math:`n` i.i.d. random variables :math:`X_1, \ldots, X_n` with
    cumulant generating function :math:`K`, the cumulant generating function :math:`K_n`
    of the sample mean :math:`\bar{X}` is given by

    .. math::
        K_{\bar{X}}(t) = \sum_{i=1}^n 1/n*K_i(t)= \sum_{i=1}^n K_i(t/n) = n K(t/n).
    """
    assert isinstance(cgf, UnivariateCumulantGeneratingFunction)
    assert isinstance(sample_size, int) and sample_size > 0
    return UnivariateCumulantGeneratingFunction(
        lambda t, n=sample_size, cgf=cgf: n * cgf.K(t / n),
        dK=lambda t, n=sample_size, cgf=cgf: cgf.dK(t / n),
        dK_inv=lambda x, n=sample_size, cgf=cgf: n * cgf.dK_inv(x),
        d2K=lambda t, n=sample_size, cgf=cgf: cgf.d2K(t / n) / n,
        d3K=lambda t, n=sample_size, cgf=cgf: cgf.d3K(t / n) / n**2,
        domain=cgf.domain.mul(sample_size / cgf.scale),
    )


def univariate_empirical(x):
    """
    Given a vector :math`x` with observations of a univariate random variable,
    draw one of then with equal probability.

    The cgf is that of the standardized sample, with ``loc`` and ``scale`` the sample mean and
    standard deviation, so numerical derivatives do not depend on the scale of the data.
    """

    x = np.asanyarray(x, dtype=float)
    loc = x.mean()
    scale = x.std()
    scale = 1.0 if not scale > 0 else scale
    x = (x - loc) / scale

    # Use logsumexp and softmax to prevent overflow for large t * x
    @type_wrapper(xloc=0)
    def K(t, x=x):
        tx = t * x if t.ndim == 0 else np.atleast_2d(t).T.dot(np.atleast_2d(x))
        y = logsumexp(tx, axis=-1) - np.log(len(x))
        return y.tolist() if t.ndim == 1 else y

    @type_wrapper(xloc=0)
    def dK(t, x=x):
        tx = t * x if t.ndim == 0 else np.atleast_2d(t).T.dot(np.atleast_2d(x))
        y = (softmax(tx, axis=-1) * x).sum(axis=-1)
        return y.tolist() if t.ndim == 1 else y

    return UnivariateCumulantGeneratingFunction(K, dK=dK, loc=loc, scale=scale)


def poisson(mu=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, mu=mu: mu * (np.exp(t) - 1),
        dK=lambda t, mu=mu: mu * np.exp(t),
        dK_inv=lambda x, mu=mu: np.log(x / mu),
        d2K=lambda t, mu=mu: mu * np.exp(t),
        d3K=lambda t, mu=mu: mu * np.exp(t),
    )


def binomial(n=1, p=0.5):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, n=n, p=p: n * np.log(p * (np.exp(t) - 1) + 1),
        dK=lambda t, n=n, p=p: n * p / ((1 - p) * np.exp(-t) + p),
        dK_inv=lambda x, n=n, p=p: -np.log((n * p / x - p) / (1 - p)),
        d2K=lambda t, n=n, p=p: n * p * (1 - p) * np.exp(-t) / ((1 - p) * np.exp(-t) + p) ** 2,
        d3K=lambda t, n=n, p=p: (
            n
            * p
            * (1 - p)
            * ((1 - p) * np.exp(-2 * t) - np.exp(-t) * p)
            / ((1 - p) * np.exp(-t) + p) ** 3
        ),
    )


def bernoulli(p=0.5):
    r"""
    Cumulant generating function of a Bernoulli distribution, :math:`K(t)=\log(1-p+pe^t)`.

    Equivalent to ``binomial(n=1, p=p)``.
    """
    return binomial(n=1, p=p)


def negative_binomial(r=1, p=0.5):
    r"""
    Negative binomial distribution of the number of failures before the :math:`r`-th success,
    with success probability :math:`p`, parametrized as ``scipy.stats.nbinom(r, p)``.

    With :math:`q=1-p`, the cumulant generating function is

    .. math::
        K(t) = r\log p - r\log(1-qe^t), \qquad t < -\log q,

    and with :math:`z=qe^t/(1-qe^t)`, :math:`K'(t)=rz`, :math:`K''(t)=rz(1+z)` and
    :math:`K'''(t)=rz(1+z)(1+2z)`. The distribution is a lattice distribution with span 1.

    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications, Exercise 1.1
    """
    if not r > 0:
        raise ValueError("r must be positive")
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    q = 1 - p

    def z(t):
        e = q * np.exp(t)
        return e / (1 - e)

    return UnivariateCumulantGeneratingFunction(
        K=lambda t, r=r: r * np.log(p) - r * np.log(1 - q * np.exp(t)),
        dK=lambda t, r=r: r * z(t),
        dK_inv=lambda x, r=r: np.log(x / (q * (r + x))),
        d2K=lambda t, r=r: r * z(t) * (1 + z(t)),
        d3K=lambda t, r=r: r * z(t) * (1 + z(t)) * (1 + 2 * z(t)),
        domain=Domain(l=-np.log(q)),
    )
