r"""
Cumulant generating functions of common distributions.

Every function in this module is a factory returning a configured
:class:`~spapprox.UnivariateCumulantGeneratingFunction` or
:class:`~spapprox.MultivariateCumulantGeneratingFunction`. Each supplies as many analytic
derivatives as are cheaply available, an analytic inverse ``dK_inv`` of the first derivative
where it exists, and a :class:`~spapprox.Domain` when the cgf is not finite everywhere.

Where a factory takes ``loc`` and ``scale``, the cgf and its domain are stated for the
standardized variable :math:`Z`, and the object represents :math:`X=\text{scale}\cdot Z+\text{loc}`
with :math:`K_X(t)=K_Z(\text{scale}\cdot t)+\text{loc}\cdot t`; the base class applies the
transformation. The parametrizations follow ``scipy.stats`` where possible.

Two factories are transformations rather than distributions: :func:`univariate_empirical`,
the cgf of drawing uniformly from a sample, and :func:`univariate_sample_mean`, the cgf of the
mean of i.i.d. copies. Composed, they give the bootstrap distribution of a sample mean, see
``examples/example_nonparametric_bootstrap.py``.

Examples
--------
>>> import spapprox as spa
>>> cgf = spa.gamma(a=2, scale=3)
>>> float(cgf.dK0), float(cgf.d2K0)
(6.0, 18.0)

References
----------
[1] Butler (2007) - Saddlepoint Approximations with Applications
[2] Johnson, Kotz, Balakrishnan (1994) - Continuous Univariate Distributions
"""

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gammaln, log_ndtr, logsumexp, softmax

from .cgf_base import (
    MultivariateCumulantGeneratingFunction,
    UnivariateCumulantGeneratingFunction,
)
from .domain import Domain
from .util import type_wrapper


def norm(loc=0, scale=1):
    r"""
    Normal distribution with mean ``loc`` and standard deviation ``scale``, as
    ``scipy.stats.norm(loc, scale)``.

    The cgf of the standard normal is exact and in closed form,

    .. math::
        K(t) = \frac{t^2}{2}, \qquad t\in\mathbb{R},

    and the saddle point approximation of the density is exact.

    Parameters
    ----------
    loc : float, optional
        Mean. Default is 0.
    scale : float, optional
        Standard deviation. Default is 1.

    Returns
    -------
    UnivariateCumulantGeneratingFunction
        The cgf, with analytic derivatives and ``dK_inv``.

    Examples
    --------
    >>> import spapprox as spa
    >>> spa.norm(loc=1, scale=2).K(0.5)
    1.0

    See ``examples/basic_example.py``.

    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications, Section 1.1
    """
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
    r"""
    Multivariate normal distribution with mean vector `loc` and scale `scale`, i.e., the
    distribution of :math:`X=\text{scale}\cdot Z+\text{loc}` for a standard normal vector
    :math:`Z`, with covariance matrix :math:`\text{scale}\cdot\text{scale}^T`.
    Scale can also be a vector of standard deviations, or the covariance matrix can be
    given through `cov`.

    The cgf is exact and in closed form, :math:`K(t)=<t,t>/2` for :math:`Z`, finite
    everywhere, as ``scipy.stats.multivariate_normal(loc, cov)``.

    Parameters
    ----------
    loc: array_like, or float
       location vector
    scale: matrix, array_like, or float
       scale matrix. When a vector is provided, it is interpreted as a vector of standard
       deviations.
    dim: int
       dimension of the multivariate normal distribution. When not specified, it is either
       inferred or defaults to 2 if, loc and scale can be scalars.
    cov: matrix, array_like, or float
       can be used instead of scale to provide the covariance matrix. The scale equals
       the Cholesky decomposition of the covariance matrix.

    Returns
    -------
    MultivariateCumulantGeneratingFunction
        The cgf, with analytic derivatives and ``dK_inv``.

    Examples
    --------
    >>> import numpy as np
    >>> import spapprox as spa
    >>> cgf = spa.multivariate_norm(loc=[0, 1], cov=[[1, 0.5], [0.5, 2]])
    >>> cgf.cov.tolist()
    [[1.0, 0.5], [0.5, 2.0]]

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
    r"""
    Exponential distribution with location ``loc`` and scale (mean) ``scale``, as
    ``scipy.stats.expon(loc, scale)``.

    The cgf of the standard exponential is exact and in closed form,

    .. math::
        K(t) = -\log(1-t), \qquad t<1,

    so :math:`t<1/\text{scale}` for the scaled variable.

    Parameters
    ----------
    loc : float, optional
        Location. Default is 0.
    scale : float, optional
        Scale, the mean of the unshifted variable. Default is 1.

    Returns
    -------
    UnivariateCumulantGeneratingFunction
        The cgf, with analytic derivatives and ``dK_inv``.

    Examples
    --------
    >>> import spapprox as spa
    >>> float(spa.exponential(scale=2).dK0)
    2.0

    See ``examples/basic_example.py``.

    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications, Section 1.1
    """
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
    r"""
    Gamma distribution with shape ``a`` and scale ``scale``, as
    ``scipy.stats.gamma(a, scale=scale)``.

    The cgf of the gamma distribution with unit scale is exact and in closed form,

    .. math::
        K(t) = -a\log(1-t), \qquad t<1,

    so :math:`t<1/\text{scale}` for the scaled variable. Unlike most other factories there
    is no ``loc`` parameter; add a constant instead, e.g., ``gamma(a) + 1``. The saddle
    point density is exact up to normalization (Stirling's approximation of
    :math:`\Gamma(a)`).

    Parameters
    ----------
    a : float, optional
        Shape parameter, positive. Default is 1, the exponential distribution.
    scale : float, optional
        Scale parameter. Default is 1.

    Returns
    -------
    UnivariateCumulantGeneratingFunction
        The cgf, with analytic derivatives and ``dK_inv``.

    Examples
    --------
    >>> import spapprox as spa
    >>> cgf = spa.gamma(a=2, scale=3)
    >>> float(cgf.dK0), float(cgf.d2K0)
    (6.0, 18.0)

    See ``examples/example_parametric_bootstrap.py``.

    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications, Section 1.1
    """
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, a=a: -a * np.log(1 - t),
        dK=lambda t, a=a: a / (1 - t),
        dK_inv=lambda x, a=a: 1 - a / x,
        d2K=lambda t, a=a: a / (1 - t) ** 2,
        d3K=lambda t, a=a: 2 * a / (1 - t) ** 3,
        domain=Domain(l=1),
        scale=scale,
    )


def inverse_gaussian(mu=1, scale=1):
    r"""
    Inverse Gaussian distribution, as ``scipy.stats.invgauss(mu, scale=scale)``, i.e., with
    mean :math:`\mu\cdot\text{scale}` and shape parameter :math:`\lambda=\text{scale}`.

    With unit scale, the cgf is exact and in closed form,

    .. math::
        K(t) = \frac{1-\sqrt{1-2\mu^2t}}{\mu}, \qquad t<\frac{1}{2\mu^2}.

    :math:`K` is finite at the boundary of its domain, but its derivatives are not, so the
    domain is taken open. As for the gamma distribution there is no ``loc`` parameter, and
    the saddle point density is exact up to normalization. The standard form
    :math:`\pi^{-1/2}u^{-3/2}e^{2-u-1/u}` of Barndorff-Nielsen and Cox (1979) is
    ``inverse_gaussian(mu=0.5, scale=2)``.

    Parameters
    ----------
    mu : float, optional
        Mean of the unscaled variable, positive. Default is 1.
    scale : float, optional
        Scale parameter, positive. Default is 1.

    Returns
    -------
    UnivariateCumulantGeneratingFunction
        The cgf, with analytic derivatives and ``dK_inv``.

    Examples
    --------
    >>> import spapprox as spa
    >>> cgf = spa.inverse_gaussian(mu=2, scale=3)
    >>> float(cgf.dK0), float(cgf.d2K0)
    (6.0, 72.0)

    References
    ----------
    [1] Barndorff-Nielsen, Cox (1979) - Edgeworth and saddle-point approximations with
    statistical applications, Section 3.4

    [2] Butler (2007) - Saddlepoint Approximations with Applications, Section 1.1
    """
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, mu=mu: (1 - np.sqrt(1 - 2 * mu**2 * t)) / mu,
        dK=lambda t, mu=mu: mu / np.sqrt(1 - 2 * mu**2 * t),
        dK_inv=lambda x, mu=mu: 1 / (2 * mu**2) - 1 / (2 * np.asanyarray(x) ** 2),
        d2K=lambda t, mu=mu: mu**3 / (1 - 2 * mu**2 * t) ** 1.5,
        d3K=lambda t, mu=mu: 3 * mu**5 / (1 - 2 * mu**2 * t) ** 2.5,
        domain=Domain(l=1 / (2 * mu**2)),
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
    [2] Kotz, Balakrishnan, Johnson (2000) - Continuous Multivariate distributions. Volume 1
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
    r"""
    Chi-squared distribution with ``df`` degrees of freedom, as ``scipy.stats.chi2(df)``.

    Equivalent to ``gamma(a=df / 2, scale=2)``, so the cgf is exact and in closed form,

    .. math::
        K(t) = -\frac{\text{df}}{2}\log(1-2t), \qquad t<1/2.

    Parameters
    ----------
    df : float, optional
        Degrees of freedom, positive. Default is 1.

    Returns
    -------
    UnivariateCumulantGeneratingFunction
        The cgf, see :func:`gamma`.
    """
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
    r"""
    Laplace (double exponential) distribution with location ``loc`` and scale ``scale``, as
    ``scipy.stats.laplace(loc, scale)``.

    The cgf of the standard Laplace distribution is exact and in closed form,

    .. math::
        K(t) = -\log(1-t^2), \qquad -1<t<1,

    so :math:`|t|<1/\text{scale}` for the scaled variable.

    Parameters
    ----------
    loc : float, optional
        Location, the mean. Default is 0.
    scale : float, optional
        Scale. Default is 1, i.e., variance 2.

    Returns
    -------
    UnivariateCumulantGeneratingFunction
        The cgf, with analytic derivatives and ``dK_inv``.

    See Also
    --------
    asymmetric_laplace, generalized_normal

    References
    ----------
    [1] Kotz, Kozubowski, Podgorski (2001) - The Laplace Distribution and Generalizations
    [2] Butler (2007) - Saddlepoint Approximations with Applications, Section 1.1
    """
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
    form, so it is evaluated by quadrature, see :func:`asymmetric_generalized_normal`:

    .. math::
        K(t) = \log\frac{\beta}{2\Gamma(1/\beta)}\int e^{tx-|x|^\beta}\,dx.

    For :math:`\beta=1` this is the Laplace distribution, see :func:`laplace`, for
    :math:`\beta=2` a normal distribution with variance :math:`1/2`, see :func:`norm`, and for
    :math:`\beta<1` the moment generating function is infinite for all :math:`t\neq 0`, so no
    saddle point approximation exists.

    References
    ----------
    [1] Nadarajah (2005) - A generalized normal distribution
    """
    if beta == 1:
        return laplace(loc=loc, scale=scale)
    if beta == 2:
        return norm(loc=loc, scale=scale / np.sqrt(2))
    return asymmetric_generalized_normal(beta=beta, sigma=1, loc=loc, scale=scale)


def asymmetric_generalized_normal(beta=2, sigma=1, loc=0, scale=1):
    r"""
    Asymmetric generalized normal (asymmetric exponential power) distribution, with shapes
    :math:`\beta=(\beta_L,\beta_R)` and scales :math:`\sigma=(\sigma_L,\sigma_R)` to the left
    and right of zero. The standardized density is

    .. math::
        f(x) = c\exp\left(-|x/\sigma_L|^{\beta_L}\right) \text{ for } x<0, \qquad
        f(x) = c\exp\left(-(x/\sigma_R)^{\beta_R}\right) \text{ for } x\geq 0,

    with :math:`1/c=\sigma_L\Gamma(1+1/\beta_L)+\sigma_R\Gamma(1+1/\beta_R)`, so the
    density is continuous at zero, and the raw moments are
    :math:`\text{E}X^k=c\left(\sigma_R^{k+1}\Gamma((k+1)/\beta_R)/\beta_R
    +(-1)^k\sigma_L^{k+1}\Gamma((k+1)/\beta_L)/\beta_L\right)`. Scalar `beta` or `sigma`
    apply to both sides; equal sides give :func:`generalized_normal`.

    In general, the cgf has no closed form and is evaluated by quadrature of the exponentially
    tilted density; its derivatives are the mean, variance and third central moment of that
    density. With both shapes two (the split normal), the tilted density is a mixture of two
    half normals, with :math:`N(t\sigma_R^2/2,\sigma_R^2/2)` truncated to :math:`x\geq 0` and
    :math:`N(t\sigma_L^2/2,\sigma_L^2/2)` truncated to :math:`x<0`, which gives the exact cgf

    .. math::
        K(t) = \log c + \log\left(\sigma_R J(t\sigma_R) + \sigma_L J(-t\sigma_L)\right),
        \qquad J(s) = \int_0^\infty e^{sy-y^2}dy = \sqrt{\pi}e^{s^2/4}\Phi(s/\sqrt{2}),

    and its derivatives from the moments of the truncated normals.
    A side with shape above one leaves the cgf finite in that direction, a side with shape one
    bounds the domain, :math:`t<1/\sigma_R` or :math:`t>-1/\sigma_L`. Shapes below one make the
    cgf infinite on that side, so no saddle point approximation exists. With both shapes one,
    this is :func:`asymmetric_laplace` with :math:`\kappa=\sqrt{\sigma_L/\sigma_R}` and scale
    :math:`\sqrt{\sigma_L\sigma_R}`, which is returned instead.

    References
    ----------
    [1] Zhu, Zinde-Walsh (2009) - Properties and estimation of asymmetric exponential power
        distribution
    """
    bl, br = np.broadcast_to(np.asanyarray(beta, float), 2)
    sl, sr = np.broadcast_to(np.asanyarray(sigma, float), 2)
    if not (bl >= 1 and br >= 1):
        raise ValueError("beta must be at least 1, otherwise the cgf is infinite on that side")
    if not (sl > 0 and sr > 0):
        raise ValueError("sigma must be positive")
    if bl == 1 and br == 1:
        return asymmetric_laplace(kappa=np.sqrt(sl / sr), loc=loc, scale=scale * np.sqrt(sl * sr))
    lognorm = -np.log(sl * np.exp(gammaln(1 + 1 / bl)) + sr * np.exp(gammaln(1 + 1 / br)))
    k = np.arange(1, 4)
    raw = np.exp(lognorm) * (
        sr ** (k + 1) * np.exp(gammaln((k + 1) / br)) / br
        + (-1) ** k * sl ** (k + 1) * np.exp(gammaln((k + 1) / bl)) / bl
    )
    cumulants = dict(
        dK0=raw[0],
        d2K0=raw[1] - raw[0] ** 2,
        d3K0=raw[2] - 3 * raw[0] * raw[1] + 2 * raw[0] ** 3,
    )
    if bl == 2 and br == 2:
        return _split_normal(sl, sr, lognorm, cumulants, loc=loc, scale=scale)

    # Generalized binomial coefficients C(b, k), k = 2, ..., 9, for the series of
    # (1 + r)^b - 1 - b r
    binom_coefs = {
        b: np.cumprod([b - j for j in range(9)])[1:] / np.cumprod(np.arange(1, 10))[1:]
        for b in {bl, br}
    }

    def h(t, x):
        # Log of the unnormalized tilted density
        return t * x - (x / sr) ** br if x >= 0 else t * x - (-x / sl) ** bl

    def tilted(t):
        # Integrate in v = (x - mode) / w, with w the width of the tilted density at its mode
        # and the exponent max subtracted, splitting at the mode and at the kink in x = 0
        if t > 0 and br > 1:
            mode = sr * (t * sr / br) ** (1 / (br - 1))
            w = np.sqrt(sr**br / (br * (br - 1) * mode ** (br - 2)))
        elif t < 0 and bl > 1:
            mode = -sl * (-t * sl / bl) ** (1 / (bl - 1))
            w = np.sqrt(sl**bl / (bl * (bl - 1) * (-mode) ** (bl - 2)))
        else:
            mode, w = 0.0, 1.0
        c = h(t, mode)

        def f(v):
            u = w * v
            if mode != 0 and u / mode > -1:
                # On the side of the mode, h(mode + u) - c = A (beta r - (1 + r)^beta + 1) with
                # r = u / mode and A = |mode / sigma|^beta, avoiding the cancellation of the
                # large terms t x and |x / sigma|^beta
                b, sig = (br, sr) if mode > 0 else (bl, sl)
                r = u / mode
                if abs(r) < 1e-2:
                    # Series, as the closed form cancels for small r
                    phi = -np.polyval(binom_coefs[b][::-1], r) * r**2
                else:
                    phi = b * r - np.expm1(b * np.log1p(r))
                logf = np.abs(mode / sig) ** b * phi
            else:
                logf = h(t, mode + u) - c
            return np.exp(logf) * v ** np.arange(4)

        # After rescaling the integrals are of order one, so an absolute tolerance is safe; it
        # is needed for pieces where the integrand underflows, which never converge otherwise.
        # Breakpoints a few widths from the mode keep long pieces, e.g., up to a far away kink,
        # from missing the mass when all their initial nodes underflow
        edges = [-np.inf, *sorted({-40.0, -10.0, 0.0, 10.0, 40.0, -mode / w}), np.inf]
        mom = sum(
            quad_vec(f, lo, hi, epsabs=1e-14, epsrel=1e-12)[0] for lo, hi in zip(edges, edges[1:])
        )
        m1, m2, m3 = mom[1:] / mom[0] * w ** np.arange(1, 4)
        return (
            lognorm + c + np.log(w * mom[0]),
            mode + m1,
            m2 - m1**2,
            m3 - 3 * m1 * m2 + 2 * m1**3,
        )

    def moment(i):
        return np.vectorize(lambda t: tilted(t)[i], otypes=[float])

    return UnivariateCumulantGeneratingFunction(
        K=moment(0),
        dK=moment(1),
        d2K=moment(2),
        d3K=moment(3),
        **cumulants,
        domain=Domain(g=-1 / sl if bl == 1 else None, l=1 / sr if br == 1 else None),
        loc=loc,
        scale=scale,
    )


def _split_normal(sl, sr, lognorm, cumulants, loc=0, scale=1):
    """
    Exact cgf of the split normal, see :func:`asymmetric_generalized_normal` with both shapes
    two: the tilted density is a mixture of two truncated normals.
    """

    def logJ(s):
        return 0.5 * np.log(np.pi) + s**2 / 4 + log_ndtr(s / np.sqrt(2))

    def half(t, sigma):
        # Mean, variance and third central moment of N(t sigma^2 / 2, sigma^2 / 2) truncated
        # to x >= 0, with lambda the inverse Mills ratio at the truncation point alpha
        sd = sigma / np.sqrt(2)
        alpha = -t * sigma**2 / 2 / sd
        lam = np.exp(-(alpha**2) / 2 - 0.5 * np.log(2 * np.pi) - log_ndtr(-alpha))
        return (
            t * sigma**2 / 2 + sd * lam,
            sd**2 * (1 + alpha * lam - lam**2),
            sd**3 * lam * (alpha**2 - 1 - 3 * alpha * lam + 2 * lam**2),
        )

    def parts(t):
        t = np.asanyarray(t, float)
        lr, ll = np.log(sr) + logJ(t * sr), np.log(sl) + logJ(-t * sl)
        logM = np.logaddexp(lr, ll)
        wr, wl = np.exp(lr - logM), np.exp(ll - logM)
        mr, vr, cr = half(t, sr)
        ml, vl, cl = half(-t, sl)
        ml, cl = -ml, -cl  # reflect the right half of -X to the left half of X
        m = wr * mr + wl * ml
        dr, dl = mr - m, ml - m
        v = wr * (vr + dr**2) + wl * (vl + dl**2)
        c3 = wr * (cr + 3 * vr * dr + dr**3) + wl * (cl + 3 * vl * dl + dl**3)
        return lognorm + logM, m, v, c3

    return UnivariateCumulantGeneratingFunction(
        K=lambda t: parts(t)[0],
        dK=lambda t: parts(t)[1],
        d2K=lambda t: parts(t)[2],
        d3K=lambda t: parts(t)[3],
        **cumulants,
        loc=loc,
        scale=scale,
    )


def univariate_sample_mean(cgf, sample_size):
    r"""
    Cumulant generating function of the sample mean of i.i.d. copies of a random variable.

    Given :math:`n` i.i.d. random variables :math:`X_1, \ldots, X_n` with
    cumulant generating function :math:`K`, the cumulant generating function
    of the sample mean :math:`\bar{X}` is given by

    .. math::
        K_{\bar{X}}(t) = \sum_{i=1}^n K(t/n) = n K(t/n),

    which is finite for :math:`t/n` in the domain of :math:`K`. The derivatives and
    ``dK_inv`` follow from those of ``cgf``: :math:`K_{\bar{X}}'(t)=K'(t/n)` and
    :math:`(K_{\bar{X}}')^{-1}(x)=n(K')^{-1}(x)`.

    Parameters
    ----------
    cgf : UnivariateCumulantGeneratingFunction
        The cgf :math:`K` of a single observation.
    sample_size : int
        The sample size :math:`n`, a positive integer.

    Returns
    -------
    UnivariateCumulantGeneratingFunction
        The cgf of :math:`\bar{X}`.

    Examples
    --------
    >>> import spapprox as spa
    >>> cgf = spa.univariate_sample_mean(spa.exponential(), 10)  # gamma(a=10, scale=0.1)
    >>> round(float(cgf.d2K0), 6)
    0.1

    See ``examples/example_parametric_bootstrap.py`` and
    ``examples/example_nonparametric_bootstrap.py``, and
    :class:`~spapprox.UnivariateSaddlePointApproxMean`.

    References
    ----------
    [1] Daniels (1954) - Saddlepoint approximations in statistics
    [2] Butler (2007) - Saddlepoint Approximations with Applications, Section 3.1
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
    r"""
    Empirical distribution of a sample: the cgf of drawing one of the observations
    :math:`x_1,\ldots,x_n` with equal probability,

    .. math::
        K(t) = \log\frac{1}{n}\sum_{i=1}^n e^{tx_i}, \qquad t\in\mathbb{R}.

    The cgf is that of the standardized sample, with ``loc`` and ``scale`` the sample mean and
    standard deviation (``scale`` is 1 for a constant sample), so numerical derivatives do not
    depend on the scale of the data. :math:`K` and its first three derivatives are exact
    (evaluated stably with ``logsumexp`` and ``softmax``): with the tilted weights
    :math:`w_i(t)\propto e^{tx_i}`, :math:`K'(t)=\sum_i w_i x_i`, and :math:`K''(t)` and
    :math:`K'''(t)` are the second and third central moments of the sample under these
    weights. The inverse of :math:`K'` is numerical. Note that the distribution is discrete, with support in
    :math:`[\min_i x_i, \max_i x_i]`.

    Combined with :func:`univariate_sample_mean` it approximates the bootstrap distribution
    of the sample mean without resampling (Davison and Hinkley, 1988).

    Parameters
    ----------
    x : array_like
        The observations.

    Returns
    -------
    UnivariateCumulantGeneratingFunction
        The cgf of the empirical distribution.

    Examples
    --------
    >>> import spapprox as spa
    >>> cgf = spa.univariate_empirical([1.0, 2.0, 6.0])
    >>> round(float(cgf.dK0), 6)
    3.0

    See ``examples/example_nonparametric_bootstrap.py``.

    References
    ----------
    [1] Davison, Hinkley (1988) - Saddlepoint approximations in resampling methods
    [2] Butler (2007) - Saddlepoint Approximations with Applications, Section 3.4
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

    def tilted_weights(t):
        tx = t * x if t.ndim == 0 else np.atleast_2d(t).T.dot(np.atleast_2d(x))
        return softmax(tx, axis=-1)

    # The derivatives are the mean and central moments of the exponentially tilted sample
    def tilted_central_moment(t, k):
        w = tilted_weights(t)
        m = (w * x).sum(axis=-1, keepdims=True)
        return (w * (x - m) ** k).sum(axis=-1)

    @type_wrapper(xloc=0)
    def dK(t):
        y = (tilted_weights(t) * x).sum(axis=-1)
        return y.tolist() if t.ndim == 1 else y

    @type_wrapper(xloc=0)
    def d2K(t):
        y = tilted_central_moment(t, 2)
        return y.tolist() if t.ndim == 1 else y

    @type_wrapper(xloc=0)
    def d3K(t):
        y = tilted_central_moment(t, 3)
        return y.tolist() if t.ndim == 1 else y

    return UnivariateCumulantGeneratingFunction(
        K, dK=dK, d2K=d2K, d3K=d3K, loc=loc, scale=scale
    )


def poisson(mu=1):
    r"""
    Poisson distribution with mean ``mu``, as ``scipy.stats.poisson(mu)``.

    The cgf is exact and in closed form,

    .. math::
        K(t) = \mu(e^t-1), \qquad t\in\mathbb{R}.

    There are no ``loc`` and ``scale`` parameters. The distribution is a lattice
    distribution with span 1; use the continuity-corrected cdf of the approximation classes
    for tail probabilities.

    Parameters
    ----------
    mu : float, optional
        Mean, positive. Default is 1.

    Returns
    -------
    UnivariateCumulantGeneratingFunction
        The cgf, with analytic derivatives and ``dK_inv``.

    Examples
    --------
    >>> import spapprox as spa
    >>> float(spa.poisson(3).d3K0)
    3.0

    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications, Section 1.2
    """
    return UnivariateCumulantGeneratingFunction(
        K=lambda t, mu=mu: mu * (np.exp(t) - 1),
        dK=lambda t, mu=mu: mu * np.exp(t),
        dK_inv=lambda x, mu=mu: np.log(x / mu),
        d2K=lambda t, mu=mu: mu * np.exp(t),
        d3K=lambda t, mu=mu: mu * np.exp(t),
    )


def binomial(n=1, p=0.5):
    r"""
    Binomial distribution with ``n`` trials and success probability ``p``, as
    ``scipy.stats.binom(n, p)``.

    The cgf is exact and in closed form,

    .. math::
        K(t) = n\log(1-p+pe^t), \qquad t\in\mathbb{R}.

    There are no ``loc`` and ``scale`` parameters. The distribution is a lattice
    distribution with span 1 and support :math:`\{0,\ldots,n\}`.

    Parameters
    ----------
    n : int, optional
        Number of trials. Default is 1.
    p : float, optional
        Success probability, in :math:`(0, 1)`. Default is 0.5.

    Returns
    -------
    UnivariateCumulantGeneratingFunction
        The cgf, with analytic derivatives and ``dK_inv``.

    Examples
    --------
    >>> import spapprox as spa
    >>> cgf = spa.binomial(n=10, p=0.3)
    >>> round(float(cgf.dK0), 6), round(float(cgf.d2K0), 6)
    (3.0, 2.1)

    References
    ----------
    [1] Butler (2007) - Saddlepoint Approximations with Applications, Section 1.2
    """
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
