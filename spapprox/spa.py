import itertools
import operator
import warnings
from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
from scipy.integrate import nquad, quad
from scipy.interpolate import RegularGridInterpolator
from statsmodels.tools.validation import PandasWrapper

from .cgfs import (
    MultivariateCumulantGeneratingFunction,
    UnivariateCumulantGeneratingFunction,
    univariate_sample_mean,
)
from .domain import Domain
from .mvn import _kolassa_orthant
from .util import fib, type_wrapper

try:
    import fastnorm
except ImportError:
    _has_fastnorm = False
else:
    _has_fastnorm = True


def _bvn_cdf(x, rho):
    """Standard bivariate normal cdf with correlation ``rho`` at the point ``x``."""
    if _has_fastnorm:
        return fastnorm.bivar_norm_cdf(x, rho)
    return sps.multivariate_normal([0, 0], np.array([[1, rho], [rho, 1]])).cdf(x)


def _lagrange_weights(nodes, x):
    """Weights of the Lagrange interpolating polynomial through ``nodes`` at ``x``."""
    nodes = np.asarray(nodes, dtype=float)
    return np.array(
        [
            np.prod([(x - nk) / (nj - nk) for k, nk in enumerate(nodes) if k != j])
            for j, nj in enumerate(nodes)
        ]
    )


def _bvn_tail_derivatives(b, c):
    r"""
    The tail probability :math:`T(\mathbf{b}) = P(\mathbf{Z}\geq\mathbf{b})` with
    :math:`\mathbf{Z}\sim N(0, A)`, :math:`A = [[1+c^2, c], [c, 1]]`, and its partial
    derivatives :math:`T_1, T_2, T_{11}, T_{12}, T_{111}, T_{112}` with respect to
    :math:`\mathbf{b}`.
    """
    s1 = np.sqrt(1 + c**2)
    g = c / s1
    T = _bvn_cdf(np.array([-b[0] / s1, -b[1]]), g)
    # Z2 | Z1 = b1 ~ N(c b1 / s1^2, 1 / s1^2), and Z1 | Z2 = b2 ~ N(c b2, 1)
    z = s1 * b[1] - g * b[0]
    p = sps.norm.pdf(b[0] / s1) / s1
    p1 = -b[0] / s1**2 * p
    p2 = (b[0] ** 2 / s1**4 - 1 / s1**2) * p
    phiz, sfz = sps.norm.pdf(z), sps.norm.sf(z)
    T1 = -p * sfz
    T2 = -sps.norm.pdf(b[1]) * sps.norm.sf(b[0] - c * b[1])
    T11 = -p1 * sfz - g * p * phiz
    T12 = s1 * p * phiz
    T111 = -p2 * sfz - 2 * g * p1 * phiz - g**2 * z * p * phiz
    T112 = s1 * p1 * phiz + g * s1 * z * p * phiz
    return T, T1, T2, T11, T12, T111, T112


class SaddlePointApprox(ABC):
    """
    Interface for saddle point approximation of probability density functions
    """

    # Half-width, in standardized units t * sqrt(K''(0)), of the window around t=0
    # in which cdf approximations are interpolated rather than evaluated directly.
    # The direct formulas suffer from catastrophic cancellation as t -> 0.
    _cdf_singular_width = 1e-2

    def __init__(self, cgf, pdf_normalization=None):
        assert isinstance(
            cgf, (UnivariateCumulantGeneratingFunction, MultivariateCumulantGeneratingFunction)
        )
        self.cgf = cgf
        self._pdf_normalization_user = pdf_normalization
        self._pdf_normalization_cache = pdf_normalization

    def pdf(self, x=None, t=None, normalize_pdf=True, fillna=np.nan, **solver_kwargs):
        r"""
        Saddle point approximation of the probability density function.
        Given by

        .. math::
            f(x) \approx \frac{1}{\sqrt{2\pi K''(t)}} \exp\left(K(t) - tx\right)

        where :math:`t` solves the saddle point equation.

        Parameters
        ----------
        x : array_like, optional (either x or t must be provided)
            The values at which to evaluate the probability density function.
        t : array_like, optional (either x or t must be provided)
            Solution of the saddle point equation. If not provided, it will be
            computed using numerical root finding.
        normalize_pdf : bool, optional
            Whether to normalize the probability density function. Default is
            True.
        fillna : float, optional
            The value to replace NaNs with.
        """
        assert x is not None or t is not None
        if x is None:
            x = self.cgf.dK(t)
        elif t is None:
            t = self._dK_inv(x, **solver_kwargs)
        wrapper = PandasWrapper(x)
        y = np.asanyarray(self._spapprox_pdf(np.asanyarray(x), np.asanyarray(t)))
        if normalize_pdf:
            y *= 1 / self._pdf_normalization
        y = np.where(np.isnan(y), fillna, y)
        return y.tolist() if y.ndim == 0 else wrapper.wrap(y)

    def cdf(self, *args, x=None, t=None, fillna=np.nan, **solver_kwargs):
        raise NotImplementedError(f"CDF not implemented for {self.__class__.__name__}")

    def ppf(self, q, *args, fillna=np.nan, t0=None, ttol=1e-4, **kwargs):
        raise NotImplementedError(f"PPF not implemented for {self.__class__.__name__}")

    @type_wrapper(xloc=1)
    def _dK_inv(self, x, **solver_kwargs):
        return self.cgf.dK_inv(x, **solver_kwargs)

    @property
    @abstractmethod
    def _pdf_normalization(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self):
        raise NotImplementedError

    def clear_cache(self):
        """
        Clear any information stored from fit functions
        """
        # Use vars, as inspect.getmembers would evaluate (expensive) properties
        for attr in [k for k in vars(self) if k.endswith("_cache")]:
            delattr(self, attr)
        # Keep a user-supplied pdf normalization
        if getattr(self, "_pdf_normalization_user", None) is not None:
            self._pdf_normalization_cache = self._pdf_normalization_user


class UnivariateSaddlePointApprox(SaddlePointApprox):
    """
    Given the cumulant generating function of a univariate random variable, this class
    provides the saddle point approximation of the probability density function
    and the cumulative distribution function.

    Parameters
    ----------
    cgf : UnivariateCumulantGeneratingFunction
        The cumulant generating function of the random variable.
    pdf_normalization : float, optional
        The normalization constant of the probability density function. If not
        provided, it will be computed using numerical integration.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Saddlepoint_approximation_method

    [2] Butler, R. W. (2007). Saddlepoint approximations with applications.

    [3] Kuonen, D. (2001). Computer-intensive statistical methods: Saddlepoint approximations in bootstrap and inference.
    """

    def __init__(self, cgf, pdf_normalization=None):
        assert isinstance(cgf, UnivariateCumulantGeneratingFunction)
        super().__init__(cgf, pdf_normalization)

    @type_wrapper(xloc=1)
    def _spapprox_pdf(self, x, t, fillna=np.nan):
        t = np.asanyarray(t)
        d2Kt = self.cgf.d2K(t, fillna=fillna)
        with np.errstate(divide="ignore", invalid="ignore"):
            retval = np.where(
                d2Kt > 0,
                np.exp(self.cgf.K(t) - t * x) * np.sqrt(np.divide(1, 2 * np.pi * d2Kt)),
                fillna,
            )
        return np.where(np.isnan(retval), fillna, retval)

    def _lattice_u(self, t, delta=None):
        r"""
        :math:`u = t\sqrt{K''(t)}`, or for a lattice with span ``delta`` the Daniels (1987)
        version :math:`2\sinh(t\delta/2)\sqrt{K''(t)}/\delta`.
        """
        if delta is None:
            return t * np.sqrt(self.cgf.d2K(t))
        return 2 * np.sinh(t * delta / 2) / delta * np.sqrt(self.cgf.d2K(t))

    @type_wrapper(xloc=1)
    def _spapprox_cdf_LR(self, x, t, fillna=np.nan, delta=None):
        r"""
        Lugannani-Rice saddle point approximation of the cumulative distribution as given by

        .. math::
            F(x) \approx \Phi(w) + \phi(w) \left(\frac{1}{w} - \frac{1}{u}\right)

        where :math:`\Phi` and :math:`\phi` are the cumulative and probability density
        of the standard normal distribution, respectively, and :math:`w` and :math:`u`
        are given by

        .. math::
            w = \text{sign}(t)\sqrt{2\left(tx - K(t)\right)},

        .. math::
            u = t\sqrt{K''(t)}.

        For :math:`t = 0`, the approximation is given by

        .. math::
            F(x) \approx \frac{1}{2} + \frac{K'''(0)}{6\sqrt{2\pi K''(0)^3}}.

        Reference
        ---------
        [1] Lugannani, R., & Rice, S. (1980). Saddle point approximation for the distribution of the sum of independent random variables. Advances in applied probability.

        [2] Kuonen, D. (2001). Computer-intensive statistical methods: Saddlepoint approximations in bootstrap and inference.
        """
        rho3 = self.cgf.d3K0 / np.power(self.cgf.d2K0, 3 / 2)
        retval = self._interpolate_cdf_near_zero(
            lambda x, t: self._lugannani_rice(x, t, delta),
            x,
            t,
            1 / 2 + rho3 / 6 / np.sqrt(2 * np.pi),
        )
        return np.where(np.isnan(retval), fillna, retval)

    def _lugannani_rice(self, x, t, delta=None):
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.sign(t) * np.sqrt(2 * (t * x - self.cgf.K(t)))
            u = self._lattice_u(t, delta)
            return sps.norm.cdf(w) + sps.norm.pdf(w) * (1 / w - 1 / u)

    @type_wrapper(xloc=1)
    def _spapprox_cdf_BN(self, x, t, fillna=np.nan, delta=None):
        r"""
        This is an alternative implementation of the Lugannani-Rice saddle point
        approximation of the cumulative distribution.

        It is given by

        .. math::
            F(x) \approx \Phi(w + \log(u/w)/w)

        where :math:`\Phi` is the cumulative distribution of the standard normal.

        References
        ----------
        [1] Barndorff-Nielsen, O. E. (1986). Inference on full or partial parameters based on the standardized signed log likelihood ratio. Biometrika.

        [2] Bandorff-Nielsen, O. E. (1990) Approximate interval probabilities. Journal of the Royal Statistical Society. Series B (Methodological).

        [3] Kuonen, D. (2001). Computer-intensive statistical methods: Saddlepoint approximations in bootstrap and inference.
        """
        rho3 = self.cgf.d3K0 / np.power(self.cgf.d2K0, 3 / 2)
        retval = self._interpolate_cdf_near_zero(
            lambda x, t: self._barndorff_nielsen(x, t, delta), x, t, sps.norm.cdf(rho3 / 6)
        )
        return np.where(np.isnan(retval), fillna, retval)

    def _barndorff_nielsen(self, x, t, delta=None):
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.sign(t) * np.sqrt(2 * (t * x - self.cgf.K(t)))
            u = self._lattice_u(t, delta)
            return sps.norm.cdf(w + np.log(u / w) / w)

    def _interpolate_cdf_near_zero(self, formula, x, t, limit):
        r"""
        Evaluate ``formula(x, t)``, but replace values with :math:`|t|<h` by quadratic
        interpolation through :math:`(-h, F(-h))`, :math:`(0, \text{limit})` and
        :math:`(h, F(h))`, where :math:`F` is the formula as a function of :math:`t`
        and :math:`h` is ``_cdf_singular_width`` in standardized units.
        """
        x, t = np.asanyarray(x), np.asanyarray(t)
        retval = formula(x, t)
        h = self._cdf_singular_width / np.sqrt(self.cgf.d2K0)
        near = np.abs(t) < h
        if np.any(near):
            nodes = np.array([-h, h])
            fm, fp = formula(self.cgf.dK(nodes), nodes)
            interp = (
                limit + (fp - fm) / (2 * h) * t + (fp - 2 * limit + fm) / (2 * h**2) * np.square(t)
            )
            retval = np.where(near, interp, retval)
        return retval

    @property
    def dim(self):
        return 1

    @property
    def _pdf_normalization(self):
        r"""
        Integral of the unnormalized pdf, computed in terms of :math:`t` as
        :math:`\int \hat f(K'(t)) K''(t) dt`.

        The bulk is integrated over the inferred :math:`t` range. On each side where the cdf
        did reach 0 (or 1), the remaining tail is integrated up to the domain bound, so that
        the result is not biased by the truncation. Elsewhere, e.g., for lattice
        distributions, the tail is ignored, as the approximation becomes unreliable there.
        """
        if not hasattr(self, "_pdf_normalization_cache") or self._pdf_normalization_cache is None:

            def integrand(t):
                return self.pdf(t=t, normalize_pdf=False, fillna=0) * self.cgf.d2K(t, fillna=0)

            a, b, a_converged, b_converged = self._infer_t_range()
            val = quad(integrand, a=a, b=b)[0]
            lower, upper = self._t_domain_bounds()
            if a_converged and lower < a:
                val += quad(integrand, a=lower, b=a)[0]
            if b_converged and b < upper:
                val += quad(integrand, a=b, b=upper)[0]
            assert not np.isnan(val) and np.isfinite(val), ()
            self._pdf_normalization_cache = val
        return self._pdf_normalization_cache

    def pdf(self, x=None, t=None, normalize_pdf=None, fillna=np.nan, delta=None, **solver_kwargs):
        r"""
        Saddle point approximation of the probability density function, see
        :meth:`SaddlePointApprox.pdf`.

        Parameters
        ----------
        normalize_pdf : bool, optional
            Whether to normalize the density by integrating it. Default is ``None``: ``True``
            for a continuous variable, ``False`` if ``delta`` is given, as the integral of the
            density is a poor normalization for a lattice variable.
        delta : float, optional
            Span of the lattice, for a discrete random variable. Default is ``None``:
            continuous. Otherwise, the probability mass function :math:`P(X=x)\approx
            \delta f(x)` is returned, for ``x`` on the lattice. This is the saddle point
            approximation of Daniels (1983, 1987).
        """
        if delta is not None and not delta > 0:
            raise ValueError("delta must be positive")
        if normalize_pdf is None:
            normalize_pdf = delta is None
        y = super().pdf(x=x, t=t, normalize_pdf=normalize_pdf, fillna=fillna, **solver_kwargs)
        return y if delta is None else y * delta

    def cdf(self, x=None, t=None, fillna=np.nan, backend="LR", delta=None, **solver_kwargs):
        r"""
        Saddle point approximation of the cumulative distribution function.

        The standard Lugannani-Rice approximation is given by

        .. math::
            F(x) \approx \Phi(w) + \phi(w) \left(\frac{1}{w} - \frac{1}{u}\right)

        where :math:`\Phi` and :math:`\phi` are the cumulative and probability density
        of the standard normal distribution, respectively, and :math:`w` and :math:`u`
        are given by

        .. math::
            w = \text{sign}(t)\sqrt{2\left(tx - K(t)\right)},

        .. math::
            u = t\sqrt{K''(t)}.

        For :math:`t = 0`, the approximation is given by

        .. math::
            F(x) \approx \frac{1}{2} + \frac{K'''(0)}{6\sqrt{2\pi K''(0)^3}}.


        The alternative Barndorff-Nielsen approximation is given by

        .. math::
            F(x) \approx \Phi(w + \log(u/w)/w)

        where :math:`\Phi` is the cumulative distribution of the standard normal.
        For :math:`t = 0`, it is given by its limit

        .. math::
            F(x) \approx \Phi\left(\frac{K'''(0)}{6 K''(0)^{3/2}}\right).

        Close to :math:`t=0` both approximations suffer from catastrophic cancellation,
        so they are quadratically interpolated between the limit and the values at
        :math:`t=\pm h`, with :math:`h` given by ``_cdf_singular_width`` in
        standardized units :math:`t\sqrt{K''(0)}`.

        Parameters
        ----------
        x : array_like, optional (either x or t must be provided)
            The values at which to evaluate the cumulative distribution function.
        t : array_like, optional (either x or t must be provided)
            Solution of the saddle point equation. If not provided, it will be
            computed using numerical root finding.
        fillna : float, optional
            The value to replace NaNs with.
        backend : str, optional
            The backend to use for the computation. Either 'LR' for Lugannani-Rice approximation
            or 'BN' for Barndorff-Nielsen approximation. Default is 'LR'.
        delta : float, optional
            Span of the lattice, for a discrete random variable. Default is ``None``:
            continuous. Otherwise ``x`` must lie on the lattice and the second continuity
            correction of Daniels (1987), see Butler (2007), Section 1.2.3, is used:
            :math:`P(X\leq x)` is approximated at :math:`x+\delta/2`, using
            :math:`u = 2\sinh(t\delta/2)\sqrt{K''(t)}/\delta`. If only ``t`` is
            given, it must solve the saddle point equation at :math:`x+\delta/2`.
        """
        assert x is not None or t is not None
        if delta is not None and not delta > 0:
            raise ValueError("delta must be positive")
        if x is None:
            x = self.cgf.dK(t)
            wrapper = PandasWrapper(x)
        else:
            wrapper = PandasWrapper(x)
            if delta is not None:
                x = np.asanyarray(x) + delta / 2
            if t is None:
                t = self._dK_inv(x, **solver_kwargs)
        if backend == "LR":
            y = np.asanyarray(
                self._spapprox_cdf_LR(np.asanyarray(x), np.asanyarray(t), delta=delta)
            )
        elif backend == "BN":
            y = np.asanyarray(
                self._spapprox_cdf_BN(np.asanyarray(x), np.asanyarray(t), delta=delta)
            )
        else:
            raise ValueError("backend must be either 'LR' or 'BN'")
        y = np.where(np.isnan(y), fillna, y)
        return y.tolist() if y.ndim == 0 else wrapper.wrap(y)

    @type_wrapper(xloc=1)
    def ppf(self, q, fillna=np.nan, t0=None, ttol=1e-4, **kwargs):
        r"""
        Percent point function computed as the inverse of the saddle point
        approximation of the cumulative distribution function.

        When available, the ppf interpolated. The method is based on
        a custom implementation. Probably we could have also used the
        Scipy's sampling pinv method.

        Parameters
        ----------
        q : array_like
            The values at which to evaluate the inverse cumulative distribution function.
        fillna : float, optional
            The value to replace NaNs with.
        """
        assert np.all((0 <= np.asanyarray(q)) & (np.asanyarray(q) <= 1))
        if hasattr(self, "_cdf_cache") and hasattr(self, "_ppf_t_cache"):
            t = np.interp(q, self._cdf_cache, self._ppf_t_cache)
        else:
            if q.ndim == 0:  # Then it is a scalar "array"
                q = q.tolist()
                kwargs["x0"] = 0 if t0 is None else t0
                bracket_methods = [
                    "bisect",
                    "brentq",
                    "brenth",
                    "ridder",
                    "toms748",
                ]
                if "method" in kwargs:
                    methods = [kwargs["method"]]
                else:
                    methods = [
                        "newton",
                        "secant",
                        "bisect",
                        "brentq",
                        "brenth",
                        "ridder",
                        "toms748",
                    ]
                for method in methods:
                    kwargs["method"] = method
                    if method in bracket_methods and "bracket" not in kwargs:
                        # find valid lb and ub
                        lb = next(
                            -1 * 0.9**i
                            for i in range(10000)
                            if ~np.isnan(self.cgf.dK(-1 * 0.9**i))
                        )
                        ub = next(
                            1 * 0.9**i for i in range(10000) if ~np.isnan(self.cgf.dK(1 * 0.9**i))
                        )
                        cdf_lb = self.cdf(x=self.cgf.dK(lb), t=lb, fillna=np.nan)
                        cdf_ub = self.cdf(x=self.cgf.dK(ub), t=ub, fillna=np.nan)
                        assert lb < ub and cdf_lb < cdf_ub, "cdf is assumed to be increasing"
                        lb_scalings = (1 - 1 / fib(i) for i in range(3, 100))
                        ub_scalings = (1 - 1 / fib(i) for i in range(3, 100))
                        lb_scaling = next(lb_scalings)
                        ub_scaling = next(ub_scalings)
                        # find lb through iterative scaling
                        while q < cdf_lb:
                            lb_new = lb / lb_scaling
                            x_new = self.cgf.dK(lb_new)
                            if not np.isnan(x_new):
                                lb = lb_new
                                cdf_lb = self.cdf(x=x_new, t=lb, fillna=np.nan)
                                continue
                            try:
                                lb_scaling = next(lb_scalings)
                            except StopIteration:
                                raise Exception("Could not find valid lb")
                        # find ub through iterative scaling
                        while q > cdf_ub:
                            ub_new = ub / ub_scaling
                            x_new = self.cgf.dK(ub_new)
                            if not np.isnan(x_new):
                                ub = ub_new
                                cdf_ub = self.cdf(x=x_new, t=ub, fillna=np.nan)
                                continue
                            try:
                                ub_scaling = next(ub_scalings)
                            except StopIteration:
                                raise Exception("Could not find valid ub")
                        assert cdf_lb <= q <= cdf_ub
                        kwargs["bracket"] = [lb, ub]
                    try:
                        res = spo.root_scalar(lambda t: self.cdf(t=t) - q, **kwargs)
                    except Exception:
                        continue
                    if res.converged and np.isclose(self.cdf(t=res.root), q, atol=ttol):
                        break
                else:
                    raise Exception("Failed to solve the saddle point equation.")
                t = np.asanyarray(res.root)
            else:
                kwargs["x0"] = np.zeros(q.shape) if t0 is None else np.asanyarray(t0)
                if "method" in kwargs:
                    methods = [kwargs["method"]]
                else:
                    methods = [
                        "hybr",
                        "lm",
                        "broyden1",
                        "broyden2",
                        "anderson",
                        "linearmixing",
                        "diagbroyden",
                        "excitingmixing",
                        "krylov",
                        "df-sane",
                    ]
                for method in methods:
                    kwargs["method"] = method
                    with np.errstate(invalid="ignore"):
                        try:
                            res = spo.root(lambda t, q=q: self.cdf(t=t) - q, **kwargs)
                        except Exception:
                            continue
                    if res.success and np.allclose(self.cdf(t=res.x), q, atol=ttol):
                        t = np.asanyarray(res.x)
                        break
                else:
                    t = np.asanyarray(
                        [self.ppf(qq, t0=None if t0 is None else t0[i]) for i, qq in enumerate(q)]
                    )
        return self.cgf.dK(t)

    def _t_domain_bounds(self):
        """
        Lower and upper bounds of the domain of the cumulant generating function in terms of
        :math:`t`, possibly infinite. Linear inequality constraints are not taken into account.
        """
        domain = self.cgf.domain.mul(1 / self.cgf.scale)
        lower = max(
            (np.nanmax(v) for v in (domain.g, domain.ge) if v is not None), default=-np.inf
        )
        upper = min((np.nanmin(v) for v in (domain.l, domain.le) if v is not None), default=np.inf)
        return lower, upper

    def infer_t_range(self, atol=1e-4, rtol=1e-4, max_iter=500):
        """
        Infers a suitable range for so that it covers roughly the entire probability mass

        The bounds are expanded until the cdf is close to zero (lower bound) or one (upper
        bound). For lattice or bounded-support distributions the saddle point cdf may never
        get there: it can turn around, leave :math:`[0, 1]`, or become ``nan``. Such a step is
        treated as the end of the usable range, and the last valid bound is returned with a
        warning.
        """
        lb, ub, _, _ = self._infer_t_range(atol=atol, rtol=rtol, max_iter=max_iter)
        return [lb, ub]

    def _infer_t_range(self, atol=1e-4, rtol=1e-4, max_iter=500):
        """
        Implements :meth:`infer_t_range`, but also returns, per bound, whether the cdf reached
        its target.
        """
        # Determine initial range
        lb = next(-1 * 0.9**i for i in range(10000) if ~np.isnan(self.cgf.dK(-1 * 0.9**i)))
        ub = next(1 * 0.9**i for i in range(10000) if ~np.isnan(self.cgf.dK(1 * 0.9**i)))
        cdf_lb = self.cdf(x=self.cgf.dK(lb), t=lb, fillna=np.nan)
        cdf_ub = self.cdf(x=self.cgf.dK(ub), t=ub, fillna=np.nan)
        if not (lb < ub and 0 <= cdf_lb < cdf_ub <= 1):
            raise ValueError(
                "Could not infer a t range: the cdf is not valid and increasing at the initial "
                f"bounds t={lb}, t={ub} (cdf {cdf_lb}, {cdf_ub})"
            )

        def expand(bound, cdf_bound, target):
            # Expand bound (away from zero) until the cdf is close to target; a step is only
            # accepted if the cdf stays valid and moves towards target
            scalings = (1 - 1 / fib(i) for i in range(3, 100))
            scaling = next(scalings)
            for _ in range(max_iter):
                if np.isclose(cdf_bound, target, atol=atol, rtol=rtol):
                    return bound, True
                new = bound / scaling
                x_new = self.cgf.dK(new)
                if np.isfinite(x_new):
                    cdf_new = self.cdf(x=x_new, t=new, fillna=np.nan)
                    if 0 <= cdf_new <= 1 and abs(cdf_new - target) < abs(cdf_bound - target):
                        bound, cdf_bound = new, cdf_new
                        continue
                try:
                    scaling = next(scalings)
                except StopIteration:
                    break
            warnings.warn(
                f"Could not expand t range until cdf={target}, stopped at t={bound} with "
                f"cdf={cdf_bound}; the saddle point cdf may not reach {target}, e.g., for "
                "lattice or bounded-support distributions",
                RuntimeWarning,
                stacklevel=4,
            )
            return bound, False

        lb, lb_converged = expand(lb, cdf_lb, 0)
        ub, ub_converged = expand(ub, cdf_ub, 1)
        assert lb <= 0 <= ub
        return lb, ub, lb_converged, ub_converged

    def fit_saddle_point_eqn(self, t_range=None, atol=1e-4, rtol=1e-4, num=1000, **solver_kwargs):
        """
        Evaluate the saddle point equation to the given range of values.
        And use linear interpolation to solve the saddle point equation, form now onwards.

        If t_range is not provided, it will be assumed that the cumulant generating
        function is defined in a neighborhood of zero. The range will be computed
        by scaling iteratively, until we little mass for higher or lower values of x.
        """
        if t_range is None:
            if hasattr(self, "_t_cache"):
                t_range = [self._t_cache[0], self._t_cache[-1]]
            else:
                t_range = self.infer_t_range(atol=atol, rtol=rtol)
        # Solve saddle point equation
        self._x_cache = np.linspace(*self.cgf.dK(t_range), num=num)
        self._t_cache = self.cgf.dK_inv(self._x_cache, **solver_kwargs)

    def fit_ppf(self, t_range=None, atol=1e-4, rtol=1e-4, num=1000):
        """
        Fit the inverse of the cumulative distribution function using linear interpolation.

        If t_range is not provided, it is taken from a previous fit (of the ppf or the
        saddle point equation), or inferred otherwise.
        """
        if t_range is None:
            if hasattr(self, "_ppf_t_cache"):
                t_range = [self._ppf_t_cache[0], self._ppf_t_cache[-1]]
            elif hasattr(self, "_t_cache"):
                t_range = [self._t_cache[0], self._t_cache[-1]]
            else:
                t_range = self.infer_t_range(atol=atol, rtol=rtol)
        x_range = np.linspace(*self.cgf.dK(t_range), num=num)
        t = self.cgf.dK_inv(x_range)
        self._cdf_cache = self.cdf(x=x_range, t=t)
        self._ppf_t_cache = t

    @type_wrapper(xloc=1)
    def _dK_inv(self, x, **solver_kwargs):
        if hasattr(self, "_x_cache") and hasattr(self, "_t_cache"):
            x = np.asanyarray(x, dtype=float)
            t = np.atleast_1d(
                np.interp(x, self._x_cache, self._t_cache, left=np.nan, right=np.nan)
            )
            # Outside the fitted grid, fall back to the root solver instead of clamping
            outside = np.atleast_1d((x < self._x_cache[0]) | (x > self._x_cache[-1]))
            if np.any(outside):
                t[outside] = super()._dK_inv(np.atleast_1d(x)[outside], **solver_kwargs)
            return t.reshape(x.shape)
        else:
            return super()._dK_inv(x, **solver_kwargs)


class UnivariateSaddlePointApproxMean(UnivariateSaddlePointApprox):
    r"""
    Given the cumulant generating function of a univariate random variable, this class
    provides the saddle point approximation of the sample mean of the random variable.

    Given :math:`n` i.i.d. random variables :math:`X_1, \ldots, X_n` with
    cumulant generating function :math:`K`, the cumulant generating function :math:`K_n`
    of the sample mean :math:`\bar{X}` is given by

    .. math::
        K_{\bar{X}}(t) = \sum_{i=1}^n 1/n*K_i(t)= \sum_{i=1}^n K_i(t/n) = n K(t/n).

    Parameters
    ----------
    cgf : UnivariateCumulantGeneratingFunction
        The cumulant generating function of the random variable.
    sample_size : int
        The sample size on which the sample mean is to be estimated.
    pdf_normalization : float, optional
        The normalization constant of the probability density function. If not
        provided, it will be computed using numerical integration.
    """

    def __init__(self, cgf, sample_size, pdf_normalization=None):
        assert isinstance(cgf, UnivariateCumulantGeneratingFunction)
        self.sample_size = sample_size
        cgf = univariate_sample_mean(cgf, sample_size)
        super().__init__(cgf, pdf_normalization=pdf_normalization)


class MultivariateSaddlePointApprox(SaddlePointApprox):
    r"""
    Given the cumulant generating function of a random vector, this class
    provides the saddle point approximation of the probability density function,
    and of the cumulative distribution function, see :meth:`_kolassa_cdf`.

    The multivariate saddle point approximation is given by

    .. math::
        f(\mathbf{x}) \approx \frac{1}{2\pi\sqrt{\left|\det H\right|}}
        \exp\left(K(\mathbf{t}) - \mathbf{x}\cdot\mathbf{t}\right)


    where :math:`H` is the Hessian matrix of the cumulant generating function,
    and :math:`\mathbf{t}` solves the saddle point equation :math:`K(\mathbf{t})=x`.

    We follow the approach as outlined in [4] Section 5.2.


    See the Kolasse book on how to remove singularities
    DasGupta claims the multidimensional case is too notationally complex

    References
    ----------
    [1] Kolassa (2006) - Series approximations in statistics
    [2] DasGupta (2008) - Asymptotic theory of statistics and probability, Chapter 14.9
    [3] Reid (1988) - Saddlepoint methods and statistical inference
    [4] Gatto (2000) - Symbolic computation for approximating distributions of some families of one and two-sample nonparametric test statistics
    [5] Gatto (2019) - Saddlepoint approximations for data in simplices
    [6] Kolassa (2003) - Multivariate saddlepoint tail probability approximations
    """

    def __init__(self, cgf, pdf_normalization=None):
        assert isinstance(cgf, MultivariateCumulantGeneratingFunction)
        self.cgf = cgf
        self._pdf_normalization_user = pdf_normalization
        self._pdf_normalization_cache = pdf_normalization

    def __getitem__(self, item):
        r"""
        Saddle point approximation of a marginal distribution.

        The cumulant generating function of the marginal follows by setting the other
        components of :math:`\mathbf{t}` to zero, see [1] Section 6.8.

        Parameters
        ----------
        item : int, list, tuple or slice
            An integer selects a single variable and returns a
            :class:`UnivariateSaddlePointApprox`. A list, tuple or slice selects a
            subvector, in the given order, and returns a
            :class:`BivariateSaddlePointApprox` for two variables, and a
            :class:`MultivariateSaddlePointApprox` otherwise.
        """
        if isinstance(item, slice):
            idx = list(range(self.dim))[item]
        elif isinstance(item, (list, tuple, np.ndarray)):
            idx = [self._normalize_index(i) for i in item]
            if len(set(idx)) != len(idx):
                raise IndexError("Indices must be unique")
        else:
            idx = self._normalize_index(item)
        key = idx if isinstance(idx, int) else tuple(idx)
        if not hasattr(self, "_marginals_cache"):
            self._marginals_cache = {}
        if key not in self._marginals_cache:
            if isinstance(idx, int):
                spa = UnivariateSaddlePointApprox(self.cgf[idx])
            elif len(idx) == 0:
                raise IndexError("Select at least one variable")
            elif len(idx) == 2:
                spa = BivariateSaddlePointApprox(self.cgf[idx])
            else:
                spa = MultivariateSaddlePointApprox(self.cgf[idx])
            self._marginals_cache[key] = spa
        return self._marginals_cache[key]

    def _normalize_index(self, i):
        try:
            i = operator.index(i)
        except TypeError:
            raise TypeError(f"Invalid index {i!r}") from None
        if not -self.dim <= i < self.dim:
            raise IndexError(f"Index {i} out of bounds for dimension {self.dim}")
        return i % self.dim

    def condition(self, i, x):
        raise NotImplementedError("Condition the approximation on the ith variable")

    # Half-width, in conditional standard deviations of the variable of interest, of the
    # window around the singular point of the conditional cdf in which it is interpolated.
    # The second order terms lose accuracy closer to it than about 2e-2 for large n.
    _conditional_singular_width = 2.5e-2

    def _conditional_inputs(self, x, given, i):
        """Validate the input of the conditional methods, return (i, J, x, given)."""
        i = self._normalize_index(i)
        J = [j for j in range(self.dim) if j != i]
        if not J:
            raise ValueError("Conditioning requires at least a two dimensional distribution")
        x = np.asarray(x, dtype=float)
        given = np.asarray(given, dtype=float)
        if given.ndim == 1:
            if len(given) != len(J):
                raise ValueError(f"given must have {len(J)} components")
            given = np.broadcast_to(given, (x.size, len(J)))
        elif given.ndim == 2 and given.shape == (x.size, len(J)):
            pass
        else:
            raise ValueError(f"given must have shape ({len(J)},) or ({x.size}, {len(J)})")
        return i, J, x, given

    def _conditional_denominator_saddle(self, i, J, xJ, **solver_kwargs):
        """Saddlepoint with component i fixed at zero, solving dK_J = x_J."""
        if len(J) == 1:
            tJ = np.atleast_1d(self.cgf[J[0]].dK_inv(xJ[0], **solver_kwargs))
        else:
            tJ = np.asarray(self.cgf[J].dK_inv(xJ, **solver_kwargs))
        tau0 = np.zeros(self.dim)
        tau0[J] = tJ
        return tau0

    def _conditional_terms(self, xfull, i, J, tau0, **solver_kwargs):
        r"""
        Return :math:`(\hat w, \hat\tau_i, \hat\sigma)` with
        :math:`\hat\sigma=\sqrt{\det K''(\hat\tau)/\det K_{JJ}''(\tau_0)}`, so that
        :math:`\hat u=\hat\tau_i\hat\sigma`, at ``xfull``, with :math:`\tau_0` the
        denominator saddlepoint.
        """
        tau = np.asarray(self.cgf.dK_inv(xfull, **solver_kwargs), dtype=float)
        H = np.asarray(self.cgf.d2K(tau))
        H0 = np.asarray(self.cgf.d2K(tau0))
        w2 = 2 * (tau @ xfull - self.cgf.K(tau) - tau0 @ xfull + self.cgf.K(tau0))
        sign, logdet = np.linalg.slogdet(H)
        sign0, logdet0 = np.linalg.slogdet(H0[np.ix_(J, J)])
        if not (np.isfinite(w2) and sign > 0 and sign0 > 0):
            return np.nan, np.nan, np.nan
        # Rounding can push w2 slightly below zero close to the singularity
        w = np.sign(tau[i]) * np.sqrt(max(w2, 0.0))
        return w, tau[i], np.exp((logdet - logdet0) / 2)

    def _d4K_central(self, t):
        """
        Fourth derivative tensor of K as a finite difference of the analytic ``d3K``.

        A fourth order stencil is used, as the truncation error of a three point stencil is
        amplified close to the singularity of the conditional cdf. Close to the boundary of
        the domain, it falls back to a three point or one-sided difference.
        """
        t = np.asarray(t, dtype=float)
        T4 = np.empty((self.dim,) * 4)
        for m in range(self.dim):
            h = 1e-3 * max(1.0, abs(t[m]))
            e = np.zeros(self.dim)
            e[m] = h
            f = {j: np.asarray(self.cgf.d3K(t + j * e)) for j in (-2, -1, 1, 2)}
            if all(np.isfinite(v).all() for v in f.values()):
                T4[..., m] = (8 * (f[1] - f[-1]) - (f[2] - f[-2])) / (12 * h)
                continue
            mid = np.asarray(self.cgf.d3K(t))
            up = f[1] if np.isfinite(f[1]).all() else mid
            down = f[-1] if np.isfinite(f[-1]).all() else mid
            T4[..., m] = (up - down) / (((up is not mid) + (down is not mid)) * h)
        return T4

    def _conditional_invariants(self, t, J=None):
        r"""
        Return :math:`(\rho_4,\rho_{13}^2,\rho_{23}^2)` of Kolassa (1996) of the
        cumulant generating function at ``t``, restricted to the components ``J`` if given.
        """
        H = np.asarray(self.cgf.d2K(t))
        T3 = np.asarray(self.cgf.d3K(t))
        T4 = self._d4K_central(t)
        if J is not None:
            H, T3, T4 = H[np.ix_(J, J)], T3[np.ix_(J, J, J)], T4[np.ix_(J, J, J, J)]
        Hi = np.linalg.inv(H)
        rho4 = np.einsum("ijkl,ij,kl->", T4, Hi, Hi)
        rho13 = np.einsum("ijk,lmn,ij,kl,mn->", T3, T3, Hi, Hi, Hi)
        rho23 = np.einsum("ijk,lmn,il,jm,kn->", T3, T3, Hi, Hi, Hi)
        return rho4, rho13, rho23

    def _conditional_second_order(self, xfull, i, J, tau0, delta=None, **solver_kwargs):
        r"""
        Return :math:`(\hat w, \hat z, c)`, with :math:`c` the bracket of Kolassa (1996),
        Theorem 6.1, eq. (10), so that the survival function is
        :math:`1-\Phi(\hat w)+\phi(\hat w)(\hat w^{-3}-\hat w^{-1}+\hat z^{-1}c)`.
        With a lattice span ``delta``, eq. (11) is used, in which :math:`\hat z` and the last
        two terms of :math:`c` are replaced. :math:`\hat z` carries a factor :math:`1/\delta`
        compared to the paper, which makes the result invariant to rescaling the lattice.
        """
        w, tau_i, sigma = self._conditional_terms(xfull, i, J, tau0, **solver_kwargs)
        if np.isnan(w):
            return np.nan, np.nan, np.nan
        tau = np.asarray(self.cgf.dK_inv(xfull, **solver_kwargs), dtype=float)
        Hi = np.linalg.inv(np.asarray(self.cgf.d2K(tau)))
        T3 = np.asarray(self.cgf.d3K(tau))
        hat = np.array(self._conditional_invariants(tau))
        # The denominator is the density of X_J, so its invariants are those of the marginal
        tilde = np.array(self._conditional_invariants(tau0, J))
        d4, d13, d23 = hat - tilde
        skew = np.einsum("k,ijk,ij->", Hi[i], T3, Hi)
        c = 1 + d4 / 8 - d13 / 8 - d23 / 12
        if delta is None:
            return w, tau_i * sigma, c - skew / (2 * tau_i) - Hi[i, i] / tau_i**2
        coth = 1 / np.tanh(tau_i * delta / 2)
        c = c - delta * skew * coth / 4 - (coth**2 / 4 - 1 / 8) * delta**2 * Hi[i, i]
        return w, 2 * np.sinh(tau_i * delta / 2) / delta * sigma, c

    def conditional_cdf(
        self, x, given, i=0, backend=None, delta=None, fillna=np.nan, **solver_kwargs
    ):
        r"""
        Saddle point approximation of the conditional cumulative distribution function
        :math:`P(X_i\leq x\mid X_J=x_J)`, with :math:`J` all other components, following
        the double saddlepoint approximation of Skovgaard (1987).

        There are two saddlepoints. The numerator saddlepoint :math:`\hat\tau` solves
        :math:`K'(\hat\tau)=(x, x_J)`. The denominator saddlepoint :math:`\tau_0` solves
        :math:`K_J'(\tau_0)=x_J` with its :math:`i`-th component equal to zero, that is, it
        is the saddlepoint of the marginal of :math:`X_J`. Then, see [1] (32.24)-(32.26)
        and [2] (4.12)-(4.13),

        .. math::
            P(X_i\leq x\mid X_J=x_J) \approx \Phi(\hat w)
            + \phi(\hat w)\left(\frac{1}{\hat w} - \frac{1}{\hat u}\right),

        with

        .. math::
            \hat w = \text{sign}(\hat\tau_i)\sqrt{2\left[\hat\tau^Tx - K(\hat\tau)
            - \tau_0^Tx + K(\tau_0)\right]},\qquad
            \hat u = \hat\tau_i\sqrt{\det K''(\hat\tau) / \det K_{JJ}''(\tau_0)}.

        With ``backend="Kolassa"`` the second order approximation of Kolassa (1996),
        Theorem 6.1, eq. (10), is used instead, which has error :math:`O(N^{-5/2})`
        rather than :math:`O(N^{-3/2})`. In terms of the survival function, with
        :math:`\hat z=\hat u`,

        .. math::
            1-F \approx 1-\Phi(\hat w) + \phi(\hat w)\left[\hat w^{-3} - \hat w^{-1}
            + \hat z^{-1}\left(1 + \tfrac18(\hat\rho_4-\tilde\rho_4)
            - \tfrac18(\hat\rho_{13}^2-\tilde\rho_{13}^2)
            - \tfrac1{12}(\hat\rho_{23}^2-\tilde\rho_{23}^2)
            - \frac{\hat k^{ik}\hat k_{ijk}\hat k^{ij}}{2\hat\tau_i}
            - \frac{\hat k^{ii}}{\hat\tau_i^2}\right)\right],

        where :math:`\hat\rho` are the invariants of the joint cgf at :math:`\hat\tau`,
        :math:`\tilde\rho` those of the marginal cgf of :math:`X_J` at :math:`\tau_0`,
        and :math:`k^{ij}` is the inverse of :math:`K''(\hat\tau)`. It needs the fourth
        derivative of the cgf, which is computed as a finite difference of ``d3K``.

        For two variables of interest, ``i=[a, b]``, the lower orthant probability
        :math:`P(X_a\leq x_a, X_b\leq x_b\mid X_J=x_J)` is approximated following Kolassa
        and Li (2010), Section 4, see :class:`_ConditionalProfile` and
        :class:`_ConditionalBivariateApprox`. The bivariate approximation of
        :meth:`BivariateSaddlePointApprox._kl_cdf` is applied to the profile cumulant
        generating function :math:`M(\tau_I)=\min_{\tau_J}[K(\tau)-\tau_J\cdot x_J]`,
        normalized to :math:`M(0)=0`, with the Jacobian
        :math:`\sqrt{\det K_{JJ}''(\tau_0)/\det K_{JJ}''}` of the conditioning components
        in its first order terms, and with the scalar conditional cdfs above as marginals.
        It reduces to the approximation above for one variable of interest, and is exact for
        the normal distribution. Requires that the saddlepoint equation can be solved at
        :math:`(x_I, x_J)`.

        The approximation is exact for the normal distribution. At :math:`\hat\tau_i=0`
        both :math:`\hat w` and :math:`\hat u` vanish. The singularity is removable, and
        within ``_conditional_singular_width`` conditional standard deviations of this
        point the approximation is replaced by a cubic interpolation.

        Parameters
        ----------
        x : array_like
            Values of the variable of interest. For two variables of interest, a vector of
            length two, or an array of shape ``(n, 2)``, in the order of ``i``.
        given : array_like
            Values of the conditioning variables, in increasing order of their index.
            Either a vector, used for all values of ``x``, or an array with one row for
            each value of ``x``.
        i : int or list of int, optional
            Index of the variable of interest, or a list with the indices of two variables
            of interest. Default is the first.
        backend : str, optional
            ``"Skovgaard"`` (default, first order) or ``"Kolassa"`` (second order). For two
            variables of interest, only ``"KL"`` is available, which is the default.
        delta : float, optional
            Span of the lattice of :math:`X_i`, for a discrete variable of interest. Default
            is ``None``: continuous. Then ``x`` must lie on the lattice, and the second
            continuity correction of [2] (4.17)-(4.18) is used: :math:`P(X_i\leq x)` is
            approximated with the survival function of :math:`X_i\geq x+\delta`, evaluated
            at :math:`x+\delta/2` with :math:`\hat u=2\sinh(\hat\tau_i\delta/2)\hat\sigma/\delta`.
            For ``backend="Kolassa"`` this is eq. (11) of [4]. The conditioning variables
            may be continuous or on a lattice; they are not corrected. For two variables of
            interest, a scalar or the spans of both, and eq. (4.8) of [5] is used, again at
            :math:`x_I+\delta/2`.
        fillna : float, optional
            The value to replace NaNs with.

        References
        ----------
        [1] Broda and Paolella (2011) - Saddlepoint approximations: a review and some new
            applications, Section 32.4.1
        [2] Butler (2007) - Saddlepoint approximations with applications, Section 4.2.1
        [3] Skovgaard (1987) - Saddlepoint expansions for conditional distributions
        [4] Kolassa (1996) - Higher-order approximations to conditional distribution functions
        [5] Kolassa and Li (2010) - Multivariate saddlepoint approximations in tail
            probability and conditional inference
        """
        if isinstance(i, (list, tuple, np.ndarray)):
            return self._conditional_cdf_vector(
                x, given, i, backend=backend, delta=delta, fillna=fillna, **solver_kwargs
            )
        backend = "Skovgaard" if backend is None else backend
        if backend not in ("Skovgaard", "Kolassa"):
            raise ValueError(f"Unknown backend {backend!r}, use 'Skovgaard' or 'Kolassa'")
        if delta is not None and not delta > 0:
            raise ValueError("delta must be positive")
        i, J, x, given = self._conditional_inputs(x, given, i)
        wrapper = PandasWrapper(x)
        y = np.array(
            [
                self._conditional_cdf_at(xi, xJ, i, J, backend, delta, **solver_kwargs)
                for xi, xJ in zip(x.ravel(), given)
            ]
        ).reshape(x.shape)
        y = np.where(np.isnan(y), fillna, y)
        return y.tolist() if y.ndim == 0 else wrapper.wrap(y)

    def _conditional_cdf_vector(
        self, x, given, i, backend=None, delta=None, fillna=np.nan, **solver_kwargs
    ):
        """:meth:`conditional_cdf` for a list ``i`` of variables of interest."""
        I = [self._normalize_index(k) for k in i]  # noqa: E741
        if len(set(I)) != len(I):
            raise ValueError("The variables of interest must be distinct")
        if len(I) == 1:
            backend = "Skovgaard" if backend is None else backend
            x = np.asarray(x, dtype=float)
            if x.ndim > 0 and x.shape[-1] == 1:
                x = x[..., 0]
            return self.conditional_cdf(x, given, I[0], backend, delta, fillna, **solver_kwargs)
        if len(I) > 2:
            raise NotImplementedError("At most two variables of interest are supported")
        if backend in (None, "Skovgaard", "KL"):
            backend = "KL"
        else:
            raise ValueError(f"Unknown backend {backend!r} for two variables, use 'KL'")
        J = [j for j in range(self.dim) if j not in I]
        if not J:
            raise ValueError("Conditioning requires at least one conditioning variable")
        if delta is not None:
            delta = np.broadcast_to(np.asarray(delta, dtype=float), (2,)).copy()
            if not (delta > 0).all():
                raise ValueError("delta must be positive")
        x = np.asarray(x, dtype=float)
        if x.ndim == 0 or x.shape[-1] != 2 or x.ndim > 2:
            raise ValueError("x must have shape (2,) or (n, 2)")
        xs = np.atleast_2d(x)
        given = np.asarray(given, dtype=float)
        if given.ndim == 1 and len(given) == len(J):
            given = np.broadcast_to(given, (len(xs), len(J)))
        elif given.shape != (len(xs), len(J)):
            raise ValueError(f"given must have shape ({len(J)},) or ({len(xs)}, {len(J)})")
        y = np.array(
            [
                self._conditional_cdf_vector_at(xI, xJ, I, J, delta, **solver_kwargs)
                for xI, xJ in zip(xs, given)
            ]
        )
        y = np.where(np.isnan(y), fillna, y)
        return y[0].tolist() if x.ndim == 1 else y

    def _conditional_cdf_vector_at(self, xI, xJ, I, J, delta=None, **solver_kwargs):  # noqa: E741
        if not (np.isfinite(xI).all() and np.isfinite(xJ).all()):
            return np.nan
        if delta is not None:
            xI = xI + delta / 2
        tau0 = self._conditional_denominator_saddle(I[0], J, xJ, **solver_kwargs)
        tau0[I] = 0.0
        xfull = np.empty(self.dim)
        xfull[I], xfull[J] = xI, xJ
        t = np.asarray(self.cgf.dK_inv(xfull, **solver_kwargs), dtype=float)[I]
        if np.isnan(t).any():
            return np.nan
        spa = _ConditionalBivariateApprox(self, I, J, xJ, tau0, delta)
        spa.profile.add_saddle(self.cgf.dK_inv(xfull, **solver_kwargs))
        return spa._kl_cdf(xI, t, **solver_kwargs)

    def _conditional_cdf_at(self, xi, xJ, i, J, backend="Skovgaard", delta=None, **solver_kwargs):
        if not (np.isfinite(xi) and np.isfinite(xJ).all()):
            return np.nan
        if delta is not None:
            xi = xi + delta / 2
        tau0 = self._conditional_denominator_saddle(i, J, xJ, **solver_kwargs)

        def point(x):
            xfull = np.empty(self.dim)
            xfull[i], xfull[J] = x, xJ
            if backend == "Kolassa":
                w, z, c = self._conditional_second_order(xfull, i, J, tau0, delta, **solver_kwargs)
                with np.errstate(divide="ignore", invalid="ignore"):
                    return sps.norm.cdf(w) - sps.norm.pdf(w) * (w**-3 - 1 / w + c / z)
            w, tau_i, sigma = self._conditional_terms(xfull, i, J, tau0, **solver_kwargs)
            u = tau_i * sigma if delta is None else 2 * np.sinh(tau_i * delta / 2) / delta * sigma
            with np.errstate(divide="ignore", invalid="ignore"):
                return sps.norm.cdf(w) + sps.norm.pdf(w) * (1 / w - 1 / u)

        # Interpolate near the singular point x_i = K_i'(tau_0), where tau_hat_i = 0
        center = float(self.cgf.dK(tau0)[i])
        H0 = np.asarray(self.cgf.d2K(tau0))
        h = self._conditional_singular_width / np.sqrt(np.linalg.inv(H0)[i, i])
        if abs(xi - center) < h:
            nodes = center + h * np.array([-4.0, -2.0, 2.0, 4.0])
            coef = np.polyfit(nodes - center, [point(v) for v in nodes], deg=3)
            return np.polyval(coef, xi - center)
        return point(xi)

    def conditional_pdf(self, x, given, i=0, fillna=np.nan, **solver_kwargs):
        r"""
        Saddle point approximation of the conditional probability density function of
        :math:`X_i` given :math:`X_J=x_J`, with :math:`J` all other components. It is the
        ratio of the joint and the marginal saddle point densities, see [1] (4.7) and
        (4.14),

        .. math::
            f(x\mid x_J) \approx \sqrt{\frac{\det K_{JJ}''(\tau_0)}{\det K''(\hat\tau)}}
            \,\phi(\hat w),

        where :math:`\hat\tau`, :math:`\tau_0` and :math:`\hat w` are as in
        :meth:`conditional_cdf`. The result is not normalized.

        Parameters
        ----------
        x, given, i, fillna :
            See :meth:`conditional_cdf`.

        References
        ----------
        [1] Butler (2007) - Saddlepoint approximations with applications, Section 4.1
        """
        i, J, x, given = self._conditional_inputs(x, given, i)
        wrapper = PandasWrapper(x)
        y = np.empty(x.size)
        for k, (xi, xJ) in enumerate(zip(x.ravel(), given)):
            if not (np.isfinite(xi) and np.isfinite(xJ).all()):
                y[k] = np.nan
                continue
            tau0 = self._conditional_denominator_saddle(i, J, xJ, **solver_kwargs)
            xfull = np.empty(self.dim)
            xfull[i], xfull[J] = xi, xJ
            w, _, sigma = self._conditional_terms(xfull, i, J, tau0, **solver_kwargs)
            y[k] = sps.norm.pdf(w) / sigma
        y = np.where(np.isnan(y), fillna, y).reshape(x.shape)
        return y.tolist() if y.ndim == 0 else wrapper.wrap(y)

    @type_wrapper(xloc=1)
    def _spapprox_pdf(self, x, t, fillna=np.nan):
        t = np.asanyarray(t)
        detd2Kt = np.linalg.det(self.cgf.d2K(t, fillna=fillna))
        with np.errstate(divide="ignore"):
            retval = np.where(
                ~np.isclose(detd2Kt, 0) & ~np.isnan(detd2Kt),
                np.exp(self.cgf.K(t) - np.multiply(t, x).sum(axis=-1))
                / np.power(2 * np.pi, self.dim / 2)
                / np.sqrt(detd2Kt),
                fillna,
            )
        return np.where(np.isnan(retval), fillna, retval)

    @property
    def _pdf_normalization(self):
        if not hasattr(self, "_pdf_normalization_cache") or self._pdf_normalization_cache is None:
            tranges = self.infer_t_ranges()
            val = nquad(
                lambda *args: (
                    self.pdf(t=args[: self.dim], normalize_pdf=False, fillna=0)
                    * np.linalg.det(self.cgf.d2K(args[: self.dim], fillna=0))
                ),
                tranges,
            )[0]
            assert not np.isnan(val) and np.isfinite(val), (
                "Failed to compute pdf normalization, value is equals NaN or Infinite"
            )
            self._pdf_normalization_cache = val
        return self._pdf_normalization_cache

    @property
    def dim(self):
        return self.cgf.dim

    def fit_saddle_point_eqn(self, t_ranges=None, atol=1e-4, rtol=1e-4, num=1000, **solver_kwargs):
        """
        Evaluate the saddle point equation to the given range of values.
        And use linear interpolation to solve the saddle point equation, form now onwards.

        If t_range is not provided, it will be assumed that the cumulant generating
        function is defined in a neighborhood of zero. The range will be computed
        by scaling iteratively, until we little mass for higher or lower values of x.
        """
        if t_ranges is None:
            if hasattr(self, "_t_cache"):
                t_ranges = np.vstack(
                    (self._t_cache.min(axis=0).min(axis=0), self._t_cache.max(axis=0).max(axis=0))
                ).T
            else:
                t_ranges = self.infer_t_ranges(atol=atol, rtol=rtol)
        x_ranges = self.cgf.dK(list(itertools.product(*t_ranges)))
        x_ranges = np.vstack((x_ranges.min(axis=0), x_ranges.max(axis=0))).T
        self._x_cache = [np.linspace(*xr, num=num) for xr in x_ranges]
        xir = np.vstack([xi.ravel() for xi in np.meshgrid(*self._x_cache, indexing="ij")]).T
        self._t_cache = self.cgf.dK_inv(xir, **solver_kwargs)
        self._t_cache = self._t_cache.reshape(*[num] * self.dim, self.dim)
        self._interp_cache = RegularGridInterpolator(self._x_cache, self._t_cache)

    @type_wrapper(xloc=1)
    def _dK_inv(self, x, **solver_kwargs):
        if hasattr(self, "_interp_cache"):
            return self._interp_cache(x)
        else:
            return super()._dK_inv(x, **solver_kwargs)

    def infer_t_ranges(self, atol=1e-4, rtol=1e-4):
        """
        Infers suitable ranges for :math:`t` so that it covers, roughly the entire probability mass.
        The ranges are inferred from the univariat case.
        """
        return [
            UnivariateSaddlePointApprox(self.cgf[i]).infer_t_range(atol=atol, rtol=rtol)
            for i in range(self.dim)
        ]

    # Map of the cdf backends to the methods that evaluate them at a single point
    _cdf_backends = {"Kolassa": "_kolassa_cdf"}

    def cdf(self, x=None, t=None, fillna=np.nan, backend="Kolassa", **solver_kwargs):
        r"""
        Saddle point approximation of the cumulative distribution function.

        Parameters
        ----------
        x : array_like, optional (either x or t must be provided)
            The values at which to evaluate the cumulative distribution function. The
            last axis indexes the components.
        t : array_like, optional (either x or t must be provided)
            Solution of the saddle point equation. If not provided, it will be
            computed using numerical root finding.
        fillna : float, optional
            The value to replace NaNs with.
        backend : str, optional
            'Kolassa' for the approximation of Kolassa (2003), see
            :meth:`_kolassa_cdf`.
        """
        assert x is not None or t is not None
        if backend not in self._cdf_backends:
            raise ValueError(f"backend must be one of {list(self._cdf_backends)}")
        if x is None:
            x = self.cgf.dK(t)
        elif t is None:
            t = self._dK_inv(x, **solver_kwargs)
        wrapper = PandasWrapper(x)
        x, t = np.asanyarray(x), np.asanyarray(t)
        y = np.asanyarray(self._spapprox_cdf(x, t, backend=backend, **solver_kwargs))
        y = np.where(np.isnan(y), fillna, y)
        return y.tolist() if y.ndim == 0 else wrapper.wrap(y)

    def _spapprox_cdf(self, x, t, backend="Kolassa", **solver_kwargs):
        # Handle vectorized evaluation
        if t.ndim > 1:
            return np.asanyarray(
                [
                    self._spapprox_cdf(xx, tt, backend=backend, **solver_kwargs)
                    for xx, tt in zip(x, t)
                ]
            )
        if t.ndim != 1 or x.ndim != 1 or len(t) != self.dim or len(x) != self.dim:
            raise ValueError(f"{self.dim} dimensional vector expected")
        return getattr(self, self._cdf_backends[backend])(x, t, **solver_kwargs)

    def _kolassa_cdf(self, x, t, **solver_kwargs):
        r"""
        Saddle point approximation of the cumulative distribution function in any
        dimension, following Kolassa (2003).

        For a subvector :math:`\mathbf{X}_A`, :math:`A\subseteq\{1,\ldots,d\}`, and signs
        :math:`\sigma\in\{-1,1\}^A`, let

        .. math::
            P_A^\sigma = P\left(\sigma_j X_j \geq \sigma_j x_j,\ j\in A\right),

        so that :math:`F(\mathbf{x}) = P_{\{1,\ldots,d\}}^{(-1,\ldots,-1)}`. The random
        vector :math:`(\sigma_jX_j)_{j\in A}` has cumulant generating function
        :math:`K_A(\sigma\odot\tau)`, with :math:`K_A` the cumulant generating
        function of the marginal :math:`\mathbf{X}_A`, and saddlepoint
        :math:`\sigma\odot\hat\tau_A`, where :math:`\hat\tau_A` is the saddlepoint of
        :math:`\mathbf{X}_A` at :math:`\mathbf{x}_A`. If all components of
        :math:`\sigma\odot\hat\tau_A` are non-negative, :math:`P_A^\sigma` is
        approximated by eq. (5) of [1], see :func:`spapprox.mvn._kolassa_orthant`.
        Otherwise, pick the first :math:`j` with :math:`\sigma_j\hat\tau_{A,j}<0`, and
        use (see the text after eq. (13) of [1])

        .. math::
            P_A^\sigma = P_{A\setminus\{j\}}^{\sigma_{-j}} - P_A^{\sigma'},

        where :math:`\sigma'` equals :math:`\sigma` with :math:`\sigma_j` flipped, and
        :math:`P_\emptyset = 1`. The second term has one negative saddlepoint component
        fewer, and the first one is a lower dimensional problem with its own
        saddlepoint, so that the recursion terminates. It visits at most :math:`3^d`
        subproblems, and solves at most :math:`2^d-2` additional saddlepoint
        equations.

        Unlike the plan of using the Lugannani-Rice formula for the one-dimensional
        problems, eq. (5) is used in every dimension, including one. If
        :math:`\hat\tau_{A,j}=0`, the marginal saddlepoint of
        :math:`\mathbf{X}_{A\setminus\{j\}}` equals :math:`\hat\tau_{A,-j}`, and the
        two orthant approximations on either side of :math:`x_j` add up exactly to the
        approximation of :math:`P_{A\setminus\{j\}}^{\sigma_{-j}}`. Hence, the
        approximation is continuous across the hyperplanes :math:`\hat\tau_j=0`, which
        would not hold with Lugannani-Rice in the one-dimensional problems. Moreover,
        the approximation is exact for the normal distribution, does not depend on the
        order of the variables, and has no singularities, so that no interpolation is
        needed near :math:`\hat\tau_j=0`. Its error is :math:`O(n^{-1})`, but not
        uniformly relative, so that the relative accuracy far in the tails is poor.

        The normal orthant probabilities and their derivatives are evaluated with
        relative accuracy, see :func:`spapprox.mvn._mvn_logtail`. This is important,
        as eq. (5) multiplies them by the potentially huge factor
        :math:`\exp(\hat\tau^T K''(\hat\tau)\hat\tau/2)`.

        Parameters
        ----------
        x : array_like
            A single point at which to evaluate the cumulative distribution function.
        t : array_like
            Solution of the saddle point equation at ``x``.

        References
        ----------
        [1] Kolassa (2003) - Multivariate saddlepoint tail probability approximations
        """
        d = self.dim
        x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
        if not (np.isfinite(x).all() and np.isfinite(t).all()):
            return np.nan
        saddles = {tuple(range(d)): t}
        cache = {}

        def saddle(A):
            if A not in saddles:
                if len(A) == 1:
                    saddles[A] = np.atleast_1d(self.cgf[A[0]].dK_inv(x[A[0]], **solver_kwargs))
                else:
                    saddles[A] = np.asarray(self[list(A)].cgf.dK_inv(x[list(A)], **solver_kwargs))
            return saddles[A]

        def orthant(A, sg):
            idx = list(A)
            tau = saddle(A)
            te = np.zeros(d)
            te[idx] = tau
            k2 = self.cgf.d2K(te)[np.ix_(idx, idx)] * np.outer(sg, sg)
            k3 = self.cgf.d3K(te)[np.ix_(idx, idx, idx)] * np.einsum("i,j,k->ijk", sg, sg, sg)
            return _kolassa_orthant(self.cgf.K(te), k2, k3, sg * tau, tau @ x[idx])

        def tail(A, sg):
            if not A:
                return 1.0
            key = (A, tuple(sg))
            if key not in cache:
                tau = saddle(A)
                if np.isnan(tau).any():
                    cache[key] = np.nan
                    return np.nan
                neg = np.flatnonzero(sg * tau < 0)
                if len(neg) == 0:
                    cache[key] = orthant(A, sg)
                else:
                    j = neg[0]
                    flipped = sg.copy()
                    flipped[j] *= -1
                    cache[key] = tail(A[:j] + A[j + 1 :], np.delete(sg, j)) - tail(A, flipped)
            return cache[key]

        return tail(tuple(range(d)), -np.ones(d))

    def fit_cdf(self, t_range=None, atol=1e-4, rtol=1e-4, num=1000):
        """
        Fit the cumulative distribution function using linear interpolation.
        """
        raise NotImplementedError


class BivariateSaddlePointApprox(MultivariateSaddlePointApprox):
    r"""
    Given the cumulant generating function of a bivariate random variable, this class
    provides the saddle point approximation of the probability density function.

    The bivariate saddle point approximation is given by

    .. math::
        f(\mathbf{x}) \approx \frac{1}{2\pi\sqrt{\left|\det H\right|}}
        \exp\left(K(\mathbf{\tilde t}) - \mathbf{x}\cdot\mathbf{\tilde t}\right)

    where :math:`\mathbf{x} = \left[x,y\right]`,
    :math:`H` is the Hessian matrix of the cumulant generating function,
    and :math:`\mathbf{\tilde t}= \left[\tilde s,\tilde t\right]` solves
    the saddle point equation :math:`K(\mathbf{\tilde t})=x`.

    The cumulative distribution function is approximated by the method of [6] by
    default, see :meth:`_kl_cdf`, by the method of [3] as outlined in [4] Section
    5.2, see :meth:`_wang_cdf`, or by the method of [8], see :meth:`_kolassa_cdf`.

    References
    ----------
    [1] Broda, Paolella (2011) - Saddlepoint approximations - a review

    [2] Butler, R. W. (2007). Saddlepoint approximations with applications.

    [3] Wang (1990) - Saddlepoint approximations for bivariate distributions

    [4] Paolella (2007) - Intermediate Probability

    [5] Kolassa (2006) - Serie approximations in statistics

    [6] Kolassa, Li (2010) - Multivariate saddlepoint approximations in tail probability
    and conditional inference

    [7] Li (2009) - Saddlepoint approximation, PhD thesis, Rutgers University

    [8] Kolassa (2003) - Multivariate saddlepoint tail probability approximations

    """

    _cdf_backends = {"KL": "_kl_cdf", "Wang": "_wang_cdf", "Kolassa": "_kolassa_cdf"}

    def cdf(self, x=None, t=None, fillna=np.nan, backend="KL", **solver_kwargs):
        r"""
        Saddle point approximation of the cumulative distribution function in
        the bivariate case.

        Parameters
        ----------
        x : array_like, optional (either x or t must be provided)
            The values at which to evaluate the cumulative distribution function.
        t : array_like, optional (either x or t must be provided)
            Solution of the saddle point equation. If not provided, it will be
            computed using numerical root finding.
        fillna : float, optional
            The value to replace NaNs with.
        backend : str, optional
            Either 'KL' for the approximation of Kolassa and Li (2010), see
            :meth:`_kl_cdf`, which has relative error :math:`O(n^{-1})`, 'Wang' for
            the approximation of Wang (1990), see :meth:`_wang_cdf`, which converges
            more slowly when the components are dependent, or 'Kolassa' for the
            approximation of Kolassa (2003), see :meth:`_kolassa_cdf`, which has
            error :math:`O(n^{-1})`, but not uniformly relative.
        """
        return super().cdf(x=x, t=t, fillna=fillna, backend=backend, **solver_kwargs)

    def _wang_cdf(self, x, t, **solver_kwargs):
        r"""
        Saddle point approximation of the cumulative distribution function in
        the bivariate case.

        The approximation is given by

        .. math::
            F(\mathbf{x}) \approx \Phi_2(\mathbf{\tilde x}, \rho)
            + \Phi(\tilde w) n + \Phi(w) \tilde n + n \tilde n,

        where vectors are denoted in bold and have components as follows:

        .. math::
            \mathbf{x} = \left[x,y\right],\\
            \mathbf{t} = \left[s, t\right],\\
            \mathbf{\tilde x} = \left[\tilde x,\tilde y\right],\\
            \mathbf{\tilde t_0} = \left[0,\tilde t\right],\\
            \mathbf{s_0} = \left[s,0\right],\\
            \mathbf{t_0} = \left[0,t\right].

        And where,

        .. math::
            \tilde x = \text{sign}(\tilde t) \sqrt{2(\mathbf{\tilde t_0}\cdot \mathbf{x}
                                                         - K(\mathbf{\tilde t_0}))},\\
            \tilde w = \text{sign}(t) \sqrt{2\left(K(\mathbf{s_0}) - K(\mathbf{t}) +
                                                     \mathbf{t_0}\cdot\mathbf{x}\right)},\\
            w = \text{sign}(s) \sqrt{2}
                       \sqrt{\mathbf{t}\cdot\mathbf{x}
                        - \mathbf{\tilde t_0}\cdot\mathbf{x}
                        + K(\mathbf{\tilde t_0}) - K(\mathbf{t})},\\
            b = \frac{\tilde w - \tilde x}{w},\\
            \tilde y = \frac{w - b \tilde x}{\sqrt{1+b^2}},\\
            \rho = \frac{-b}{\sqrt{1+b^2}},\\
            n = \phi(w)\left(\frac{1}{w}-\frac{1}{u}\right),\\
            \tilde n = \phi(\tilde x)\left(\frac{1}{\tilde w}-\frac{1}{\tilde u}\right),\\
            u = s \sqrt{\frac{\text{det} K''(\mathbf{t})}{K''_{tt}(\mathbf{t})}},\\
            \tilde u = t \sqrt{K''_{tt}(\mathbf{t})}

        and :math:`\mathbf{t}=[s,t]` is found by solving the saddle point equation

        .. math::
           \nabla K(\mathbf{t}) = \mathbf{x},

        And, :math:`\mathbf{\tilde t_0}=[0,\tilde t]` is found by solving a second saddlepoint
        equation:

        .. math::
            \partial_{t} K(\mathbf{\tilde t_0}) = y.

        We follow the approach as outlined in [4] Section 5.2, with simplified notation.
        The :math:`w_0` in their formula for :math:`\tilde n_0` (their notation), which is
        not defined there, is :math:`\tilde w_0`, as in [3] eq. (25).

        The formula is not symmetric in the two variables: above, the inner integral is
        over :math:`y` (the second variable). As in [3] Section 5 and [4] Remark (a), the
        order of integration is chosen per point: we integrate first over the variable for
        which :math:`\tilde w` is smaller. That is, the formula above is used if
        :math:`\tilde w` above is not larger than its counterpart with the roles of the
        variables reversed (see below), and the reversed formula is used otherwise. With the
        wrong order, the approximation error can be an order of magnitude larger, e.g.,
        under strong dependence.

        There are several cases that lead to singularities.
        First, following [4], if :math:`t=0` and :math:`s\neq0`, then :math:`\tilde n` should be replaced
        by its limiting value

        .. math::
            \tilde n = \frac{\phi(\tilde x)}{6}\frac{K_{ttt}(\mathbf{s_0})}{K_{tt}(\mathbf{s_0})^{3/2}}.

        Second, following [4], if :math:`s=0` and :math:`t\neq0`, then we reverse the role of :math:`X`
        and :math:`Y`, to reduce it to the first case. The reversed formula, which is also
        used by the order rule above, is obtained by swapping the roles of
        :math:`s` and :math:`t`, :math:`\tilde s` and :math:`\tilde t`, :math:`x` and :math:`y`,
        :math:`\tilde x` and :math:`\tilde y`, and :math:`\mathbf{s_0}` and :math:`\mathbf{t_0}`.
        This leads to

        .. math::
            \tilde y = \text{sign}(\tilde s) \sqrt{2(\mathbf{\tilde s_0}\cdot \mathbf{x}
                                                         - K(\mathbf{\tilde s_0}))},\\
            \tilde w = \text{sign}(s) \sqrt{2\left(K(\mathbf{t_0}) - K(\mathbf{t}) +
                                                     \mathbf{s_0}\cdot\mathbf{x}\right)},\\
            w = \text{sign}(t) \sqrt{2}
                       \sqrt{\mathbf{t}\cdot\mathbf{x}
                        - \mathbf{\tilde s_0}\cdot\mathbf{x}
                        + K(\mathbf{\tilde s_0}) - K(\mathbf{t})},\\
            \tilde x = \frac{w - b \tilde y}{\sqrt{1+b^2}},\\
            u = t \sqrt{\frac{\text{det} K''(\mathbf{t})}{K''_{ss}(\mathbf{t})}},\\
            \tilde u = s \sqrt{K''_{ss}(\mathbf{t})},

        And, where :math:`b`, :math:`\rho`, :math:`n`, and :math:`\tilde n` are as above
        Additionally, we introduce :math:`\mathbf{\tilde s_0}=[\tilde s,0]` as the solution of the alternative
        second saddepoint equation:

        .. math::
            \partial_{s} K(\mathbf{\tilde s_0}) = x.

        The limits above are only used when a component of :math:`\mathbf{t}` is exactly
        zero. Close to zero, the formulas suffer from catastrophic cancellation in the
        terms :math:`1/w - 1/u` and :math:`1/\tilde w - 1/\tilde u`; the roundoff error
        grows roughly like :math:`|t_j|^{-3}`. Therefore, the following is done, which goes
        beyond [3] and [4]. Let

        .. math::
            h_j = \frac{\delta}{\sqrt{K''_{jj}(\mathbf{0})}}, \quad j=1,2,

        where :math:`\delta` is ``_cdf_singular_width`` (0.01 by default), i.e., the
        width is fixed in standardized units. For every component with
        :math:`|t_j| < h_j`, the approximation is replaced by quadratic Lagrange
        interpolation in that component through the nodes :math:`t_j \in \{-h_j, 0, h_j\}`.
        If both components are near zero, a tensor product of these interpolations is
        used, so that 9 nodes are evaluated. At the nodes, the formula is evaluated
        directly, using the exact limits above for nodes on an axis. If only one component
        is near zero, all nodes integrate first over that variable, so that the
        interpolated function is smooth and has a closed-form limit on the axis. The interpolation
        error is :math:`O(h^3)`.

        At the double singularity :math:`\mathbf{t}=\mathbf{0}` no closed-form limit is
        available. Instead, let :math:`A(h)` denote the average of the approximation at
        the four points :math:`(\pm h_1, 0)` and :math:`(0, \pm h_2)`, which are evaluated
        using the exact one-component limits. By symmetry, the odd-order terms cancel, so
        that :math:`A(h) = F(\mathbf{0}) + c h^2 + O(h^4)`. The value at the origin is then
        obtained by Richardson extrapolation:

        .. math::
            F(\mathbf{0}) \approx \frac{4 A(h) - A(2h)}{3},

        which has error :math:`O(h^4)`.

        Parameters
        ----------
        x : array_like
            A single point at which to evaluate the cumulative distribution function.
        t : array_like
            Solution of the saddle point equation at ``x``.
        """
        h = self._cdf_singular_width / np.sqrt(np.diag(self.cgf.d2K0))
        near = np.abs(t) < h
        if not near.any():
            return self._bivariate_formula(x, t, **solver_kwargs)
        # Close to a single axis, integrate first over the variable whose saddlepoint
        # component is near zero, so that every interpolation node uses the same formula
        # and the node on the axis has a closed-form limit.
        inner = None if near.all() else int(np.flatnonzero(near)[0])
        nodes, weights = [], []
        for tj, hj, nj in zip(t, h, near):
            if nj:
                nodes.append(np.array([-hj, 0.0, hj]))
                # Quadratic Lagrange basis polynomials for the nodes -h, 0, h
                weights.append(
                    np.array([tj * (tj - hj) / 2, hj**2 - tj**2, tj * (tj + hj) / 2]) / hj**2
                )
            else:
                nodes.append(np.array([tj]))
                weights.append(np.ones(1))
        vals = np.array(
            [
                [
                    self._bivariate_formula_at(np.array([a, b]), h, inner, **solver_kwargs)
                    for b in nodes[1]
                ]
                for a in nodes[0]
            ]
        )
        return weights[0] @ vals @ weights[1]

    def _bivariate_formula_at(self, t, h, inner=None, **solver_kwargs):
        if np.all(t == 0):
            # No closed form for the double singularity: extrapolate the averages over
            # the neighbouring points on the axes, A(h) = F(0) + c h^2 + O(h^4).
            def axis_mean(hh):
                pts = [[hh[0], 0], [-hh[0], 0], [0, hh[1]], [0, -hh[1]]]
                return np.mean([self._bivariate_formula_at(np.array(p), h) for p in pts])

            return (4 * axis_mean(h) - axis_mean(2 * h)) / 3
        return self._bivariate_formula(self.cgf.dK(t), t, inner=inner, **solver_kwargs)

    def _inner_w0(self, x, t, inner):
        """
        The quantity :math:`\\tilde w_0` of :meth:`_spapprox_cdf` when integrating first
        over the variable with index ``inner``.
        """
        t_outer = t.copy()
        t_outer[inner] = 0
        return np.sign(t[inner]) * np.sqrt(
            2 * (self.cgf.K(t_outer) - self.cgf.K(t) + t[inner] * x[inner])
        )

    def _bivariate_formula(self, x, t, inner=None, **solver_kwargs):
        """
        Direct evaluation of the formula documented in :meth:`_spapprox_cdf` for a
        single point, using the exact limit when the inner component of ``t`` is 0.

        ``inner`` is the index of the variable that is integrated over first. If it is
        not given, it is chosen by the rule of [3]: the order with the smaller
        :math:`\\tilde w_0` is used.
        """
        if t[0] == 0:
            # The formula is singular when the outer component is zero
            inner = 0
        elif t[1] == 0:
            inner = 1
        elif inner is None:
            inner = 1 if self._inner_w0(x, t, 1) <= self._inner_w0(x, t, 0) else 0
        outer = 1 - inner
        t_outer = t.copy()
        t_outer[inner] = 0
        # Note, slicing a component of a cgf sets the other variables to zero
        ti = self.cgf[inner].dK_inv(x[inner], **solver_kwargs)
        ti0 = np.zeros(2)
        ti0[inner] = ti
        # Calculate components
        x1 = np.sign(ti) * np.sqrt(2 * (ti * x[inner] - self.cgf.K(ti0)))
        tw = self._inner_w0(x, t, inner)
        w = np.sign(t[outer]) * np.sqrt(2 * (np.dot(t - ti0, x) + self.cgf.K(ti0) - self.cgf.K(t)))
        b = (tw - x1) / w
        y1 = (w - b * x1) / np.sqrt(1 + np.square(b))
        tx = np.array([x1, y1])
        rho = -b / np.sqrt(1 + np.square(b))
        d2K = self.cgf.d2K(t)
        u = t[outer] * np.sqrt(np.linalg.det(d2K) / d2K[inner, inner])
        tu = t[inner] * np.sqrt(d2K[inner, inner])
        n = sps.norm.pdf(w) * (1 / w - 1 / u)
        assert np.isfinite(n), "Something is wrong"
        if t[inner] != 0:
            tn = sps.norm.pdf(x1) * (1 / tw - 1 / tu)
        else:
            d3K = self.cgf.d3K(t_outer)[..., inner, inner, inner]
            d2K_outer = self.cgf.d2K(t_outer)[..., inner, inner]
            tn = sps.norm.pdf(x1) / 6 * (d3K / d2K_outer ** (3 / 2))
        # Put everything together
        retval = _bvn_cdf(tx, rho)
        retval += sps.norm.cdf(tw) * n + sps.norm.cdf(w) * tn + n * tn
        return retval

    def _kl_cdf(self, x, t, **solver_kwargs):
        r"""
        Saddle point approximation of the cumulative distribution function in the
        bivariate case, following Kolassa and Li [6] and Li [7] Chapter 3.

        The approximation is for an upper orthant probability, and has relative error
        :math:`O(n^{-1})` uniformly, also when the components are dependent. We
        evaluate it for the orthant in which both saddlepoint components are positive,
        so that the orthant probability is small, and obtain the cdf by
        inclusion-exclusion. Let :math:`\sigma_j=\text{sign}(t_j)` (1 if :math:`t_j=0`),
        and let :math:`J = P(\sigma_1 X_1 \geq \sigma_1 x_1, \sigma_2 X_2 \geq \sigma_2 x_2)`
        which is approximated by applying the upper tail approximation to the random
        vector with cumulant generating function :math:`K^*(\tau)=K(\sigma\odot\tau)`.
        Then

        .. math::
            F(\mathbf{x}) =
            \begin{cases}
                J & \text{if } \sigma=(-1,-1),\\
                F_1(x_1) - J & \text{if } \sigma=(-1,1),\\
                F_2(x_2) - J & \text{if } \sigma=(1,-1),\\
                F_1(x_1) + F_2(x_2) - 1 + J & \text{if } \sigma=(1,1),
            \end{cases}

        where the marginal cdfs :math:`F_j` are approximated by the Lugannani-Rice
        formula. Without the reflection, the approximation breaks down when the orthant
        probability is close to one.

        In the remainder, :math:`K` denotes the reflected :math:`K^*`, and the reflected
        point and saddlepoint are denoted by :math:`\mathbf{x}` and :math:`\hat\tau`.
        Following [6], define

        .. math::
            \hat w_2 = \text{sign}(\tilde\tau_2)\sqrt{2(\tilde\tau_2 x_2 - K(0, \tilde\tau_2))},\\
            \hat w_1 = \text{sign}(\hat\tau_1)\sqrt{2(\hat\tau\cdot\mathbf{x} - K(\hat\tau)
                - \tilde\tau_2 x_2 + K(0, \tilde\tau_2))},\\
            \check w_2 = \hat w_2 - \text{sign}(\hat\tau_2)\sqrt{2(\hat\tau\cdot\mathbf{x}
                - K(\hat\tau) - \hat\tau_1 x_1 + K(\hat\tau_1, 0))},

        where :math:`\tilde\tau_2` solves the marginal saddlepoint equation
        :math:`\partial_2 K(0, \tilde\tau_2) = x_2`. In the parametrization of [6], the
        pole :math:`\tau_2=0` is at :math:`w_2=\tilde w_2(w_1)`, with
        :math:`\check w_2=\tilde w_2(\hat w_1)`. Let :math:`c = \tilde w_2'(\hat w_1)`
        and :math:`d = \tilde w_2''(\hat w_1)`, which are computed analytically from the
        derivatives of :math:`K` up to third order at :math:`\hat\tau` (see
        :meth:`_kl_tail`). Furthermore, let

        .. math::
            \hat{\mathbf{u}} = (\hat w_1, \hat w_2 - \check w_2), \quad
            A = \begin{bmatrix} 1 + c^2 & c \\ c & 1\end{bmatrix}, \quad
            \mathbf{b} = A\hat{\mathbf{u}},\quad
            C = \exp\left(-\tfrac{1}{2}(\hat w_1^2 + \hat w_2^2)
                + \tfrac{1}{2}\hat{\mathbf{u}}^T A \hat{\mathbf{u}}\right),

        and let :math:`T(\mathbf{b})=P(\mathbf{Z}\geq \mathbf{b})` with
        :math:`\mathbf{Z}\sim N(0, A)`. Then :math:`J\approx I^\emptyset + I^{\{1\}}
        + I^{\{2\}}`, with

        .. math::
            I^\emptyset = C\left(T - \frac{d}{6}\left[3c(\partial_1+\hat u_1)^3
                + 3(\partial_1+\hat u_1)^2(\partial_2+\hat u_2)\right]T\right),\\
            I^{\{1\}} = -C\left(\frac{1}{\hat\tau_1 \sqrt{K_{11}-K_{12}^2/K_{22}}}
                - \frac{1}{\hat w_1}\right)\partial_1 T,\\
            I^{\{2\}} = \left(\frac{1}{\tilde\tau_2\sqrt{K_{22}(0, \tilde\tau_2)}}
                - \frac{1}{\hat w_2}\right)\bar\Phi(\hat w_1)\phi(\hat w_2),

        where the second derivatives in :math:`I^{\{1\}}` are evaluated at
        :math:`\hat\tau`, and the derivatives of :math:`T` are with respect to
        :math:`\mathbf{b}`. The cubic correction in :math:`I^\emptyset` is not given in
        [6] and [7], which refer to [8]. It follows by expanding the exponent
        up to third order around :math:`\hat{\mathbf{u}}`: only :math:`\tilde w_2` is
        not quadratic, so the only non-zero third derivatives are
        :math:`g_{111}=3cd` and :math:`g_{112}=d`. The resulting Gaussian integrals are
        obtained by applying :math:`-\partial_{b_j} - \hat u_j` to :math:`T`. Without
        this term, the relative error is :math:`O(n^{-1/2})`.

        The approximation is not symmetric in the two variables. Moreover, with the
        reflection, it is discontinuous across the axis :math:`t_2=0`: at
        :math:`\hat\tau_2=0`, the sum of the two reflected orthant approximations
        :math:`I^\emptyset + I^{\{1\}} + I^{\{2\}}` differs from the Lugannani-Rice
        approximation of the marginal tail of :math:`X_1` by :math:`O(n^{-1})`. Across the
        axis :math:`t_1=0`, it is continuous, as :math:`\hat w_2` is the signed root of the
        marginal of :math:`X_2`. Therefore, going beyond [6], we use a weighted average
        of the approximations :math:`F_{12}` and :math:`F_{21}` with the variables in
        either order:

        .. math::
            F(\mathbf{x}) \approx \frac{z_2^2}{z_1^2+z_2^2}F_{12}(\mathbf{x})
                + \frac{z_1^2}{z_1^2+z_2^2}F_{21}(\mathbf{x}),
                \quad z_j = t_j\sqrt{K''_{jj}(\mathbf{0})},

        which is continuous, and does not depend on the order of the variables. The
        accuracy is the same as for either order.

        The formulas are singular when a component of :math:`\hat\tau` is zero, and
        :math:`I^{\{2\}}` is singular when :math:`\tilde\tau_2=0`. Close to these
        singularities, the formulas suffer from catastrophic cancellation, as in
        :meth:`_wang_cdf`. Let :math:`h_j` be as in :meth:`_wang_cdf`. For every
        component with :math:`|t_j| < h_j`, :math:`J` is replaced by cubic Lagrange
        interpolation in that component through the nodes
        :math:`t_j\in\{-2h_j,-h_j,h_j,2h_j\}`, with a tensor product if both components
        are close to zero. The orientation :math:`\sigma` is kept fixed across the
        nodes. Similarly, if :math:`|\tilde\tau_2| < h_2`, the first factor of
        :math:`I^{\{2\}}` is interpolated in :math:`\tilde\tau_2`.

        Parameters
        ----------
        x : array_like
            A single point at which to evaluate the cumulative distribution function.
        t : array_like
            Solution of the saddle point equation at ``x``.
        """
        z2 = np.square(t) * np.diag(self.cgf.d2K0)
        w = z2[1] / z2.sum() if z2.sum() > 0 else 0.5
        retval = 0.0
        if w > 0:
            retval += w * self._kl_cdf_ordered(x, t, **solver_kwargs)
        if w < 1:
            retval += (1 - w) * self._kl_swapped()._kl_cdf_ordered(
                x[::-1], t[::-1], **solver_kwargs
            )
        return retval

    def _kl_swapped(self):
        """The approximation with the two variables in reverse order."""
        return self[[1, 0]]

    def _kl_marginal_cdf(self, i, x, t, **solver_kwargs):
        """Marginal cdf of variable ``i`` used in the inclusion-exclusion of :meth:`_kl_cdf`."""
        return self[i].cdf(x=x, t=t, **solver_kwargs)

    def _kl_first_factor(self, t, ts0, e1, w1h):
        r"""
        The factor :math:`1/(\hat\tau_1\sqrt{K_{11}-K_{12}^2/K_{22}}) - 1/\hat w_1` of
        :math:`I^{\{1\}}` in :meth:`_kl_cdf`, with ``ts0`` the reflected
        :math:`\hat\tau_1`, ``e1`` the square root and ``t`` the unreflected saddlepoint.
        """
        return 1 / (ts0 * e1) - 1 / w1h

    def _kl_cdf_ordered(self, x, t, **solver_kwargs):
        """
        The approximation of :meth:`_kl_cdf` with the variables in the given order.
        """
        h = self._cdf_singular_width / np.sqrt(np.diag(self.cgf.d2K0))
        sg = np.where(t < 0, -1.0, 1.0)
        tm = [self.cgf[i].dK_inv(x[i], **solver_kwargs) for i in range(2)]
        F = [self._kl_marginal_cdf(i, x[i], tm[i], **solver_kwargs) for i in range(2)]
        near = np.abs(t) < h
        if not near.any():
            J = self._kl_tail(x, t, sg, tm[1], h[1])
        else:
            nodes, weights = [], []
            for tj, hj, nj in zip(t, h, near):
                if nj:
                    nodes.append(np.array([-2, -1, 1, 2]) * hj)
                    weights.append(_lagrange_weights(nodes[-1], tj))
                else:
                    nodes.append(np.array([tj]))
                    weights.append(np.ones(1))
            vals = np.empty((len(nodes[0]), len(nodes[1])))
            for i, a in enumerate(nodes[0]):
                for j, b in enumerate(nodes[1]):
                    p = np.array([a, b])
                    xp = self.cgf.dK(p)
                    ttp = self.cgf[1].dK_inv(xp[1], t0=tm[1], **solver_kwargs)
                    vals[i, j] = self._kl_tail(xp, p, sg, ttp, h[1])
            J = weights[0] @ vals @ weights[1]
        if sg[0] < 0 and sg[1] < 0:
            return J
        elif sg[0] < 0:
            return F[0] - J
        elif sg[1] < 0:
            return F[1] - J
        return F[0] + F[1] - 1 + J

    def _kl_marginal_factor(self, tt):
        r"""
        The factor :math:`1/(\tilde\tau_2\sqrt{K_{22}(0,\tilde\tau_2)}) - 1/\hat w_2` of
        :math:`I^{\{2\}}` in :meth:`_kl_cdf`, in the original (unreflected) coordinates.
        """
        t0 = np.array([0.0, tt])
        w = np.sign(tt) * np.sqrt(2 * (tt * self.cgf.dK(t0)[1] - self.cgf.K(t0)))
        return 1 / (tt * np.sqrt(self.cgf.d2K(t0)[1, 1])) - 1 / w

    def _kl_tail(self, x, t, sg, tt, h2):
        r"""
        Approximation of :math:`P(\sigma_1 X_1 \geq \sigma_1 x_1, \sigma_2 X_2 \geq
        \sigma_2 x_2)` of :meth:`_kl_cdf` for a single point, where ``tt`` is the
        marginal saddlepoint of :math:`X_2` and ``h2`` the width used to interpolate the
        factor of :math:`I^{\{2\}}` when ``tt`` is close to zero.

        The derivatives of :math:`\tilde w_2` are obtained as follows. Parametrize the
        curve :math:`(\tau_1, \tilde\tau_2(\tau_1))`, with :math:`\tilde\tau_2(\tau_1)`
        the minimizer of :math:`K(\tau_1, \tau_2) - \tau_2 x_2`, by :math:`\tau_1`. Let
        :math:`e = w_1 - \hat w_1` and :math:`f = \tilde w_2 - \hat w_2` along this curve.
        Then :math:`\frac12 e^2 = L(\hat\tau_1) - L(\tau_1)` and
        :math:`\frac12 f^2 = D(\tau_1)`, with
        :math:`L(\tau_1)= \tau_1 x_1 + \tilde\tau_2 x_2 - K(\tau_1, \tilde\tau_2)` and
        :math:`D(\tau_1) = \tilde\tau_2 x_2 - K(\tau_1, \tilde\tau_2) + K(\tau_1, 0)`.
        Implicit differentiation at :math:`\hat\tau_1` gives

        .. math::
            e' = \sqrt{S},\quad e'' = \frac{S'}{3e'},\quad
            f' = \frac{K_1(\hat\tau_1, 0) - x_1}{f},\quad
            f'' = \frac{K_{11}(\hat\tau_1, 0) - S - f'^2}{f},

        where :math:`S = K_{11} - K_{12}^2/K_{22}` and :math:`S'` is its derivative along
        the curve, which involves the third derivatives of :math:`K`. Finally,
        :math:`\tilde w_2' = f'/e'` and :math:`\tilde w_2'' = (f''e' - f'e'')/e'^3`.
        """
        K, dK, d2K, d3K = self.cgf.K, self.cgf.dK, self.cgf.d2K, self.cgf.d3K
        ts = sg * t
        t10 = np.array([t[0], 0.0])
        t02 = np.array([0.0, tt])
        Lh = np.dot(t, x) - K(t)
        m2 = tt * x[1] - K(t02)
        w2h = sg[1] * np.sign(tt) * np.sqrt(2 * m2)
        w1h = np.sign(ts[0]) * np.sqrt(2 * (Lh - m2))
        w2c = w2h - np.sign(ts[1]) * np.sqrt(2 * (Lh - t[0] * x[0] + K(t10)))
        # Derivatives of the reflected cumulant generating function
        k2 = d2K(t) * np.outer(sg, sg)
        k3 = d3K(t) * np.einsum("i,j,k->ijk", sg, sg, sg)
        r = k2[0, 1] / k2[1, 1]
        S = k2[0, 0] - r * k2[0, 1]
        dS = (k3[0, 0, 0] - 2 * r * k3[0, 0, 1] + r**2 * k3[0, 1, 1]) - r * (
            k3[0, 0, 1] - 2 * r * k3[0, 1, 1] + r**2 * k3[1, 1, 1]
        )
        e1 = np.sqrt(S)
        e2 = dS / (3 * e1)
        f = w2c - w2h
        f1 = sg[0] * (dK(t10)[0] - x[0]) / f
        f2 = (d2K(t10)[0, 0] - S - f1**2) / f
        c = f1 / e1
        d = (f2 * e1 - f1 * e2) / e1**3
        # Leading term and its cubic correction
        u = np.array([w1h, w2h - w2c])
        A = np.array([[1 + c**2, c], [c, 1.0]])
        b = A @ u
        C = np.exp(-(w1h**2 + w2h**2) / 2 + u @ A @ u / 2)
        T, T1, T2, T11, T12, T111, T112 = _bvn_tail_derivatives(b, c)
        D111 = T111 + 3 * u[0] * T11 + 3 * u[0] ** 2 * T1 + u[0] ** 3 * T
        D112 = (
            T112 + 2 * u[0] * T12 + u[0] ** 2 * T2 + u[1] * (T11 + 2 * u[0] * T1 + u[0] ** 2 * T)
        )
        I0 = C * (T - d / 6 * (3 * c * D111 + 3 * D112))
        I1 = -C * self._kl_first_factor(t, ts[0], e1, w1h) * T1
        if np.abs(tt) >= h2:
            fac = self._kl_marginal_factor(tt)
        else:
            nodes = np.array([-2, -1, 1, 2]) * h2
            fac = _lagrange_weights(nodes, tt) @ [self._kl_marginal_factor(n) for n in nodes]
        I2 = sg[1] * fac * sps.norm.sf(w1h) * sps.norm.pdf(w2h)
        return I0 + I1 + I2

    def _dK_inv(self, x, **solver_kwargs):
        # TODO: implement multivariate interpolation with radial basis functions or something like that
        return super()._dK_inv(x, **solver_kwargs)


def _vectorize_points(f, shape=()):
    """Apply ``f``, defined for a single point, to an array of points in the last axis."""

    def g(t):
        t = np.asarray(t, dtype=float)
        if t.ndim == 1:
            return f(t)
        out = np.array([f(tt) for tt in t.reshape(-1, t.shape[-1])])
        return out.reshape(t.shape[:-1] + shape)

    return g


class _ConditionalProfile:
    r"""
    Profile of the cumulant generating function :math:`K` of :math:`X=(X_I, X_J)` over the
    components :math:`J`, for the distribution of :math:`X_I` given :math:`X_J=x_J`:

    .. math::
        M(\tau_I) = \min_{\tau_J}\left[K(\tau_I,\tau_J)-\tau_J\cdot x_J\right]
            - \min_{\tau_J}\left[K(0,\tau_J)-\tau_J\cdot x_J\right],

    with minimizer :math:`\tilde\tau_J(\tau_I)`, so that :math:`M(0)=0`. By the implicit
    function theorem, :math:`M'=K_I`, :math:`M''` is the Schur complement
    :math:`K_{II}-K_{IJ}K_{JJ}^{-1}K_{JI}` and :math:`M'''=K'''[V,V,V]`, with
    :math:`V_I=1` and :math:`V_J=-K_{JJ}^{-1}K_{JI}`, all evaluated at
    :math:`(\tau_I,\tilde\tau_J(\tau_I))`.

    The nested signed roots of Kolassa and Li (2010) for the components :math:`I` in the
    conditional case are those of :math:`M`, as the nested minimizers range over all
    components after the fixed ones, :math:`J` included. The Jacobian of the components
    :math:`J` in their Lemma 4.1 is the weight :math:`r(\tau_I)=\sqrt{\det K_{JJ}(\tau_0)/
    \det K_{JJ}(\tau_I,\tilde\tau_J(\tau_I))}`, with :math:`r(0)=1`, where :math:`\tau_0`
    is the saddlepoint of the marginal of :math:`X_J`. Thus, the conditional distribution is
    approximated as the distribution with cumulant generating function :math:`M`, with the
    factor :math:`r` in the inversion integrand.
    """

    def __init__(self, cgf, I, J, xJ, tau0):  # noqa: E741
        self.cgf = cgf
        self.I, self.J = list(I), list(J)
        self.xJ = np.asarray(xJ, dtype=float)
        self.tau0 = np.asarray(tau0, dtype=float)
        self._saddles = {}
        self._start = self.tau0[self.J].copy()
        self.c0 = self.cgf.K(self.tau0) - self.tau0[self.J] @ self.xJ
        H0 = np.asarray(self.cgf.d2K(self.tau0))
        self.logdet0 = np.linalg.slogdet(H0[np.ix_(self.J, self.J)])[1]

    def _objective(self, tau):
        return self.cgf.K(tau) - tau[self.J] @ self.xJ

    def saddle(self, tI):
        """Full saddlepoint :math:`(\\tau_I,\\tilde\\tau_J(\\tau_I))`, NaN if not found."""
        tI = np.asarray(tI, dtype=float)
        key = tuple(tI)
        if key not in self._saddles:
            tau = self._solve(tI, self._start)
            if np.isnan(tau).any():
                tau = self._continuation(tI)
            if not np.isnan(tau).any():
                self._start = tau[self.J].copy()
            self._saddles[key] = tau
        return self._saddles[key]

    def add_saddle(self, tau):
        """Register a known full saddlepoint, with :math:`K_J'(\tau)=x_J`."""
        tau = np.asarray(tau, dtype=float)
        self._saddles[tuple(tau[self.I])] = tau

    def _continuation(self, tI, max_halvings=10):
        """Solve along the path :math:`s\tau_I`, from the known solution at zero."""
        s, tau, step = 0.0, self.tau0, 0.25
        while s < 1:
            new = self._solve(min(s + step, 1.0) * tI, tau[self.J])
            if np.isnan(new).any():
                step /= 2
                if step < 0.25 / 2**max_halvings:
                    return new
                continue
            s, tau, step = min(s + step, 1.0), new, 2 * step
        return tau

    def _solve(self, tI, start, max_iter=100):
        J = self.J
        tau = np.empty(self.cgf.dim)
        tau[self.I] = tI
        tau[J] = start
        f = self._objective(tau)
        tol = 1e-12 * (1 + np.max(np.abs(self.xJ)))
        for _ in range(max_iter):
            if not np.isfinite(f):
                break
            g = np.asarray(self.cgf.dK(tau))[J] - self.xJ
            if np.max(np.abs(g)) < tol:
                return tau
            H = np.asarray(self.cgf.d2K(tau))[np.ix_(J, J)]
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            s = 1.0
            while s > 1e-12:
                new = tau.copy()
                new[J] -= s * step
                fn = self._objective(new)
                if np.isfinite(fn) and fn <= f - 1e-4 * s * (g @ step) + 1e-14 * (1 + abs(f)):
                    break
                s /= 2
            else:
                break
            tau, f = new, fn
        return np.full(self.cgf.dim, np.nan)

    def K(self, tI):
        tau = self.saddle(tI)
        return self._objective(tau) - self.c0

    def dK(self, tI):
        return np.asarray(self.cgf.dK(self.saddle(tI)), dtype=float)[self.I]

    def _tangent(self, H):
        I, J = self.I, self.J  # noqa: E741
        V = np.zeros((self.cgf.dim, len(I)))
        V[I] = np.eye(len(I))
        V[J] = -np.linalg.solve(H[np.ix_(J, J)], H[np.ix_(J, I)])
        return V

    def d2K(self, tI):
        tau = self.saddle(tI)
        if np.isnan(tau).any():
            return np.full((len(self.I),) * 2, np.nan)
        H = np.asarray(self.cgf.d2K(tau), dtype=float)
        V = self._tangent(H)
        return V.T @ H @ V

    def d3K(self, tI):
        tau = self.saddle(tI)
        if np.isnan(tau).any():
            return np.full((len(self.I),) * 3, np.nan)
        H = np.asarray(self.cgf.d2K(tau), dtype=float)
        V = self._tangent(H)
        T3 = np.asarray(self.cgf.d3K(tau), dtype=float)
        return np.einsum("pqr,pa,qb,rc->abc", T3, V, V, V)

    def r(self, tI):
        tau = self.saddle(tI)
        if np.isnan(tau).any():
            return np.nan
        H = np.asarray(self.cgf.d2K(tau), dtype=float)
        sign, logdet = np.linalg.slogdet(H[np.ix_(self.J, self.J)])
        return np.exp((self.logdet0 - logdet) / 2) if sign > 0 else np.nan

    def as_cgf(self):
        d0 = len(self.I)
        zero = np.zeros(d0)
        return MultivariateCumulantGeneratingFunction(
            K=_vectorize_points(self.K),
            dK=_vectorize_points(self.dK, (d0,)),
            d2K=_vectorize_points(self.d2K, (d0, d0)),
            d3K=_vectorize_points(self.d3K, (d0, d0, d0)),
            dim=d0,
            dK0=self.dK(zero),
            d2K0=self.d2K(zero),
            d3K0=self.d3K(zero),
            domain=Domain(dim=d0),
        )


class _ConditionalBivariateApprox(BivariateSaddlePointApprox):
    r"""
    Approximation of the conditional distribution of :math:`X_I` given :math:`X_J=x_J`, for
    two variables of interest, by the bivariate approximation of Kolassa and Li (2010)
    applied to the profile :math:`M` of :class:`_ConditionalProfile`, Section 4 of their
    paper. Compared to the unconditional case, the weight :math:`r` of the profile enters
    the factors of :math:`I^{\{1\}}` and :math:`I^{\{2\}}`, and the marginal cdfs in the
    inclusion-exclusion are the conditional cdfs of :math:`X_{I_k}` given :math:`X_J`. For a
    lattice with spans ``delta``, :math:`\tau_k` in these factors is replaced by
    :math:`2\sinh(\tau_k\delta_k/2)/\delta_k`, eq. (4.8), and the approximation should be
    evaluated at the continuity corrected point :math:`x_I+\delta/2`.
    """

    def __init__(self, parent, I, J, xJ, tau0, delta=None):  # noqa: E741
        self.profile = _ConditionalProfile(parent.cgf, I, J, xJ, tau0)
        super().__init__(self.profile.as_cgf())
        self.parent, self.I, self.J = parent, list(I), list(J)
        self.xJ, self.tau0 = np.asarray(xJ, dtype=float), np.asarray(tau0, dtype=float)
        self.delta = None if delta is None else np.asarray(delta, dtype=float)

    def _kl_swapped(self):
        if not hasattr(self, "_swapped_cache"):
            delta = None if self.delta is None else self.delta[::-1]
            self._swapped_cache = _ConditionalBivariateApprox(
                self.parent, self.I[::-1], self.J, self.xJ, self.tau0, delta
            )
        return self._swapped_cache

    def _rho(self, tau, k):
        if self.delta is None:
            return tau
        return 2 * np.sinh(tau * self.delta[k] / 2) / self.delta[k]

    def _kl_marginal_cdf(self, i, x, t, **solver_kwargs):
        spa = self.parent[[self.I[i]] + self.J]
        if self.delta is None:
            return spa.conditional_cdf(x, self.xJ, i=0, **solver_kwargs)
        d = self.delta[i]
        return spa.conditional_cdf(x - d / 2, self.xJ, i=0, delta=d, **solver_kwargs)

    def _kl_first_factor(self, t, ts0, e1, w1h):
        r = self.profile.r(np.array([t[0], 0.0]))
        return r / (self._rho(ts0, 0) * e1) - 1 / w1h

    def _kl_marginal_factor(self, tt):
        t0 = np.array([0.0, tt])
        w = np.sign(tt) * np.sqrt(2 * (tt * self.cgf.dK(t0)[1] - self.cgf.K(t0)))
        u = self._rho(tt, 1) * np.sqrt(self.cgf.d2K(t0)[1, 1]) / self.profile.r(t0)
        return 1 / u - 1 / w
