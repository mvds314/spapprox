# `spa.py` -- saddle point approximation API reference

```
SaddlePointApprox (ABC)
+-- UnivariateSaddlePointApprox
|   `-- UnivariateSaddlePointApproxMean
`-- MultivariateSaddlePointApprox
    `-- BivariateSaddlePointApprox
```

## Contents

- [The base class](#the-base-class)
- [Univariate pdf](#univariate-pdf)
- [Univariate cdf: LR and BN backends](#univariate-cdf-lr-and-bn-backends)
- [ppf](#ppf)
- [Normalization](#normalization)
- [Speeding things up: infer_t_range, fit_*, clear_cache](#speeding-things-up-infer_t_range-fit_-clear_cache)
- [Sample mean](#sample-mean)
- [Multivariate and bivariate](#multivariate-and-bivariate)

## The base class

```python
SaddlePointApprox(cgf, pdf_normalization=None)
```

Stores `self.cgf` and seeds `_pdf_normalization_cache`. Defines the shared
`pdf`, leaves `cdf`/`ppf` raising `NotImplementedError` for subclasses to
provide, and requires subclasses to implement `_spapprox_pdf`,
`_pdf_normalization` and `dim`.

`clear_cache()` is reflective: it walks `inspect.getmembers` and deletes every
non-routine attribute whose name ends in `_cache`. Any new cached value should
therefore use that suffix -- otherwise it silently survives a cache clear.

## Univariate pdf

```python
approx.pdf(x=None, t=None, normalize_pdf=True, fillna=np.nan, **solver_kwargs)
```

The `x`/`t` resolution step at the top of `pdf` (and `cdf`) is shared:

```python
assert x is not None or t is not None
if x is None:
    x = self.cgf.dK(t)                     # cheap
elif t is None:
    t = self._dK_inv(x, **solver_kwargs)   # expensive: root finding per point
```

so `**solver_kwargs` are forwarded to the root finder and only matter when you
pass `x`. The core formula lives in `_spapprox_pdf`:

$$f(x) \approx \frac{1}{\sqrt{2\pi K''(t)}}\exp\big(K(t)-tx\big)$$

guarded by `~np.isclose(d2Kt, 0) & ~np.isnan(d2Kt)` -- where $K''(t)$ vanishes or
is undefined the result becomes `fillna` rather than infinity.

## Univariate cdf: LR and BN backends

```python
approx.cdf(x=None, t=None, fillna=np.nan, backend="LR", **solver_kwargs)
```

Both backends share the auxiliary quantities

$$w = \operatorname{sign}(t)\sqrt{2(tx - K(t))}, \qquad u = t\sqrt{K''(t)}.$$

**`"LR"` -- Lugannani-Rice (1980), the default:**

$$F(x) \approx \Phi(w) + \phi(w)\left(\frac{1}{w} - \frac{1}{u}\right)$$

**`"BN"` -- Barndorff-Nielsen (1986/1990):**

$$F(x) \approx \Phi\!\left(w + \frac{\log(u/w)}{w}\right)$$

Both expressions are singular at $t = 0$ (where $w = u = 0$) but the singularity
is removable, and both are patched with the same limit:

$$F(x) \approx \frac{1}{2} + \frac{K'''(0)}{6\sqrt{2\pi}\,K''(0)^{3/2}}$$

selected via `np.where(~np.isclose(t, 0), retval, ...)`. This is why `d3K0` must
be available for any CGF you intend to take a cdf of, and why the whole
raw/transformed cumulant caching machinery in `cgf_base.py` exists.

Note the implementations compute the singular expression anyway (inside
`np.errstate(divide="ignore", invalid="ignore")`) and discard it -- vectorized
masking rather than branching.

## `ppf`

```python
approx.ppf(q, fillna=np.nan, t0=None, ttol=1e-4, **kwargs)
```

Asserts `0 <= q <= 1`. If `fit_ppf()` has been called (`_cdf_cache` and
`_t_cache` both present) it is a plain `np.interp`. Otherwise it inverts the cdf
numerically with the same method cascade used by `cgf.dK_inv` -- `newton`,
`secant`, then bracketing methods -- locating a bracket by walking `+/-0.9**i` and
widening with Fibonacci scalings `1 - 1/fib(i)`.

## Normalization

The saddle point density does not integrate to one, so `pdf` divides by
`_pdf_normalization` unless `normalize_pdf=False`. The constant is computed by
integrating the *unnormalized* approximation in `t`-space, substituting
$dx = K''(t)\,dt$:

```python
quad(lambda t: self.pdf(t=t, normalize_pdf=False, fillna=0) * self.cgf.d2K(t, fillna=0), a, b)
```

over `infer_t_range()`, and multivariate uses `nquad` over `infer_t_ranges()`
with `det(d2K)` as the Jacobian. Note `fillna=0` here -- outside the domain the
integrand must vanish, not be NaN.

This integration is usually the dominant cost of the first `pdf` call. Pass
`pdf_normalization=` to the constructor if you already know it, and beware that
`examples/example_nonparametric_bootstrap.py` flags normalization as "not really
implemented well" for some heavy-tailed cases.

## Speeding things up: `infer_t_range`, `fit_*`, `clear_cache`

- `infer_t_range(atol=1e-4, rtol=1e-4)` -- finds `[lb, ub]` bracketing
  essentially all probability mass, by starting from the largest `+/-0.9**i` where
  `dK` is not NaN and expanding by Fibonacci-derived scalings until the cdf is
  within tolerance of 0 and 1. Asserts `lb <= 0 <= ub`.
- `fit_saddle_point_eqn(t_range=None, atol, rtol, num=1000, **solver_kwargs)` --
  fills `_x_cache = np.linspace(*cgf.dK(t_range), num)` and
  `_t_cache = cgf.dK_inv(_x_cache)`. Afterwards `_dK_inv` is `np.interp` on those
  arrays instead of per-point root finding.
- `fit_ppf(...)` -- fills `_cdf_cache` and `_t_cache` so `ppf` interpolates.
- `clear_cache()` -- drops all of the above plus the normalization constant.

Reach for these when evaluating at many `x` values; skip them when you can drive
everything from `t`.

## Sample mean

```python
UnivariateSaddlePointApproxMean(cgf, sample_size)
```

A thin subclass that wraps `cgf` in `univariate_sample_mean(cgf, sample_size)`
before delegating to `UnivariateSaddlePointApprox`. Combined with
`univariate_empirical(sample)`, this is the "bootstrap the transform domain"
application: the distribution of a bootstrapped sample mean without drawing a
single bootstrap sample.

## Multivariate and bivariate

`MultivariateSaddlePointApprox(cgf)` implements only the density,

$$f(\mathbf{x}) \approx (2\pi)^{-d/2}\,\lvert\det H\rvert^{-1/2}
\exp\big(K(\mathbf{t}) - \mathbf{x}\cdot\mathbf{t}\big)$$

with $H = K''(\mathbf{t})$, following Gatto (2000) Sec. 5.2. Explicitly **not**
implemented: `cdf`, `__getitem__` (marginal approximation) and `condition`
(conditional approximation) all raise `NotImplementedError`. The README lists
conditional distributions as future work.

`BivariateSaddlePointApprox` adds a `cdf` for $d = 2$, following
Wang (1990) / Paolella (2007). It approximates via a bivariate normal cdf term
$\Phi_2(\tilde{\mathbf{x}}, \rho)$ plus correction terms, and optionally uses the
`fastnorm` package (probed as `_has_fastnorm`) for the bivariate normal
evaluation. The $\mathbf{t} \approx 0$ singularity is handled by averaging
evaluations at $\pm 10^{-7}$ rather than by a closed-form limit.
