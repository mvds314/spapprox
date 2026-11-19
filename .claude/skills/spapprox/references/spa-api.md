# `spa.py` -- saddle point approximation API reference

```
SaddlePointApprox (ABC)
+-- UnivariateSaddlePointApprox
|   `-- UnivariateSaddlePointApproxMean
`-- MultivariateSaddlePointApprox
    `-- BivariateSaddlePointApprox

EstimatingEquationSaddlePointApprox   (standalone, wraps the above per tau)
MultivariateEstimatingEquationSaddlePointApprox   (standalone, vector tau)
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

**Lattice (`delta=span`):** `cdf(x, delta=...)` and `pdf(x, delta=...)` handle a discrete
variable; `x` must lie on the lattice. The cdf is the second continuity correction of Daniels
(1987) / Butler 1.2.3: $P(X\leq x)$ is evaluated at $x+\delta/2$ (so the saddlepoint solves
$K'(t)=x+\delta/2$; with `t=` given, it must solve there), with
$u = 2\sinh(t\delta/2)\sqrt{K''(t)}/\delta$; $w$ is unchanged. The $t=0$ limit is the same
as the continuous one, so the near-zero interpolation is reused. The pmf is
$P(X=x)\approx\delta f(x)$, not normalized by default (`normalize_pdf=None` means `True`
only for a continuous variable; integrating the density is a poor lattice normalization). `poisson(3)` at $x=3$: 0.646 vs exact 0.647.
`correction=1` selects the first continuity correction `Pr1` (Butler (1.27)-(1.29)): the
saddlepoint solves $K'(t)=x+\delta$, $u_1=(1-e^{-t\delta})\sqrt{K''(t)}/\delta$, and the $t=0$
limit gains $-\delta/(2\sqrt{K''(0)})$ inside the bracket (`_cdf_limit_term`). Reproduces Butler
Tables 1.7 and 1.12. `correction` without `delta` raises `ValueError` (`_check_correction`).

## `ppf`

```python
approx.ppf(q, fillna=np.nan, t0=None, ttol=1e-4, **kwargs)
```

Asserts `0 <= q <= 1`. If `fit_ppf()` has been called (`_cdf_cache` and
`_t_cache` both present) it is a plain `np.interp`. Otherwise it inverts the cdf
numerically with the same method cascade used by `cgf.dK_inv` -- `newton`,
`secant`, then bracketing methods -- locating a bracket by walking `+/-0.9**i` and
widening with Fibonacci scalings `1 - 1/fib(i)`.

`q == 0` and `q == 1` short-circuit to `support()`: the limits of $K'(t)$ at the
bounds of the domain of $K$, found by stepping $t$ geometrically towards each bound
and Aitken-extrapolating; divergence gives `-inf`/`inf`. Cached in `_support_cache`.

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

Multivariate: `fit_saddle_point_eqn(t_ranges=None, ..., num=1000)` builds a tensor grid in
`x` (per component, the marginal `cgf[j].dK` of the inferred `t` range, see `_x_axes`; the
joint `dK` at the corners of the `t` box can leave the domain). Grid points without a
solution (outside a non-rectangular support) are NaN in `_t_cache`. `_dK_inv` polishes the
linearly interpolated `t` by up to five Newton steps (the cdf backends need `x = dK(t)`
accurately; Wang raises otherwise), and solves directly, from the interpolated start, where
the residual stays large, interpolation gives NaN, or `x` lies outside the grid.
`fit_cdf(t_ranges=None, atol, rtol, num=None, backend=None, method="pchip")` tabulates the
cdf on such a grid (default `num = round(2500 ** (1 / d))`) and `cdf` then interpolates
(`RegularGridInterpolator`, `method` as given) when `backend` is `None` or the fitted one;
outside the grid, or for another backend, it evaluates directly. The default backend is
the class attribute `_cdf_default_backend` (`"Kolassa"`, `"KL"` for bivariate). The fit
solves the joint saddlepoints with warm starts (`_grid_saddles`, validated by
`dK(t) == x`), and memoizes marginal saddlepoints and cdfs through `_memoized` /
`_fit_memo` (set only during the fit). Failed or out-of-`[0, 1]` values are filled with
the max over the lower neighbours. About 35 s for a bivariate fit at the default grid.
Plan: `docs/multivariate_fit_cdf_plan.md`. `fit_cdf(delta=)` tabulates what
`cdf(x, delta=)` returns (the smooth corrected approximation, saddlepoints solved on the
shifted grid) on the usual grid, no lattice origin needed; stored in `_cdf_delta_cache`, and
`cdf` uses the table only for the same backend and `delta` (`_uses_cdf_table`). `cdf` routes
through `_lattice_spans` (parse, all-zero -> `None`, backend in `_lattice_backends`),
`_cdf_evaluator(backend, delta)` (offset, object, kwargs) and `_direct_cdf`.

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

with $H = K''(\mathbf{t})$, following Gatto (2000) Sec. 5.2. Marginals are
available as `spa[i]` / `spa[[i, j]]`; conditional distributions via `conditional_cdf` /
`conditional_pdf` (the old `condition` stub was removed).

`conditional_cdf(x, given, i=0)` and `conditional_pdf` give $X_i \mid X_J = x_J$
(equality conditioning only): Skovgaard's double saddlepoint,
$F \approx \Phi(w) + \phi(w)(1/w - 1/u)$ (Broda & Paolella 32.24-32.26), error
$O(n^{-1})$, exact for the normal. The numerator saddlepoint is the full one, the
denominator one is the marginal saddlepoint of $X_J$. Near $w=u=0$ the cdf is
cubic-interpolated. The pdf is normalized by default (`normalize_pdf=True`), per distinct
`given`, cached in `_conditional_pdf_normalization_cache`: the integral is done in $\tau_i$
along the `_ConditionalProfile` saddle path, with Jacobian the Schur complement, so no saddle
point equations are solved. `EstimatingEquationSaddlePointApprox` uses the unnormalized one. `backend="Kolassa"` selects the second order
Kolassa (1996) eq. (10), error about $n^{-3}$ in the gamma test (vs $n^{-1}$): the bracket
uses invariants $\rho_4,\rho_{13}^2,\rho_{23}^2$ of the joint cgf at $\hat\tau$ minus those of the
**marginal** of $X_J$ (J blocks) at $\tau_0$; $k_{ijkl}$ is a finite difference of `d3K`.
`delta=span` handles a lattice $X_i$ (second continuity correction, Kolassa eq. (11) for the
second order; $\hat u$ carries $1/\delta$ for scale invariance); `x` must lie on the lattice.
`correction=1` is Butler's `P1` (4.15)-(4.16): numerator saddlepoint at $x_i+\delta$,
$\tilde u_1=(1-e^{-\hat\tau_i\delta})\sigma/\delta$; Skovgaard backend, one variable of interest.
Plan: `docs/conditional_cdf_plan.md`.

`conditional_cdf(x, given, i=[a, b, ...])` gives the lower orthant
$P(X_I\le x_I \mid X_J=x_J)$, Kolassa & Li (2010) Sec. 4 (`x` of shape `(d0,)` or
`(n, d0)`, only `backend="KL"`, `delta` scalar or per component). `_ConditionalProfile` builds the profile pseudo-cgf
$M(\tau_I)=\min_{\tau_J}[K-\tau_J x_J]$ (inner Newton solve, continuation from $\tau_I=0$ if a
warm start is infeasible; $M''$ Schur complement, $M'''=K'''[V,V,V]$) and the weight
$r=\sqrt{\det K_{JJ}(\tau_0)/\det K_{JJ}}$. `_ConditionalBivariateApprox` runs the bivariate KL
code on $M$ through the hooks `_kl_marginal_cdf` (scalar Skovgaard conditionals) and
`_kl_weight` ($r(\tau)$, used at $(\hat\tau_1,0)$, $(0,\tilde\tau_2)$ and $\hat\tau$), plus
$\rho(\tau)=2\sinh(\tau\delta/2)/\delta$ for a lattice. For $d_0\ge3$ (F2),
`_ConditionalMultivariateApprox` runs the general-`d` KL engine (`kl.py`) on $M$ with the same
hooks; `_ConditionalBivariateApprox` derives from it and only adds the bivariate order swap.
Reproduces the paper's Table 3 to about
3 digits; with the $I^{\{1,2\}}$ term, which the paper drops, it is more accurate than the
paper. Both KL variants include $I^{\{1,2\}}$, so their relative error is $O(1/n)$ also at a
fixed large deviation point.
Plan: `docs/conditional_vector_plan.md`.

`cdf(x=None, t=None, fillna=np.nan, backend="Kolassa")` works in any dimension
(phase A of `docs/multivariate_cdf_plan.md`). `_kolassa_cdf` evaluates eq. (5) of
Kolassa (2003) for orthants in which the (reflected) saddlepoint is non-negative,
and otherwise recurses via $P_A^\sigma = P_{A\setminus j}^{\sigma_{-j}} -
P_A^{\sigma'}$, solving the marginal saddlepoint equations of the subvectors. The
1-D base case is also eq. (5) (Robinson-type), not Lugannani-Rice: that makes the
approximation exactly continuous across $\hat\tau_j = 0$; LR would jump by
$O(n^{-1})$. It has no singularities, is exact for the normal, and is permutation
invariant. Error $O(n^{-1})$, about 1.5x that of `"KL"` in 2-D. Independent components
do not give the product of the 1-D results (eq. (5) adds the cubic corrections), the
difference is $O(n^{-1})$.

`cdf(x=None, t=None, backend="KL")` (phase B, `spapprox/kl.py`): Kolassa & Li (2010) in any
dimension; `d = 2` reproduces `BivariateSaddlePointApprox` `"KL"`. `_KLSolver` caches the
profiles $\min L$ over the later components (nested minimizers), `_KLOrder` evaluates one
order of the variables (jets of the nested signed roots, the cubic term of $I^\emptyset$ via
`_kolassa_orthant`, all $2^d$ subset terms with nested divided differences), `_kl_orthant`
blends the $d!$ orders (first variable $j$ with weight $\propto\prod_{k\ne j}z_k^2$,
recursively) and `_kl_cdf` reflects and does inclusion-exclusion with LR marginals. Hooks on
the approximation class: `_kl_weight`, `_rho`, `_kl_marginal_cdf` (shared by the lattice and
conditional subclasses). About 0.7 s per point for `d = 3`, 1.7-2.6x more accurate than
`"Kolassa"` (absolute error, 3-D gamma). Default stays `"Kolassa"`. Derivation and caveats:
phase B of `docs/multivariate_cdf_plan.md`.

`cdf(x, delta=...)` (`"Kolassa"`, `"KL"`; bivariate also `"Wang"`): lattice spans, scalar or per
component, `None`/0 for continuous components, so lattice and continuous components may be
mixed (`_parse_delta`). Evaluated at $\mathbf{x}+\delta/2$; `_kolassa_orthant(..., delta=)`
applies eq. (13): the factor $\prod_j\hat\tau_j/\rho_j(\hat\tau_j)$ and the linear term
$-\sum_j a_j(\partial_j+\hat\tau_j)T$, $a_j=1/\hat\tau_j-(\delta_j/2)\coth(\hat\tau_j\delta_j/2)$
(`_lattice_factors`, series for small $\tau\delta$). The recursion is unchanged in corrected
coordinates. Reproduces Boole's law 0.01709 of Kolassa (2003) section 5 (example 2 of
`examples/example_Kolassa2003.py`). `fit_cdf(delta=)` tabulates it. Plan:
`docs/multivariate_lattice_cdf_plan.md`, including when a component counts as continuous.
With `backend="KL"` (`_LatticeMultivariateApprox`, eq. (4.8) of Kolassa and Li (2010) at
$\mathbf{x}+\delta/2$, via `_cdf_evaluator`) all spans must be positive for $d>2$; mixed spans
raise `NotImplementedError` there (the bivariate class supports them).

`spapprox/mvn.py` holds the numerics: `_mvn_logtail` (normal orthant probability
in log-space with **relative** accuracy, via Genz's separation of variables and a
tensor Gauss-Legendre rule in $y$-space; near machine precision for $d \le 4$),
`_mvn_tail_derivatives` (derivatives up to order 3, Kolassa (2003) eq. (8) plus
the product rule), and `_kolassa_orthant`. Do not use
`scipy.stats.multivariate_normal.cdf` there: it controls only the absolute error
(`releps` is ignored), which the factor $\exp(\hat\tau^T\Sigma\hat\tau/2)$
blows up in the tails.

`BivariateSaddlePointApprox` adds a `cdf` for $d = 2$, following
Wang (1990) / Paolella (2007). It approximates via a bivariate normal cdf term
$\Phi_2(\tilde{\mathbf{x}}, \rho)$ plus correction terms, and optionally uses the
`fastnorm` package (probed as `_has_fastnorm`) for the bivariate normal
evaluation. The $\mathbf{t} \approx 0$ singularity is handled by averaging
evaluations at $\pm 10^{-7}$ rather than by a closed-form limit.

**Lattice:** `BivariateSaddlePointApprox.cdf(x, delta=span)` (scalar or per component,
`None`/0 entries continuous, so mixed is allowed for every backend; `x` on the lattice).
`_cdf_evaluator` picks the shift and object: `"KL"` solves the saddle point at
$\mathbf{x}+\delta/2$ and runs `_LatticeBivariateApprox` (cached per `delta` in
`_lattice_cache`), which sets `self.delta`: `_rho` replaces $\tau_j$ by
$2\sinh(\tau_j\delta_j/2)/\delta_j$ in `_kl_first_factor` / `_kl_marginal_factor` / $I^{\{1,2\}}$,
and `_kl_marginal_cdf` uses the univariate lattice cdf, Kolassa and Li (2010) Section 3;
continuous components (`delta[k] == 0`) keep $\tau$ and the plain LR marginal.
`_ConditionalBivariateApprox` shares these hooks. `"Wang"` runs the same object at
$\mathbf{x}+\delta$ with `_wang_rho` $=(1-e^{-t\delta})/\delta$ in $u,\tilde u$ of
`_bivariate_formula` (Wang 1990 Thms 2-3; like univariate `correction=1`; the axis limit gets
$-\delta/(2\sqrt{K_{ii}})$). `"Kolassa"` forwards to the multivariate path. Independent
Poissons (3, 4) at (2, 3): KL 0.181, Wang 0.1835 vs exact 0.183. Wang is less accurate under
dependence (4% on the Kolassa-Li binomial sums vs 0.1% for KL). All-zero `delta` is the
continuous cdf.

## Estimating equations

`EstimatingEquationSaddlePointApprox(cgf, given=None, decreasing=True, bracket=None)`
approximates the distribution of the root $\hat\tau$ of $S(\tau)=0$. `cgf` is a
*callable* $\tau \mapsto$ cgf of $S(\tau)$ (univariate), or of $(T_1, S(\tau))$
(bivariate, $S$ last) when `given` $=t_1$ is set. For decreasing $S$,
$P(\hat\tau \le \tau) = P(S(\tau) \le 0 \mid T_1 = t_1)$, so `cdf` evaluates the
univariate LR cdf, or `conditional_cdf(0, given, i=1)`, at each $\tau$.

- `from_resampling(psi, x, weights, given)` builds `cgf` with the existing algebra:
  `from_univariate(*[weights] * n).ldot([1; psi(x, tau)])`. Poisson weights given $n$ =
  multinomial bootstrap (a cut: equals `UnivariateSaddlePointApproxMean` on
  `univariate_empirical`, tested), exponential given $n$ = Dirichlet bootstrap,
  Bernoulli(1/2) given $n-m$ = delete-$m$ jackknife.
- `pdf` = $|\mathbb{E}_{\hat s}[\partial_\tau S]| \cdot f_{S}(0)$ (Daniels 1983), with the
  tilted Jacobian computed generically as $\partial_\tau K(\hat s;\tau)/\hat s_S$ by
  central differences in $\tau$, interpolated linearly around $\hat s_S = 0$. Normalized
  by default over `bracket` (trapezoid, `_pdf_normalization_num` points): unnormalized it
  is about 10% too high for the jackknife example.
- `fit_cdf(bracket, num)` tabulates the cdf; `ppf` (interpolate + `brentq`), `mean` and
  `var` (integrals of the tabulated cdf) use it. Opt-in `interpolate=True` on `ppf` (skip
  `brentq`) and `cdf` (linear interpolation inside the table, direct evaluation outside)
  avoids further saddle point solves, at the accuracy of the table. NaNs near the support boundary are filled
  with 0/1, and the table is made monotone outward from the median, because the cdf can
  return spurious values right at the boundary.
- `delta=span` (plus `lattice_point(tau)`, a point on the lattice of $S(\tau)$; defaulted by
  `from_resampling` to `given * psi(x, tau)[0]`) evaluates the lattice corrected cdf at the
  largest lattice point $\le 0$ (decreasing) or $< 0$ (increasing). The estimator is then
  discrete, so `pdf` raises `ValueError`.
- Each $\tau$ builds a new cgf and solves a saddle point equation: about 0.07 s per
  conditional cdf point for $n=25$.

### Vector-valued estimating equations

`MultivariateEstimatingEquationSaddlePointApprox(psi, x, weights, given=None, dpsi=None,
bracket=None)` approximates the density of the root $\hat\tau\in\mathbb{R}^p$ of
$\sum_i \psi(x_i;\tau) W_i = 0$, with `psi(x, tau)` returning shape `(n, p)` and `dpsi`
shape `(n, p, p)` (central differences by default). Unlike the scalar class it needs the
weighted structure, since the scalar Jacobian trick $\partial_\tau K/\hat s_S$ has no matrix
analogue:

$$f(\tau) \approx \Big|\det \textstyle\sum_i K_W'(a_i)\,\partial\psi_i/\partial\tau^\top\Big|
\, f_{T,S}(t, 0) / f_T(t), \qquad a_i = s_T + s_S^\top \psi_i .$$

- The joint saddlepoint is solved directly (damped Newton, vectorized weight cgf, warm
  start from the previous point); no cgf objects are built per point, about 3-7 ms per point.
  NaN where it fails, e.g. outside the convex hull.
- `fit_pdf(bracket, num)` tabulates on a `num**p` tensor grid; normalization, `marginal_pdf`,
  `marginal_cdf`, `marginal_ppf`, `mean` and `cov` use the table.
- Checks (tested): `p=1` equals the scalar class's unnormalized pdf; the vector mean under
  Poisson weights given $n$ equals the multivariate empirical-mean density (the cut).
