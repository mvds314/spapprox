---
name: spapprox
description: Complete map of the spapprox codebase -- a Python library for saddle point approximation of distributions from cumulant generating functions (CGFs). Use this skill whenever working in the spapprox repository or whenever the user mentions saddle point approximations, cumulant generating functions, CGFs, K(t)/dK/d2K/d3K derivatives, Lugannani-Rice or Barndorff-Nielsen cdf approximations, the SaddlePointApprox classes, transform-domain bootstrapping, or asks where something lives / how something works in this package -- even if they only name a single file like cgf_base.py, spa.py, diff.py, or domain.py.
---

# spapprox

`spapprox` approximates the pdf/cdf of a random variable (or vector) from its
**cumulant generating function** (CGF), following Butler (2007), *Saddlepoint
Approximations with Applications*. The README marks the package "UNDER
CONSTRUCTION" -- expect rough edges and `tofix`-marked tests.

## Why the design looks the way it does

The whole library rests on one fact: for a random variable $X$,

$$K(t) = \log \mathbb{E}[e^{tX}]$$

determines the distribution, is **additive over independent variables**, and has
derivatives at zero equal to the cumulants ($K'(0)=\mathbb{E}X$,
$K''(0)=\operatorname{Var}X$). So instead of convolving densities, you *add* CGFs
and then recover the density numerically. That is why the codebase is organized
as "build a CGF object -> do algebra on it -> hand it to an approximation class",
and why so much machinery exists for derivatives and domains: the saddle point
formula needs $K$, $K'$, $K''$ (and $K'''$ at zero), evaluated only where $K$ is
actually finite.

The core density formula is

$$f(x) \approx \frac{1}{\sqrt{2\pi K''(t)}}\exp\big(K(t)-tx\big),
\qquad \text{where } t \text{ solves } K'(t)=x.$$

That equation $K'(t)=x$ is *the saddle point equation*, and solving it -- or
cleverly avoiding solving it -- drives most of the API surface below.

## Layout

```
spapprox/
+-- util.py        (32 loc)   type_wrapper, Timer, fib
+-- diff.py       (569 loc)   numerical differentiation via findiff
+-- domain.py     (587 loc)   Domain: where K is finite
+-- cgf_base.py  (1919 loc)   CGF base + Univariate/Multivariate classes  <- the heart
+-- cgfs.py       (276 loc)   concrete distribution factories
+-- spa.py        (799 loc)   the saddle point approximation classes
`-- __init__.py               public API re-exports
tests/            test_cgfs, test_mvar_cgfs, test_diff, test_domain, test_spa
examples/         standalone matplotlib demo scripts (not part of the test suite)
```

Dependencies run strictly one way:
`util / diff / domain -> cgf_base -> cgfs -> spa`. Nothing in `diff.py` or
`domain.py` knows about CGFs; keep it that way when adding code.

Public API (`spapprox/__init__.py`): the two CGF classes, `Domain`, the four
approximation classes, `Timer`, and the distribution factories `norm`,
`multivariate_norm`, `exponential`, `gamma`, `bivariate_gamma`, `chi2`,
`laplace`, `poisson`, `binomial`, `univariate_empirical`,
`univariate_sample_mean`.

## The typical flow

```python
import numpy as np, spapprox as spa

cgf = spa.norm(loc=0, scale=1)          # or gamma(...), univariate_empirical(sample), ...
cgf = cgf + spa.exponential(scale=2)    # sum of independent variables == sum of CGFs
approx = spa.UnivariateSaddlePointApprox(cgf)

t = np.linspace(-3, 3)
x = cgf.dK(t)                           # cheap: x from t, no root finding
approx.pdf(t=t)                         # or approx.pdf(x=x) -> solves K'(t)=x
approx.cdf(t=t)                         # Lugannani-Rice by default
```

**The `x` vs `t` duality is the single most important performance concept.**
Every `pdf`/`cdf` accepts either. Given `t`, the code computes `x = cgf.dK(t)`
directly. Given `x`, it must invert via `cgf.dK_inv(x)` -- numerical root finding,
per point. So when sweeping a curve, generate `t` and derive `x`. When you
genuinely need specific `x` values repeatedly, call
`approx.fit_saddle_point_eqn()` first: it tabulates the mapping over an inferred
`t` range and switches `_dK_inv` to `np.interp`. `fit_ppf()` does the same for
quantiles, and `clear_cache()` discards all of it.

## Module reference

For depth beyond the summaries below, read the reference files -- they exist so
this file stays navigable:

- `references/cgf-api.md` -- the CGF classes: construction, algebra, derivative
  plumbing, caching, and multivariate specifics (`ldot`, `__getitem__`,
  `from_univariate`).
- `references/spa-api.md` -- the approximation classes: pdf/cdf/ppf, the LR and
  BN cdf backends, normalization, and the fitting/interpolation accelerators.
- `references/diff-and-domain.md` -- `diff.py` and `domain.py` internals.

### `cgf_base.py` -- the heart

`CumulantGeneratingFunction` (ABC) -> `UnivariateCumulantGeneratingFunction` and
`MultivariateCumulantGeneratingFunction`.

A CGF wraps a callable `K` plus *optional* analytic `dK`, `dK_inv`, `d2K`, `d3K`,
a `Domain`, and optional cumulants-at-zero `dK0`/`d2K0`/`d3K0`. Anything not
supplied is computed numerically and cached.

Three conventions matter more than anything else here:

1. **Everything you pass in describes the *standardized* variable.** With
   `loc`/`scale`, the object represents $X = \text{scale}\cdot Z + \text{loc}$
   where your `K` is $K_Z$. The class applies the transform itself:
   $K_X(t) = K_Z(\text{scale}\cdot t) + \text{loc}\cdot t$ and
   $K_X'(t) = \text{scale}\cdot K_Z'(\text{scale}\cdot t) + \text{loc}$.
   So never bake `loc`/`scale` into a factory's `dK`/`d2K`/`d3K` -- they would be
   applied twice. Multivariate is the matrix analogue: `loc` is a vector,
   `scale` a matrix (a 1-D `scale` means a vector of standard deviations, i.e. a
   diagonal matrix), and `K(t) = K_Z(scale^T t) + <loc, t>`.

2. **Out-of-domain input is masked, not passed through.** Every evaluation does
   the same dance: compute the scaled argument, `cond = domain.is_in_domain(st)`,
   *substitute 0 for the out-of-domain entries* so the user's `K` never sees them
   (it might overflow, warn, or raise), evaluate, then
   `np.where(cond, val, fillna)`. Replicate this in any new evaluation method --
   skipping the substitution step is the usual source of spurious overflow
   warnings.

3. **`_raw` means unscaled/untranslated; the public name means transformed.**
   `_dK0_raw` is $K_Z'(0)$, `dK0` is $K_X'(0)$. Caches follow the same split
   (`_dK0_raw_cache` vs `_dK0_cache`), and the algebra methods deliberately
   forward the raw caches into the new object so hard-won numerical work
   survives.

Algebra (all return new objects unless `inplace=True`, and all propagate the
domain through the matching `Domain` operation):

| Expression | Meaning |
|---|---|
| `cgf + 3`, `cgf - 3` | shift: adjusts `loc` |
| `cgf1 + cgf2` | **sum of independent variables**; domains intersect |
| `2 * cgf`, `cgf / 2` | scaling: adjusts `loc` and `scale` |
| `mcgf[i]`, `mcgf[[0, 2]]`, `mcgf[1:]` | marginals (other components set to zero) |
| `mcgf.ldot(A)` | $AX$; matrix -> multivariate, **vector -> univariate** |
| `mcgf.stack(other)` / `from_cgfs(...)` | join into a bigger random vector |
| `MultivariateCumulantGeneratingFunction.from_univariate(*cgfs)` | stack *independent* univariates |

The numerical differentiation backend is chosen per object via
`numdiff_backend="numdifftools"` (default) or `"findiff"`, and the derivative
object is built lazily on first use. Note that
`MultivariateCumulantGeneratingFunction.d3K` is **findiff-only** (there is a TODO
for a numdifftools path).

### `cgfs.py` -- distributions

Plain factory functions returning a configured CGF. Each supplies as many
analytic derivatives as are cheaply available, plus a `Domain` when the CGF is
not finite everywhere. Note the domain is stated for the **standardized**
variable, consistent with rule 1 above: `exponential` and `gamma` both declare
`Domain(l=1)`, i.e. $t < 1$ for $Z$, which the scale transform turns into
$t < 1/\text{scale}$ for $X$. Add new distributions here, matching this shape:

```python
def norm(loc=0, scale=1):
    return UnivariateCumulantGeneratingFunction(
        K=lambda t: t**2 / 2,
        dK=lambda t: t,
        dK_inv=lambda x: x,
        d2K=lambda t: np.ones(np.asanyarray(t).shape),
        d3K=lambda t: np.zeros(np.asanyarray(t).shape),
        loc=loc, scale=scale,
    )
```

Two factories are not distributions but transformations, and they are what make
the bootstrap use case work: `univariate_empirical(x)` is the CGF of drawing
uniformly from a sample, and `univariate_sample_mean(cgf, n)` maps
$K \mapsto nK(t/n)$. Composed, they give the bootstrap distribution of a sample
mean *without bootstrapping* -- see `examples/example_nonparametric_bootstrap.py`.

### `spa.py` -- the approximations

- `UnivariateSaddlePointApprox(cgf)` -- `pdf`, `cdf`, `ppf`.
- `UnivariateSaddlePointApproxMean(cgf, sample_size)` -- sample mean; thin
  subclass wiring in `univariate_sample_mean`.
- `MultivariateSaddlePointApprox(cgf)` -- `pdf` only; `cdf`, marginals
  (`__getitem__`) and `condition` raise `NotImplementedError`.
- `BivariateSaddlePointApprox(cgf)` -- adds a bivariate `cdf`.

`cdf` has two interchangeable backends selected by `backend=`: `"LR"`
(Lugannani-Rice, the default) and `"BN"` (Barndorff-Nielsen). Both are singular
at $t=0$ and both special-case it with a $K'''(0)$-based expression -- which is
why `d3K0` must be obtainable for any CGF you intend to take a cdf of.

`pdf(..., normalize_pdf=True)` is the default and triggers numerical integration
(`quad`/`nquad`) of the unnormalized approximation, cached in
`_pdf_normalization_cache`. The saddle point density is not guaranteed to
integrate to one, hence the correction; it is also the expensive part, so pass a
precomputed `pdf_normalization=` to the constructor when you have one.

### `diff.py`, `domain.py`, `util.py`

`diff.py` wraps `findiff` (**version >= 0.11, using the `Diff` API -- `FinDiff` is
deprecated**). `PartialDerivative` builds a small finite-difference grid around
the requested point and applies a findiff operator; `TensorDerivative` and its
subclasses `Gradient`, `Hessian`, `Tressian` assemble full derivative tensors of
order 1/2/3. `transform_rank3_tensor` and `block_diag_3d` exist because a
`loc`/`scale` transform has to be pushed through a rank-3 derivative tensor,
which numpy has no one-liner for.

`domain.py` defines where a CGF is finite: scalar bounds `l`, `le`, `ge`, `g`
(strict/inclusive lower and upper, required to satisfy `g > ge > le > l`) plus
linear inequalities `A x <= a` and `B x < b`. `Domain` mirrors the CGF algebra
(`add`, `mul`, `ldot`, `ldotinv`, `intersect`, `stack`, `from_domains`) so
domains propagate automatically when CGFs are combined. `is_in_domain(t)` is the
mask used throughout `cgf_base.py`.

`util.py` is tiny but pervasive: `type_wrapper(xloc=N)` decorates nearly every
evaluation method, converting the `N`-th argument to a numpy array and wrapping
the result back into the caller's container type (pandas in -> pandas out; 0-d
array out -> plain Python scalar). If you add a public method taking array-like
`t` or `x`, decorate it too, or you will silently break the pandas round-trip
that the tests check.

## Working in this repo

```bash
pip install -e ".[findiff,numdiff,fastnorm]"   # editable install with all extras
pytest                                          # full suite (~11 min)
pytest tests/test_diff.py                       # one file
pytest tests/test_spa.py::test_name             # one test
pytest -m "not slow"                            # skip slow tests
pytest -m "not tofix"                           # skip known-broken tests
ruff check .                                    # line-length 99
```

Optional dependencies and their probe flags -- all three are genuinely optional
at runtime, and CI installs **only** `numdifftools`, so any code path reachable
by default must work without the other two:

| Package | Flag | Defined in | Purpose |
|---|---|---|---|
| `findiff>=0.11` | `_has_findiff` | `diff.py` | fast finite differences |
| `numdifftools` | `has_numdifftools` | `cgf_base.py` | default numdiff backend |
| `fastnorm` | `_has_fastnorm` | `spa.py` | fast bivariate normal cdf |

Tests gate on these with
`pytest.mark.skipif(not has_findiff, reason="No findiff")`, frequently combined
with `pytest.mark.slow` inside `pytest.param(..., marks=[...])` lists. Follow
that pattern for new tests touching an optional backend. The markers `slow` and
`tofix` are registered in `pyproject.toml`; `tofix` documents known-failing
behavior rather than serving as a to-do note.

Numerical tests compare against `scipy.stats` reference distributions with
tolerances. When a change shifts results slightly, work out whether it is an
accuracy regression or a legitimately better approximation before touching a
tolerance.

## Things that will bite you

- **Docstrings are LaTeX-heavy reStructuredText in raw strings** (`r"""`) with a
  `References` section citing the literature. Match this when adding public API;
  the math *is* the documentation here.
- **Caching is by `_..._cache` attribute plus `hasattr` checks**, not
  `functools.cache`. Code frequently branches on
  `hasattr(self, "_dK0_cache") or self._dK0_raw_cache is not None` to decide
  whether a cumulant is cheaply available. `SaddlePointApprox.clear_cache()`
  reflectively deletes every attribute ending in `_cache`, so any new cached
  value should use that suffix to stay discoverable.
- **`fillna` is a parameter, not a policy.** Out-of-domain and degenerate results
  become `fillna` (default `np.nan`), and several internal callers deliberately
  pass `fillna=0` during integration. Preserve the parameter when wrapping or
  overriding evaluation methods.
- **`examples/` is untested and partly stale** -- `example_findiff.py`,
  `example_findiff_multivariate.py` and `example_gen_lin_opr.py` still use the
  deprecated `FinDiff` API. Do not treat them as behavioral references, and note
  they need `matplotlib`, which is not a declared dependency.
- **Multivariate support is incomplete by design, not by accident.** Several
  methods raise `NotImplementedError` (multivariate cdf, marginal and conditional
  approximations). The README lists conditional distributions and quotients of
  random variables as future work.
