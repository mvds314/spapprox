# spapprox — Agent Instructions

Saddle point approximation library: given a random variable's (or vector's) cumulant
generating function (CGF), approximate its pdf/cdf via the saddle point method
(Butler, 2007, "Saddlepoint Approximations with Applications").

## Setup / build

No `setup.py`/build step beyond editable install (flit backend):

```bash
pip install -e .
```

Install optional extras to exercise all functionality/tests:

```bash
pip install -e ".[findiff,numdiff,fastnorm]"
```

- `findiff` (>=0.11, uses the `Diff` API — `FinDiff` is deprecated) — fast numerical
  differentiation backend.
- `numdifftools` — alternative numerical differentiation backend (slower).
- `fastnorm` — faster bivariate normal cdf evaluation.

All three are optional at runtime: modules probe for them with `try/import` and expose
`_has_findiff`, `has_numdifftools`, `_has_fastnorm` booleans used to skip tests/branches
when a dependency is missing.

## Test / lint commands

```bash
pytest                                   # full suite
pytest tests/test_diff.py                # single file
pytest tests/test_diff.py::test_name     # single test
pytest -k "some_pattern"                 # by name pattern
pytest -m "not slow"                     # skip slow tests
pytest -m "not tofix"                    # skip known-broken/WIP tests
ruff check .                             # lint (line-length 99, see pyproject.toml)
```

Custom pytest markers (registered in `pyproject.toml`): `slow` and `tofix` (tests
documenting not-yet-fixed behavior). Many tests are parametrized with
`pytest.mark.skipif(not has_findiff, ...)` / `not has_numdifftools` since numerical
differentiation backends are optional dependencies — follow this pattern for any new
test that depends on findiff/numdifftools/fastnorm.

CI (`.github/workflows/python-app.yml`) runs on Python 3.11 and only installs
`numdifftools` before `pytest` — keep default (non-optional) code paths runnable
without findiff/fastnorm installed.

## Architecture

- `cgf_base.py` — core abstractions: `CumulantGeneratingFunction` (ABC) with
  `UnivariateCumulantGeneratingFunction` and `MultivariateCumulantGeneratingFunction`
  subclasses. A CGF wraps `K(t)` plus optional analytic `dK`/`d2K`/`d3K` callables; when
  not supplied, derivatives are computed numerically (backend selected via
  `numdiff_backend="numdifftools"|"findiff"` in the constructor) and cached as
  `dK0`/`d2K0`/`d3K0` (derivatives at `t=0`, i.e. moments/cumulants).
  CGFs support `loc`/`scale` (affine transforms of the underlying standardized
  variable) and `__add__` (summing independent random variables sums their CGFs).
- `diff.py` — thin wrapper around `findiff` providing `PartialDerivative`,
  `Gradient`, `Hessian`, `Tressian` (3rd-order tensor) used by
  `MultivariateCumulantGeneratingFunction` for numerical differentiation, plus tensor
  helpers (`transform_rank3_tensor`, `block_diag_3d`) for transforming derivative
  tensors under linear maps (needed because `loc`/`scale` transforms change the domain
  the derivatives are taken over).
- `cgfs.py` — concrete CGF constructors (`norm`, `exponential`, `gamma`, `poisson`,
  `binomial`, `chi2`, `laplace`, `multivariate_norm`, `bivariate_gamma`,
  `univariate_empirical`, `univariate_sample_mean`, ...). New distributions are added
  here as factory functions returning a `UnivariateCumulantGeneratingFunction` or
  `MultivariateCumulantGeneratingFunction`.
- `domain.py` — `Domain` describes the support of a CGF/distribution via scalar bound
  constraints (`l`, `le`, `g`, `ge`, requiring `g > ge > le > l`) and/or linear
  inequalities (`Ax <= a`, `Bx < b`). Used to restrict where the saddle point equation
  is solved.
- `spa.py` — the approximation layer: `SaddlePointApprox` (ABC) with
  `UnivariateSaddlePointApprox`, `UnivariateSaddlePointApproxMean`,
  `BivariateSaddlePointApprox`, `MultivariateSaddlePointApprox`. Given a `cgf`, solves
  the saddle point equation (`scipy.optimize`) for `t` given `x` (or vice versa) and
  evaluates the pdf/cdf approximation.
- `util.py` — `type_wrapper` decorator: converts pandas input to numpy for computation
  and wraps the result back to the caller's original input type (Series/array/scalar)
  via `statsmodels`' `PandasWrapper`. Used throughout `cgf_base.py`/`cgfs.py` on methods
  that take array-like `t`/`x` arguments. `Timer` is a small `with`-block stopwatch used
  for ad hoc profiling in examples.

Data flow: a `cgfs.py` factory builds a CGF → optionally combined via `+`/`loc`/`scale`
→ passed into a `spa.py` approximation class → `.pdf()`/`.cdf()` solve the saddle point
equation using `dK`/`d2K` (analytic or numerically differentiated via `diff.py`) →
results are wrapped back to the input container type by `util.type_wrapper`.

## Conventions

- Derivatives/moments follow the naming `dK`, `d2K`, `d3K` (1st/2nd/3rd derivative of
  the CGF) and `dK0`/`d2K0`/`d3K0` for their values at `t=0`; keep this naming for any
  new derivative-related code.
- Analytic derivatives (`dK`, `d2K`, `d3K`) are always expressed for the *standardized*
  (unscaled, untranslated) variable; `loc`/`scale` are applied afterward by the base
  class — don't bake `loc`/`scale` into a distribution factory's `dK`/`d2K`/`d3K`.
- `examples/` contains standalone demo/scratch scripts (including `example_findiff*.py`
  which still use the deprecated `FinDiff` API) — not part of the test suite; don't rely
  on them for behavior verification.
