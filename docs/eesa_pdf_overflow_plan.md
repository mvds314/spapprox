# Plan: overflow warnings in `EstimatingEquationSaddlePointApprox.pdf`

## Problem

`examples/example_bootstrap_dirichlet.py` emits repeated
`cgfs.py:299: RuntimeWarning: overflow encountered in exp` (the `poisson` K). The stack is
`pdf -> _pdf_normalization -> _pdf_at -> _tilted_jacobian -> g -> K`, for the multinomial
(Poisson weights) case only.

Cause: `_pdf_normalization` integrates over `bracket = (x.min(), x.max())` including the
endpoints. There, `S(tau) = sum_i (x_i - tau) W_i` cannot change sign, so the saddle point
`s_S` escapes to -inf/+inf; `dK_inv` returns an unconverged value (e.g. -50) that still
satisfies the saddle point equation numerically. `_tilted_jacobian` then evaluates
`K(s; tau +- h)` with `exp(50 * 24)`, which overflows. Exponential weights do not warn because
their domain `t < 1` masks those arguments; Poisson has no domain bound.

The end result is NaN, which `nan_to_num` turns into 0, so today's numbers are fine and the
warning is cosmetic. That is partly luck: at the endpoints the *density* itself is wrong
(0.0077 and 0.0019, versus ~1e-10 just inside), and it is only the NaN Jacobian that hides
it. The saddle point does not always escape to infinity either (4.31 at `x.max`). The path
is also fragile: an overflow could produce `inf` instead of NaN, and `nan_to_num(inf)` =
1.8e308 would wreck the normalization.

## Scope

`EstimatingEquationSaddlePointApprox._pdf_at` (`spapprox/spa.py`) only:

1. Evaluate only the two `K` calls in `_tilted_jacobian.g` under
   `np.errstate(over="ignore", invalid="ignore")`; all warnings originate there, and
   `dK_inv` / `conditional_pdf` stay unsilenced so genuine interior failures still warn.
2. In `_pdf_at`, return `np.nan` when the result is not finite, consistent with the
   library's out-of-domain -> `fillna` convention. `pdf` maps NaN to `fillna`, and
   `_pdf_normalization` maps NaN to 0; no `inf` can reach `nan_to_num`.

Out of scope (known limitations): `MultivariateEstimatingEquationSaddlePointApprox._pdf_at`
(not reported); detecting support boundaries in `dK_inv`; the wrong endpoint density when the
Jacobian happens to be finite; finite-but-wrong Jacobians outside the bracket (negligible
density); and `pdf` failing at the endpoints without `given` (`AssertionError` /
`Could not find valid lb`), which is unrelated.

## Validation

- New regression test in `tests/test_spa.py`, next to `test_estimating_equation_cut`:
  Poisson weights with `given=n` on `_BB_X`. Under
  `warnings.simplefilter("error", RuntimeWarning)`, `pdf(..., normalize_pdf=False)` at
  `x.min()`, `x.max()` and an interior point emits no warning, returns `fillna` at the
  endpoints, and a finite value inside. Cheap: no full normalization.
- `test_estimating_equation_cut` covers unchanged interior values (rtol 1e-4).
- `pytest tests/test_spa.py -k estimating_equation`, then `pytest -m "not slow"`,
  `ruff check .`.
- Rerun `examples/example_bootstrap_dirichlet.py`: no overflow warnings, unchanged table.
- Bump the version in `pyproject.toml` (26.11.14 -> 26.11.15), required for the PR.

## Outcome

Implemented as planned, after a Code Architect review: `errstate` is scoped to the two `K`
calls in `_tilted_jacobian.g` (not all of `_pdf_at`), and `_pdf_at` maps non-finite results
to NaN. Regression test `test_estimating_equation_pdf_at_support_boundary`.

Validation:

- `pytest tests/test_spa.py -k "support_boundary or estimating_equation_cut"`: new test fails
  without the fix, both pass with it.
- `examples/example_bootstrap_dirichlet.py`: no overflow warnings, table unchanged.
- `ruff check spapprox tests examples docs`: passes.
- `pytest -m "not slow"`: left to CI at the user's request.

No documentation update beyond this plan: behavior for valid inputs is unchanged.
