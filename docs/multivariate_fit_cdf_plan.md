# Plan: `fit_cdf` for multivariate saddlepoint approximations

Status: done. See "As built" at the end for deviations from the plan. Implements the `NotImplementedError` stub
`MultivariateSaddlePointApprox.fit_cdf`, see `docs/TODO.md` ("`fit_cdf` for the multivariate
case is not implemented").

## Goal

Evaluating the multivariate cdf is expensive: one saddlepoint solve per point plus extra
root solves. Measured on a dependent bivariate gamma:

| backend | per point | of which root solving |
| ------- | --------- | --------------------- |
| `"KL"` (bivariate default) | ~60 ms | ~50% |
| `"Kolassa"`, d = 2 | ~20 ms | ~25% |
| `"Kolassa"`, d = 3 | ~35 ms | |

`fit_cdf` tabulates the cdf once on a tensor grid in `x`. After that, `cdf(x=...)`
interpolates the table, just as `fit_saddle_point_eqn` makes `_dK_inv` interpolate. This
helps plotting, likelihood or test-statistic evaluation over many points, and repeated
rectangle probabilities.

## Design

### Grid

- Build the grid per component in `x`. Compute
  `t_ranges[j] = UnivariateSaddlePointApprox(cgf[j]).infer_t_range(atol, rtol)` (already
  `infer_t_ranges`), map it with the **marginal** `cgf[j].dK`, and use
  `np.linspace(*x_range_j, num)`.
- This replaces the corner construction of `fit_saddle_point_eqn`, which evaluates the
  joint `dK` at the corners of the t box. Those corners can fall outside the domain: for
  `X = A Z` with independent gammas, `A^T t` leaves `t < 1` at corner `(0.9, 0.9)`. That
  gives NaN x-ranges and a `RegularGridInterpolator` error. This fails today for
  `from_univariate(gamma(3), gamma(2)).ldot([[1, 0], [0.5, 1]])`. We share a helper
  `_x_grid(t_ranges, num)` and fix `fit_saddle_point_eqn` in the same PR.
- Default `num` keeps the table near 2,500 points: `num = round(2500 ** (1 / d))`, which
  gives 50 for d = 2 and 14 for d = 3.

### Evaluation on the grid

- Joint saddlepoints: one `dK_inv` per grid point. Reuse `_t_cache` when
  `fit_saddle_point_eqn` was fitted on the same grid.
- Marginal saddlepoints depend only on the coordinates involved, so on a tensor grid
  they repeat. Both backends solve them:
  - `_kl_cdf_ordered`: `cgf[i].dK_inv(x[i])`;
  - `_kolassa_cdf`: `saddle(A)` for every subset `A`.
  
  During the fit we memoize them in a dict keyed by `(A, x_A)`. For d = 2 that is
  `2 * num` solves instead of `2 * num**2`. For the bivariate KL it removes about half of
  the cost. The memo is a `_..._cache` attribute, so `clear_cache` drops it.
- Points that come out NaN (no saddlepoint, outside the support): use 0 on the lower
  boundary faces. Elsewhere, the max over the already filled lower neighbours, which
  keeps the table monotone. Points outside the support of a lattice-free variable are rare
  with inferred ranges.
- `backend` defaults to the class default (`"KL"` for bivariate, `"Kolassa"` otherwise).
  It is stored with the table.

### Interpolation and use

- `self._cdf_interp_cache = RegularGridInterpolator(axes, F, method="linear")`. Multilinear
  interpolation of a table whose rectangle masses are non-negative is itself a valid cdf.
- `cdf(x=..., backend=None)` uses the table if it is fitted and `backend` is `None` or
  matches the fitted backend; otherwise it evaluates directly as now. `cdf(t=...)` maps
  `x = dK(t)` first. `fillna` and the pandas wrapping are kept.
- Points outside the grid fall back to direct evaluation, like the univariate `_dK_inv`.
  Clamping to the grid would be cheaper, but biased by up to the left-out tail mass
  `atol`.
- Monotonicity is not enforced. We report the minimum rectangle mass of the table in the
  tests; if it turns out negative in practice, add an option later.

### Signature

```python
def fit_cdf(self, t_ranges=None, atol=1e-4, rtol=1e-4, num=None, backend=None, **solver_kwargs)
```

This mirrors `fit_saddle_point_eqn(t_ranges, atol, rtol, num, **solver_kwargs)`. The stub's
`t_range` is renamed to `t_ranges`, consistent with the multivariate class. Nobody can
depend on the stub, since it raises.

## Checks (tests)

- Normal, d = 2 and 3 (Kolassa is exact for the normal): the fitted cdf equals
  `scipy.stats.multivariate_normal.cdf` up to the interpolation error, about 1e-3.
- Dependent bivariate gamma, `"KL"`: at random off-grid points the fitted cdf matches the
  direct `cdf` to within about 2e-3. On grid nodes it matches exactly.
- The `ldot` gamma cgf above: `fit_saddle_point_eqn` and `fit_cdf` no longer fail
  (regression test).
- Outside the grid, the result equals direct evaluation. With `backend` different from the
  fitted one, `cdf` evaluates directly. `clear_cache` removes the table.
- Timing is not tested, but the fit should take about 1 min or less for the d = 2 KL
  default.

## Not done

- Multivariate `ppf` (not uniquely defined), and rectangle probability helpers (easy to add
  on top of the table later).
- Interpolation on a normal-score scale `Phi^-1(F)`, which would be more accurate in the
  tails. Revisit if the linear table is too coarse.
- Sparse or adaptive grids for d >= 4; the tensor grid grows as `num**d`.

## As built

Deviations from the plan above, and what we learned implementing it:

- **Interpolation is tensor PCHIP by default** (`method="pchip"`, any
  `RegularGridInterpolator` method can be passed). Linear was too coarse: against an exact
  table, the error of the default grid is about 0.0056 (d = 2) and 0.021 (d = 3, 14 points
  per axis). PCHIP gives about 0.0022 and stays monotone along each axis; cubic is more
  accurate (about 0.0005) but can overshoot. PCHIP costs about 0.5 ms per evaluated point,
  versus about 1 µs for linear, still far below a direct evaluation. PCHIP needs at least
  4 points per axis.
- **Joint saddlepoints** are solved by `_grid_saddles` (`method="hybr"`, warm starts
  from the neighbours below, then a cold start) instead of reusing `_t_cache`: 13 s for
  2,500 points instead of 115 s. A solution counts only if `dK(t) == x` within `1e-6`:
  for a non-rectangular support, e.g. `X = (Z1, Z1 / 2 + Z2)` with support
  `x2 > x1 / 2`, the tensor grid contains points outside the support, where the solver
  returns a point that is no solution. `fit_saddle_point_eqn` applies the same check, and
  `_dK_inv` solves directly where interpolation gives NaN or `x` is outside the grid.
- **Marginal saddlepoints** are solved per axis by `_axis_saddles`, warm started outward
  from the mean (0.2 s instead of 8 s), and seeded into the memo. The memo, `_fit_memo`, is
  set only during the fit (not a `_cache` attribute) and covers the KL marginal
  saddlepoints and cdfs (also for its swapped ordering), the Kolassa subset saddlepoints,
  and Wang's inner saddlepoint.
- **Failures:** values outside `[-1e-3, 1 + 1e-3]` count as failures too, then all
  failures are filled with the max over the lower neighbours (0 on the lower faces), and
  the table is clipped to `[0, 1]`. Monotonicity is not enforced otherwise; for the KL
  table of the `ldot` gamma, a few rectangle masses along the support boundary are
  negative.
- **Found along the way:** univariate `dK_inv` failed when `x` equals `dK` at the end of
  its initial bracket (fixed, `docs/TODO.md` item 58); the bivariate `"KL"` cdf fails near
  the domain boundary (item 59) and `"Wang"` raises for the `ldot` gamma (item 60), both
  open.
- **Timings** (d = 2, 2,500 points): KL about 34 s, Kolassa about 35 s. d = 3 normal,
  Kolassa, 14^3 points: about 91 s, dominated by Kolassa itself.
