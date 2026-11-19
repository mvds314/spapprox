# Lattice (and mixed) multivariate cdf: Kolassa (2003) eq. (13)

Goal: `MultivariateSaddlePointApprox.cdf(x, delta=...)` for random vectors whose components
lie on lattices, in any dimension, with `backend="Kolassa"`. Components may be mixed: some
on a lattice, others continuous. Today `delta` raises `NotImplementedError` in the
multivariate cdf, and only the bivariate `"KL"` backend supports (pure) lattices.

## Theory

Kolassa (2003), Lemma: the inversion integral (1) for `P[T >= t]` holds for a lattice with
`rho(tau) = 2 sinh(tau/2)` (unit span) and `t` corrected for continuity. The proof works
component by component (sum instead of integral), so it also holds for mixed vectors, with
`rho_j(tau) = tau` for a continuous component. For span `delta_j`,
`rho_j(tau) = 2 sinh(tau delta_j / 2) / delta_j`, as in `BivariateSaddlePointApprox._rho`.

Eq. (13), the lattice counterpart of eq. (5), adds to `_kolassa_orthant`:

- the prefactor `prod_j tau_j / rho_j(tau_j)`, and
- the linear term `sum_j a_j I(Sigma, s, e_j, b, tau)`, with
  `a_j = 1/tau_j - (delta_j / 2) coth(tau_j delta_j / 2)`. A factor `tau_j - tau_hat_j` in
  the integrand is `(-d_j - tau_hat_j) T`, as for the cubic term.

For a continuous component, the factor is 1 and `a_j = 0`. Near `tau_j = 0`, use the series
`tau / rho = 1 - (delta tau)^2 / 24 + ...` and `a = -delta^2 tau / 12 + ...` to avoid 0/0.
At `tau_j = 0` both reduce to the continuous values, so the continuity argument of
`_kolassa_cdf` across the hyperplanes `tau_j = 0` still holds.

Continuity correction: `P[X <= x] = P[-X >= -x]`; with `x` on the lattice, evaluate at
`x + delta/2` (`delta_j = 0` for continuous components). The orthant recursion of
`_kolassa_cdf` (`P_A = P_{A\j} - P_A^{sigma'}`) is exact in corrected coordinates: flipping
`P[X_j >= k]` to `P[X_j <= k - delta]` maps the corrected point `k - delta/2` to itself.
Reflection does not change `delta`, `tau/rho` is even and `a` is odd in `tau`, so passing
the reflected saddlepoint `sg * tau` is enough.

### When a component counts as continuous

The mixed formula needs every continuous component to have a bounded density given the
lattice components. A sum `N + G` of a lattice `N` and an independent continuous `G` is
continuous, so `delta_j = 0`. A component fixed by the lattice ones, e.g. `0.7 N`, is not:
it lies on a lattice with span `0.7 delta_N` and needs that `delta`.

If `G` is small compared with the span `delta` of `N`, the continuous approximation of
`N + G` is poor: the inversion integral has alias saddlepoints at `tau + 2 pi i k / delta`,
which it ignores. Their weight relative to the main saddlepoint is
`|M_G(tau + 2 pi i / delta) / M_G(tau)|`, `exp(-2 pi^2 sigma^2 / delta^2)` for normal `G`
with standard deviation `sigma`: 3e-9, 0.007, 0.17 for `sigma / delta` = 1, 0.5, 0.3. Rule
of thumb: continuous is fine for `sigma >~ delta / 2`; below about `delta / 4` the lattice
correction is closer. Document this in the `delta` docstring. Adding the alias terms, which
interpolates between both, is a follow-up in `docs/TODO.md`.

## Design

1. `spapprox/mvn.py`: `_kolassa_orthant(K, k2, k3, tau, tx, delta=None)`. `delta` is a
   vector of spans, 0 for continuous components. With `delta=None` or all zeros the result
   is unchanged. The linear term reuses the first derivatives already computed for the cubic
   term.
2. `spapprox/spa.py`, `MultivariateSaddlePointApprox`:
   - `cdf(x=None, t=None, fillna=np.nan, backend=None, delta=None, **solver_kwargs)`:
     `delta` is a scalar or one entry per component; `None` entries or 0 mean continuous;
     negative entries raise `ValueError`. Only `backend="Kolassa"` accepts it (else
     `NotImplementedError`). The cdf is evaluated at `x + delta/2`, with the saddlepoint
     solved there; if only `t` is given, `x = dK(t)` first, as in the bivariate cdf. The
     docstring states the bounded-density condition and the `sigma >~ delta / 2` rule above.
     The
     `fit_cdf` table is bypassed when `delta` is given.
   - `_kolassa_cdf(x, t, delta=None, ...)` passes `delta[idx]` to `_kolassa_orthant` for
     every subproblem in the recursion.
   - `_reject_lattice` remains for the methods that still have no lattice support.
3. `BivariateSaddlePointApprox.cdf`: `"Kolassa"` accepts any `delta` (forwarded to the
   multivariate path). `"KL"` (default) keeps pure lattices; mixed `delta` raises
   `NotImplementedError` naming `backend="Kolassa"`. `"Wang"` with `delta` still raises.

## Validation

- Independent Poissons, `d = 2, 3`: compare with the product of the one-dimensional eq. (13)
  results, and with the exact cdf.
- Trivariate Poisson sum (shared component, as in `_poisson_sum`) vs the exact cdf, for
  increasing means: error `O(1/n)`.
- Continuity across `tau_j = 0` (points around a marginal mean).
- `delta` with all zeros equals the continuous result.
- Mixed: Poisson + gamma (dependent through a shared Poisson term, or independent: product
  check) vs numerical summation/integration. Include a continuous component `N0 + G` that
  shares `N0` with a lattice component.
- Input validation: negative `delta`, wrong length, `delta` with `backend="KL"` in `d > 2`,
  mixed `delta` with bivariate `"KL"`.
- `examples/example_Kolassa2003.py`, example 2: full inclusion-exclusion (Boole's law) with
  the lattice pair and triple orthants. Target: the paper's 0.01709 (exact 0.01713). The
  current example only sums the three univariate tails, which likely explains the open
  discrepancy noted in `docs/TODO.md`.

## Out of scope

`fit_cdf(delta=)`, lattice `"Wang"` and mixed bivariate `"KL"` were done later, see
`docs/bivariate_lattice_cdf_plan.md`. The lattice `"KL"` for `d > 2` (all spans positive) and
the lattice `conditional_cdf` for three or more variables of interest (F2) came with phase B
of `docs/multivariate_cdf_plan.md`.
