# Plan: unconditional bivariate cdf for lattice variables

Follow-up of the Alhejaili and AlGhamedi (2025) review, see `docs/TODO.md` (lattice items, bug 62).

## Problem

`BivariateSaddlePointApprox.cdf` has no continuity correction: `delta` raises
`NotImplementedError`. Uncorrected, two independent Poissons (3, 4) at (2, 3) give 0.102 against
the exact 0.183.

## Method

Kolassa and Li (2010), Section 3: for unit lattice variables, the tail probability approximation
is the continuous one with

- the point corrected for continuity, and
- :math:`\rho(\tau)=2\sinh(\tau/2)` instead of :math:`\tau` in the factors
  :math:`1/\rho(\tau_j) - 1/w_j` of :math:`G(\tau)`; the leading term and its cubic correction are
  unchanged, since :math:`2\sinh(x/2)/x\to1`.

For span :math:`\delta_j`, :math:`\rho_j(\tau)=2\sinh(\tau\delta_j/2)/\delta_j`. This is what
`_ConditionalBivariateApprox` already does for the conditional case, Section 4, eq. (4.8).

With the reflection of `_kl_cdf`, the orthant is :math:`P(X\le x)` (lower) or
:math:`P(X\ge x+\delta)` (upper) in each component, both corrected to :math:`x+\delta/2`. So the
whole approximation is evaluated at :math:`x+\delta/2`, and the marginal cdfs of the
inclusion-exclusion are the lattice-corrected univariate cdfs at :math:`x`.

## Implementation

- Move `_rho` and the :math:`\rho` in `_kl_first_factor`, `_kl_marginal_factor` from
  `_ConditionalBivariateApprox` into `BivariateSaddlePointApprox` (`delta = None`: unchanged).
  The lattice marginal cdf goes in `_kl_marginal_cdf`.
- `_LatticeBivariateApprox(cgf, delta)`: private subclass holding `delta`, with a lattice
  `_kl_swapped`. Cached per `delta` by `BivariateSaddlePointApprox._lattice`.
- `BivariateSaddlePointApprox.cdf(..., delta=)`: scalar or one span per component, positive;
  backend `"KL"` only. Solves the saddle point equation at :math:`x+\delta/2`; a given `t` is
  only used to compute `x`.

## Out of scope

Nothing of the original list is left: `"Kolassa"` with `delta` (any `d`, also mixed) was done in
`docs/multivariate_lattice_cdf_plan.md`, the rest in the follow-ups below.

## Follow-ups (done)

1. **Mixed lattice/continuous components for `"KL"`.** Same convention as the `"Kolassa"`
   lattice cdf (`_parse_delta`): `None` or 0 entries are continuous. Continuous components keep
   :math:`\tau` (no :math:`\rho`) and no offset, lattice components use :math:`\rho` and
   :math:`x+\delta/2`; the marginal cdfs of the inclusion-exclusion are lattice corrected for
   lattice components and plain Lugannani-Rice for continuous ones. `_rho` and
   `_kl_marginal_cdf` per component; `cdf` no longer rejects mixed spans.
2. **`delta` for `"Wang"`.** Wang (1990), Theorems 2-3: in the factors
   :math:`1/w-1/u` of :math:`I_{12}, I_{21}, I_{22}`, :math:`t_0` is replaced by
   :math:`1-e^{-t_0}`, for span :math:`\delta`: :math:`(1-e^{-t_0\delta})/\delta`, without an
   offset. The kernel :math:`1/(1-e^{-t})` of his eq. (28) is that of an upper tail
   :math:`P(X\geq x)`, as in Daniels' first continuity correction, so the formula at
   :math:`x+\delta` approximates :math:`F(x)`, as `correction=1` of the univariate cdf. The
   review of Alhejaili and AlGhamedi has :math:`1-e^{t_0}`, which, at :math:`x` itself, is the
   reflected (lower tail) version. Both are checked numerically against exact values, the more
   accurate one is used (see the docstring and `docs/TODO.md`). The exact limits at
   :math:`t_j=0` get the extra term :math:`\mp\delta/(2\sqrt{K_{jj}})`, as in
   `_cdf_limit_term`. Mixed components work the same way, as for `"KL"`.
3. **`fit_cdf(delta=)`.** The lattice approximation, e.g. :math:`\hat F(\mathbf{x}+\delta/2)`
   with the lattice factors, is a smooth function of :math:`\mathbf{x}` whose values at the
   lattice points are the answers. So the table tabulates exactly what `cdf(x, delta=)` returns
   (the saddle point equations solved at the shifted grid), on the usual equidistant grid,
   and interpolates it. This needs no knowledge of the lattice origin, which `delta` does not
   fix (e.g. :math:`2N+1`), and interpolation error is that of the continuous case. A step
   lookup on lattice points would need the origin and gives nothing at off-lattice points that
   the user cannot get by flooring. The table stores `delta`; `cdf` only uses it for the same
   `backend` and `delta`.

## Validation

- Independent Poissons: product of exact marginal cdfs.
- Dependent: :math:`(A+B, B+C)`, independent binomials, exact by convolution, and the upper
  orthant probabilities of Kolassa and Li (2010), Table 2 (:math:`A,B,C\sim` Bin(80, 0.2)).
- `delta=None` results unchanged; the conditional lattice tests still pass.
- Follow-ups: `"Wang"` with `delta` on the same independent and dependent cases; mixed:
  Poisson and an independent gamma (product of the marginal cdfs), and a dependent
  :math:`(N_0+N_1, N_0+G)` by summation over :math:`N_0`; `fit_cdf(delta=)` reproduces the
  direct `cdf(x, delta=)` at lattice points.
