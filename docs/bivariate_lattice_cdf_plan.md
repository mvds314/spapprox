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

## Out of scope (stay `NotImplementedError`)

- `delta` with `"Wang"` (Wang 1990, Theorems 2-3) and `"Kolassa"` (Kolassa 2003, eq. (13)).
- `fit_cdf(delta=)`, and `delta` for `MultivariateSaddlePointApprox.cdf` with `d > 2`.
- Mixed lattice/continuous components.

## Validation

- Independent Poissons: product of exact marginal cdfs.
- Dependent: :math:`(A+B, B+C)`, independent binomials, exact by convolution, and the upper
  orthant probabilities of Kolassa and Li (2010), Table 2 (:math:`A,B,C\sim` Bin(80, 0.2)).
- `delta=None` results unchanged; the conditional lattice tests still pass.
