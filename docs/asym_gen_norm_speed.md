# Plan: faster cgf of the (asymmetric) generalized normal

Status: option A done (PR #58).

Goal: make `generalized_normal` and `asymmetric_generalized_normal` (in `spapprox/cgfs.py`)
fast enough for routine use, with an accuracy of about `1e-8` in `K`, `dK`, `d2K`, `d3K`,
without changing the public interface. The exact special cases stay: shape 1 (Laplace,
asymmetric Laplace) and shape 2 (normal, split normal).

## Current state

For shapes other than 1 and 2 the cgf has no closed form. For every `t` separately,
`tilted(t)` integrates the exponentially tilted density with `scipy.integrate.quad_vec`
(adaptive, `epsrel=1e-12`), in the variable `v = (x - mode) / w` around the mode of the tilted
density, split at the mode, at the kink `x = 0` and at a few fixed breakpoints. `K`, `dK`,
`d2K`, `d3K` are each a separate `np.vectorize` of `tilted`, so every derivative redoes the
same integral.

Measured on `generalized_normal(beta=1.5)`, before this change:

| Operation | Time |
| --------- | ---- |
| `K` at 50 points | 2.5 s |
| `pdf(t=...)` at 50 points (normalized) | 40 s |
| `cdf(x=...)` at 2 points | 1.6 s |
| `cdf(t=...)` of the sample mean, `n = 10`, at 50 points | 8.7 s |

## What has to be integrated

With the standardized density `c exp(-|x/sigma_L|^beta_L)` for `x < 0` and
`c exp(-(x/sigma_R)^beta_R)` for `x >= 0`, for each `t`:

```
I_k(t) = int x^k exp(t x - |x/sigma|^beta) dx,   k = 0, 1, 2, 3
K = log(c I_0),  dK = m1,  d2K = m2 - m1^2,  d3K = m3 - 3 m1 m2 + 2 m1^3,  m_k = I_k / I_0
```

Example, `beta = 1.5`, `sigma = 1`, `t = 1`: `I_0..I_3 = 2.658, 2.152, 4.278, 8.752`,
`K = 0.38684`, `dK = 0.80941`, `d2K = 0.95426`, `d3K = 0.44509`.

## Role of the shape `beta`

- Existence: for `beta > 1` the cgf is finite for all `t`; for `beta = 1` only for
  `-1/sigma_L < t < 1/sigma_R`; for `beta < 1` only at `t = 0`, so no saddle point
  approximation exists (all moments are finite, but the mgf is not). In the asymmetric case
  `beta_R` governs `t > 0` and `beta_L` governs `t < 0`.
- Location: the tilted density peaks at `x* = sigma (t sigma / beta)^(1 / (beta - 1))`. For
  `beta` near 1 this moves out very fast, e.g. `beta = 1.1`, `t = 2`: `x* ~ 400`, width
  `~ 44`, exponent `~ 72`. Centering, rescaling and subtracting the maximum exponent are
  essential.
- Kink at zero: `|x|^beta` is smooth only for even integer `beta`; for `beta = 1.5` the second
  derivative is infinite at 0, which slows Gauss-type rules. It matters for small `|t|`, when
  the mode is close to 0.
- Tails: decay like `exp(-x^beta)`, long for `beta` near 1, box like with steep edges at
  `|x| ~ 1` for large `beta`.
- `beta` cannot be scaled out (`sigma` can, `K_sigma(t) = K_1(sigma t)`): it is a genuine
  shape parameter. A substitution `u = x^beta` only moves it into the tilt term.

The hard cases are `beta` near 1 and large `beta`.

## Special functions

The one sided integral is a Fox-Wright function:

```
J_beta(s) = int_0^inf exp(s y - y^beta) dy = (1 / beta) sum_k Gamma((k + 1) / beta) s^k / k!
```

and `M(t) ~ sigma_R J_beta_R(t sigma_R) + sigma_L J_beta_L(-t sigma_L)`. Special cases:
`beta = 1`: `1 / (1 - s)`; `beta = 2`: `sqrt(pi) exp(s^2 / 4) Phi(s / sqrt(2))` (used for the
split normal); `beta = 3`: the Scorer function `Hi`; rational `beta`: Meijer G / sums of
`pFq`. No fast implementation for general `beta` exists: not in scipy or Boost; mpmath is
arbitrary precision and slow. Published Chebyshev type approximations exist only for the
Scorer functions (Gil, Segura, Temme). The R package `mhn`, for a related Fox-Wright
function, also falls back to quadrature around the peak. As a function of two continuous
parameters `(beta, s)` (four outputs, plus the ratio `sigma_L / sigma_R`), there is no
precomputed approximation to reuse.

## Options

Common to all: compute `K`, `dK`, `d2K`, `d3K` in one pass, and cache the last result so that
`K(t)` followed by `dK(t)` does not recompute (3-4x by itself, no loss of accuracy).

### A. Vectorized fixed node quadrature (chosen)

Keep the centering in `v = (x - x*) / w` and the max subtraction, but replace the adaptive
per point `quad_vec` with a fixed tanh-sinh (double exponential) rule, evaluated for all `t`
at once in numpy:

1. Mode `x*(t)` and width `w(t)` for all `t` (existing formulas).
2. Split the `v` line at the kink `v = -x*/w`; truncate the tails where the integrand is
   negligible (below `~1e-17` relative to the peak).
3. Integrate each piece with fixed tanh-sinh nodes; their clustering at the endpoints handles
   the kink. The integrand is an array `(n_t, n_nodes)`, multiplied by `v^k`, `k = 0..3`, and
   contracted with the weights.
4. `K`, `dK`, `d2K`, `d3K` from `I_0..I_3` in one pass; cache the last call.

Pros: no setup cost, no range limit in `t`, works for any `beta`, little code. Cons: the node
count must be chosen to reach `1e-8` over the supported range; `beta` near 1 and large `beta`
may need more nodes.

### B. Tabulation and interpolation

Evaluate the exact quadrature on a grid in `t`, interpolate (Chebyshev or spline), with an
asymptotic (Laplace) expansion for large `|t|`. Very fast lookups, but seconds of setup per
`(beta, sigma)` (or a two dimensional table in `(beta, t)`), a finite range, and more code.

### C. Series and asymptotics

The Fox-Wright series for small `t` and Laplace / Watson asymptotics for large `|t|`. For
`s >= 0` all terms are positive (no cancellation, symmetric case stable for all `t`), but the
number of terms grows like `t^(beta / (beta - 1))` (about 30 for `beta = 1.5`, `t = 3`; 1000
for `t = 10`; prohibitive for `beta` near 1); for `s < 0` the series alternates and cancels.
Intermediate `t` is not covered.

### D. Hybrid

Series where cheap and stable, the asymptotic series
`J(s) ~ sum_n (-1)^n Gamma(n beta + 1) / (n! |s|^(n beta + 1))` for large negative `s`, and
option A elsewhere. Faster and more exact where it applies, but more code than A.

## Decision

Do A, measure it against the current implementation (speed, and accuracy for `beta` in about
`[1.05, 10]`, small and large `t`, asymmetric `sigma`). Consider D or B only if A is still too
slow.

## Implementation (option A)

`_gn_half_line` in `spapprox/cgfs.py` builds the nodes per side of zero (a half line with
tilt `t` on the right and `-t` on the left), `_gn_tilted` combines them into `K`, `dK`, `d2K`,
`d3K`. Two refinements of the sketch above:

- Instead of one tanh-sinh rule over `v`, each half line is split at the mode `y*`. Beyond the
  mode an exp-sinh rule `delta = d exp(pi sinh(tau_j) / 2)`, with `d` the distance where the
  log density dropped by one (bisection in `log(delta)`; the log density is concave). Before
  the mode `delta = y* expit(pi sinh(tau_j) / 2 - log(y* / d_L))`, which starts as the same
  exp-sinh rule and clusters double exponentially at the kink `y = 0`. A plain tanh-sinh rule
  on `[0, y*]` was too coarse at the mode when `y* >> d`.
- The step is `h = 1 / (8 ceil(max(2, beta)))` over `tau_j` in `[-3.5, 3.5]`: the integrand
  `exp(-(delta / d)^beta)` is analytic in a strip of width proportional to `1 / beta` in
  `log(delta)`, so large shapes need smaller steps (113 nodes per piece for `beta <= 2`, 449
  for `beta = 8`). `h = 1 / (4 ceil(max(2, beta)))` only reached about `1e-6` in `d3K`.

The drop from the mode is computed as `A (beta r - (1 + r)^beta + 1)`, `r = delta / y*`, as
before, so the large terms `t y` and `(y / sigma)^beta` do not cancel. `K(t)`, `dK(t)`, ...
with the same `t` share one evaluation (cache of the last call).

## Results

Accuracy against the old adaptive quadrature, on 12 configurations (`beta` from 1.05 to 10,
asymmetric shapes and scales, shape one on one side) and `t` in `[-4, 4]` plus `+-1e-3`,
`1e-6`: worst error `7e-12` in `K`, `1.6e-10` in `dK`, `d2K`, `d3K` (relative to the width of
the tilted density, at `beta = 1.05`), about `1e-12` elsewhere. At `t = 1e-6` the old code was
wrong: `K = -40.5` for `beta = 1.05` and `-2.4` for `beta = 10` instead of about zero (the
mode is then tiny and its width estimate breaks down); this is now a regression test.
Finite difference checks of `dK`, `d2K`, `d3K` pass to about `1e-8` (the finite difference
error) for `beta` up to 60 and down to 1.01.

Timings on `generalized_normal(beta=1.5)`, same machine and session:

| Operation | Before | After |
| --------- | ------ | ----- |
| `K` at 50 points | 4.7 s | 0.014 s |
| `pdf(t=...)` at 50 points (normalized) | 92 s | 1.0 s |
| `cdf(x=...)` at 2 points | 3.4 s | 0.055 s |
| `cdf(t=...)` of the sample mean, `n = 10`, at 50 points | 13 s | 0.05 s |

(The "before" timings are slower than in the table above because the machine was loaded.)
The normalized pdf is now dominated by the normalization integral, which evaluates the cgf at
scalar points. Options B to D are not needed.

## Validation

- Existing tests in `tests/test_cgfs.py` (comparison with `scipy.stats.gennorm`, special
  cases, asymmetric version) must pass unchanged.
- Agreement with the current adaptive quadrature to `~1e-8` over a grid of `beta`, `sigma`
  and `t`.
- Timings of the table above, before and after.
