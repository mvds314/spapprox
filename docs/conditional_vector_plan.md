# Plan: conditional tail of a vector of interest (Kolassa & Li 2010, section 4)

Status: F1 (two variables of interest) done, see `conditional_cdf(i=[a, b])`; F2 (`d0 >= 3`) done, see `conditional_cdf(i=[a, b, c, ...])` and the F2 outcome below. Parent: phase F of `docs/conditional_cdf_plan.md`.

Goal: `MultivariateSaddlePointApprox.conditional_cdf`-style approximation of

```
P(X_I >= x_I | X_J = x_J),   |I| = d0 >= 1,  |J| = d - d0 >= 1,
```

continuous or lattice, relative error `O(1/n)`. The scalar case `d0 = 1` is already done
(`conditional_cdf`, Skovgaard and Kolassa 1996); this is the extension to `d0 > 1`.

Source: Kolassa & Li (2010), Bernoulli 16(4), 1191-1207: section 2 (nested signed roots),
section 3 (unconditional terms `I^s`), section 4 (Lemma 4.1, eqs. (4.1)-(4.8)). File:
`~/pCloudDrive/My Books/Saddlepoint approximation/Multivariate/`.

## What the paper says

Notation: coordinates ordered with the `d0` of interest first. `T` is the mean of `n` iid
vectors, `K` the cgf, `t* = t` (continuous) or `t` corrected for continuity (lattice),
`rho(tau) = tau` or `2 sinh(tau / 2)`.

1. **Nested signed roots** (2.2)-(2.6). `tau_tilde^j(gamma^j)` minimizes
   `K(gamma) - gamma.t*` over the last `d - j` components with the first `j` fixed at
   `gamma^j`. Then

   ```
   w_hat_j = sign(tau_tilde_j(0_{j-1})) * sqrt(-2 [ L(tau_tilde^{j-1}(0)) - L(tau_tilde^j(0)) ])
   w_j     = w_hat_j + sign(tau_j - tau_tilde_j(tau^{j-1}))
                       * sqrt(-2 [ L(tau_tilde^{j-1}(tau^{j-1})) - L(tau_tilde^j(tau^j)) ])
   ```

   with `L(g) = K(g) - g.t*` and `tau_tilde^0 = tau_hat`. Order dependent. The map
   `tau -> w` is triangular: `w_j` depends on `tau^j` only. `w_tilde_j(w^{j-1})` is defined by
   `tau_j(w_1, ..., w_tilde_j) = 0`; `w_check_j = w_tilde_j(w_hat^{j-1})` (2.6).
2. **Inversion integral** (2.1)/(4.1) becomes, after `tau -> w` (2.10), an integral with
   denominator `prod_{j<=d0} (w_j - w_tilde_j(w^{j-1}))` and numerator
   `exp(n[w.w/2 - w_hat.w])`, times `G(tau)`. Continuous (4.4); lattice (4.8): replace
   `tau_j` in the denominator by `2 sinh(tau_j / 2)`.
3. **Lemma 4.1**: the conditional case equals the unconditional recipe with the conditioning
   components of `w` frozen at `w_hat_{-d0}`, up to a factor `1 + O(1/n)`. The extra Jacobian
   `prod_{j>d0} dtau_j/dw_j` at `(w_{d0}, w_hat_{-d0})` appears in `G` (eq. (2.9)). The
   marginal density `f_{T_{-d0}}(t_{-d0})` in the denominator of the conditional is replaced by
   (4.3), the same integral with `w_{d0} = 0`, so the extra Jacobian cancels at `w = 0`.
4. **Decomposition** `G = sum_{s subset U} H^s`, `U = {1..d0}`, keep `|s| <= 1` (2.11):
   - `I^empty` (3.3)-(3.4): the main term, `C^empty * Nbar(ybar^empty, Sigma^empty)`, an upper
     orthant probability of a `d0`-dim normal. `Sigma` and `ybar` come from the Hessian
     `g^{jk}` of the reparametrized exponent at `u_hat`, `u_j = w_j - w_tilde_j(w^{j-1})`; plus
     a cubic correction `n/6 g^{jkl}` that the paper only refers to Kolassa (2003) for.
   - `I^{r}` for each `r in U` (3.5)-(3.10), (4.7): one term per variable of interest, a
     1-D normal density times a `(d0 - 1)`-dim normal orthant probability after a second
     change of variables, times `[G^{r}(tau) - G(0)] / (w_r - w_tilde_r)` at `t_hat_r`.
5. **Result**: `sum_{|s|<=1} I^s / f_{T_{-d0}}(t_{-d0})`. Error `O(1/n)`, relative.

Typo alert (kept from the parent plan): the summary after Lemma 4.1 writes
`T_{-d0} > t_{-d0}`; the equality `T_{-d0} = t_{-d0}` is meant. (3.5) / (4.7) also carry a
`(2 pi d)^d` that should read `(2 pi i)^d`.

## Reading of the structure that suggests a cheaper route

Not in the paper, to be verified numerically before relying on it. Fix the conditioning
components of `tau` at the constrained minimizer, i.e. define the profile exponent

```
Lambda(gamma_I) = L(tau_tilde^{d0}(gamma_I)),    gamma_I in R^{d0}
```

Then `w_1..w_{d0}` are exactly the nested signed roots of `Lambda`, and the paper's
`dtau_j/dw_j` factors for `j > d0` combine to `det K''_JJ(tau_tilde(gamma_I))^{-1/2}`
(the Jacobian of the inner minimization). The conditional tail is then the unconditional
`d0`-dim Kolassa & Li tail for the profile `Lambda`, with the extra factor
`sqrt(det K''_JJ(tau_0) / det K''_JJ(tau_tilde(.)))` already seen in `d0 = 1` as
`sigma_hat = sqrt(det K''(tau_hat) / det K''_JJ(tau_0))` in `_conditional_terms`.
Consequences:

- Everything beyond the profile (inner Newton solves for the `J` block, `det K''_JJ`) is shared
  with the `d0 = 1` code. The new work is the `d0`-dim part.
- For `d0 = 2` the existing bivariate machinery (`BivariateSaddlePointApprox._kl_cdf_ordered`,
  `_kl_tail`, `_bivariate_formula`, the derived cubic terms) is the starting point, applied to
  `Lambda` instead of `K`.

## Scope and phases

The unconditional method for `d > 2` (`multivariate_cdf_plan.md`, phase B) needs the cubic
term `g^{jkl}` for general `d`, which the paper omits and which is not derived anywhere in
the repo. Phase F inherits that. So split by `d0`:

- **F1: `d0 = 2`, any `d > 2`.** Needs only the bivariate-level derivation, which exists. This
  is the useful case (two parameters of interest, nuisance conditioned out) and
  covers the paper's Table 3. Do this first.
- **F2: `d0 >= 3`.** Was blocked on phase B of the multivariate cdf plan (general-`d` cubic
  term, order blending); done now that phase B exists, see the F2 outcome below.
- **F0 (sanity, no new feature): `d0 = 1`.** The paper's formula must reduce to the existing
  first-order `conditional_cdf(backend="Skovgaard")`. Use it as a unit test of the nested
  roots code, not as a new code path.

### F1 steps

F1.1. **Profile solver.** `_profile_saddle(self, gamma_I, J, xJ)`: solve
      `dK_J(gamma_I, tau_J) = x_J` for `tau_J`, warm-started (reuse the machinery behind
      `_conditional_denominator_saddle`, which is the `gamma_I = 0` case). Vectorize over a
      batch of `gamma_I` so the bivariate code can evaluate `Lambda` on a grid or stencil.
      Return `tau_J`, `Lambda`, and `det K''_JJ`.
F1.2. **Profile derivatives.** `Lambda` gradient and Hessian in `gamma_I` follow from the
      implicit function theorem (Schur complement of `K''` w.r.t. `J`), and the third
      derivatives from the fourth-order cumulant plumbing already in `_d4K_central`
      (finite difference of `d3K`). Do not differentiate numerically through the solver.
      Check against finite differences of `Lambda`.
F1.3. **Nested roots for `d0 = 2`.** `w_hat_1, w_hat_2, w_check_2, w_tilde_2(w_1)`,
      `dtau/dw` from (2.7)-(2.9), on `Lambda`. Cross-check `w` against brute-force
      minimization.
F1.4. **Terms.** `I^empty` (bivariate normal tail with `Sigma^empty`, cubic correction), and
      `I^{1}`, `I^{2}`: each a 1-D density times a 1-D normal tail (`(3.10)`, `d0 - 1 = 1`).
      Divide by the denominator density `f_{T_{-d0}}` from (4.3), i.e. the same expression
      at `w_{d0} = 0`, which is what the marginal saddlepoint `spa[J].pdf` approximates
      to first order; prefer the self-consistent (4.3) form so the Jacobians cancel.
F1.5. **Orthant reflection and removable singularities.** `P(X_I >= x_I)` for the other
      sign patterns by reflection (`ldot(diag(sg))`) and inclusion-exclusion, as in the
      bivariate cdf. Handle `w_hat_j = 0`, `tau_tilde_2(0) = 0` and `w_hat_2 = 0` (the paper
      omits those Table 3 rows for the same reason) with the same near-zero interpolation
      used by `_conditional_cdf_at` and `_interpolate_cdf_near_zero`.
F1.6. **Order dependence.** `w` depends on the order of the two variables of interest; the
      bivariate fix (average both orders, `_kl_cdf`) carries over. Decide after measuring the
      discontinuity across `x_I` hyperplanes.
F1.7. **Lattice.** `rho = 2 sinh(tau / 2)` with a continuity corrected `t*` (4.8), per
      variable of interest, with `delta` per variable and the same `1/delta` scale-invariance
      fix as in `_conditional_second_order`. Conditioning variables are not corrected.
F1.8. **API.** Extend `conditional_cdf` rather than adding a method:
      `conditional_cdf(x, given, i=[0, 1], backend="KL", delta=None)` where `i` may be a
      list; `x` has one column per variable of interest. `backend` names follow the
      bivariate cdf (`"KL"`; `"Skovgaard"` / `"Kolassa"` stay scalar-only and raise for a
      list `i`). The result is `P(X_I >= x_I | ...)` or the lower orthant? Decide with
      the existing scalar convention (lower tail, `P(X_i <= x)`) and reflect internally.
      Update `docs/TODO.md`, the README, and `.claude/skills/spapprox/references/spa-api.md`.

## Validation

Exact references that need no paper:

- **Three iid exponentials** (paper Table 3): `Y = (X_2, X_3, X_1 + X_2 + X_3)`, `n = 10`, the
  conditional law given `Y_3 = c` is `Dirichlet(n, n, n)` on `(S_2, S_3, S_1) / S`, so
  `P(Y1 >= a, Y2 >= b | Y3 = c) = P(D_2 >= a/c, D_3 >= b/c)`, a 2-D quadrature. Checked
  here against the paper's exact column (`n = 10`):

  | y1 | y2 | y3 | exact (here) | paper exact | paper P. approx. |
  | -- | -- | -- | ------------ | ----------- | ---------------- |
  | 2.0 | 2.0 | 7.0 | 4.378e-1 | 4.38e-1 | 4.42e-1 |
  | 2.5 | 2.5 | 7.0 | 6.319e-2 | 6.32e-2 | 6.25e-2 |
  | 3.0 | 3.0 | 7.0 | 3.458e-4 | 3.46e-4 | 3.02e-4 |
  | 2.0 | 2.0 | 6.5 | 2.908e-1 | 2.91e-1 | 2.93e-1 |
  | 3.0 | 3.0 | 6.5 | 1.098e-6 | 1.10e-6 | 9.63e-7 |

  The paper's relative errors (1% to 13% at `n = 10`) are the acceptance bar for the
  implementation, not the regression value. Because the exact values are cheap, test the
  `1/n` rate with the same construction at `n = 10 * 2^k`.
- **Multivariate normal**: conditional on `X_J`, `X_I` is normal. The approximation must be
  exact (prefactor 1, cubic terms vanish), for `d0 = 2`, `d = 3, 4`.
- **`d0 = 1` reduction** (F0): the new code path with a single variable of interest equals
  `conditional_cdf(backend="Skovgaard")` to rounding.
- **Lattice**: Poisson or negative binomial vectors with the conditioning on a sum
  (multinomial / Dirichlet-multinomial conditionals have closed forms), as in
  `test_conditional_lattice_*`.
- Reflection consistency, `delta -> 0` limit, continuity across `w_hat = 0`, and permutation
  of the two variables of interest (to the tolerance of the order-dependent error).

Tests that need `findiff` must follow the CI rule in `.github/copilot-instructions.md` (only `numdifftools`
installed there); mark the quadrature-heavy ones `slow`.

## Risks

- **The paper omits the cubic term.** For `d0 = 2` we derive it from the bivariate
  work already in `spa.py`; any `d0 >= 3` needs a new derivation.
- **The profile reduction is my reading, not a statement of the paper.** F1.1-F1.4 start by
  checking it numerically against Table 3 and the normal case; if it fails, fall back to the
  literal recipe (nested minimizers for every `j`, derivatives (2.7)-(2.9)).
- **Cost.** Each point needs an inner solve per stencil node of `Lambda`; batch and warm-start.
- **Accuracy is limited**: 10% to 14% relative error at `n = 10` in the far tail (Table 3), so
  tolerances must follow the `n`-sequence, not a fixed small number.
- **Discontinuity in the variable order** (as in the bivariate `"KL"`).

## F2 outcome

`_ConditionalMultivariateApprox` applies the general-`d` engine (`spapprox/kl.py`, phase B of
`multivariate_cdf_plan.md`) to the profile cgf of the `d0` variables of interest, with the
same hooks as F1: the weight `sqrt(det K''_JJ(tau_0) / det K''_JJ)`, the conditional
Skovgaard cdfs as 1-D marginals, and, for lattice variables of interest (`delta=`, all
spans positive), `rho(tau) = 2 sinh(tau delta / 2) / delta` at `x_I + delta / 2`.
`_ConditionalBivariateApprox` now derives from it and only keeps the bivariate order swap, so
`d0 = 2` is unchanged. Validation (`tests/test_spa.py`, `test_conditional_vector_three_*`):

- Normal, `d = 5`, `I = [3, 0, 1]`, `J = [2, 4]`: exact to `1e-9`.
- Independent variables of interest: the product of the conditional marginals to `5e-11`.
- Dirichlet(`n, n, n, n`) as 4 iid gammas given their total, `P(D_1 <= u_1, D_2 <= u_2,
  D_3 <= u_3)` against quadrature: relative error at most `2.6%` for `n = 2`, `0.36%` for
  `n = 8`.
- Multinomial as independent Poissons (`lambda = (2, 3, 4, 5)`) given the total `14`,
  lattice: relative error at most `1.25%`.

Cost: a point takes seconds (up to about 14 s near the origin of the saddlepoint, where the
tensor interpolation multiplies the inner profile solves).

