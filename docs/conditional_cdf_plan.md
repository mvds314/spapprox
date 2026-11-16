# Plan: conditional distribution via double saddlepoint approximations

Status: not started. Branch `feature/conditional-skovgaard` (draft PR #4).

Goal: approximate the cdf (and pdf) of one component `X_1` of a random vector given the
other components, `X_{-1} = x_{-1}`, from the joint cgf `K`. Builds on the marginal
approximations (`spa[i]`, `spa[[i, j]]`) and the multivariate machinery in `spa.py`.

## Sources

| Ref | Content | Use |
| --- | ------- | --- |
| Skovgaard (1987), J. Appl. Prob. 24, 275-287 | Original double-saddlepoint conditional cdf, continuous and lattice | Basis of phase A and B |
| Broda & Paolella (2011), ch. 32, section 32.4.1, eqs. (32.22)-(32.27) | Clean statement of the bivariate Skovgaard formula | Phase A formulas |
| Butler (2007), ch. 4 (4.1, 4.2.1-4.2.2, 4.4.3-4.4.4) and 5.4.5 | Multivariate conditioning, pdf (4.7)/(4.14), lattice corrections (4.15)-(4.18), `t_hat = 0` limit | Phase A, B, pdf |
| Kolassa (1996), Ann. Statist. 24(1), 353-364, Theorem 6.1, eqs. (2), (3), (10), (11) | Second-order terms, error `O(N^-5/2)` instead of `O(N^-3/2)`, continuous and lattice | Phase C |
| Paolella (2007), Intermediate Probability, sections 5.2-5.3 (eq. 5.18) | Same formula as Broda & Paolella | Cross-check only (read via the literature skill, not the book) |
| Kolassa & Li (2010), Bernoulli 16(4), section 4, eqs. (4.1)-(4.8) | Conditional tail `P(T_{1..d0} >= t | T_{-} = t_-)` for a **vector** of interest (`d0 >= 1`), relative error `O(1/n)`, continuous and lattice, via the same nested signed roots as the unconditional case | Phase F |
| Wang (1993), J. Appl. Prob. 30, 397-404, Theorems 1-2, eqs. (3)-(9) | Conditional cdf and density of `T = X_bar` given `Z = g(X_bar, Y_bar) = z` for **non-linear** `g`, relative error `O(1/n)`; contains Skovgaard as the linear case; p-dim generalization stated | Phase G |
| Wang (1990), J. Appl. Prob. 27, 586-597 | Bivariate cdf only (basis of `"Wang"` backend); mentions the conditional case only as a pointer to Skovgaard | No conditional content |
| Broda & Paolella (2011), section 32.3 and 32.6 | Partial expectation `G(x) = ∫ y f(y) dy` and expected shortfall of one variable (Martin 2006), eq. (32.19), with `w_hat`, `u_hat` as in the univariate LR | Conditional *expectation* given an event on the same variable, not conditioning on another variable |
| DiCiccio, Martin & Young (1993), Biometrika 80, 781-790 | Laplace-based marginal density then a tail formula; nonlinear statistics, ancillary conditioning | Not planned, see phase E |
| Gatto & Jammalamadaka (1999), JASA 94, 533-541 | Conditional saddlepoint for testing problems | Not planned, see phase E |

Files: `~/pCloudDrive/My Books/Saddlepoint approximation/{Overview,Conditional}/`.

## Notation

The sources use different letters. We follow Broda & Paolella and Kolassa; Butler swaps
the roles of `X` and `Y`.

| Concept | Broda & Paolella | Kolassa (1996) | Butler (2007) |
| ------- | ---------------- | -------------- | ------------- |
| Variable of interest | `X`, scalar | `X^1` | `Y`, scalar (or vector in the density) |
| Conditioning variable | `Y`, scalar | `X^2..X^d` | `X`, vector |
| Numerator saddlepoint | `(s~, t~)` | `b_hat` | `(s_hat, t_hat)` (roles swapped) |
| Denominator saddlepoint | `t~_0` | `b~` (`b~_1 = 0`) | `s_hat_0` |

In the code we call the target index `i`, the conditioning indices `J`, the numerator
saddlepoint `tau_hat` (all `d` components) and the denominator saddlepoint `tau_0`
(component `i` fixed at 0, the others solving `dK_J = x_J`).

## Formulas

Let `K` be the joint cgf, `X = (X_i, X_J)`, observed `x = (x_i, x_J)`.

- **Numerator saddlepoint** `tau_hat`: `dK(tau_hat) = x` (all `d` equations). We already
  solve this in `_dK_inv`.
- **Denominator saddlepoint** `tau_0`: `tau_0[i] = 0` and `dK_J(tau_0) = x_J`. This is the
  saddlepoint of the marginal of `X_J`, so `self[J].cgf.dK_inv(x_J)` embedded with a zero
  at position `i`.

### First order (Skovgaard 1987)

Broda & Paolella (32.24)-(32.26) for scalar conditioning, Butler (4.12)-(4.13) and
Kolassa (2) for vector conditioning:

```
w_hat = sgn(tau_hat[i]) * sqrt( 2 * [ tau_hat.x - K(tau_hat) - tau_0.x + K(tau_0) ] )
u_hat = tau_hat[i] * sqrt( det K''(tau_hat) / det K''_JJ(tau_0) )

P(X_i <= x_i | X_J = x_J) ~ Phi(w_hat) + phi(w_hat) * (1/w_hat - 1/u_hat),   tau_hat[i] != 0
```

- `tau_0 . x = tau_0[J] . x_J` since `tau_0[i] = 0`.
- `K''_JJ(tau_0)` is the `J` block of the Hessian at `tau_0`, which is the Hessian of the
  marginal at its own saddlepoint. For scalar conditioning it is `K''_tt(0, t~_0)`.
- The argument of the square root is positive: the unconstrained supremum of
  `tau.x - K(tau)` is attained at `tau_hat`, the constrained one at `tau_0`.
- Exact for the normal. Error `O(1/n)` relative to the (small) conditional probability.
- Survival function: `1 - F`, same expression with the signs of the correction flipped.
  Kolassa (2) writes it that way.

### First-order density (Butler 4.1, eq. (4.7)/(4.14))

```
f(x_i | x_J) ~ det K''_JJ(tau_0)^(1/2) / det K''(tau_hat)^(1/2) * phi(w_hat)
```

This is the ratio of the joint and marginal saddlepoint densities. It is not normalized;
Butler 4.4.3 normalizes by a change of variable to `tau_hat[i]`. We reuse the existing
`normalize_pdf` machinery of `SaddlePointApprox.pdf` if it applies, otherwise skip it.

### Singularity at `tau_hat[i] = 0`

`w_hat = u_hat = 0` there and the formula is `0/0` (Butler p. 113). Options, cheapest
first:

1. Interpolate across `|tau_hat[i]| < h`, as `UnivariateSaddlePointApprox` does with
   `_cdf_singular_width`. Same approach as the existing LR code.
2. Evaluate at `x_i` slightly on each side and average (Butler suggests `1e-8` with 30
   digits, which does not work in double precision; use a larger offset or option 1).
3. The exact limit, Skovgaard (1987) eq. (2.11). Needs third derivatives of `K` in all
   index combinations; do this only if 1 turns out inaccurate.

The singular point is `x_i = dK_i(tau_0)`, where `tau_hat = tau_0`. It is computable
directly, so the interpolation window can be centered on it.

### Lattice (Skovgaard 1987; Butler (4.15)-(4.18); Kolassa (3))

`X_i` on a lattice with span `delta` (usually 1). The conditioning variables may be
continuous or lattice. Survival function, first continuity correction:

```
P1(X_i >= k | X_J = j) = 1 - Phi(w_hat) - phi(w_hat) * (1/w_hat - 1/u1_hat)
u1_hat = (1 - exp(-delta * tau_hat[i])) / delta * sqrt( det K''(tau_hat) / det K''_JJ(tau_0) )
```

- Butler (4.15)-(4.16) has `delta = 1` and no `1/delta`; Butler 5.4.5 (5.79) has the
  `delta` version. Check both against a `delta = 2` example before trusting the `delta`
  factor.
- `tau_hat` solves `dK(tau_hat) = (k, x_J)`; `tau_0` is unchanged.
- Second continuity correction (Butler (4.17)-(4.18)): solve for the offset saddlepoint
  at `k - delta/2`, `w_tilde_2` with that `tau_tilde`, and
  `u2 = 2 sinh(tau_tilde[i] * delta / 2) / delta * sqrt(det K''(tau_tilde) / det K''_JJ(tau_0))`.
  `tau_0` is unchanged, but `w_tilde_2` uses the offset `K` value; read (4.18) once more
  while implementing, the extraction of the PDF is garbled there.
- Butler recommends `P1` and `P2` for the right tail and has the left tail follow from
  `F(k) = 1 - P(X_i >= k + 1)`. Do not use his `F1a`: it is incompatible with `P1`.
- Kolassa (3) uses `2 sinh(b_hat_1 / 2)` and evaluates `b_hat` at the continuity
  corrected `x_1 - delta/2`. Same thing as `P2`.

### Second order (Kolassa 1996, Theorem 6.1, eqs. (10), (11))

Kolassa derives the conditional tail for means of `N` iid vectors with per-summand cgf
`K_Y`; the error is `O(N^-5/2)`. Since `K = N K_Y`, we put `N = 1` and apply the formulas to the full cgf `K`: `sqrt(N) w_hat(K_Y)`
is `w_hat(K)`, and the invariants `rho` of `K` are `1/N` times those of `K_Y`, which absorbs
the explicit `1/N` in the bracket. Verify the scaling with the normal and a gamma/Poisson
example.

With `b_hat = tau_hat`, `b~ = tau_0`, `w_hat_1 = w_hat`, and

```
z_hat = tau_hat[i] * sigma_hat,   sigma_hat = sqrt( det K''(tau_hat) / det K''_JJ(tau_0) )   (= u_hat)
```

continuous case, survival function:

```
1 - F~ = 1 - Phi(w) + phi(w) * [ 1/w^3 - 1/w
        + (1/z) * ( 1 + [ (rho4_hat - rho4_tilde)/8
                          - (rho13sq_hat - rho13sq_tilde)/8
                          - (rho23sq_hat - rho23sq_tilde)/12
                          - (1/2) * k_hat^{1k} k_hat_{ijk} k_hat^{ij} / tau_hat[i]
                          - k_hat^{11} / tau_hat[i]^2 ] ) ]
```

where, with `k_{ij}`, `k_{ijk}`, `k_{ijkl}` the derivatives of `K` and `k^{ij}` the inverse
Hessian (all `d x d` over every index, including the conditioned ones):

```
rho4     = k_{ijkl} k^{ij} k^{kl}
rho13sq  = k^{gij} k^{hkl} k_{gh} k_{ij} k_{kl}   i.e. k_{ijk} k_{lmn} k^{ij} k^{kl} k^{mn}
rho23sq  = k_{ijk} k_{lmn} k^{il} k^{jm} k^{kn}
```

`rho*_hat` are evaluated at `tau_hat`, `rho*_tilde` at `tau_0`, for the **marginal** cgf of `X_J`, i.e. built from the `J` blocks
of the Hessian and of the third and fourth derivative tensors, with the inverse of the `J` block
(verified numerically on the bivariate gamma: using the full `d x d` quantities converges at the
first-order rate, the `J` blocks at about `n^-3`). `k_hat^{11}` is the `(i, i)` element of
the inverse Hessian at `tau_hat`. Pages 359-360 of the paper are the authority; the
extracted text lost the index positions, I read them from the rendered page. Treat the
index placement in the third and fourth terms of the bracket as the first thing to
verify numerically (see validation).

Lattice, eq. (11): the same with `z_hat = 2 sinh(tau_hat[i] * delta / 2) * sigma_hat`,
the continuity corrected `x_1`, and the two last terms of the bracket replaced by

```
- (1/4) delta k_{1k} k^{ijk} k_{ij} * coth(tau_hat[i] * delta / 2)
- ( coth(tau_hat[i] * delta / 2)^2 / 4 - 1/8 ) * delta^2 * k_{11}
```

(index placement again to verify; reduces to the continuous terms as `delta -> 0`, which
is a useful consistency check).

Properties and caveats:

- Needs the **fourth** derivative tensor `k_{ijkl}` of the joint cgf at two points.
  `CumulantGeneratingFunction` currently stops at `d3K`. The cheapest path is the
  contraction `rho4 = k_{ijkl} k^{ij} k^{kl}` computed with a numerical directional
  derivative of `d2K` along the third derivative, or a `d4K` in `diff.py`. Decide in phase C.
- Kolassa's logistic regression example (Table 1: normal 0.0260, double saddlepoint
  0.0315, higher order 0.0321, exact 0.0352) is a regression target for the lattice
  version, but needs the data from the paper.
- The same removable singularity at `tau_hat[i] = 0` appears, now with more terms; use
  the interpolation of phase A.

## API proposal

Keep it analogous to `cdf`, vectorized over points, and put the machinery on
`MultivariateSaddlePointApprox`; `BivariateSaddlePointApprox` inherits it (`d = 2`).

```python
spa.conditional_cdf(x_i, given, i=0, backend="Skovgaard", lattice=None, **solver_kwargs)
spa.conditional_pdf(x_i, given, i=0, **solver_kwargs)
```

- `given`: dict `{j: x_j}` or an array for the remaining components. Open question:
  accept several points of `x_i` for one `given` (the typical case, one saddlepoint
  `tau_0` shared) and broadcast `given` when it is a matrix.
- `backend`: `"Skovgaard"` (default), `"Kolassa"` (second order, phase C).
- `lattice`: span `delta` of `X_i`, `None` for continuous. Maybe inferred from the cgf
  later; explicit for now.
- Reuse `spa[J]` for the marginal saddlepoint and Hessian, and the `_dK_inv` solver for
  the numerator.
- Possibly also `ppf` through root finding on `conditional_cdf`.

Alternative considered: return a conditional approximation object (a
`UnivariateSaddlePointApprox`-like with `cdf`/`pdf`/`ppf`) via `spa.condition(given)`.
Nicer to use but the object would hold `x_J`-dependent state; revisit after phase A.

## Phases

### Phase A: first-order continuous

A1. `_denominator_saddle(x_J)` through `self[J].cgf.dK_inv`, embedded at index `i`.
A2. `w_hat`, `u_hat`, the formula and the singularity interpolation (`h` in standardized
    units, as in the univariate LR).
A3. `conditional_cdf` (vectorized), `conditional_pdf`.
A4. Tests (see validation).

### Phase B: lattice continuity corrections

B1. `P1` and `P2` survival functions, left tail via `F(k) = 1 - P(X >= k + 1)`.
B2. `lattice` argument, `delta` handling, and conditioning on continuous or lattice
    variables.
B3. Tests on Poisson to binomial and negative binomial to Polya.

### Phase C: Kolassa (1996) second order (continuous case done)

Done as `conditional_cdf(..., backend="Kolassa")`. The third and fourth bracket terms use the
inverse Hessian at `tau_hat` as written above (confirmed by fitting against the exact gamma
conditional). `k_{ijkl}` is a central difference of `d3K` (`_d4K_central`), no `d4K` was added.
The lattice version (eq. 11) remains open.

C1. Fourth derivatives (see caveats). Decide `d4K` vs numerical contraction.
C2. `rho4`, `rho13sq`, `rho23sq`, and the bracket, at `tau_hat` and `tau_0`.
C3. `backend="Kolassa"`, continuous and lattice.
C4. Compare orders: error against exact conditionals, check the convergence rate in `n`.

### Phase D: docs

D1. Update `spapprox` skill (`spa-api.md`), README, TODO.
D2. Remove the TODO marker in `tests/test_spa.py`.

### Phase F: vector of interest (Kolassa & Li 2010, section 4)

`P(X_I >= x_I | X_J = x_J)` with `|I| >= 1`. Same structure as the multivariate cdf
phases (nested signed roots, orthant reflection), with the components of `w_hat` of the
conditioning variables substituted. The `d0 = 1` case should reproduce phase A at first
order. This inherits the difficulty of the unconditional Kolassa & Li (2010) method
(see `multivariate_cdf_plan.md`, phase B), so only do it if the vector case is needed.
Their Table 3 (three iid exponentials, `Y = (X_2, X_3, X_1 + X_2 + X_3)`, `n = 10`; exact values
from Wang's unpublished report [15]) is a regression target. Typo alert: the summary after Lemma 4.1
writes `T_{-d0} > t_{-d0}` where the equality is meant.

### Conditioning on inequalities (no source found)

None of Skovgaard, Butler ch. 4, Broda & Paolella, Kolassa (1996), Kolassa & Li (2010)
or Wang (1990) treats `X_J >= x_J` or interval events as conditioning. They all condition on
equality. Wang's 1993 paper on conditional inference is not in the library and its title
points to the same equality case. Proposal, not literature: use the definition,

```
P(X_i <= x_i | X_J >= x_J) = P(X_i <= x_i, X_J >= x_J) / P(X_J >= x_J)
```

with the numerator from the existing orthant/cdf code (`"Kolassa"`, `"KL"`, `"Wang"`
backends and reflection) and the denominator from `spa[J].cdf`. Intervals are differences.
Accuracy is that of the two orthant approximations, to be checked against exact
conditionals (e.g. bivariate gamma by quadrature) before documenting it.

### Phase G: non-linear conditioning (Wang 1993)

Condition on `Z = g(X, Y) = z` instead of a component. Bivariate statement (the paper says
the `p`-dim case follows the same way with `Y` and `Z` of dimension `p - 1`, `D` the
`p x p` Hessian determinant). Write `K` for the joint cgf (`n` absorbed, so drop `n`);
the transform `(X, Y) = (T, q(T, Z))` is one-to-one with inverse `q`.

```
L(t)       = K(u(t), v(t)) - u(t) t - v(t) q(t, z)      with (u, v) solving dK = (t, q(t, z))
t_1        : L'(t_1) = 0,  L''(t_1) < 0                 (one extra 1-D solve, "numerator" = joint saddlepoint at (t, q))
a_0        = sgn(t_0 - t_1) * sqrt(2 [L(t_1) - L(t_0)])
b_0        = -L'(t_0) * |J_1 / J_0| * sqrt( D(u_0, v_0) / (-L''(t_1) D(u_1, v_1)) )
H(t_0 | z) ~ Phi(a_0) + phi(a_0) * (1/a_0 - 1/b_0)                      (eq. (8))
h(t | z)   ~ sqrt( -L''(t_1) D(u_1, v_1) / (2 pi D(u, v)) ) * |J/J_1| * exp(L(t) - L(t_1))   (eq. (9))
L'(t)      = -u - v q_t(t, z)
L''(t)     = [ -K_vv + 2 q_t K_uv - q_t^2 K_uu ] / D - v q_tt          (all at (u(t), v(t)))
J          = q_z(t, z) = 1 / g_y
```

(`t_0` is the evaluation point; `t_1` the maximizer of `L`; `u_0,v_0` and `u_1,v_1` the
saddlepoints at `t_0` and `t_1`.) Read `b_0` from p. 400 of the paper, my transcription of the
garbled text could still contain typos; the page is the authority.

- **Special case**: `g` linear gives Skovgaard; for the normal and `g = s^2` it is exact
  (Wang, Example 1).
- **Cost**: each point needs a 1-D maximization over `t`, each step solving a joint saddlepoint
  at `(t, q(t, z))`, plus second derivatives of `q` for `L''`.
  Accuracy is `O(1/n)`; Wang suggests renormalizing the density numerically.
- **Needs from the user**: `q` with its `t`/`z` derivatives (or a numerical inverse of `g`).
- **Uses**: conditional bootstrap, e.g. the mean given the empirical second moment, via the same
  construction as Davison & Hinkley (1988) and Wang's example 1.

Only worth doing once the linear case works; reuses `dK_inv` and the Hessian code.

### Phase E: not planned (decide later)

- DiCiccio, Martin & Young (1993): different principle (non-linear conditioning is Wang 1993, phase G) (Laplace and Phillips-type
  marginal density), covers nonlinear statistics and conditioning on ancillaries.
- Gatto & Jammalamadaka (1999): conditional saddlepoint for testing; probably a
  special case of the lattice formula.
- Butler 4.4.4: single saddlepoint of an explicit conditional cgf. Not needed, the user
  can build that cgf directly and hand it to `UnivariateSaddlePointApprox`.
- Conditional probability of sets for vector `X_i` (Butler: importance sampling,
  Butler et al. 2005).

## Validation

Exact conditionals available in closed form (Butler Table 4.1 and 4.2.3):

| Joint | Conditional | Notes |
| ----- | ----------- | ----- |
| Multivariate normal | normal | Exact for A, also for C (all corrections vanish). Pins down index placement in C. |
| Independent gammas `(Z_1, Z_1 + Z_2)` | `Z_1 / (Z_1 + Z_2) * x` with beta | Continuous; exact beta cdf via `scipy.stats.beta` |
| Poisson `(Z_1, Z_1 + Z_2)` | binomial | Lattice, Butler (4.8): equals the single saddlepoint of the binomial |
| Negative binomials | Polya | Lattice, Butler 4.2.3 |
| Gamma, sum of independent | Dirichlet | `d >= 3`, multi-variable conditioning |
| Independent `X`, `Y` | marginal of `Y` | Conditional approximation equals the marginal one, Butler 4.1.1 |

Checks:

- Normal, `d = 2, 3`: exact to solver tolerance, both backends.
- Independence: `conditional_cdf` equals `spa[i].cdf` for all `x_i`.
- `d = 2`: matches the existing univariate LR on the marginal only when independent;
  otherwise compare against the exact beta/binomial.
- Continuity across `tau_hat[i] = 0` and symmetric behavior for symmetric examples
  (Butler 4.2.5).
- Convergence rate in `n` for sums of iid vectors (`bivariate_gamma([2, 1, 3] * k)` is a
  ready example): relative error `O(1/n)` for A, `O(1/n^2)` expected for C.
- `delta -> 0` limit of the lattice formulas reproduces the continuous ones.
- Tests that need findiff are gated as in the other tests; CI only has `numdifftools`,
  so nothing at module level may construct findiff objects.

## Risks and open questions

- **Index placement in Kolassa (10)/(11).** The text extraction is unusable and the
  paper's own notation mixes derivative and inverse indices. The normal check
  (everything should vanish) and a lattice-to-continuous limit check catch most errors,
  but a Poisson/binomial and gamma/beta comparison against the exact conditional is
  needed to confirm the signs of the `1/tau_hat[i]` and `1/tau_hat[i]^2` terms.
- **Fourth derivatives** are not in the cgf API (phase C only).
- **Hessian blocks and determinants.** `det K''` and `det K''_JJ` get ill-conditioned
  for `d` large and strongly correlated; use `slogdet`.
- **Conditioning on a boundary**, e.g. `x_J` at the edge of the support: the denominator
  saddlepoint does not exist. Return `fillna`, as the other methods do.
- **Left versus right tail** conventions: sources give survival functions for the lattice
  case (`P(X >= k)`), the continuous cdf is `Phi(w) + ...`. Decide the public
  convention in the docstring: `conditional_cdf` is `P(X_i <= x_i | ...)`.
- **Name of the entry point** (`conditional_cdf` vs `spa.condition(...)`), see API
  proposal.
