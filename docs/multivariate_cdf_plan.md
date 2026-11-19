# Plan: saddlepoint cdf for random vectors of dimension d > 2

Status: phase A done, see [Phase A: outcome](#phase-a-outcome).
`MultivariateSaddlePointApprox.cdf(backend="Kolassa")` works in any dimension, and
`BivariateSaddlePointApprox` offers it next to `"KL"` (default) and `"Wang"`. Phase B is
not started.

## Sources

| Ref | Content | Relevance |
| --- | ------- | --------- |
| Kolassa (2003), Ann. Statist. 31(1), 274-286 | Approximation `Q(t)` to `P[T >= t]` for any `d`, continuous and lattice; recursion for orthants with negative saddlepoint components | Basis of phase A |
| Kolassa & Li (2010), Bernoulli 16(4), 1191-1207, sections 2-3 | Lugannani-Rice type approximation for any `d`, relative error `O(1/n)`; the 2-D case is our `backend="KL"` | Basis of phase B |
| Li (2009), PhD thesis, ch. 3 | Bivariate details of Kolassa & Li | Already used for `"KL"` |
| Kolassa (2006), Series approximation methods in statistics, ch. 6 | Background, nested signed roots | Reference |

Files: `~/pCloudDrive/My Books/Saddlepoint approximation/Multivariate/`.

## The two candidate methods

### Kolassa (2003): `Q(t)` (Robinson-type, phase A)

For `T` with cgf `K` and saddlepoint `t_hat` with **all components positive**
(Kolassa (2003) eq. (5); `n` absorbed in `K`):

```
P[T >= t] ~ exp(K(t_hat) - t_hat.t + t_hat' K''(t_hat) t_hat / 2)
            * [ I(Sigma, s, 0, Sigma t_hat, t_hat)
                + 1/6 K'''_jkl(t_hat) I(Sigma, s, e_j + e_k + e_l, Sigma t_hat, t_hat) ]
```

with `Sigma = K''(t_hat)`, `s = (-1, ..., -1)` and

```
I(Sigma, r, m, b, t_hat) = (2 pi i)^-d  ∫ exp(tau' Sigma tau / 2 - tau' b)
                                         prod_l tau_l^r_l (tau_l - t_hat_l)^m_l dtau.
```

- `I(Sigma, s, 0, b)` is the upper orthant probability `N_bar(b, Sigma)` of `N(0, Sigma)`.
- The cubic term needs `I` with `prod (tau_l - t_hat_l)^{m_l}`, `|m| = 3`. Expanding the
  binomials reduces it to `I(Sigma, r, 0, ...)` with `r_l` in `{-1, 0, 1, 2}`. These are
  derivatives of `N_bar` with respect to `b` up to order 3 (eq. (7)). Closed forms:
  - distinct indices `A`: `(-1)^|A| d^A N_bar = phi(b_A; Sigma_A) N_bar(conditional)`
    (eq. (8)). That is a `|A|`-dim normal density times a `(d-|A|)`-dim normal orthant
    probability with conditional mean and covariance.
  - repeated indices: recursions (9)-(11).
- Negative saddlepoint components (text after eq. (13)): if `t_hat_j < 0`, then
  `P[T >= t] = P[T_{-j} >= t_{-j}] - P[U >= u]`, where `U` is `T` with component `j`
  negated. `U` has one negative component fewer. `T_{-j}` is a new (d-1)-dim problem with
  its own saddlepoint. Recurse; the 1-D base case can use Lugannani-Rice.
- The cdf follows from `P[T <= x] = P[-T >= -x]`.
- Lattice variables: eq. (13) gives the continuity-corrected version (`rho = 2 sinh(tau/2)`).

Properties:
- **Exact for the normal**: the prefactor is 1 and the cubic term vanishes.
- **Symmetric in the variables**: there is no nested ordering, so no blending over
  orders is needed (unlike the bivariate KL).
- Error `O(1/n)`, but not uniformly relative (Kolassa (2003) section 3), so relative
  accuracy far in the tails is poor. Kolassa (2003) Fig. 1 finds it comparable to
  Wang (1990) in 2-D.
- Needs only the saddlepoint, `K''`, `K'''` and normal orthant probabilities up to
  dimension `d`. No nested optimizations.

### Kolassa & Li (2010): Lugannani-Rice type (phase B)

The `d`-dim generalization of our bivariate `backend="KL"`.

- The nested signed roots `w_hat_j`, `w_check_j` (eqs. (2.2)-(2.6)) need constrained
  minimizers `tau_tilde^j(gamma^j)`. For each `j`, minimize `K(gamma) - gamma.t` over the
  last `d-j` components with the first `j` fixed. That is `O(d)` extra saddlepoint
  solves per point, and more for the derivatives `w_check_jk` (eq. (2.7)) and `dtau/dw`
  (eqs. (2.8)-(2.9)).
- `I^∅` (eq. (3.3)) needs `g^{jk}` and the cubic `g^{jkl}` of the reparametrized exponent.
  The paper omits the cubic part ("algebraically complicated"). For `d = 2` we derived it
  ourselves (only `w_tilde_2(w_1)` is non-quadratic). For `d >= 3`, every
  `w_tilde_j(w^{j-1})` contributes, with second derivatives in `j-1` variables. That is a
  new derivation, and it is needed to reach `O(1/n)`.
- `I^{r}` (eqs. (3.5)-(3.10)): one term per variable, each a 1-D density times a
  `(d-1)`-dim normal orthant probability after a further change of variables.
- The same orthant reflection and inclusion-exclusion as in 2-D.
- **Not symmetric in the variables.** With reflection it is discontinuous across
  coordinate hyperplanes, as in 2-D. The 2-D fix (blending both orders) would become a
  weighted average over `d!` orders: 6 for `d = 3`, 24 for `d = 4`, each with its own
  nested solves.

### Verdict

Phase A is a moderate, well-specified job: every formula is in Kolassa (2003). Phase B is
a research project: the cubic term must be derived for general `d`, and continuity and
cost get worse as `d` grows. Do A first; do B only if A turns out not to be accurate
enough. A quick 2-D comparison of A against our `"KL"` (step A6) will show whether B is
worth it.

## Phase A: implementation steps

A1. **Normal orthant probabilities and derivatives** (module-level helpers in `spa.py`,
    or a new `spapprox/mvn.py`):
   - `_mvn_tail(b, Sigma)`: `P[Z >= b]` for `Z ~ N(0, Sigma)`, in any dimension.
     - `d = 0`: return 1.
     - `d = 1`: `norm.sf`.
     - `d = 2`: `_bvn_cdf` (fastnorm or scipy; exact).
     - `d >= 3`: `scipy.stats.multivariate_normal.cdf` (Genz quasi-Monte Carlo) with
       tight `abseps`/`releps` and a fixed `rng` for reproducibility.
   - `_mvn_tail_derivative(b, Sigma, idx)`: `d^|idx| N_bar / db_idx` for a multiset `idx`
     of size <= 3. Use eq. (8) for distinct indices and (9)-(11) for repeats. Test
     against finite differences and, for `d = 2`, against `_bvn_tail_derivatives`.

A2. **`_kolassa_tail(x, t)`**: `Q` for a single point with all `t_hat > 0`. It combines the
    prefactor, `N_bar`, and `1/6 K'''_jkl I(..., e_j+e_k+e_l)` summed over the `d^3` index
    triples (exploit symmetry: unique multisets only).

A3. **Orthant recursion**: `P[T >= t]` for any sign pattern, via Kolassa's
    `P[T_{-j} >= t_{-j}] - P[U >= u]`.
   - Use the marginal and reflected cgfs: `self.cgf[idx]`, `self[idx]` from the
     marginals feature, and `cgf.ldot(diag(sg))` or equivalent for the reflection.
   - Memoize subproblems by `(index set, sign pattern)` per evaluation point.
   - 1-D base case: `UnivariateSaddlePointApprox.cdf` (LR).

A4. **Near-zero components**: first measure continuity across `t_hat_j = 0` (expected to
    be good; see the risks below). Interpolate across `|t_hat_j| < h_j`
    (`_cdf_singular_width`), as the bivariate code does, only if the direct formula
    shows cancellation.

A5. **API**: `MultivariateSaddlePointApprox.cdf(x=None, t=None, fillna=np.nan,
    backend="Kolassa", **solver_kwargs)`, vectorized over points like the bivariate
    `_spapprox_cdf`. `BivariateSaddlePointApprox` inherits it as an extra backend
    (`backend="Kolassa"`) and keeps `"KL"` as its default.

A6. **Validation**:
   - Normal, `d = 3, 4`, with correlation: exact against scipy, to QMC accuracy.
   - 2-D: compare against `"KL"`, `"Wang"` and exact quadrature for
     `bivariate_gamma([2, 1, 3] * k)`, k = 1..256 (the TODO 56 table), and check the
     `1/n` rate.
   - 3-D gamma with a common component, `X_i = Z_0 + Z_i`: the exact cdf is a 1-D
     quadrature, `F(x) = ∫ f_0(z) prod_i G_i(x_i - z) dz`. Build the cgf from independent
     gammas with `ldot`, or add a `multivariate_gamma` factory (already on the TODO
     plans list).
   - Continuity across coordinate hyperplanes, and permutation invariance.
   - Reflection consistency:
     `P[T <= x] + P[T_1 > x_1, T_{-1} <= x_{-1}] = P[T_{-1} <= x_{-1}]`.

A7. **Tests** in `tests/test_spa.py` (mark as slow where QMC makes them expensive),
    docstrings with the formulas, a `docs/TODO.md` entry, ruff, and `pytest -m "not slow"`.

## Phase A: outcome

Implemented in `spapprox/mvn.py` (numerics) and `MultivariateSaddlePointApprox._kolassa_cdf`
(recursion); tests in `tests/test_spa.py`. Deviations from the steps above:

- **Orthant probabilities (A1).** `scipy.stats.multivariate_normal.cdf` is unusable: it
  controls only the absolute error (`releps` is ignored, also in 2-D), and the prefactor
  `exp(t_hat' Sigma t_hat / 2)` blows that up in the tails. `_mvn_logtail` uses Genz's
  separation of variables in log-space instead, with a tensor Gauss-Legendre rule in the
  `y` variables on truncated intervals `[a_i, a_i + s_i]`. The integrands are analytic
  there, so it converges exponentially: relative error ~1e-14 for `d <= 3` against
  adaptive quadrature, also for probabilities of 1e-24. Gauss-Legendre in Genz's
  `w`-space converged only algebraically (~1e-5) because of the endpoint singularities
  of `Phi^-1`; Sobol QMC was similar. The truncation needs the mass to be near the
  corner `b` (`Sigma^-1 b >= 0`), which holds for all orthants in eq. (5), including the
  conditional ones in eq. (8). The nodes per dimension shrink for `d >= 5`, to keep
  the total under 2e5.
- **Derivatives (A1).** One recursion handles distinct and repeated indices: eq. (8)
  plus the product and chain rules. Results are memoized per evaluation, and the
  prefactor is applied in log-space, so nothing over- or underflows.
- **1-D base case (A3).** Eq. (5) with `d = 1` (Robinson-type), not Lugannani-Rice. When
  `t_hat_j = 0`, the two reflected orthant approximations add up *exactly* to the
  lower-dimensional one, so the recursion is continuous, but only if the 1-D level is
  also eq. (5). With LR, the cdf of the 3-D gamma below would jump by ~3e-3 across
  the hyperplanes.
- **Near zero (A4).** Not needed: eq. (5) has no singularity at `t_hat_j = 0`, and the
  measured jumps across the hyperplanes are O(eps).

Measured (A6):

| Case | Result |
| ---- | ------ |
| Normal, `d = 3, 4, 5`, correlated | exact to ~1e-9 (the accuracy of the QMC reference) |
| `bivariate_gamma([2, 1, 3] * k)`, max abs error | k=1: 7.4e-3, 4: 1.7e-3, 16: 4.0e-4, 64: 1.0e-4, 256: 2.6e-5 |
| same, `"KL"` | k=1: 4.1e-3, 4: 9.2e-4, 16: 2.5e-4, 64: 6.5e-5, 256: 1.6e-5 |
| 3-D gamma `Z_0 + Z_i`, `a = [2, 1, 3, 1.5] * k` | k=1: 1.2e-2, 4: 3.1e-3, 16: 7.5e-4, 64: 1.9e-4 |
| permutation invariance, reflection consistency | to machine precision |
| cost | ~0.05-0.2 s per point for `d = 3`, ~0.3 s for `d = 5` (normal) |

So phase A has a clean `O(1/n)` rate, with errors about 1.5x those of `"KL"` in 2-D.
That is good enough to postpone phase B.

## Risks and open questions

- **QMC noise for d >= 3.** (Resolved: deterministic Gauss-Legendre SOV, see the outcome.) Genz's algorithm has an absolute error of about 1e-6 and is
  randomized, so fix the seed. The cubic term needs up to three derivatives, each a
  lower-dim orthant probability. That works, but the cost grows with the number of index
  triples.
- **Prefactor overflow.** `exp(t_hat' Sigma t_hat / 2)` multiplies a tiny orthant
  probability far in the tails. The absolute error stays moderate, but evaluate the
  product in a numerically stable way (e.g., in log-space, or by conditioning on the
  largest component) if needed.
- **Continuity at `t_hat_j = 0`.** When `t_hat_j = 0`, the marginal saddlepoint of
  `T_{-j}` equals `t_hat_{-j}`, and the normal-tail identities hold term by term, so the
  recursion is expected to be continuous. Verify numerically (A4).
- **Cost.** The recursion visits at most `3^d` subproblems per point, which is
  acceptable up to about `d = 5`.
- **Accuracy vs `"KL"` in 2-D.** Kolassa (2003) reports that "neither is clearly
  superior" against Wang; we have not compared against our `"KL"`. The outcome decides
  phase B.

## Phase B: plan and derivation

Goal: `MultivariateSaddlePointApprox.cdf(backend="KL")` for any `d`, the Lugannani-Rice type
approximation of Kolassa & Li (2010), sections 2-3, generalizing the bivariate `"KL"` code
(`BivariateSaddlePointApprox._kl_cdf`, `_kl_tail`) so that `d = 2` reproduces it, then use it
for `d0 >= 3` variables of interest in `conditional_cdf` (F2 of
`docs/conditional_vector_plan.md`). Notation: `n` absorbed in `K`, `L(tau) = K(tau) - tau.x`,
one fixed order of the variables `1..d`, all saddlepoint components positive (the orthant is
reflected, see B4).

### B1. Nested roots and their jets

For a prefix `gamma^k` (first `k` components fixed) let `tau_tilde^k(gamma^k)` minimize `L` over
the others and `l_k(gamma^k) = L(tau_tilde^k(gamma^k))` (profile; `l_0 = L(tau_hat) =
-w_hat.w_hat/2`). Then `(w_j - w_hat_j)^2 / 2 = D_j(tau^j) := l_j(tau^j) - l_{j-1}(tau^{j-1})`,
and `w_tilde_j(w^{j-1}) = w_hat_j - sqrt(2 E_j(tau^{j-1}))`, `E_j(gamma) = l_j(gamma, 0) -
l_{j-1}(gamma)`. Derivatives of the profiles (implicit function theorem, as in
`_ConditionalProfile`): gradient `K_a - x_a`, Hessian the Schur complement `H_k = V'K''V`,
third derivatives `K'''[V, V, V]`, `V = [I; -K_FF^-1 K_Fk]`, `F` the free components, at
`tau_tilde^k`.

2-jet of `w_k(tau^k)` at `tau_hat^k`, with `s = w_k - w_hat_k`, `s^2/2 = D_k`, `s = 0` there:
`s_a s_b = D_ab` gives `s_a = H_k[a, k] / sqrt(H_k[k, k])` (the Hessian of `D_k` is the
rank-one `H_k[:, k] H_k[k, :] / H_k[k, k]`, because `H_{k-1}` is the Schur complement of
`H_k`). The third derivative gives `s_a s_bc + s_b s_ac + s_c s_ab = D_abc`, solved for the
symmetric `s_bc` in the order `(k, k)`, `(k, c)`, `(b, c)`, since `s_k > 0`. Here `D_abc =
l_k''' - l_{k-1}'''` (padded). The inverse map `tau^{j-1}(w^{j-1})` has Jacobian `B =
(dw/dtau)^-1` (lower triangular) and second derivatives `-B_ik s^(k)_ab B_ap B_bq`.

2-jet of `F_j(gamma) = w_hat_j - sqrt(2 E_j)` at `tau_hat^{j-1}`, with `r = sqrt(2 E_j) =
w_hat_j - w_check_j`: `E_a = K_a(p_j) - x_a`, `E_ab = H_j(p_j)[<j, <j] - H_{j-1}(tau_hat)`, at
`p_j = tau_tilde^j(tau_hat^{j-1}, 0)`; `F_a = -E_a / r`, `F_ab = -E_ab / r + E_a E_b / r^3`.
Then `w_tilde_j` has gradient `c_j = B'F'` and Hessian `B'F''B + sum_a F_a d^2 tau_a / dw^2`
at `w_hat^{j-1}`. For `d = 2` this is `c = f'/e'`, `d = (f''e' - f'e'')/e'^3` of `_kl_tail`
(checked term by term).

### B2. The cubic term of `I^∅`

`u_j = w_j - w_tilde_j(w^{j-1})` (`u_1 = w_1`), `w_j = u_j + w_tilde_j(w^{j-1}(u))`, Jacobian
`dw/du = M` unit lower triangular, `g(u) = w.w/2 - w_hat.w`. At `u_hat` the gradient of `g` in
`w` vanishes, so `g'' = A = M'M`, and

```
g_abc = sum_j (M_ja W^j_bc + M_jb W^j_ac + M_jc W^j_ab),
W^j_ab = d^2 w_j / du_a du_b = sum_{l,m<j} w_tilde_j^{lm} M_la M_mb + sum_{l<j} w_tilde_j^l W^l_ab,
```

recursively, `W^1 = 0`. Expanding `exp(g)` to third order, a factor `u_a - u_hat_a` is
`-d/db_a - u_hat_a` applied to `T(b) = P(Z >= b)`, `Z ~ N(0, A)`, `b = A u_hat`:

```
I^∅ = C [T - 1/6 g_abc (d_a + u_hat_a)(d_b + u_hat_b)(d_c + u_hat_c) T],
C = exp(-w_hat.w_hat/2 + u_hat'A u_hat/2).
```

This is `_kolassa_orthant(K=-w_hat.w_hat/2, k2=A, k3=g''', tau=u_hat, tx=0)`, which already
evaluates it in log-space. For `d = 2`: `g_111 = 3cd`, `g_112 = d`, the formula of `_kl_tail`.

### B3. The other terms, all subsets

`G(tau) = r(tau) prod_j phi_j(tau^j)`, `phi_j = u_j / rho_j(tau_j) * dtau_j/dw_j` (`r = 1` and
`rho(tau) = tau` unconditional and continuous; the profile weight and `2 sinh(tau delta/2) /
delta` otherwise), `dtau_j/dw_j = (w_j - w_hat_j) / (K_j(tau_tilde^j) - x_j)`, or
`1 / sqrt(H_j[j, j])` where `w_j = w_hat_j`. `G = sum_A H^A`, `H^A = sum_{S in A} (-1)^|A-S|
G(tau^S)`, `h_A = H^A / prod_{j in A} u_j` analytic. The bivariate code keeps all four subsets
(`I^{1,2}` is needed for a relative error `O(1/n)` at large deviations, and makes it factorize
under independence). In general, keep all `2^d` subsets, and treat the poles that a term
does not absorb as in Lugannani-Rice: decompose `h_A` once more over the remaining poles.
With `P` the set of remaining poles and `B = U \ P`:

```
J = sum_P N_P a_P,      a_P = sum_{A in B} c_{A, B-A},
```

`c_{A,R}` is `h_A` with the poles `j in R` removed by *nested* differences, built from the
last position inwards: walking the order, component `j` of the point is `0` if `j` is in
neither `A` nor `R`, `tau_tilde_j(prefix)` if `j in A`, and if `j in R` the value is
`[f(tau_tilde_j) - f(0)] / u_j(prefix, tau_tilde_j)`, with `f` the recursion over the later
components and `u_j` evaluated at the actual prefix. (A flat inclusion-exclusion
`sum_Q (-1)^|R-Q| h_A(...) / prod u_j` with all `u_j` at one point, as first written here,
is not correct for `|R| >= 2`: the divided differences do not commute with the nested
minimizers.)

- `N_P` is the Gaussian integral with poles `P` in the coordinates `w_j` (`j in B`) and `u_j`
  (`j in P`): Jacobian `M_B` (rows `j in B` unit vectors), `A_B = M_B'M_B`, expansion point
  `(w_hat_B, u_hat_P)`, `b = A_B v_hat`; `N_P = C_B phi(b_B; A_BB) P(Z_P >= b_P | Z_B = b_B)`
  (density times conditional orthant, log-space via `_mvn_logtail`, the conditional point is
  the Schur complement times `u_hat_P >= 0`). `N_∅ = prod phi(w_hat_j)`. `N_U` is `I^∅`
  including the cubic term.
- `tau^(P)` is the point with `w_j = w_hat_j` for `j` not in `P` and `tau_j = 0` for `j in
  P`: sequentially `tau_j = 0` (`j in P`) or `tau_j = tau_tilde_j(tau^{j-1})` (property
  (2.5c)), so no inversion of `tau -> w` is needed. `tau^(∅) = tau_hat`.
- `u_j(tau) = w_j(tau^j) - w_j(tau^{j-1}, 0)`, computed from the profiles.

For `d = 2` this is exactly `_kl_tail`: `c_{{1}, ∅}` is the factor of `I^{1}`, `c_{{2}, ∅} =
f_2` (at `tau^({1}) = (0, tau_tilde_2)`), `c_{{1,2}, ∅} + c_{{2}, {1}}` is the bracket of
`I^{1,2}` plus the remainder of `I^{2}` (the `G(0, tau_hat_2)` terms cancel), and `c_{{1},
{2}} = 0`. Properties: exact for the normal (`G = 1`, `g` quadratic), the product of the
Lugannani-Rice marginals under independence (`u = w`, all `A_B = I`, `h_A` depends only on
`w_A`), `O(1/n)` relative error. Cost: `2^d` points `tau^(P)`, and the profiles at the
prefixes of the points `tau^S`, memoized by prefix.

Near-equal `w_j ~ w_hat_j` at a point `tau^S` (e.g., near independence) makes `(w_j -
w_hat_j) / (K_j - x_j)` a 0/0: there, `dw_j / dtau_j = sqrt(S*) (1 + (S(tau_j) - S*) /
(3 S*))` with `S = H_j[j, j]` along `tau_j` and `S*` at `tau_tilde_j(tau^{j-1})`, error
`O(delta^2)`.

### B4. Orthants, near-zero components, order blending

- **Reflection/inclusion-exclusion**: the recursion of `_kolassa_cdf`, with the KL tail in
  every orthant in which the reflected marginal saddlepoint is non-negative. 1-D tails: the
  Lugannani-Rice cdf (as in 2-D), via the hook `_kl_marginal_cdf`.
- **Near zero**: Lagrange interpolation in the saddlepoint components with `|t_j| < h_j`, fixed
  orientation, tensor product, as in `_kl_cdf_ordered`. Intermediate removable singularities
  (a component of a point `tau^(P)` close to 0, like `tau_tilde_2` in 2-D): interpolate the
  tail in that component of `t` the same way (the 2-D code interpolates the factor `f_2` in
  `tau_tilde_2` instead, so `d = 2` differs inside that band only, at the level of the
  interpolation error).
- **Blending**: with a fixed order, the approximation is continuous across `t_j = 0` only for
  the first variable (`w_hat_2..d` are then the nested roots of the marginal of the others).
  Weight the `d!` orders recursively: the first variable is `j` with weight
  `omega_j = z_j^-2 / sum_k z_k^-2`, `z_j = t_j sqrt(K''_jj(0))`, and the rest is weighted the
  same way among the remaining variables. For `d = 2`, `omega_1 = z_2^2 / (z_1^2 + z_2^2)`,
  the bivariate rule. At `z_j = 0` only orders starting with `j` get weight, and their weights
  are those of the `(d-1)`-dim approximation of `X_{-j}`, which is the term of the
  inclusion-exclusion, so the blend is continuous across every hyperplane `t_j = 0`, and it is
  symmetric (relabeling invariant). Orders with weight below `1e-12` are skipped. Cost: up to
  `d!` orders (6 for `d = 3`, 24 for `d = 4`).

### B5. Validation

Normal exact (`d = 3, 4`); `d = 2` against `BivariateSaddlePointApprox` `"KL"`; independence
(product of LR marginals); 3-D gamma with a common component `Z_0 + Z_i` vs quadrature, error
rate as shape parameters scale by `k`, against `"Kolassa"`; relabeling invariance; continuity
across `t_j = 0`; reflection consistency.

## Phase B: outcome

Implemented in `spapprox/kl.py` (`_KLSolver`: cached profiles and nested minimizers;
`_KLOrder`: one order of the variables, the jets of B1, the cubic term of B2 via
`_kolassa_orthant`, the subset terms of B3; `_kl_orthant`: the blend of B4; `_kl_cdf`: the
reflection and inclusion-exclusion) and wired in as
`MultivariateSaddlePointApprox.cdf(backend="KL")` with the hooks `_kl_weight`, `_rho` and
`_kl_marginal_cdf`, so the lattice version (all spans positive, eq. (4.8)) and the
conditional profile (F1/F2) reuse it. The default backend stays `"Kolassa"`.

Accuracy:

- `d = 2`: agrees with `BivariateSaddlePointApprox` `"KL"` to `1e-11` away from zero, `~2e-9`
  in the near-zero band (different interpolation nodes). Lattice `d = 2` to `1e-13`.
- Normal `d = 3, 4`: exact to `~1e-13`; relabeling invariance to `4e-14`.
- 3-D gamma with a common component (`Z_0 + Z_i`, shapes times `k`), maximum absolute error
  over a grid, KL vs Kolassa: `k = 1`: `4.3e-3` vs `1.1e-2`; `k = 4`: `1.2e-3` vs `2.6e-3`;
  `k = 16`: `3.5e-4` vs `6.0e-4`, so `O(1/k)`. Relative error in the lower tail at fixed
  saddlepoints: `k = 4`: `3.9e-3` vs `5.3e-2`; `k = 16`: `1.4e-3` vs `1.3e-2`; `k = 64`:
  `4.0e-4` vs `2.4e-3` (probabilities down to `1e-69`); at `k = 1`, `t = (-2, -2, -2)`:
  `1.25%` vs `16%`.
- 3-D Poisson with a common shock (lattice), relative error KL vs Kolassa: `k = 1`: `2.5%` vs
  `6.3%`; `k = 4`: `0.65%` vs `2.1%`; `k = 16`: `0.17%` vs `0.55%`.

Known properties and limitations:

- Continuous across a single hyperplane `t_j = 0` (to `~1e-9`). Where two or more components
  of the saddlepoint vanish together the limit depends on the direction of approach (`~1e-4`
  at `k = 1`), because the blend weights are angular; the bivariate code has the same
  property at `t = 0` (`~3e-5`).
- For `d = 3` the slope has a small kink across `t_j = 0` of order `n^{-3/2}` (standardized
  jump `2.8e-3`, `3.8e-4`, `4.8e-5` for `k = 1, 4, 16`), below the `O(1/n)` error; not present
  for `d = 2`.
- Orders whose nested minimizers leave the domain of the cgf are dropped from the blend (the
  result is `NaN` if all are).
- Two orders share profile cache entries, so equal values can differ by rounding; `w_j` is
  snapped to its expansion within `1e-7` standard deviations of the minimizer
  (`_snap_width`), and the saddlepoint is polished by Newton steps.
- Cost: about `0.7 s` per point for `d = 3`, up to `d!` orders and `2^d` subset terms;
  near-zero tensor interpolation multiplies this.
- Lattice spans go through `_cdf_evaluator` (`"KL"` is in `_lattice_backends`), so
  `fit_cdf(delta=..., backend="KL")` works too. For `d > 2` all spans must be positive: mixed
  lattice/continuous spans raise `NotImplementedError` (use `"Kolassa"`). The bivariate class
  keeps its own mixed-span `"KL"` and `"Wang"` (`_LatticeBivariateApprox`); the
  `delta[k] == 0` convention of `_rho` / `_kl_marginal_cdf` is shared by both.

