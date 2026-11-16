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

## Phase B outline (only if needed)

B1. Derive `g^{jk}` and `g^{jkl}` for general `d` from the nested parametrization; check
    them against the 2-D formulas in `_kl_tail`.
B2. Implement the nested constrained minimizers `tau_tilde^j` and the derivatives
    (2.7)-(2.9).
B3. Implement `I^∅` with the cubic correction, and the `I^{r}` terms via eq. (3.10).
B4. Add reflection and inclusion-exclusion. Blend over variable orders for continuity,
    or find a cheaper symmetrization.
B5. Validate as in A6 and compare with phase A.
