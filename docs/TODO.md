# Priorities

- Go through bug backlog
- [x] Review implementation against Alhejaili, AlGhamedi (2025) - A review of saddlepoint
  approximation: done, see "Review: Alhejaili and AlGhamedi (2025)" below. Follow-ups: bug 62 (fixed),
  `Pr1` (done), and the lattice item unconditional bivariate cdf (done)
- Finish bivariate approximation:
  - basically this case: test_bvar_cdf_near_zero
  - are there more edge cases?
- Multivariate approximation: cdf phase A done (`backend="Kolassa"`); phase B only if
  needed, see `docs/multivariate_cdf_plan.md`

- Bootstrap distributions (Booth and Butler): done, see `docs/estimating_equations_plan.md`

# Bug backlog

Findings from two repo-wide reviews. Every item was reproduced empirically
(unless marked _suspected_):

- **Review 1** at `b0cd9e4` (branch `dev`): items 1-18.
- **Review 2** at `6b4b894` (library code identical to `b0cd9e4`): re-confirmed
  all of 1-18, resolved 17 and 18, and added items 19-56.

Items marked `[x]` below are fixed. The fast suite (`pytest -m "not slow"`, 186 tests)
passes, so none of these are caught by existing tests.

## Status legend

- [ ] not started
- [x] fixed + regression test added

---

## High severity

Wrong numbers, hangs, or hard crashes on normal use.

### [x] 1. `poisson()` declares a wrong domain

- **Where:** `spapprox/cgfs.py:321`
- **What:** declares `Domain(ge=0)`, but `K(t) = mu * (exp(t) - 1)` is finite on
  all of R. Everything below the mean is masked to `nan`.
- **Repro:** `poisson(mu=1).K(-1.0)` -> `nan`; true value `-0.6321205588285577`.
  Knock-on: `UnivariateSaddlePointApprox(poisson(3)).cdf(x=1.)` -> `nan`, and it
  feeds the infinite loop in item 24.
- **Fix:** drop the domain restriction (full real line).
- **Why it hid:** existing tests only probe `t > 0`.
- **Resolved:** domain restriction removed. Regression tests:
  `test_poisson_negative_t` (no optional backend needed), plus `t = -0.4` added to
  the poisson cases of `test_basic` and `test_dKinv` in `tests/test_cgfs.py`.

Manual note: confirmed, see Butler 2007, page 9, example 1.1.6

### [x] 2. `d2K0` / `d3K0` mutate the raw cache in place

- **Where:** `spapprox/cgf_base.py:1133-1135`, `1146-1149`
- **What:** uses `*=` on the same object held by `_d2K0_raw_cache`, so the
  scaling compounds on every access after a scale assignment. Because algebra
  methods forward raw caches, the corruption also leaks into derived objects.
- **Repro:**
  ```python
  c = multivariate_norm(scale=np.array([1., 2.]))
  c.cov  # [[1, 0], [0, 4]]
  c.scale = np.array([1., 2.]); c.cov  # [[1, 0], [0, 16]]
  c.scale = np.array([1., 2.]); c.cov  # [[1, 0], [0, 64]]
  # also: c.cov; c2 = c.mul(1.0); c2.cov -> diag [1, 16], and c.cov now [1, 16]
  ```
- **Fix:** build a new array (`out = raw * factor`) instead of in-place `*=`.
- **Resolved:** 1-D scale branches now build new arrays (`np.outer` / `einsum`).
  Regression test: `test_vector_scale_does_not_mutate_raw_cumulants` in
  `tests/test_cgfs.py` (repeated scale assignment, raw caches, `mul` propagation).

Manual note: I don't believe you, just checked it, but I don't find the result you claim

### [x] 3. `Domain.from_domains` crashes

- **Where:** `spapprox/domain.py:568`, `583`
- **What:** `i, j = 0` -- unpacking an `int`.
- **Repro:** `Domain.from_domains(Domain(dim=2, A=[[1, 1]], a=[1]), Domain(dim=1, l=1))`
  -> `TypeError: cannot unpack non-iterable int object` (both the `A` and `B`
  paths).
- **Fix:** `i, j = 0, 0`.
- **Blast radius:** reaches `from_univariate`, `from_cgfs`, `stack` for any
  domain carrying linear constraints.

### [x] 4. `Domain.ldotinv` sign error on `ge` bounds

- **Where:** `spapprox/domain.py:394-399` (wrong sign at `398`)
- **What:** appends `+ge` where `x >= ge` requires `-x <= -ge`. The strict `g`
  branch alongside it does it correctly. Invisible when `ge == 0`.
- **Repro:**
  ```python
  d = Domain(dim=2, ge=1.0)
  [0., 0.] in d              # False (correct)
  [0., 0.] in d.ldotinv(np.eye(2))  # True (wrong -- domain inverted)
  ```
- **Fix:** negate to match the `g` branch.
- **Resolved:** `ge` is now negated. Regression test: `test_ldotinv_ge_bound` in
  `tests/test_domain.py` (positive, negative and per-component `ge`; identity and
  scaled transforms).

Manual note: indeed, this seems to need a fix

### [x] 6. `dK_inv` can loop forever

- **Where:** `spapprox/cgf_base.py:484`
- **What:** the bracket-expansion call omits `loc=0, scale=1`, which every
  neighbouring call passes. With `loc != 0` the loop never satisfies its exit
  condition.
- **Repro (hangs):**
  `UnivariateCumulantGeneratingFunction(K=lambda t: -np.log(1 - t), domain=Domain(l=1), loc=5.).dK_inv(5.01, method="brentq")`
  (expected about `-49`; correction: the exact value is `-99`, since
  `dK(t) = 1/(1-t) = 0.01`).
- **Fix:** add `loc=0, scale=1` to match siblings, and cap the iterations.
- **Note:** guard the regression test with a timeout or bounded iteration count.
- **Resolved:** `loc=0, scale=1` added; both bracket-expansion loops are capped at
  1000 steps and raise if no bracket is found (previously an unreachable `x`, e.g.
  below the infimum of `dK`, also looped forever). Regression test:
  `test_dKinv_bracket_search_with_loc` in `tests/test_cgfs.py`, run in a
  subprocess with a timeout.

### [x] 7. `fit_ppf` raises `UnboundLocalError`

- **Where:** `spapprox/spa.py:492-499` (read at `497`)
- **What:** `x_range` is only bound in the branch where `_x_cache` is absent,
  but is read unconditionally afterwards.
- **Repro:**
  ```python
  u = UnivariateSaddlePointApprox(norm())
  u.fit_saddle_point_eqn()
  u.fit_ppf()  # UnboundLocalError: cannot access local variable 'x_range'
  ```
- **Fix:** derive `x_range` from the cache in the other branch too, and honour
  the `t_range` argument. See also item 40.
- **Resolved (together with item 40):** `fit_ppf` always builds its grid, honours
  `t_range`/`num`, and stores it in its own `_ppf_t_cache`. Regression tests:
  `test_fit_ppf_after_fit_saddle_point_eqn` and `test_fit_ppf_honours_t_range` in
  `tests/test_spa.py`.

### [x] 8. `atol=1.0 - 8` typo

- **Where:** `spapprox/spa.py:713` (`BivariateSaddlePointApprox.cdf`; recursion
  at `827`)
- **What:** evaluates to `-7.0`; `1e-8` was intended. A negative tolerance means
  the near-`t=0` guard never fires -> `RecursionError`. Exactly `t=0` works only
  because `isclose` treats equal values as close.
- **Repro:** `BivariateSaddlePointApprox(multivariate_norm(dim=2)).cdf(t=np.array([1e-9, 1e-9]))`
  -> `RecursionError`.
- **Fix:** `atol=1e-8` (and see item 23 for the tolerance itself).

### [x] 9. `np.asanayarray` typo (5 sites)

- **Where:** `spapprox/cgf_base.py:518`, `1723`, `1837`, `1885`;
  `spapprox/spa.py:396`
- **What:** misspelling of `np.asanyarray` -> `AttributeError` whenever `t0=` is
  supplied (`dK_inv`, `ppf`), or whenever the `fillna` path is taken (`d2K`,
  `d3K` with a single out-of-domain `t`).
- **Repro:** `u.ppf(np.array([.2, .7]), t0=np.array([0., 0.]))` -> `AttributeError`.
- **Fix:** mechanical rename; confirm no other spellings via
  `grep -rn "asanayarray" --include=*.py .`

### [x] 10. Derived-CGF domains are in the wrong `t` units when `scale != 1`

- **Where:** `spapprox/cgfs.py:260` (`univariate_sample_mean`);
  `spapprox/cgf_base.py:737` (univariate `add`)
- **What:** the parent domain is reused without rescaling by `1/scale`, so the
  domain and `K` disagree. This breaks the package-wide masking contract, where
  `t in cgf.domain` is supposed to predict whether `K(t)` is finite. With
  `scale < 1` it masks valid points instead.
- **Repro:**
  ```python
  sm = univariate_sample_mean(exponential(scale=2), 5)
  4.0 in sm.domain   # True  (true bound is t < 2.5)
  sm.K(4.0)          # nan

  s = exponential(scale=2) + norm()
  0.8 in s.domain    # True
  s.K(0.8)           # nan
  (exponential(scale=0.5) + norm()).K(1.5)  # nan, true 2.511
  ```
- **Fix:** intersect `self.domain.mul(1/self.scale)` with
  `other.domain.mul(1/other.scale)`; same idea for the sample mean.
- **Same family:** items 14, 21, 22.
- **Resolved:** univariate `add` and `univariate_sample_mean` now transform the
  operand domains by `1/scale` (and `n/scale`). Needed item 26 to be fully
  effective. Regression tests: `test_derived_domain_matches_K` and
  `test_derived_domain_scale_values` in `tests/test_cgfs.py`.

### [x] 17. Swapped pairing in the bivariate Lugannani-Rice-style cdf -- RESOLVED: code is wrong

- **Where:** `spapprox/spa.py:912`
- **What:** pairs `Phi(w_tilde) * n` with `Phi(w) * n_tilde`. The docstring's
  pairing is the correct one.
- **Evidence:** for independent components the formula must reduce to
  `(Phi(w_X) + n_X)(Phi(w_Y) + n_Y)`. With `from_univariate(gamma(a=3), gamma(a=5))`
  the current code is off by up to **-0.043** from the exact product of gamma
  cdfs; with the swapped pairing it matches the product of the two univariate
  LR cdfs to <= 1e-4. A correlated bivariate normal cannot discriminate (all
  variants are exact there).
- **Fix:** `Phi(w_tilde) * n + Phi(w) * n_tilde + n * n_tilde` (per docstring).
- **Resolved:** pairing swapped to match the docstring. At 16 points with both
  saddlepoint components nonzero, the max deviation from the product of the
  univariate LR cdfs dropped from 0.041 to 1e-7, and from the exact gamma
  product from 0.040 to 1.5e-4. Regression test:
  `test_bvar_cdf_independent_reduces_to_product` in `tests/test_spa.py`. The
  case where one component is zero still needs item 18.

### [x] 18. `sps.norm.pdf(w)` vs documented `phi(x_tilde)` -- RESOLVED: code is wrong

- **Where:** `spapprox/spa.py:862`, `892`
- **What:** matters when one saddlepoint component is 0. Same gamma pair at
  `t=(0.5, 0)` and `(0, 0.4)`: with only item 17 fixed, errors are **-0.034**
  and **-0.038**; with both fixed, -8e-5 and -5e-5.
- **Fix:** `sps.norm.pdf(tx.T[0])` at `862`, `sps.norm.pdf(tx.T[1])` at `892`.
- **Resolved:** the singular-case `n_tilde` now uses `phi(x_tilde)` (now at
  lines `868` and `898`). At `t=(0.5,0)`, `(-0.8,0)`, `(0,0.4)` and `(0,-1)`, the
  max deviation from the exact gamma product dropped from 0.038 to 7.6e-5.
  Regression test: `test_bvar_cdf_independent_one_saddlepoint_component_zero`
  in `tests/test_spa.py`.

### [x] 19. Cumulants at zero crash on CGFs without analytic derivatives

- **Where:** `spapprox/cgf_base.py:635`, `644`, `653` (univariate), `1160`,
  `1169`, `1178` (multivariate)
- **What:** `_dK0_raw` / `_d2K0_raw` / `_d3K0_raw` call `self._dK(0)` etc.
  before the lazy numerical-derivative object has been built.
- **Repro:** `UnivariateCumulantGeneratingFunction(K=lambda t: t**2 / 2).dK0`
  -> `TypeError: 'NoneType' object is not callable`. Same for
  `UnivariateSaddlePointApprox(...).cdf(t=0.)`, for `univariate_empirical(...).d2K0`
  read before any `d2K(t)` call, and for the multivariate analogues.
- **Fix:** call `self.dK(0, loc=0, scale=1)` etc., which initialises the backend.
- **Resolved:** the six `_d*K0_raw` properties now call the public `dK`/`d2K`/`d3K`
  with `loc=0, scale=1`. Regression tests: `test_cumulants_at_zero_without_analytic_derivatives`
  in `tests/test_cgfs.py` and `tests/test_mvar_cgfs.py`, and
  `test_cdf_at_zero_without_analytic_derivatives` in `tests/test_spa.py`.

### [x] 20. Multivariate findiff gradient applies `loc`/`scale` twice

- **Where:** `spapprox/cgf_base.py:1602`
- **What:** the findiff `Gradient` is built on the transformed `self.K`, and the
  result is then transformed again.
- **Repro:**
  `MultivariateCumulantGeneratingFunction(K=lambda t: np.sum(t**2 / 2 + t, axis=-1), dim=2, loc=[1, 2], scale=2., numdiff_backend="findiff").dK(np.zeros(2))`
  -> `[7, 10]`; expected `[3, 4]` (numdifftools gives `[3, 4]`).
- **Fix:** `Gradient(lambda tt: self.K(tt, loc=0, scale=1), self.domain.dim)`.
- **Resolved:** the findiff `Gradient` is now built on the standardized `K`,
  matching the numdifftools path and the findiff `Hessian`/`Tressian`.
  Regression test: `test_numerical_dK_applies_loc_scale_once` in
  `tests/test_mvar_cgfs.py` (both backends, vector and matrix scale).

### [x] 21. Multivariate + univariate `add` ignores both operands' scale in the domain

- **Where:** `spapprox/cgf_base.py:1362`
- **What:** same family as item 10.
- **Repro:** `s = multivariate_norm(dim=2) + exponential(scale=2)`;
  `[.4, .4] in s.domain` -> True, `s.K([.4, .4])` -> `nan`.
- **Fix:** `self.domain.ldotinv(self.scale_mat.T).intersect(other.domain.mul(1/other.scale).ldotinv(np.ones((1, dim))))`.
- **Resolved:** applied the suggested fix. The domain now matches the analytic
  `K` exactly on a grid for `exp(2)`, `exp(0.5)`, `-exp(2)` and a scaled
  multivariate operand (previously 29-211 mismatching points). Regression
  test: `test_add_univariate_domain_matches_K` in `tests/test_mvar_cgfs.py`.

### [x] 22. `from_univariate` / `from_cgfs` combine domains of the standardized variables

- **Where:** `spapprox/cgf_base.py:1973`, `2067`
- **What:** the operands' domains are stacked untransformed, but the new object
  has scale 1.
- **Repro:** `MultivariateCumulantGeneratingFunction.from_univariate(exponential(scale=.5), norm()).K([1.5, 0])`
  -> `nan` (true 1.386). With `scale=2`, `[0.8, 0]` is reported in-domain but
  `K` is `nan`.
- **Fix:** transform each domain by its operand's scale before `from_domains`.
- **Note:** confirmed again while fixing items 14/26: the sum of
  `from_univariate(exponential(scale=2), norm(loc=3))` and
  `from_univariate(norm(loc=-1), gamma(a=2, scale=0.5))` reports `[0.6, 0]` in
  domain while `K` is `nan`.
- **Resolved:** both methods now stack each operand's domain in actual `t`
  units via a new helper `_domain_in_t_units` (`mul(1/scale)` for univariate,
  `ldotinv(scale_mat.T)` for multivariate operands). The domain matches the
  analytic `K` exactly on a grid in all tested cases, including the note above
  (previously 34-184 mismatching points). Regression test:
  `test_stacked_domain_matches_K` in `tests/test_mvar_cgfs.py`.

### [x] 23. LR/BN `t=0` guard is far too tight -> catastrophic cancellation near the mean

- **Where:** `spapprox/spa.py:182`, `215`; bivariate `827`, `854`, `885`
- **What:** `np.isclose(t, 0)` (atol 1e-8) only catches `t` essentially at 0;
  just outside it the `1/w - 1/u` terms cancel catastrophically.
- **Repro:** `UnivariateSaddlePointApprox(gamma(a=3)).cdf(t=1e-7)` -> **-12028**
  (expected 0.5768); at `t=1e-5` -> 0.5663 (error 0.01). After
  `fit_saddle_point_eqn()`, `cdf(x=3.0)` (the mean) -> `nan`. Bivariate
  `t=(0.3, 1e-6)` -> about 5.0.
- **Fix:** widen the guard (e.g. `|w| < 1e-4`) or use a series expansion near 0.
- **Resolved:** within `|t| sqrt(K''(0)) < 1e-2` (`_cdf_singular_width`) the
  cdf is now quadratically interpolated through the exact limit at 0 and the
  direct formula at `+-h`. That window was chosen from measurements against
  mpmath: roundoff is ~1e-11 at the nodes, interpolation error ~1e-8. The
  bivariate cdf interpolates per near-zero component (tensor product); the
  value at `t=(0,0)` uses Richardson extrapolation of axis averages, replacing
  the old average at `+-1e-7`, which was itself broken (`cdf(t=(0,0))` was
  ~1e8). Also fixed the BN `t=0` value: it used the LR limit instead of
  `Phi(rho3/6)` (off by 4.7e-4). Errors near 0 are now <= 5e-8 (univariate and
  bivariate). Regression tests in `tests/test_spa.py`:
  `test_cdf_near_zero_no_cancellation`,
  `test_cdf_at_mean_after_fit_saddle_point_eqn`,
  `test_bvar_cdf_near_axes_no_cancellation`.

### [x] 24. `infer_t_range` loops forever on lattice / bounded-support CGFs

- **Where:** `spapprox/spa.py:444-466`
- **What:** the expansion loop has no iteration cap and waits for the LR cdf to
  reach 0/1, which it never does for lattice CGFs. Default `normalize_pdf=True`
  calls it, so `pdf()` hangs.
- **Repro (hangs):** `UnivariateSaddlePointApprox(binomial(10, .3)).pdf(t=.1)`;
  same for `poisson(3)` (worse with item 1: `lb=-0.0` never moves).
- **Fix:** cap iterations; treat `nan`, cdf < 0 or cdf > 1 as end of range;
  raise if no range is found.
- **Resolved:** each bound is expanded by a helper with an iteration cap
  (`max_iter=500`). A step is accepted only if `dK` is finite and the cdf stays in
  [0, 1] and moves towards its target; otherwise a smaller step is tried, and when
  those run out the last valid bound is returned with a `RuntimeWarning`. (For
  lattice CGFs the LR cdf turns around in the lower tail, e.g. it bottoms out at
  ~0.015 for `binomial(10, .3)`.) Raises `ValueError` if the initial bounds give
  no valid increasing cdf. Ranges for continuous distributions are unchanged.
  Regression test: `test_infer_t_range_lattice_terminates` in `tests/test_spa.py`
  (subprocess with timeout).

### [x] 25. CI install step cannot work

- **Where:** `.github/workflows/python-app.yml:30`
- **What:** runs `python setup.py install`, but there is no `setup.py` (flit
  backend). The workflow only triggers on `main` and has never run, so this has
  gone unnoticed. Also uses outdated `@v3` actions and tests only 3.11.
- **Fix:** `pip install -e ".[numdiff]"`; bump actions; consider a version
  matrix that includes the declared floor (see item 16).

---

## Medium severity

Contract violations and silently wrong results in narrower situations.

### [x] 5. `__getitem__` slices cumulants on one axis only

- **Where:** `spapprox/cgf_base.py:1261`, `1266`
- **What:** `d2K0[item]` where `d2K0[np.ix_(item, item)]` is meant; `d3K0` needs
  the rank-3 equivalent.
- **Repro:** slicing a cached 3-D CGF with `[0, 2]` yields a 2x3 "covariance"
  `[[4, 1, 0.5], [0.5, 0.2, 1]]` instead of `[[4, 0.5], [0.5, 1]]`.
- **Fix:** use `np.ix_` on all axes.
- **Resolved:** cached cumulants are now sliced with `np.ix_(item, item)` and
  `np.ix_(item, item, item)`. Regression test: `test_slicing_cached_cumulants` in
  `tests/test_mvar_cgfs.py`, which compares against analytic cumulants of a linear map
  of independent gammas. (The existing `test_slicing` cases never cached the cumulants
  before slicing, which is why this went unnoticed.)

### [x] 11. In-place scaling mutates caller / user arrays

- **Where:** `spapprox/diff.py:48` (`transform_rank3_tensor`, `T *= ...`);
  `spapprox/cgf_base.py:1804`, `1865` (multivariate `d2K` / `d3K`)
- **What:** writes through to the caller's array and returns the same object.
  When a user-supplied `d2K`/`d3K` returns a fixed array, it compounds on every
  call. With an int `T` and float diagonal it raises `UFuncTypeError`.
- **Repro:** `C = np.eye(2)`; `MultivariateCumulantGeneratingFunction(..., d2K=lambda t: C, scale=2.)`;
  three calls to `d2K(0)` -> `4, 16, 64` and `C` is modified (expected `4` each
  time). `d3K` with a vector scale: `8, 64, 512`.
- **Fix:** `y = y * scale**2`; `T = T * Ai.reshape(shape)` (copy, not in place).
- **Resolved:** applied as proposed, plus the same change for the scalar-scale
  branch of `d3K` (`y = y * scale**3`). Regression tests:
  `test_scaled_derivatives_do_not_mutate_user_arrays` in `tests/test_mvar_cgfs.py`
  (scalar, vector and matrix scale) and
  `test_transform_rank3_tensor_does_not_mutate_input` in `tests/test_diff.py`.

### [x] 12. `transform_rank3_tensor` returns index-reversed output

- **Where:** `spapprox/diff.py:50`
- **What:** `tensordot` moves each contracted axis to the front, reversing the
  axis order. With three distinct matrices the result equals
  `einsum('il,jm,kn,lmn->ijk', A, B, C, T).transpose(2, 1, 0)`. Also wrong for a
  single matrix with non-symmetric `T`, and for mixed vector/matrix inputs.
- **Note:** masked in current use because callers pass one matrix and a
  symmetric `T`.
- **Fix:** `np.einsum('il,jm,kn,lmn->ijk', A, B, C, T)`.
- **Resolved:** applied as proposed; vector inputs are converted to diagonal
  matrices first. Regression test: `test_transform_rank3_tensor_axis_order` in
  `tests/test_diff.py` (distinct matrices, non-symmetric `T`, mixed inputs).

### [x] 13. `bivariate_gamma` domain omits `t1 + t2 < 1`

- **Where:** `spapprox/cgfs.py:217`
- **Repro:** `t = [0.9, 0.9]` reports `in domain` -> `True` while `K(t)` -> `nan`.
- **Fix:** add `B=[[1, 1]], b=[1]`.
- **Resolved:** the constraint is added when `a[0] > 0`. For `a[0] = 0` the
  components are independent and `K` is finite for `t1 + t2 >= 1`, but `K`
  returned `0 * log(negative) = nan` there; that term is now skipped. Regression
  test: `test_bivariate_gamma_domain_matches_K` in `tests/test_mvar_cgfs.py`.

### [x] 14. Multivariate `add` builds the wrong domain

- **Where:** `spapprox/cgf_base.py:1405-1407`
- **What:** uses `domain.ldot(S).add(loc)`. For `Y = loc + S X` the t-domain is
  `{t : S^T t in D}`: `loc` plays no role, and the operation should be
  `ldotinv(S.T)`. Separately, `self.loc_vect` is used for `other` (copy-paste).
- **Repro:** `multivariate_norm(loc=[3, 3]) + M(K=-sum(log(1 - t)), domain l=[1, 1])`:
  `[2, 2]` is reported in-domain but `K` is `nan`. Also wrong with `scale=2` on
  the other operand.
- **Fix:** `self.domain.ldotinv(self.scale_mat.T).intersect(other.domain.ldotinv(other.scale_mat.T))`.
- **Resolved:** applied as proposed. Surfaced by the item 26 fix (the union-style
  intersect had hidden it; `test_stack` started failing). Regression test:
  `test_add_domain_matches_K` in `tests/test_mvar_cgfs.py`.

### [x] 15. `from_cgfs` crashes on mixed univariate/multivariate input

- **Where:** `spapprox/cgf_base.py:2048`, `2064`
- **What:** fails when cumulants are already cached and the inputs mix
  dimensionalities.
- **Repro:** `norm()` and `multivariate_norm(dim=2)` with cached cumulants;
  `from_cgfs(a, b)` -> `ValueError: zero-dimensional arrays cannot be concatenated`.
- **Fix:** `np.atleast_1d(cgf.dK0)` and `np.reshape(cgf.d3K0, (d, d, d))`.
- **Resolved:** applied as proposed, plus `np.atleast_2d` on `d2K0` for
  robustness. Regression test: `test_from_cgfs_mixed_dims_cached_cumulants` in
  `tests/test_mvar_cgfs.py` (both orders, compared to analytic cumulants).

### [x] 26. `Domain.intersect` returns a union for like-kind bounds

- **Where:** `spapprox/domain.py:446-495`
- **What:** takes the max of upper bounds and min of lower bounds (reversed).
  NaN entries also drop bounds.
- **Repro:** `1.5 in Domain(l=1).intersect(Domain(l=2))` -> True (expected
  False); same for `g`, `le`, `ge`, and in 2-D.
  `Domain(dim=2, l=[nan, 1]).intersect(Domain(dim=2, l=[1, nan])).l` ->
  `[nan, nan]` (expected `[1, 1]`).
- **Note:** used by univariate `cgf + cgf` (`cgf_base.py:737`); actual impact
  there may be dampened because each component masks its own domain.
- **Fix:** `min`/`np.fmin` for `l`/`le`, `max`/`np.fmax` for `g`/`ge`.
- **Resolved:** applied as proposed via a NaN-aware helper. Regression test:
  `test_intersect` in `tests/test_domain.py`.

### [x] 27. Bound-ordering validation is wrong and inconsistent

- **Where:** `spapprox/domain.py:12`, `46`, `61`
- **What:** the code enforces `l > le > ge > g` while docstring and messages say
  `g > ge > le > l`. Valid combinations are rejected: `Domain(l=1, le=2)`,
  `Domain(l=inf, le=inf)`, and `Domain(le=1, ge=1)` in 1-D (accepted in 2-D,
  where the check is non-strict). Consequently
  `Domain(l=1).intersect(Domain(le=2))` -> `AssertionError`.
- **Fix:** only require upper bounds >= lower bounds, consistently across
  dimensions; fix the docstring.
- **Resolved:** applied as proposed: a single elementwise check (NaN entries
  ignored) for every dimension; equal strict bounds (empty domain) are allowed.
  Regression tests: `test_valid_bound_combinations`,
  `test_invalid_bound_combinations`, `test_intersect_mixed_strictness` in
  `tests/test_domain.py`.

### [x] 28. List bounds are only converted to arrays when several are given

- **Where:** `spapprox/domain.py:62-65`
- **Repro:** `d = Domain(dim=2, l=[1, 2])`: `d + 1` -> `TypeError` (list
  concatenation); `d * 2` -> `AssertionError` (list repeated); `d.l_vect`,
  `d.stack(...)` -> `ValueError`.
- **Fix:** convert outside the `len(specified_bounds) > 1` block.
- **Resolved:** applied as proposed. Regression test:
  `test_single_list_bound_converted_to_array` in `tests/test_domain.py`.

### [x] 29. `Domain.ldotinv` with a 1-D `C` calls `ldot`

- **Where:** `spapprox/domain.py:376-380`
- **Repro:** `1.0 in Domain(dim=2, l=1).ldotinv([2., 3.])` -> True (with `l=5`);
  expected False (`y < 1/3`). The 2-D column form is correct.
- **Fix:** `return self.ldotinv(other[:, None])`.
- **Resolved:** applied as proposed. Regression test:
  `test_ldotinv_vector_matches_column` in `tests/test_domain.py` (compares with
  the definition `C y in D` pointwise).

### [x] 30. `Domain.ldot` ignores negative entries of `C`

- **Where:** `spapprox/domain.py:330-360`
- **What:** the result is not even a superset of the true domain; it can be
  disjoint from it.
- **Repro:** `[0, 0] in Domain(dim=2, l=1).ldot(-np.eye(2))` -> False; true
  domain is `y > -1`.
- **Fix:** route through `ldotinv(inv(C))` when invertible; otherwise combine
  `C+ @ upper` and `C- @ lower` per output component.
- **Resolved:** applied as proposed. The interval path uses the tightest of the
  strict/inclusive bounds per input component and marks an output bound strict
  if any contributing bound is. Regression test: `test_ldot_matches_image` in
  `tests/test_domain.py` (exact for invertible `C`; superset with attained
  bounds otherwise).

### [x] 31. `PartialDerivative` with `acc > 2` always crashes

- **Where:** `spapprox/diff.py:173`
- **What:** grid size is `2*order+1` regardless of `acc`.
- **Repro:** `PartialDerivative(np.exp, 1, acc=4)(0.3)` -> `ValueError`
  (broadcast); `Tressian(..., acc=4)` -> `IndexError`.
- **Fix:** size the grid from `acc` (e.g. `2*((order+1)//2 - 1 + acc//2) + 1`)
  plus one-sided headroom.
- **Resolved:** findiff applies the stencil to the whole grid (forward/backward
  stencils at the edges), which needs `center_half + forward_max` points; the
  half-width is now derived from `fd.coefficients(order, acc)` (at least
  `order`, so `acc=2` grids are unchanged). Regression tests:
  `test_partial_derivative_higher_acc`, `test_tensor_derivatives_higher_acc` in
  `tests/test_diff.py`.

### [x] 32. Default finite-difference step is poorly chosen

- **Where:** `spapprox/diff.py:192`
- **What:** `min(1e-9**(1/n), 1e-4)` gives `h=1e-9` for first derivatives and
  caps third derivatives at `1e-4`.
- **Evidence (exp at 0.3):** d1 error 2.7e-8 (vs 2e-11 at `h=1e-5`); d3 error
  1.6e-4 (vs 3e-7 at `h=1e-3`). Log-sum-exp `Tressian` error 1.9e-4 (vs 1.6e-7);
  under findiff 0.11 one 2-D entry is ~2% off.
- **Fix:** `max(...)`, or `eps**(1/(n+2))` scaling.
- **Findings (left as is, by decision):** `eps**(1/(n+acc))` is much better for
  well-scaled functions (max rel. error, old -> new: d3 exp/poisson/empirical
  `1e-4 -> 2e-7..4e-6`; d1 `~3e-8 -> ~1e-11`), but worse for CGFs evaluated
  close to a domain singularity, where `K` is small and steep: d3 of
  `laplace(scale=3)` at `t=0.2` `1.7e-6 -> 9.4e-5`, `-log(1-t)` at `1-t=0.01`
  `3e-4 -> 1.7e-2`. It also breaks the findiff reference (not the CGF) in
  `test_basic[univariate laplace from multivariate]`. A scale-aware step would
  be needed to win in both regimes.
- **Resolved:** scale-aware default step, `eps**(1/(n+acc)) * L` with `L` the minimum of one
  and the distance to the nearest NaN of `f` along each axis (probed at `t +- 2^-k`).
  Max rel. error, old -> new: d1 exp `2e-8 -> 6e-12`; d3 exp/poisson/empirical/laplace
  `1e-4..7e-4 -> 1e-7..1e-6`; d3 `-log(1-t)` at `1-t=0.01` `3e-4 -> 3e-5`, at `1e-3`
  `3e-2 -> 2e-6`; log-sum-exp `Tressian` `2e-4 -> 3e-7`. A function that is badly scaled
  away from any boundary gets worse (`exp(50t)` d3 `6e-6 -> 3e-4`); CGFs are differentiated
  standardized, and `univariate_empirical` now standardizes its sample (it was off by
  `3e4` in d3 for data with std 0.01, with either step). The probing costs one evaluation of
  `f` at `t +- 1` per axis, plus 53 points per side within distance one of a boundary
  (slow `test_basic` cases with `quad` reference CGFs: up to 4x slower). Regression tests: `test_default_step` in `tests/test_diff.py`,
  `test_univariate_empirical_scale` in `tests/test_cgfs.py`.

### [x] 33. `block_diag_3d` takes its dtype from the first tensor

- **Where:** `spapprox/diff.py:63`
- **Repro:** `block_diag_3d(np.ones((1, 1, 1), int), 0.5 * np.ones((1, 1, 1)))[1, 1, 1]`
  -> 0 (expected 0.5). Lists -> `AttributeError`; no arguments -> `IndexError`.
- **Fix:** `np.result_type(*tensors)` and `np.asarray` on each input.
- **Resolved:** applied as proposed, plus a `ValueError` for no arguments.
  Regression test: `test_block_diag_3d_dtype_and_inputs` in `tests/test_diff.py`.

### [x] 34. Singular-scale range check in multivariate `dK_inv` ignores `loc`

- **Where:** `spapprox/cgf_base.py:1720`
- **Repro:** `(multivariate_norm(dim=2).ldot([[1, 1], [1, 1]]) + [1, 1]).dK_inv([2, 2])`
  -> `[nan, nan]` (a valid `t` exists; works with `loc=0`).
- **Fix:** `np.allclose(scale @ scale_inv @ (x - loc), x - loc)`.
- **Resolved:** applied as proposed. Regression test:
  `test_dK_inv_singular_scale_with_loc` in `tests/test_mvar_cgfs.py`.

### [x] 35. Numerical derivatives use `self.dim` instead of `self.domain.dim`

- **Where:** `spapprox/cgf_base.py:1602`, `1789`, `1853`
- **What:** breaks projections where the underlying dimension differs.
- **Repro:** `M(K=-sum(log(1 - t)), dim=2, domain=...).ldot(np.array([[1., 1.]]))`:
  with findiff, `dK`/`d2K`/`d3K` raise "Dimensions do not match"; with
  numdifftools, `d3K` does.
- **Fix:** pass `self.domain.dim`.
- **Resolved:** applied as proposed for `Gradient`, `Hessian` and `Tressian`.
  Regression test: `test_numerical_derivatives_of_projection` in
  `tests/test_mvar_cgfs.py`.

### [x] 36. `from_cgfs` with only univariate inputs drops keyword arguments

- **Where:** `spapprox/cgf_base.py:1994`
- **Repro:** `from_cgfs(norm(), norm(), d2K0=7 * np.eye(2), numdiff_backend="findiff")`
  -> `d2K0` is `I`, backend `numdifftools`.
- **Fix:** forward the kwargs to `from_univariate`.
- **Resolved:** applied as proposed. Regression test:
  `test_from_cgfs_univariate_forwards_kwargs` in `tests/test_mvar_cgfs.py`.

### [x] 37. Multivariate `add` defaults to `inplace=True`

- **Where:** `spapprox/cgf_base.py:1278`
- **What:** univariate `add` defaults to `False`. `c.add(np.array([1, 1]))`
  mutates `c.loc` and returns `c`; `c.add(other_mcgf)` ->
  `AssertionError: inplace not supported`.
- **Fix:** default `inplace=False`.
- **Resolved:** applied as proposed. Regression test:
  `test_add_not_inplace_by_default` in `tests/test_mvar_cgfs.py`.

### [x] 38. Multivariate + univariate `add` late-binds `self`

- **Where:** `spapprox/cgf_base.py:1332-1341`
- **What:** the lambdas read `self.loc`/`self.scale` at call time, so mutating
  an operand changes an already-built sum.
- **Repro:** `s = a + norm(); s.K([.5, .5])` -> `0.75`; after `a.loc = [10, 10]`
  -> `10.75`.
- **Fix:** capture as lambda defaults, like the multivariate + multivariate
  branch.
- **Resolved:** `loc`/`scale` of both operands are bound at construction and
  passed explicitly (the univariate `other` late-bound too). Regression test:
  `test_add_univariate_binds_operands` in `tests/test_mvar_cgfs.py`.

### [x] 39. `clear_cache` evaluates every property

- **Where:** `spapprox/spa.py:100`
- **What:** `inspect.getmembers` triggers the expensive `_pdf_normalization`
  integration (hangs for `binomial`, item 24) and deletes a user-supplied
  `pdf_normalization`.
- **Fix:** iterate over `vars(self)`.
- **Resolved:** applied as proposed; a user-supplied `pdf_normalization` is
  kept (stored separately and restored). Regression tests:
  `test_clear_cache_does_not_evaluate_properties`,
  `test_clear_cache_keeps_user_pdf_normalization` in `tests/test_spa.py`.

### [x] 40. `fit_ppf` and `fit_saddle_point_eqn` share `_t_cache`

- **Where:** `spapprox/spa.py:486`, `499`, `317`
- **Repro:** `fit_ppf()` then `fit_saddle_point_eqn(num=200)`; `ppf(0.5)` ->
  `ValueError: fp and xp are not of the same length`.
- **Fix:** give the ppf grid its own cache names.
- **Resolved (together with item 7):** ppf grid now lives in `_ppf_t_cache`.
  Regression test: `test_fit_ppf_and_saddle_point_eqn_caches_are_independent`.

### [x] 41. Interpolated `dK_inv` silently clamps outside the fitted grid

- **Where:** `spapprox/spa.py:504`
- **Repro:** `gamma(a=3)`, grid ends at `x=30`; after fitting, `pdf(x=40)` ->
  5.34e-15 vs 3.49e-15 unfitted (50% off).
- **Fix:** `np.interp(..., left=np.nan, right=np.nan)` and fall back to the root
  solver.
- **Resolved:** applied as proposed. Regression test:
  `test_fitted_dK_inv_outside_grid_uses_solver` in `tests/test_spa.py`.

### [x] 42. `laplace().dK_inv(0)` is `0/0`

- **Where:** `spapprox/cgfs.py:231`
- **What:** `cdf(x=0)` and `pdf(x=0)` -> `nan` (cdf should be 0.5); same at
  `x=loc`.
- **Fix:** `x / (1 + sqrt(1 + x**2))` -- stable and 0 at 0.
- **Resolved:** applied as proposed. Regression test:
  `test_laplace_dK_inv_at_mean` in `tests/test_cgfs.py`.

---

## Low severity / robustness

### [x] 43. `ndarray * Domain` is hijacked by numpy

- **Where:** `spapprox/domain.py:304-308`
- **Repro:** `type(np.array([2, -1]) * Domain(dim=2, l=1))` -> `ndarray` (object
  array of Domains). `Domain * array` is fine.
- **Fix:** `__array_ufunc__ = None` on the class.
- **Fixed:** Regression test: `test_ndarray_times_domain` in `tests/test_domain.py`.

### [x] 44. Plain lists rejected for `Domain` vector add/mul

- **Where:** `spapprox/domain.py:197-208`, `237-245`
- **Repro:** `Domain(dim=2, l=1) + [1, 2]` -> `ValueError`.
- **Fix:** `is_list_like` + `np.asarray`.
- **Fixed:** List-likes are converted with `np.asanyarray`; `Domain - list` fixed too. Regression
  test: `test_domain_list_add_mul` in `tests/test_domain.py`.

### [x] 45. A list `h` breaks `PartialDerivative`

- **Where:** `spapprox/diff.py:225`
- **Repro:** `PartialDerivative(lse, 1, 1, 0, h=[1e-3] * 3)(t0)` -> `ValueError`.
- **Fix:** use `self._h_vect`.
- **Fixed:** Grid construction and domain shifts now use `_h_vect`. Regression test:
  `test_partial_derivative_vector_h` in `tests/test_diff.py`.

### [x] 46. All-zero derivative orders crash

- **Where:** `spapprox/diff.py:192`
- **Repro:** `PartialDerivative(f, 0)` -> `ZeroDivisionError`.
- **Fix:** validate `sum(orders) >= 1`.
- **Fixed:** `orders` must be non-negative with at least one positive order. Regression test:
  `test_partial_derivative_invalid_orders` in `tests/test_diff.py`.

### [x] 47. `type_wrapper` breaks on keyword arguments

- **Where:** `spapprox/util.py:35`
- **Repro:** `Domain(l=1).is_in_domain(t=0.5)` -> `IndexError`.
- **Fix:** fall back to `kwargs` by parameter name.
- **Fixed:** Falls back to the keyword argument with the decorated parameter's name. Regression
  test: `test_is_in_domain_keyword_argument` in `tests/test_domain.py`.

### [x] 48. Multivariate `cgf - list` fails

- **Where:** `spapprox/cgf_base.py:216`
- **What:** `-1 * other` on a list yields `[]`.
- **Repro:** `multivariate_norm(dim=2) - [1., 2.]` -> `AssertionError`.
- **Fix:** `self.add(-np.asanyarray(other))`.
- **Fixed:** Regression test: `test_multivariate_list_and_numpy_scalar_algebra` in
  `tests/test_mvar_cgfs.py`.

### [x] 49. NumPy scalars rejected in CGF algebra

- **Where:** `spapprox/cgf_base.py:676`, `757`, `1304`, `1427`
- **Repro:** `norm() + np.int64(2)`, `norm() * np.float32(2)` -> `ValueError`.
- **Fix:** `numbers.Real` / `pd.api.types.is_number`.
- **Fixed:** NumPy scalars and 0-d arrays are unwrapped to Python scalars (`_unwrap_scalar`).
  Regression tests: `test_numpy_scalar_algebra` in `tests/test_cgfs.py`,
  `test_multivariate_list_and_numpy_scalar_algebra` in `tests/test_mvar_cgfs.py`.

### [x] 50. Unformatted error message and list `loc` rejected

- **Where:** `spapprox/cgf_base.py:928`
- **Repro:** `M(K=..., dim=2, loc=[1., 2.])` -> message
  `"...vector of length {self.dim}"`.
- **Fix:** add the `f` prefix; `np.asanyarray(loc)` before validation.
- **Fixed:** The multivariate `loc` and `scale` setters convert to arrays before validation.
  Regression test: `test_multivariate_list_loc` in `tests/test_mvar_cgfs.py`.

### [x] 51. `univariate_empirical` overflows

- **Where:** `spapprox/cgfs.py:273`, `281`
- **Repro:** data `[1000, 1001, 1003]`: `K(1)` -> `inf` (true 1002.07),
  `dK(1)` -> `nan`.
- **Fix:** logsumexp / max-shifted softmax.
- **Fixed:** Regression test: `test_univariate_empirical_no_overflow` in `tests/test_cgfs.py`.

### [x] 52. `ppf(0)` / `ppf(1)` return arbitrary finite values

- **Where:** `spapprox/spa.py:315`, `361-380`
- **Repro:** `norm()`: `ppf(0.)` -> -64, `ppf(1.)` -> 16.
- **Fix:** return the support bounds (e.g. `-inf`/`inf`).
- **Fixed:** `ppf(0)` and `ppf(1)` return `support()`, the limits of `K'(t)` at the bounds of
  the domain of `K`, found by stepping `t` geometrically towards each bound (Aitken
  extrapolated), e.g. `(0, inf)` for `gamma`, `(min(x), max(x))` for `univariate_empirical`.

### [x] 53. pdf normalization over a truncated `t` range biases exact cases

- **Where:** `spapprox/spa.py:248`
- **What:** normalized `norm` pdf has relative error +1e-4.
- **Fix:** tighter tolerance or integrate over the full domain.
- **Fixed:** The tail beyond the inferred `t` range is now integrated up to the domain bound on
  each side where the cdf reached 0/1 (lattice tails stay excluded). The remaining
  exponential bias came from `_spapprox_pdf` treating `d2K < 1e-8` as zero (via
  `np.isclose`), which also made the pdf `nan` everywhere for small scales, e.g.
  `norm(scale=1e-5)`; it now requires `d2K > 0`. Regression test:
  `test_normalized_pdf_exact` in `tests/test_spa.py`.

### [x] 54. _Suspected:_ `K` returning shape `(1,)` breaks multivariate `loc`

- **Where:** `spapprox/cgf_base.py:1554`
- **Repro:** `M(K=lambda t: np.atleast_1d(np.sum(t**2) / 2), dim=2, loc=[1, 2]).K([1, 1])`
  -> `ValueError: shapes (2,1) and (2,) not aligned`.
- **Fix:** `val += np.dot(t, loc)`. Arguably user error; decide on the contract.
- **Fixed:** Contract: a shape `(1,)` return for a single point is accepted and treated as
  a scalar, i.e., `K` squeezes it and the `loc` term is now `np.sum(loc * t)`. Derivatives,
  `dK_inv`, `ldot`, marginals and the pdf were checked to match a scalar-returning `K` on
  both backends. Regression test: `test_multivariate_K_returning_shape_one` in
  `tests/test_mvar_cgfs.py`.

### [x] 55. README and config drift

- `README.md:49`: example uses `spa.SaddlePointApprox`, which doesn't exist
  (should be `UnivariateSaddlePointApprox`).
- `README.md:65`: optional dependencies omit `findiff`.
- `README.md:67-77`: mismatched code fences; refers to `setup.py`.
- `pyproject.toml:14`: ruff comment claims "bugbear, isort", but only `E/F/W`
  are selected.
- **Fixed:** All four points fixed.

### [x] 57. Multivariate CGFs handle wrongly shaped points inconsistently

- **Where:** `spapprox/cgf_base.py`, `MultivariateCumulantGeneratingFunction.K`, `dK`,
  `d2K`, `d3K`, `dK_inv`
- **What:** a point is a vector of length `dim`, also for `dim=1`, but this was not
  enforced. For `dim=1`, a `(3,)` input made `K` return `[14, 14, 14]` (treated as one
  3-vector), `dK` treat it as 3 points, and `d2K` crash. A scalar `x` in `dK_inv` crashed
  (`dim=1`) or raised an `AssertionError` (`dim=2`).
- **Fixed:** `_validate_points` broadcasts scalars to a point and raises a `ValueError`
  for any other shape whose last axis does not match the dimension. Regression test:
  `test_multivariate_point_shapes` in `tests/test_mvar_cgfs.py`.

### [x] 58. Univariate `dK_inv` fails at the end of its initial bracket

- **Where:** `spapprox/cgf_base.py`, `UnivariateCumulantGeneratingFunction.dK_inv`
- **What:** when `x` equals `dK` at an end of the initial bracket, the bracket expansion
  stopped on equality, after which the strict bracket assertion failed. Hit by
  `fit_saddle_point_eqn`/`fit_cdf`, whose grid ends are `dK` at the inferred `t` range.
- **Fixed:** the expansion continues only while `x` lies strictly outside the bracket.
  Regression test: `test_dK_inv_at_initial_bracket_end` in `tests/test_cgfs.py`.

### [x] 61. Vector estimating equation solver stalls near the minimum

- **Where:** `spapprox/spa.py`, `MultivariateEstimatingEquationSaddlePointApprox._solve`
- **What:** close to the minimum, the Newton decrement (e.g. 1e-16) is below the
  resolution of the objective, so the Armijo test fails by roundoff and backtracking shrinks
  the step until it no longer moves the iterate. The solver then stalled for all iterations
  and `pdf` returned NaN, for about 0.5% of points in the bootstrap of a vector mean.
  Whether a given point stalls depends on roundoff, which made
  `test_vector_estimating_equation_mean_cut` fail intermittently in CI.
- **Fixed:** below a decrement of 1e-10, full Newton steps are taken without line search,
  until the decrement stops decreasing. Regression test:
  `test_vector_estimating_equation_newton_roundoff` in `tests/test_spa.py`.

### [x] 59. Bivariate `"KL"` cdf fails close to the domain boundary

- **Where:** `spapprox/spa.py`, `BivariateSaddlePointApprox._kl_cdf`
- **What:** for `from_univariate(gamma(a=3), gamma(a=2)).ldot([[1, 0], [0.5, 1]])`,
  the `"KL"` cdf is NaN or far outside `[0, 1]` (e.g. -4.24, -12.45) at valid points in the
  support whose saddlepoint has `t1 >~ 0.97`, near the boundary `t1 < 1`, e.g.
  `x = (4.325, 3.025)`. `"Kolassa"` is fine there. `fit_cdf` treats such values as failures
  and fills them from the neighbours.
- **Fix:** the prefactor exp(b'A^{-1}b/2) overflowed while the normal tail probability T
  underflowed to 0, so the cubic correction, a near-total cancellation, was garbage.
  `_bvn_tail_derivatives` now takes a `logscale` and computes T and its derivatives in log
  space (T via `_mvn_logtail` for large scales), and `_kl_tail` passes the log prefactor.
  Close to the boundary, KL is now within about 0.02 of `"Kolassa"` and Monte Carlo.
  Regression test: `test_bvar_kl_cdf_near_domain_boundary`.

### [x] 60. Bivariate `"Wang"` cdf raises for a dependent gamma

- **Where:** `spapprox/spa.py`, `BivariateSaddlePointApprox._wang_cdf`
- **What:** for the cgf of item 59, `cdf(x=[[2, 3], [4, 5]], backend="Wang")` raises
  `AssertionError: Something is wrong`. `"KL"` and `"Kolassa"` give 0.21 and 0.68.
- **Correction:** on a fresh object it works; it raised after a coarse
  `fit_saddle_point_eqn` (e.g. `num=8`), because `_dK_inv` returned linearly
  interpolated saddlepoints that were far off (t = -4.6 instead of 0.08), so that x was not
  the gradient at t. `"KL"` and `"Kolassa"` silently gave wrong values there.
- **Fix:** `MultivariateSaddlePointApprox._dK_inv` polishes the interpolated t by up to five
  Newton steps, and solves directly, starting from it, where the residual stays large.
  Regression test: `test_bvar_fit_saddle_point_eqn_polishes_interpolation`.

### [x] 62. Multivariate `cdf(..., delta=...)` is forwarded to the root finder

- **Where:** `MultivariateSaddlePointApprox.cdf` / `BivariateSaddlePointApprox.cdf`, which take
  `**solver_kwargs` but have no `delta`, unlike the univariate and conditional cdfs.
- **What:** `delta=1` ends up in `dK_inv`. For `from_univariate(poisson(3), poisson(4))` at
  `x=(2, 3)`, `"KL"` and `"Wang"` raise "Failed to solve the saddle point equation", and
  `"Kolassa"` returns `nan`: the message hides the real cause.
- **Fix:** `cdf` and `fit_cdf` take an explicit `delta=None` and raise a clear
  `NotImplementedError` otherwise, until the lattice cdf below is implemented. Regression test:
  `test_mvar_cdf_rejects_delta`. The bivariate `"KL"` cdf now supports `delta` (see the lattice
  items); `"Wang"`, `"Kolassa"`, `fit_cdf` and `d > 2` still raise.

---

## Tooling / packaging

### [x] 16. Python version floor conflicts with the syntax in use

- **Where:** `spapprox/diff.py:345`, `496` use `retval.T[*sel]` /
  `self._partials[*ijk]` -- star-in-index is Python 3.11+, while
  `pyproject.toml` declares `requires-python = ">=3.9"`.
- **Symptom:** `ruff check .` fails with 7 errors: 4 invalid-syntax (2 here, 2
  in `examples/`), 2 F401, 1 E741.
- **Decide:** either raise the floor to `>=3.11`, or rewrite the indexing
  expressions as `tuple(...)` subscripts.
- **See also:** item 25 (CI).
- **Fixed:** Raised the floor to `requires-python = ">=3.11"`, and fixed the remaining F401 and
  E741 in `examples/`; `ruff check .` passes.

---

## Needs a literature check (unresolved)

### [x] 56. Bivariate cdf converges too slowly under dependence

- **Where:** `spapprox/spa.py:836-912`
- **What:** even with items 17 and 18 fixed, the error for dependent pairs only
  shrinks like `n^(-1/2)` as shape parameters grow by a factor `k`, whereas a
  second-order method should give `1/n` (as the independent case does).
- **Evidence (max abs error):**

  | Case                                                    | k=1   | k=4   | k=16   | k=64   |
  | ------------------------------------------------------- | ----- | ----- | ------ | ------ |
  | `(Z1, Z1+Z2)`, exact by quadrature                      | 0.043 | 0.042 | 0.029  | 0.0175 |
  | `bivariate_gamma([2, 1, 3]) * k` vs 4M-draw Monte Carlo | 0.023 | 0.016 | 0.0095 | --     |

- **Diagnosis:** the code matches Paolella (2007) eq. (5.22) and Wang (1990) eq. (25)
  term by term (Paolella's undefined `w0` is Wang's `w_{u0}`, i.e. `w_tilde`). What was
  missing is the order-of-integration rule of Wang (1990) section 5 / Paolella Remark
  (a): integrate first over the variable whose `w_tilde_0` is smaller. The code always
  integrated over the second variable first. With the rule, Wang's Table 1 (n=10,
  bivariate gamma) is reproduced to the printed digits; without it, e.g. 0.053 instead of
  0.125.
- **Partially fixed:** `_bivariate_formula` picks the order per point (consistently
  across a near-axis interpolation stencil). Max abs error for `bivariate_gamma([2, 1, 3] * k)`
  vs exact quadrature:

  | k                 | 1     | 4      | 16     | 64     | 256     |
  | ----------------- | ----- | ------ | ------ | ------ | ------- |
  | before            | 0.245 | 0.076  | 0.033  | 0.015  | 0.0073  |
  | with Wang's order | 0.035 | 0.0050 | 0.0023 | 0.0013 | 0.00073 |

  Regression tests: `test_bvar_cdf_wang_1990_table_1`, `test_bvar_cdf_dependent_accuracy`,
  `test_bvar_cdf_invariant_to_relabeling`.
- **Limitation of Wang's method:** the error still shrinks like `n^(-1/2)` at fixed
  standardized points.
  This is intrinsic to Wang's formula (Wang's own table has errors of ~0.009 at n=10;
  Kolassa & Li (2010) note that Wang's argument is incomplete near the pole). Two more
  findings:
  - Wang's formula can exceed 1 in the upper-right corner, e.g. 1.0012 for
    `bivariate_gamma([10, 10, 10])` at `x=(40, 40)`.
  - Replacing the factor `1/w_tilde_0 - 1/u_tilde_0` in Wang's `I21` with the marginal
    Lugannani--Rice correction `1/x_tilde - 1/(t_tilde_0 sqrt(K_tt))` is 3--20x more
    accurate at k=1, but does not change the rate. It is not in the literature, so it was
    not adopted.
- **Resolved:** the default is now `cdf(..., backend="KL")`, the method of Kolassa & Li
  (2010) / Li (2009, PhD thesis, ch. 3), which has relative error `O(1/n)`. Wang's method
  is kept as `backend="Wang"`. Max abs error for `bivariate_gamma([2, 1, 3] * k)` vs
  exact quadrature:

  | k    | 1      | 4      | 16     | 64     | 256    |
  | ---- | ------ | ------ | ------ | ------ | ------ |
  | Wang | 0.035  | 0.0050 | 0.0023 | 0.0013 | 7.3e-4 |
  | KL   | 4.1e-3 | 8.9e-4 | 2.5e-4 | 6.5e-5 | 1.7e-5 |

  Three things were needed that the paper omits:
  - The cubic term of `I^∅`, which the thesis only references to Kolassa (2003). Only
    `w_tilde_2(w_1)` is non-quadratic, so `g_111 = 3 w2' w2''`, `g_112 = w2''`, and the
    other cubic derivatives vanish. The Gaussian integrals follow by applying
    `(-d/db_i - u_hat_i)` to the bivariate normal tail. Without this term the error is
    `O(n^(-1/2))` and worse than Wang's. (Also: the thesis prints the correlation as
    `c/(1+c^2)`; it is `c/sqrt(1+c^2)`.)
  - Applying the upper-tail formula in the orthant where both saddlepoint components are
    positive (reflect `K(±s, ±t)`), then inclusion--exclusion with univariate
    Lugannani--Rice marginals. Without this, `C = exp(...)` blows up when the orthant
    probability is close to 1.
  - The reflection makes the result jump across the axis `t_2 = 0` (up to 3.4e-3 at
    k=1), because the `X_1` marginal tail implied by KL differs from Lugannani--Rice at
    `O(1/n)`. It is continuous across `t_1 = 0`. The two variable orders are therefore
    blended with weight `z_2^2/(z_1^2+z_2^2)` on the given order (`z` the standardized
    saddlepoint). This is continuous (jumps < 1e-7), symmetric in the variables, and as
    accurate as either order. Picking the order with the smaller `|z|` still left jumps of
    6.6e-4.

  Regression tests: `test_bvar_cdf_dependent_accuracy` (both backends),
  `test_bvar_cdf_kl_continuous_near_singularities`, `test_bvar_cdf_invariant_to_relabeling`
  (both backends), KL checks in the independence and near-axis tests.
- **Remaining findings (not fixed):**
  - KL does not factorize exactly under independence: max error 4.6e-3 for
    `gamma(3) x gamma(5)`, versus 1.3e-4 for the product of the Lugannani--Rice marginals
    (and for Wang, which factorizes exactly). Adding the `O(1/n)` term
    `I^{12} = phi(w1) phi(w2) [G(t) - G(t1, 0) - G(0, t2) + 1] / (w1 (w2 - w2_check))`
    restores factorization (1.3e-4), but trades accuracy under dependence: max error with
    vs without it 2.1e-3 vs 5.1e-3 at correlation 0.09, but 2.0e-2 vs 6.6e-3 at 0.75. Not
    adopted.
  - The "Wang can exceed 1" and `I21` variant findings above still apply to
    `backend="Wang"`.

---

## Working notes

- Full suite takes ~16 min (~4 min with `-m "not slow"`); iterate with targeted `pytest tests/test_X.py`.
- Numerical tests compare against `scipy.stats` with tolerances. If a fix shifts
  results, establish whether it is a regression or a genuinely better
  approximation _before_ touching any tolerance.
- Check for `tofix`-marked tests that may already document some of these
  behaviours -- those markers should be removed as the corresponding items land.
- Several items share a root cause; fix them together:
  - domain/scale units: 10, 14, 21, 22 (and 26 for `intersect`);
  - in-place mutation: 2, 11.
- The univariate lattice cdf is continuity-corrected with `cdf(..., delta=span)`
  (`poisson(3)` at `x=3`: 0.646 vs exact 0.647; without `delta` 0.538).
- Suggested first batch (unambiguous one-liners, high impact): items 9, 3, 8, 6,
  7, 19, 1, 17, 18. Then the hangs (24) and the near-zero cdf guard (23).

# Plans

The plan is to work on applications, and add more functionality based on what can be found in the literature.

All the funtionality in this package should make the results achieved in most papers trivial.

## Additional distributions

These would go into cgfs.py

- add asymmetric laplace?
- add generalized normal distribution
- add asymmetric generalized normal distribution
- add Cheriyan and Ramabhadran's multivariate gamma distribution (Kotz 2000)
- add Jensen's multivariate Gamma (See Butler 2007, page 77)
- multinomial distribution (see Butler 2007, page 80)
- add bivariate poisson distribution (see Butler 2007, page 81)
- add multivariate exponential as special case of gamma
- also see Butler 2007, page 90 for higher order expansions
- add multinomial bootstrap of the mean
- see the exercise in Butler for negative binomial distribution, and multivariate logistic distribution
- Saddlepoint approximation of de EDF (Kuonen 2001, chapter 5)

## Applications

First application would be bootstrapping, and the various froms it can take.

Additional approximations would go into spa.py
Or into a separate model, e.g., bootstrap.py

- conditional distribution, `X_i | X_J = x_J` (done on branch `feature/conditional-skovgaard`, see
  `docs/conditional_cdf_plan.md`): `MultivariateSaddlePointApprox.conditional_cdf` (Skovgaard,
  `backend="Kolassa"` for second order, `delta=` for a lattice `X_i`) and `conditional_pdf`.
  Marginal approximations are available as `spa[i]`, `spa[[i, j]]`. Follow ups, in order:
  - [x] normalize `conditional_pdf`: `normalize_pdf=True` by default, quadrature in `tau_i`
    along the profile saddle path. The second-order pdf correction is still open
  - [x] the `condition()` stub was removed, use `conditional_cdf` / `conditional_pdf`
  - [x] `P1`, the first continuity correction of Butler (4.15)-(4.16), for comparison with `P2`:
    `conditional_cdf(..., delta=..., correction=1)`, Skovgaard backend and one variable of
    interest only. On the binomial test it is ~4x less accurate than `P2`
  - vector of interest, `P(X_I >= x_I | X_J = x_J)`: phase F, Kolassa & Li (2010) section 4,
    see `docs/conditional_vector_plan.md`. F1, two variables of interest, is done:
    `conditional_cdf(x, given, i=[a, b], delta=...)`, continuous and lattice. Open: F2,
    three or more variables of interest
  - the bivariate `"KL"` cdf, also conditional, drops the `I^{1,2}` term of Kolassa & Li
    (2010). Its relative error therefore does not vanish at a fixed large deviation point:
    for two independent gamma(n) means at (1.3, 1.25) it is -2.4%, -1.2%, -0.85% for
    n = 10, 40, 160, where the product of the Lugannani-Rice marginals converges at 1/n.
    For independent components the missing term is `phi(w1) phi(w2) f1 f2`, with
    `f_j = 1/u_j - 1/w_j`. Consider adding it
  - non-linear conditioning `Z = g(X, Y)`: phase G, Wang (1993). Lower priority
- resampling distributions, Booth and Butler (1990): done with
  `EstimatingEquationSaddlePointApprox` (scalar estimating equations, multinomial and Dirichlet
  bootstrap, delete-m jackknife) plus examples for the permutation tests using the existing
  algebra, see `docs/estimating_equations_plan.md`. Follow ups:
  - [x] vector-valued estimating equations (several parameters):
    `MultivariateEstimatingEquationSaddlePointApprox`, see
    `docs/vector_estimating_equations_plan.md` and `examples/example_bootstrap_regression.py`.
    Follow up: marginal tail approximations (Jensen & Wood) instead of the tabulated grid
  - [x] continuity correction for the permutation and jackknife distributions:
    `EstimatingEquationSaddlePointApprox(..., delta=span)`, for a lattice `S(tau)`, e.g. a
    mean of integer data, see `examples/example_bootstrap_lattice.py`; Wilcoxon rank sum test
    in `examples/example_permutation_two_sample.py`. The paper calls the correction for its
    Table 1 negligible, but for the pain relief data it moves the one-sided level from 0.0486
    to 0.0502 (exact 0.0507)
  - the analgesia data of Booth and Butler (1990), Table 1 (b), do not reproduce the published
    exact significance level (0.012; we get 0.044 one-sided); find Robinson (1982), Lehmann
    (1975) p. 92 to check the data
- implement multivariate saddle point approximations
  - cdf for any dimension: phase A of `docs/multivariate_cdf_plan.md` is done,
    `MultivariateSaddlePointApprox.cdf(backend="Kolassa")` (Kolassa 2003). Error
    `O(1/n)`, about 1.5x that of `"KL"` in 2-D; exact for the normal; continuous;
    symmetric. Phase B (Kolassa & Li 2010 for `d > 2`) is open, and only worth it if
    phase A turns out not to be accurate enough
  - [x] `fit_cdf` for the multivariate case: tabulates the cdf on a tensor grid in `x` and
    interpolates (PCHIP by default), see `docs/multivariate_fit_cdf_plan.md`
  - lattice variables: continuity-corrected version of Kolassa (2003) eq. (13)
  - [x] bivariate `"KL"` and `"Wang"` cdf returned NaN, plus `invalid value in sqrt` warnings,
    in `examples/example_Wang1990.py`, e.g. cells (x=0, y=1.0) and (x=-1.0, y=0.0). Cause: `x`
    at a marginal mean up to roundoff, so the *marginal* saddlepoint is ~1e-15 and
    `2 (t x - K(t))` came out slightly negative. Fixed by clipping these arguments at zero
    (`_signed_root`); regression test `test_bvar_cdf_at_marginal_mean_from_t`
- implement other bootstraps
- conditional distributions
- approximation for discrete distributions (lattice variables). Open items:
  - [x] univariate cdf continuity correction (Daniels, 1987; Butler ch. 1.2.3):
    `cdf(..., delta=span)`, LR and BN. `infer_t_range` still excludes lattice tails
  - [x] univariate lattice pmf: `pdf(..., delta=span)` returns `delta * f(x)`, not
    normalized by default (integrating the density is a poor normalization for a lattice,
    e.g. 41% relative error for `poisson(0.5)` vs 8% unnormalized). Possible refinement:
    normalize by summing over the lattice, and Daniels' second-order terms
  - [x] univariate first continuity correction `Pr1` (Daniels 1987; Butler (1.27)-(1.29)):
    `u1 = (1 - exp(-t)) sqrt(K''(t))` at the uncorrected saddlepoint of `k`, and
    `F(k - 1) = 1 - Pr1(X >= k)`. Butler 1.2.6 recommends computing both `Pr1` and `Pr2`,
    as neither dominates (Pr1 often slightly better, e.g. Poisson/binomial left tails,
    Tables 1.12-1.13). Done: `cdf(..., delta=..., correction=1)` reproduces Butler Tables 1.7
    and 1.12; the default `correction=2` is `Pr2`.
    `Pr3` and the alternative left tail `F1a` are not competitors, skip them
  - [x] unconditional bivariate cdf for lattice variables:
    `BivariateSaddlePointApprox.cdf(x, delta=...)`, `"KL"` only, Kolassa & Li (2010) Section 3
    at `x + delta/2` with `rho(tau) = 2 sinh(tau delta / 2) / delta` (`_LatticeBivariateApprox`;
    plan: `docs/bivariate_lattice_cdf_plan.md`). Two independent Poissons (3, 4) at (2, 3):
    0.181 vs exact 0.183 (uncorrected 0.102); reproduces the paper's Table 2 to within 0.7%.
    Open: `fit_cdf(delta=)`, `"Wang"` (Wang (1990) Theorems 2-3, which replace `t0` by
    `1 - exp(-t0)` in the factors, without an offset), `"Kolassa"` and `d > 2`
    (Kolassa (2003) eq. (13), see above), and mixed lattice/continuous components
  - conditional cdf for a vector of interest `P(X_I >= x_I | X_J = x_J)`, continuous and
    lattice: phase F above (Kolassa & Li, 2010, section 4). Done for two variables of
    interest, `conditional_cdf(i=[a, b], delta=...)`; open for three or more
  - `bernoulli(p)` factory in `cgfs.py` (thin wrapper around `binomial(n=1, p)`, plus export
    and test). Then simplify example 2 of `examples/example_Kolassa2003.py`: its hand-written
    logistic CGF equals `MultivariateCumulantGeneratingFunction.from_univariate(*[binomial(
    n=m_j, p=0.5)]).ldot(Z.T)` (checked to machine precision). Show both tail approximations
    side by side: the current first continuity correction and the package's
    `1 - cdf(x=c - 1, delta=1)` (second correction). Boole sums: 0.01756 (first), 0.01727
    (second), paper 0.01709, exact 0.01713; the paper's figure is still unexplained
- bootstrap for robust estimators (Kuonen, 1998): `EstimatingEquationSaddlePointApprox` with
  Poisson weights covers monotone M-estimators. Done: Huber location estimate in
  `examples/example_bootstrap_robust_estimator.py`
- quadratic forms (Kuonen 2001)
- mixture models

## Review: Alhejaili and AlGhamedi (2025)

"A review of saddle-point approximation: theory and applications", Adv. Appl. Stat. 92(3),
343-390. Sections 1-2 and 4 are a literature survey; section 3 states the formulas. Coverage of
section 3 by the package, checked against the primary sources (Butler 2007, Wang 1990):

| Review | Formula | Status |
|---|---|---|
| (4)-(5) | univariate pdf/pmf | `pdf`, `pdf(delta=)` |
| (6)-(7) | Lugannani-Rice, limit at the mean | `cdf(backend="LR")`, interpolated near `t=0` |
| (8)-(9) | first continuity correction `Pr1` | `cdf(..., delta=, correction=1)` |
| (10)-(12) | second continuity correction `Pr2` | `cdf(delta=)`, matches Butler Table 1.7 |
| (13) | third approximation `Pr3` | missing, by choice (Butler: not a competitor) |
| (14)-(21) | Wang (1990) bivariate cdf | `cdf(backend="Wang")`, plus `"KL"` and `"Kolassa"` |
| (22)-(23) | Wang (1990) discrete versions | missing; the KL lattice version is `cdf(delta=)` |
| (24)-(25) | multivariate pdf | `MultivariateSaddlePointApprox.pdf` |
| (27)-(33) | double saddlepoint pdf | `conditional_pdf` (normalized) |
| (39)-(41) | Skovgaard conditional cdf | `conditional_cdf`, plus Kolassa (1996) second order |
| (42)-(44) | conditional `Pr1` | `conditional_cdf(..., delta=, correction=1)` |
| (45)-(48) | conditional `Pr2` | `conditional_cdf(delta=)` |

Typos in the review, do not implement from it: in (6) the roles of `phi` and `Phi` are swapped
in the text; Theorem 3.2 has `1 - e^{t0}`, Wang (1990) has `1 - e^{-t0}`; (44) should read
`K_t'(t0, 0) = j`; (48) should use the offset saddlepoint `(t~, u~)` in the Hessian; in (32)
`u0^T x` should be `t0^T x`.

Mentioned in the survey but not implemented, candidate applications: non-normal base
distributions (Wood, Booth and Butler 1993), saddlepoint approximations from a truncated CGF
for general statistics (Easton and Ronchetti 1986; Gatto and Ronchetti 1996), empirical and
extended empirical saddlepoint (Feuerverger 1989; Fasiolo et al. 2018), and saddlepoint
smoothers for discrete data (Davison and Wang 2002). The remark on non-convex support (exclude
points outside the support when normalizing a multivariate pdf) does not apply yet: no factory
has a non-convex support.

## Literature to review

- Find complete version of Alya Omar's master thesis (2015)
