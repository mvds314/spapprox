# Priorities

- Go through bug backlog
- Maybe review implementation with this in mind: Alhejaili, AlGhamedi (2025) - A review of saddlepoint approximation
- Finish bivariate approximation:
  - basically this case: test_bvar_cdf_near_zero
  - are there more edge cases?
- Conditional approximation (Skovgard approximation)
- Multivariate approximation

- Work on bootstrap distributions (see Booth and Butler)

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

### [ ] 18. `sps.norm.pdf(w)` vs documented `phi(x_tilde)` -- RESOLVED: code is wrong

- **Where:** `spapprox/spa.py:862`, `892`
- **What:** matters when one saddlepoint component is 0. Same gamma pair at
  `t=(0.5, 0)` and `(0, 0.4)`: with only item 17 fixed, errors are **-0.034**
  and **-0.038**; with both fixed, -8e-5 and -5e-5.
- **Fix:** `sps.norm.pdf(tx.T[0])` at `862`, `sps.norm.pdf(tx.T[1])` at `892`.

### [ ] 19. Cumulants at zero crash on CGFs without analytic derivatives

- **Where:** `spapprox/cgf_base.py:635`, `644`, `653` (univariate), `1160`,
  `1169`, `1178` (multivariate)
- **What:** `_dK0_raw` / `_d2K0_raw` / `_d3K0_raw` call `self._dK(0)` etc.
  before the lazy numerical-derivative object has been built.
- **Repro:** `UnivariateCumulantGeneratingFunction(K=lambda t: t**2 / 2).dK0`
  -> `TypeError: 'NoneType' object is not callable`. Same for
  `UnivariateSaddlePointApprox(...).cdf(t=0.)`, for `univariate_empirical(...).d2K0`
  read before any `d2K(t)` call, and for the multivariate analogues.
- **Fix:** call `self.dK(0, loc=0, scale=1)` etc., which initialises the backend.

### [ ] 20. Multivariate findiff gradient applies `loc`/`scale` twice

- **Where:** `spapprox/cgf_base.py:1602`
- **What:** the findiff `Gradient` is built on the transformed `self.K`, and the
  result is then transformed again.
- **Repro:**
  `MultivariateCumulantGeneratingFunction(K=lambda t: np.sum(t**2 / 2 + t, axis=-1), dim=2, loc=[1, 2], scale=2., numdiff_backend="findiff").dK(np.zeros(2))`
  -> `[7, 10]`; expected `[3, 4]` (numdifftools gives `[3, 4]`).
- **Fix:** `Gradient(lambda tt: self.K(tt, loc=0, scale=1), self.domain.dim)`.

### [ ] 21. Multivariate + univariate `add` ignores both operands' scale in the domain

- **Where:** `spapprox/cgf_base.py:1362`
- **What:** same family as item 10.
- **Repro:** `s = multivariate_norm(dim=2) + exponential(scale=2)`;
  `[.4, .4] in s.domain` -> True, `s.K([.4, .4])` -> `nan`.
- **Fix:** `self.domain.ldotinv(self.scale_mat.T).intersect(other.domain.mul(1/other.scale).ldotinv(np.ones((1, dim))))`.

### [ ] 22. `from_univariate` / `from_cgfs` combine domains of the standardized variables

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

### [ ] 23. LR/BN `t=0` guard is far too tight -> catastrophic cancellation near the mean

- **Where:** `spapprox/spa.py:182`, `215`; bivariate `827`, `854`, `885`
- **What:** `np.isclose(t, 0)` (atol 1e-8) only catches `t` essentially at 0;
  just outside it the `1/w - 1/u` terms cancel catastrophically.
- **Repro:** `UnivariateSaddlePointApprox(gamma(a=3)).cdf(t=1e-7)` -> **-12028**
  (expected 0.5768); at `t=1e-5` -> 0.5663 (error 0.01). After
  `fit_saddle_point_eqn()`, `cdf(x=3.0)` (the mean) -> `nan`. Bivariate
  `t=(0.3, 1e-6)` -> about 5.0.
- **Fix:** widen the guard (e.g. `|w| < 1e-4`) or use a series expansion near 0.

### [ ] 24. `infer_t_range` loops forever on lattice / bounded-support CGFs

- **Where:** `spapprox/spa.py:444-466`
- **What:** the expansion loop has no iteration cap and waits for the LR cdf to
  reach 0/1, which it never does for lattice CGFs. Default `normalize_pdf=True`
  calls it, so `pdf()` hangs.
- **Repro (hangs):** `UnivariateSaddlePointApprox(binomial(10, .3)).pdf(t=.1)`;
  same for `poisson(3)` (worse with item 1: `lb=-0.0` never moves).
- **Fix:** cap iterations; treat `nan`, cdf < 0 or cdf > 1 as end of range;
  raise if no range is found.

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

### [ ] 5. `__getitem__` slices cumulants on one axis only

- **Where:** `spapprox/cgf_base.py:1261`, `1266`
- **What:** `d2K0[item]` where `d2K0[np.ix_(item, item)]` is meant; `d3K0` needs
  the rank-3 equivalent.
- **Repro:** slicing a cached 3-D CGF with `[0, 2]` yields a 2x3 "covariance"
  `[[4, 1, 0.5], [0.5, 0.2, 1]]` instead of `[[4, 0.5], [0.5, 1]]`.
- **Fix:** use `np.ix_` on all axes.

### [ ] 11. In-place scaling mutates caller / user arrays

- **Where:** `spapprox/diff.py:48` (`transform_rank3_tensor`, `T *= ...`);
  `spapprox/cgf_base.py:1804`, `1865` (multivariate `d2K` / `d3K`)
- **What:** writes through to the caller's array and returns the same object.
  When a user-supplied `d2K`/`d3K` returns a fixed array, it compounds on every
  call. With an int `T` and float diagonal it raises `UFuncTypeError`.
- **Repro:** `C = np.eye(2)`; `MultivariateCumulantGeneratingFunction(..., d2K=lambda t: C, scale=2.)`;
  three calls to `d2K(0)` -> `4, 16, 64` and `C` is modified (expected `4` each
  time). `d3K` with a vector scale: `8, 64, 512`.
- **Fix:** `y = y * scale**2`; `T = T * Ai.reshape(shape)` (copy, not in place).

### [ ] 12. `transform_rank3_tensor` returns index-reversed output

- **Where:** `spapprox/diff.py:50`
- **What:** `tensordot` moves each contracted axis to the front, reversing the
  axis order. With three distinct matrices the result equals
  `einsum('il,jm,kn,lmn->ijk', A, B, C, T).transpose(2, 1, 0)`. Also wrong for a
  single matrix with non-symmetric `T`, and for mixed vector/matrix inputs.
- **Note:** masked in current use because callers pass one matrix and a
  symmetric `T`.
- **Fix:** `np.einsum('il,jm,kn,lmn->ijk', A, B, C, T)`.

### [ ] 13. `bivariate_gamma` domain omits `t1 + t2 < 1`

- **Where:** `spapprox/cgfs.py:217`
- **Repro:** `t = [0.9, 0.9]` reports `in domain` -> `True` while `K(t)` -> `nan`.
- **Fix:** add `B=[[1, 1]], b=[1]`.

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

### [ ] 15. `from_cgfs` crashes on mixed univariate/multivariate input

- **Where:** `spapprox/cgf_base.py:2048`, `2064`
- **What:** fails when cumulants are already cached and the inputs mix
  dimensionalities.
- **Repro:** `norm()` and `multivariate_norm(dim=2)` with cached cumulants;
  `from_cgfs(a, b)` -> `ValueError: zero-dimensional arrays cannot be concatenated`.
- **Fix:** `np.atleast_1d(cgf.dK0)` and `np.reshape(cgf.d3K0, (d, d, d))`.

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

### [ ] 27. Bound-ordering validation is wrong and inconsistent

- **Where:** `spapprox/domain.py:12`, `46`, `61`
- **What:** the code enforces `l > le > ge > g` while docstring and messages say
  `g > ge > le > l`. Valid combinations are rejected: `Domain(l=1, le=2)`,
  `Domain(l=inf, le=inf)`, and `Domain(le=1, ge=1)` in 1-D (accepted in 2-D,
  where the check is non-strict). Consequently
  `Domain(l=1).intersect(Domain(le=2))` -> `AssertionError`.
- **Fix:** only require upper bounds >= lower bounds, consistently across
  dimensions; fix the docstring.

### [ ] 28. List bounds are only converted to arrays when several are given

- **Where:** `spapprox/domain.py:62-65`
- **Repro:** `d = Domain(dim=2, l=[1, 2])`: `d + 1` -> `TypeError` (list
  concatenation); `d * 2` -> `AssertionError` (list repeated); `d.l_vect`,
  `d.stack(...)` -> `ValueError`.
- **Fix:** convert outside the `len(specified_bounds) > 1` block.

### [ ] 29. `Domain.ldotinv` with a 1-D `C` calls `ldot`

- **Where:** `spapprox/domain.py:376-380`
- **Repro:** `1.0 in Domain(dim=2, l=1).ldotinv([2., 3.])` -> True (with `l=5`);
  expected False (`y < 1/3`). The 2-D column form is correct.
- **Fix:** `return self.ldotinv(other[:, None])`.

### [ ] 30. `Domain.ldot` ignores negative entries of `C`

- **Where:** `spapprox/domain.py:330-360`
- **What:** the result is not even a superset of the true domain; it can be
  disjoint from it.
- **Repro:** `[0, 0] in Domain(dim=2, l=1).ldot(-np.eye(2))` -> False; true
  domain is `y > -1`.
- **Fix:** route through `ldotinv(inv(C))` when invertible; otherwise combine
  `C+ @ upper` and `C- @ lower` per output component.

### [ ] 31. `PartialDerivative` with `acc > 2` always crashes

- **Where:** `spapprox/diff.py:173`
- **What:** grid size is `2*order+1` regardless of `acc`.
- **Repro:** `PartialDerivative(np.exp, 1, acc=4)(0.3)` -> `ValueError`
  (broadcast); `Tressian(..., acc=4)` -> `IndexError`.
- **Fix:** size the grid from `acc` (e.g. `2*((order+1)//2 - 1 + acc//2) + 1`)
  plus one-sided headroom.

### [ ] 32. Default finite-difference step is poorly chosen

- **Where:** `spapprox/diff.py:192`
- **What:** `min(1e-9**(1/n), 1e-4)` gives `h=1e-9` for first derivatives and
  caps third derivatives at `1e-4`.
- **Evidence (exp at 0.3):** d1 error 2.7e-8 (vs 2e-11 at `h=1e-5`); d3 error
  1.6e-4 (vs 3e-7 at `h=1e-3`). Log-sum-exp `Tressian` error 1.9e-4 (vs 1.6e-7);
  under findiff 0.11 one 2-D entry is ~2% off.
- **Fix:** `max(...)`, or `eps**(1/(n+2))` scaling.

### [ ] 33. `block_diag_3d` takes its dtype from the first tensor

- **Where:** `spapprox/diff.py:63`
- **Repro:** `block_diag_3d(np.ones((1, 1, 1), int), 0.5 * np.ones((1, 1, 1)))[1, 1, 1]`
  -> 0 (expected 0.5). Lists -> `AttributeError`; no arguments -> `IndexError`.
- **Fix:** `np.result_type(*tensors)` and `np.asarray` on each input.

### [ ] 34. Singular-scale range check in multivariate `dK_inv` ignores `loc`

- **Where:** `spapprox/cgf_base.py:1720`
- **Repro:** `(multivariate_norm(dim=2).ldot([[1, 1], [1, 1]]) + [1, 1]).dK_inv([2, 2])`
  -> `[nan, nan]` (a valid `t` exists; works with `loc=0`).
- **Fix:** `np.allclose(scale @ scale_inv @ (x - loc), x - loc)`.

### [ ] 35. Numerical derivatives use `self.dim` instead of `self.domain.dim`

- **Where:** `spapprox/cgf_base.py:1602`, `1789`, `1853`
- **What:** breaks projections where the underlying dimension differs.
- **Repro:** `M(K=-sum(log(1 - t)), dim=2, domain=...).ldot(np.array([[1., 1.]]))`:
  with findiff, `dK`/`d2K`/`d3K` raise "Dimensions do not match"; with
  numdifftools, `d3K` does.
- **Fix:** pass `self.domain.dim`.

### [ ] 36. `from_cgfs` with only univariate inputs drops keyword arguments

- **Where:** `spapprox/cgf_base.py:1994`
- **Repro:** `from_cgfs(norm(), norm(), d2K0=7 * np.eye(2), numdiff_backend="findiff")`
  -> `d2K0` is `I`, backend `numdifftools`.
- **Fix:** forward the kwargs to `from_univariate`.

### [ ] 37. Multivariate `add` defaults to `inplace=True`

- **Where:** `spapprox/cgf_base.py:1278`
- **What:** univariate `add` defaults to `False`. `c.add(np.array([1, 1]))`
  mutates `c.loc` and returns `c`; `c.add(other_mcgf)` ->
  `AssertionError: inplace not supported`.
- **Fix:** default `inplace=False`.

### [ ] 38. Multivariate + univariate `add` late-binds `self`

- **Where:** `spapprox/cgf_base.py:1332-1341`
- **What:** the lambdas read `self.loc`/`self.scale` at call time, so mutating
  an operand changes an already-built sum.
- **Repro:** `s = a + norm(); s.K([.5, .5])` -> `0.75`; after `a.loc = [10, 10]`
  -> `10.75`.
- **Fix:** capture as lambda defaults, like the multivariate + multivariate
  branch.

### [ ] 39. `clear_cache` evaluates every property

- **Where:** `spapprox/spa.py:100`
- **What:** `inspect.getmembers` triggers the expensive `_pdf_normalization`
  integration (hangs for `binomial`, item 24) and deletes a user-supplied
  `pdf_normalization`.
- **Fix:** iterate over `vars(self)`.

### [x] 40. `fit_ppf` and `fit_saddle_point_eqn` share `_t_cache`

- **Where:** `spapprox/spa.py:486`, `499`, `317`
- **Repro:** `fit_ppf()` then `fit_saddle_point_eqn(num=200)`; `ppf(0.5)` ->
  `ValueError: fp and xp are not of the same length`.
- **Fix:** give the ppf grid its own cache names.
- **Resolved (together with item 7):** ppf grid now lives in `_ppf_t_cache`.
  Regression test: `test_fit_ppf_and_saddle_point_eqn_caches_are_independent`.

### [ ] 41. Interpolated `dK_inv` silently clamps outside the fitted grid

- **Where:** `spapprox/spa.py:504`
- **Repro:** `gamma(a=3)`, grid ends at `x=30`; after fitting, `pdf(x=40)` ->
  5.34e-15 vs 3.49e-15 unfitted (50% off).
- **Fix:** `np.interp(..., left=np.nan, right=np.nan)` and fall back to the root
  solver.

### [ ] 42. `laplace().dK_inv(0)` is `0/0`

- **Where:** `spapprox/cgfs.py:231`
- **What:** `cdf(x=0)` and `pdf(x=0)` -> `nan` (cdf should be 0.5); same at
  `x=loc`.
- **Fix:** `x / (1 + sqrt(1 + x**2))` -- stable and 0 at 0.

---

## Low severity / robustness

### [ ] 43. `ndarray * Domain` is hijacked by numpy

- **Where:** `spapprox/domain.py:304-308`
- **Repro:** `type(np.array([2, -1]) * Domain(dim=2, l=1))` -> `ndarray` (object
  array of Domains). `Domain * array` is fine.
- **Fix:** `__array_ufunc__ = None` on the class.

### [ ] 44. Plain lists rejected for `Domain` vector add/mul

- **Where:** `spapprox/domain.py:197-208`, `237-245`
- **Repro:** `Domain(dim=2, l=1) + [1, 2]` -> `ValueError`.
- **Fix:** `is_list_like` + `np.asarray`.

### [ ] 45. A list `h` breaks `PartialDerivative`

- **Where:** `spapprox/diff.py:225`
- **Repro:** `PartialDerivative(lse, 1, 1, 0, h=[1e-3] * 3)(t0)` -> `ValueError`.
- **Fix:** use `self._h_vect`.

### [ ] 46. All-zero derivative orders crash

- **Where:** `spapprox/diff.py:192`
- **Repro:** `PartialDerivative(f, 0)` -> `ZeroDivisionError`.
- **Fix:** validate `sum(orders) >= 1`.

### [ ] 47. `type_wrapper` breaks on keyword arguments

- **Where:** `spapprox/util.py:35`
- **Repro:** `Domain(l=1).is_in_domain(t=0.5)` -> `IndexError`.
- **Fix:** fall back to `kwargs` by parameter name.

### [ ] 48. Multivariate `cgf - list` fails

- **Where:** `spapprox/cgf_base.py:216`
- **What:** `-1 * other` on a list yields `[]`.
- **Repro:** `multivariate_norm(dim=2) - [1., 2.]` -> `AssertionError`.
- **Fix:** `self.add(-np.asanyarray(other))`.

### [ ] 49. NumPy scalars rejected in CGF algebra

- **Where:** `spapprox/cgf_base.py:676`, `757`, `1304`, `1427`
- **Repro:** `norm() + np.int64(2)`, `norm() * np.float32(2)` -> `ValueError`.
- **Fix:** `numbers.Real` / `pd.api.types.is_number`.

### [ ] 50. Unformatted error message and list `loc` rejected

- **Where:** `spapprox/cgf_base.py:928`
- **Repro:** `M(K=..., dim=2, loc=[1., 2.])` -> message
  `"...vector of length {self.dim}"`.
- **Fix:** add the `f` prefix; `np.asanyarray(loc)` before validation.

### [ ] 51. `univariate_empirical` overflows

- **Where:** `spapprox/cgfs.py:273`, `281`
- **Repro:** data `[1000, 1001, 1003]`: `K(1)` -> `inf` (true 1002.07),
  `dK(1)` -> `nan`.
- **Fix:** logsumexp / max-shifted softmax.

### [ ] 52. `ppf(0)` / `ppf(1)` return arbitrary finite values

- **Where:** `spapprox/spa.py:315`, `361-380`
- **Repro:** `norm()`: `ppf(0.)` -> -64, `ppf(1.)` -> 16.
- **Fix:** return the support bounds (e.g. `-inf`/`inf`).

### [ ] 53. pdf normalization over a truncated `t` range biases exact cases

- **Where:** `spapprox/spa.py:248`
- **What:** normalized `norm` pdf has relative error +1e-4.
- **Fix:** tighter tolerance or integrate over the full domain.

### [ ] 54. _Suspected:_ `K` returning shape `(1,)` breaks multivariate `loc`

- **Where:** `spapprox/cgf_base.py:1554`
- **Repro:** `M(K=lambda t: np.atleast_1d(np.sum(t**2) / 2), dim=2, loc=[1, 2]).K([1, 1])`
  -> `ValueError: shapes (2,1) and (2,) not aligned`.
- **Fix:** `val += np.dot(t, loc)`. Arguably user error; decide on the contract.

### [ ] 55. README and config drift

- `README.md:49`: example uses `spa.SaddlePointApprox`, which doesn't exist
  (should be `UnivariateSaddlePointApprox`).
- `README.md:65`: optional dependencies omit `findiff`.
- `README.md:67-77`: mismatched code fences; refers to `setup.py`.
- `pyproject.toml:14`: ruff comment claims "bugbear, isort", but only `E/F/W`
  are selected.

---

## Tooling / packaging

### [ ] 16. Python version floor conflicts with the syntax in use

- **Where:** `spapprox/diff.py:345`, `496` use `retval.T[*sel]` /
  `self._partials[*ijk]` -- star-in-index is Python 3.11+, while
  `pyproject.toml` declares `requires-python = ">=3.9"`.
- **Symptom:** `ruff check .` fails with 7 errors: 4 invalid-syntax (2 here, 2
  in `examples/`), 2 F401, 1 E741.
- **Decide:** either raise the floor to `>=3.11`, or rewrite the indexing
  expressions as `tuple(...)` subscripts.
- **See also:** item 25 (CI).

---

## Needs a literature check (unresolved)

### [ ] 56. _Suspected:_ bivariate cdf converges too slowly under dependence

- **Where:** `spapprox/spa.py:836-912`
- **What:** even with items 17 and 18 fixed, the error for dependent pairs only
  shrinks like `n^(-1/2)` as shape parameters grow by a factor `k`, whereas a
  second-order method should give `1/n` (as the independent case does).
- **Evidence (max abs error):**

  | Case                                                    | k=1   | k=4   | k=16   | k=64   |
  | ------------------------------------------------------- | ----- | ----- | ------ | ------ |
  | `(Z1, Z1+Z2)`, exact by quadrature                      | 0.043 | 0.042 | 0.029  | 0.0175 |
  | `bivariate_gamma([2, 1, 3]) * k` vs 4M-draw Monte Carlo | 0.023 | 0.016 | 0.0095 | --     |

- **Next step:** all 16 Phi/phi argument pairings were tried without success;
  likely culprits are `u_tilde`, `b` or `rho`. Re-derive against Wang (1990) /
  Kolassa & Li (2010); Paolella (2007) section 5.2 via the linked
  `paolella-saddlepoint` skill.

---

## Working notes

- Full suite takes ~11 min; iterate with targeted `pytest tests/test_X.py`.
- Numerical tests compare against `scipy.stats` with tolerances. If a fix shifts
  results, establish whether it is a regression or a genuinely better
  approximation _before_ touching any tolerance.
- Check for `tofix`-marked tests that may already document some of these
  behaviours -- those markers should be removed as the corresponding items land.
- Several items share a root cause; fix them together:
  - domain/scale units: 10, 14, 21, 22 (and 26 for `intersect`);
  - in-place mutation: 2, 11.
- Lattice cdf is not continuity-corrected (`poisson(3)` at `x=3`: 0.538 vs exact
  0.647); already a known TODO in the code, not listed as a bug.
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

- add the conditional distribution
- implement Dirichlet bootstrap
- implement multivariate saddle point approximations
- implement other bootstraps
- jackknife stuff
- conditional distributions
- approximation for discrete distributions
- bootstrap for estimating equations (Kuonen, 1998, particularly bootstrap for robust estimators)
- quadratic forms (Kuonen 2001)
- mixture models

## Literature to review

- Find complete version of Alya Omar's master thesis (2015)
