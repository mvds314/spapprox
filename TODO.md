# Bug backlog

Findings from a repo-wide review. Every item below was reproduced empirically
against the working tree at `b0cd9e4` (branch `dev`). Nothing has been fixed yet.

## Status legend

- [ ] not started
- [x] fixed + regression test added

---

## High severity

Wrong numbers or hard crashes on normal use.

### [ ] 1. `poisson()` declares a wrong domain

- **Where:** `spapprox/cgfs.py:321`
- **What:** declares `Domain(ge=0)`, but `K(t) = mu * (exp(t) - 1)` is finite on
  all of R. Everything below the mean is masked to `nan`.
- **Repro:** `poisson(mu=1).K(-1.0)` -> `nan`; true value `-0.6321205588285577`.
  Likewise `dK(-0.5)` -> `nan`, true `0.6065306597126334`.
- **Fix:** drop the domain restriction (full real line).
- **Why it hid:** existing tests only probe `t > 0`.

### [ ] 2. `d2K0` / `d3K0` mutate the raw cache in place

- **Where:** `spapprox/cgf_base.py:1132-1135`, `1145-1149`
- **What:** uses `*=` on the same object held by `_d2K0_raw_cache`, so the
  scaling compounds on every access after a scale assignment.
- **Repro:**
  ```python
  c = multivariate_norm(scale=np.array([1., 2.]))
  c.cov  # [[1, 0], [0, 4]]
  c.scale = np.array([1., 2.]); c.cov  # [[1, 0], [0, 16]]
  c.scale = np.array([1., 2.]); c.cov  # [[1, 0], [0, 64]]
  ```
- **Fix:** build a new array (`out = raw * factor`) instead of in-place `*=`.

### [ ] 3. `Domain.from_domains` crashes

- **Where:** `spapprox/domain.py:568`, `583`
- **What:** `i, j = 0` -- unpacking an `int`.
- **Repro:** `Domain.from_domains(Domain(dim=2, A=[[1, 1]], a=[1]), Domain(dim=1, l=1))`
  -> `TypeError: cannot unpack non-iterable int object`.
- **Fix:** `i, j = 0, 0`.
- **Blast radius:** reaches `from_univariate`, `from_cgfs`, `stack` for any
  domain carrying linear constraints.

### [ ] 4. `Domain.ldotinv` sign error on `ge` bounds

- **Where:** `spapprox/domain.py:394-399`
- **What:** appends `+ge` where `x >= ge` requires `-x <= -ge`. The strict `g`
  branch alongside it does it correctly, so this is an inconsistency inside one
  method.
- **Repro:**
  ```python
  d = Domain(dim=2, ge=1.0)
  [0., 0.] in d              # False (correct)
  [0., 0.] in d.ldotinv(np.eye(2))  # True (wrong -- domain inverted)
  ```
- **Fix:** negate to match the `g` branch.

### [ ] 5. `__getitem__` slices cumulants on one axis only

- **Where:** `spapprox/cgf_base.py:1261`, `1266`
- **What:** `d2K0[item]` where `d2K0[np.ix_(item, item)]` is meant; `d3K0` needs
  the rank-3 equivalent.
- **Repro:** slicing a 3-D CGF with `[0, 2]` yields a 2x3 "covariance"
  `[[4, 1, 0.5], [0.5, 0.2, 1]]` instead of `[[4, 0.5], [0.5, 1]]`.
- **Fix:** use `np.ix_` on all axes.

### [ ] 6. `dK_inv` can loop forever

- **Where:** `spapprox/cgf_base.py:484`
- **What:** the bracket-expansion call omits `loc=0, scale=1`, which every
  neighbouring call passes. With `loc != 0` the loop never satisfies its exit
  condition.
- **Fix:** add `loc=0, scale=1` to match siblings.
- **Note:** hangs rather than raising, so guard the regression test with a
  timeout or a bounded iteration count.

### [ ] 7. `fit_ppf` raises `UnboundLocalError`

- **Where:** `spapprox/spa.py:492-499`
- **What:** `x_range` is only bound in the branch where `_x_cache` is absent,
  but is read unconditionally afterwards.
- **Repro:**
  ```python
  u = UnivariateSaddlePointApprox(norm())
  u.fit_saddle_point_eqn()
  u.fit_ppf()  # UnboundLocalError: cannot access local variable 'x_range'
  ```
- **Fix:** derive `x_range` from the cache in the other branch too.

### [ ] 8. `atol=1.0 - 8` typo

- **Where:** `spapprox/spa.py:713` (`BivariateSaddlePointApprox.cdf`)
- **What:** evaluates to `-7.0`; `1e-8` was intended. A negative tolerance means
  the near-`t=0` guard never fires -> `RecursionError`.
- **Fix:** `atol=1e-8`.

### [ ] 9. `np.asanayarray` typo (5 sites)

- **Where:** `spapprox/cgf_base.py:518`, `1723`, `1837`, `1885`;
  `spapprox/spa.py:396`
- **What:** misspelling of `np.asanyarray` -> `AttributeError` whenever `t0=` is
  supplied, or whenever the `fillna` path is taken.
- **Fix:** mechanical rename; confirm no other spellings via
  `grep -rn "asanayarray" --include=*.py .`

---

## Medium severity

Contract violations and silently wrong results in narrower situations.

### [ ] 10. Derived-CGF domains are in the wrong `t` units when `scale != 1`

- **Where:** `spapprox/cgfs.py:260` (`univariate_sample_mean`), and `add` in
  `spapprox/cgf_base.py`
- **What:** the parent domain is reused without rescaling by `1/scale`, so the
  domain and `K` disagree. This breaks the package-wide masking contract, where
  `t in cgf.domain` is supposed to predict whether `K(t)` is finite.
- **Repro:**
  ```python
  sm = univariate_sample_mean(exponential(scale=2), 5)
  4.0 in sm.domain   # True
  sm.K(4.0)          # nan

  s = exponential(scale=2) + norm()
  0.8 in s.domain    # True
  s.K(0.8)           # nan
  ```

### [ ] 11. `transform_rank3_tensor` mutates its input

- **Where:** `spapprox/diff.py:48`
- **What:** writes through to the caller's array and returns the same object.
- **Repro:** input array is modified in place; `id(out) == id(T)`.
- **Fix:** copy before transforming.

### [ ] 12. `transform_rank3_tensor` returns index-reversed output for 3 matrices

- **Where:** `spapprox/diff.py:50`
- **What:** with three distinct matrices the result equals
  `einsum('il,jm,kn,lmn->ijk', A, B, C, T).transpose(2, 1, 0)`, not the
  un-transposed contraction. Verified: `allclose(got, ref)` False,
  `allclose(got, ref.T)` True.
- **Note:** masked in current use because callers pass a single symmetric
  matrix, where the transposition is invisible.

### [ ] 13. `bivariate_gamma` domain omits `t1 + t2 < 1`

- **Where:** `spapprox/cgfs.py:217`
- **Repro:** `t = [0.9, 0.9]` reports `in domain` -> `True` while `K(t)` -> `nan`.

### [ ] 14. Multivariate `add` translates the domain by `loc`, and uses the wrong `loc`

- **Where:** `spapprox/cgf_base.py:1405-1407`
- **What:** the domain lives in `t`-space, so translating it by `loc` is not
  meaningful; separately, `self.loc_vect` is used where `other`'s should be
  (copy-paste).

### [ ] 15. `from_cgfs` crashes on mixed univariate/multivariate input

- **Where:** `spapprox/cgf_base.py:2043`, `2054`
- **What:** fails when cumulants are already cached and the inputs mix
  dimensionalities.

---

## Tooling / packaging

### [ ] 16. Python version floor conflicts with the syntax in use

- **Where:** `spapprox/diff.py:345`, `496` use `retval.T[*sel]` /
  `self._partials[*ijk]` -- star-in-index is Python 3.11+, while
  `pyproject.toml` declares `requires-python = ">=3.9"`.
- **Symptom:** `ruff check .` fails (7 errors). CI runs 3.11 so it passes there,
  but the package is uninstallable-as-advertised on 3.9/3.10.
- **Decide:** either raise the floor to `>=3.11`, or rewrite the two indexing
  expressions as `tuple(...)` subscripts.

---

## Needs a literature check (unresolved)

Check against Paolella (2007) section 5.2 -- the linked `paolella-saddlepoint`
skill in `.claude/skills/` is the natural reference.

### [ ] 17. Possibly swapped pairings in the Lugannani-Rice-style combination

- **Where:** `spapprox/spa.py:912`
- **What:** pairs `Phi(w_tilde) * n` with `Phi(w) * n_tilde`, which is the
  opposite of what the surrounding docstring describes.

### [ ] 18. `sps.norm.pdf(w)` vs documented `phi(x_tilde)`

- **Where:** `spapprox/spa.py:857`, `888`
- **What:** the code evaluates the normal pdf at `w` where the docstring says
  `x_tilde`. Determine which is correct.

---

## Working notes

- Full suite takes ~11 min; iterate with targeted `pytest tests/test_X.py`.
- Numerical tests compare against `scipy.stats` with tolerances. If a fix shifts
  results, establish whether it is a regression or a genuinely better
  approximation *before* touching any tolerance.
- Check for `tofix`-marked tests that may already document some of these
  behaviours -- those markers should be removed as the corresponding items land.
- Suggested first batch (unambiguous one-liners, high impact): items 9, 3, 8, 6, 7.
