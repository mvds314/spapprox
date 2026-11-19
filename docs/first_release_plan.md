# Plan: first official release on PyPI

Goal: publish a first official, installable `spapprox` on PyPI (`pip install spapprox`),
with a README that reads well on both GitHub and PyPI, documentation through
docstrings plus the `examples/` scripts, and a lean distribution.

## Current state (survey, Nov 2026)

- The name `spapprox` is still free on PyPI.
- `flit_core` backend. The wheel contains only the `spapprox/` package (13 files), so
  `docs/`, `tests/`, `examples/` and `.claude/` already stay out of it. The sdist
  is not configured explicitly.
- Metadata is minimal: no classifiers, one keyword, one URL, and `license` uses the
  legacy `{ text = ... }` table.
- Version is CalVer `26.11.31`, bumped on every PR.
- CI tested Python 3.11 only (now 3.11-3.14). There is no release workflow.
- README still says "UNDER CONSTRUCTION". It lists "approximating conditional
  distributions" as to be implemented, but they are implemented and documented
  further down. It installs from GitHub, links relative paths (`examples/...`),
  and uses `$...$` math, which PyPI does not render.
- Docstrings: public API coverage is partial. Most modules have no module
  docstring, and many `cgfs.py` factories and `cgf_base.py`/`domain.py`/`util.py`
  methods have no docstring.
- Examples: 20 scripts, all need `matplotlib` (not declared). `example_findiff.py`,
  `example_findiff_multivariate.py` and `example_gen_lin_opr.py` use the deprecated
  `FinDiff` API. `basic_example.py` and `expon_example.py` overlap. None of them run
  in CI.
- `CITATION.cff` is stale: version `23.10` and a typo in the URL (`sparrox`).
- `statsmodels` is a hard dependency, used only for `PandasWrapper` and `cov2corr`.
- GitHub issues #1 and #2 can probably be closed (see `docs/TODO.md`).

## Decisions

1. **Version scheme: keep CalVer** (`YY.MM.N`, decided). PyPI orders versions by
   PEP 440 and that order is permanent: after a `26.x` release, a SemVer `1.0.0`
   would sort below it. The `version-bump` check stays as it is. "Official" is
   expressed by the README and the classifiers (`Development Status :: 4 - Beta`
   for now).
2. **Releases are cut manually on GitHub** (decided), not on every merge to `main`.
   `main` keeps bumping the version per PR, and a GitHub Release (tag `vYY.MM.N`)
   is created by hand when worthwhile. Publishing to PyPI is a later step (work
   package 5).
3. **sdist contents:** package, `README.md`, `LICENSE`, `CITATION.cff` and `tests/`
   (downstream packagers such as conda-forge run them). `docs/`, `examples/`,
   `.claude/`, `.github/` and `CLAUDE.md` are excluded. The wheel contains only the
   package.
4. **`statsmodels` dependency.** Open. It is heavy for two helpers (`PandasWrapper`,
   `cov2corr`), and both are easy to replace. Not a release blocker.

## Work packages

Each package is one PR (worktree + version bump, as usual). Items 1 and 2 were
done together with this plan. Items 3 and 4 are independent. Item 6 depends on
the others, except the PyPI part of item 5.

### 1. Packaging and metadata (`pyproject.toml`) -- done

- `license = "BSD-3-Clause"` (SPDX string) plus `license-files = ["LICENSE"]`.
  This needs `flit_core >=3.12`, so bump the build requirement.
- A more descriptive `description`, e.g. "Saddle point approximations of
  distributions from their cumulant generating functions".
- `keywords`: statistics, saddlepoint, cumulant generating function, bootstrap,
  resampling, approximation.
- `classifiers`: Development Status, Intended Audience :: Science/Research, License
  :: OSI Approved :: BSD License, Programming Language :: Python :: 3 :: Only,
  3.11/3.12/3.13, Topic :: Scientific/Engineering :: Mathematics, Operating System ::
  OS Independent.
- `[project.urls]`: Homepage, Source, Issues, Changelog (GitHub releases).
- Check the lower bounds of runtime dependencies (`numpy`, `scipy`, `pandas`, and
  `statsmodels` unless it is dropped), and check that numpy 2 works.
- Explicit sdist configuration, per decision 3:
  ```toml
  [tool.flit.sdist]
  include = ["tests/", "CITATION.cff"]
  exclude = ["docs/", "examples/", ".claude/", ".github/", "CLAUDE.md"]
  ```
- Optionally, an `examples = ["matplotlib"]` extra and an `all` extra.
- `__version__` in `spapprox/__init__.py` via `importlib.metadata.version`.
- `__all__` in `spapprox/__init__.py`, to make the public API explicit.
- Verify: `python -m build`, then `twine check dist/*`, then list the contents of
  both artifacts, then install the wheel into a clean venv and `import spapprox`.

### 2. README cleanup -- done

- Remove "UNDER CONSTRUCTION". Fix the "to be implemented" list: conditional
  distributions are done, quotients are not.
- Structure: one-paragraph pitch, then installation (`pip install spapprox`,
  extras), quick start, features, available distributions (the factories in
  `cgfs.py`), approximation classes, an examples index (one line per script), and
  references (Butler 2007, Daniels 1983, Booth & Butler 1990, Lugannani & Rice,
  Barndorff-Nielsen, Skovgaard 1987, Kolassa 2003, Kolassa & Li 2010, Wang 1990).
  End with how to cite, a short development section and the license.
- PyPI rendering: use absolute GitHub URLs for links to `examples/` and `docs/`,
  and replace `$...$` math by code-formatted or unicode formulas, since PyPI shows
  raw dollar signs.
- Check the result with `twine check` and by previewing the long description.

### 3. Docstrings -- done

Docstrings plus examples are the documentation, so no Sphinx site for now. The
style stays as it is: LaTeX-heavy raw-string reST with a `References` section.

- Module docstrings for `__init__`, `cgf_base`, `cgfs`, `domain`, `spa` and
  `util`.
- Every name in `spapprox.__all__` gets a docstring with parameters, return value,
  a short example where useful, and a pointer to the relevant `examples/` script.
  That means every `cgfs.py` factory (parameterization, domain, and whether the
  cgf is exact or uses quadrature), the public CGF methods (`K`, `dK`, `d2K`,
  `d3K`, `dK_inv`, algebra, `ldot`, `from_univariate`, ...), `Domain`, and every
  approximation class and its public methods.
- Add a small fast test that fails when a name in `__all__`, or a public method of
  one, has no docstring, so coverage does not regress: `tests/test_docstrings.py`
  (inherited docstrings count; ruff `D` rules stay off).
- Ruff's `D` (pydocstyle) rules were left off as too noisy; the test above covers
  the public API instead.

### 4. Examples -- done

- Deleted the development scratch scripts (`example_findiff.py`,
  `example_findiff_multivariate.py`, `example_gen_lin_opr.py`, `example_einsum.py`) and
  the unfinished `example_hypergeometric_bootstrap.py`, which used a removed API. Added
  `example_numdiff_backends.py` to show the two numerical differentiation backends.
- Merged `expon_example.py` into `basic_example.py`, and gave every script a header
  docstring.
- `tests/test_examples.py` runs every script with the `Agg` backend (markers `slow` and
  `examples`, so not in PR CI). The Examples workflow (`examples.yml`) runs them after
  each merge into `main`, and on demand.
- `release.yml` runs when a GitHub release is published, and fails unless the Examples
  workflow passed on the released commit. The PyPI publish job (item 5) will depend on it.

### 5. CI and publishing (later)

- [x] Test matrix: Python 3.11-3.14 on the fast suite. The aggregate `test` job keeps
  the single required status check of the `main` ruleset.
- Optionally add a job with all extras (`findiff`, `fastnorm`).
- A `build` job on every PR: `python -m build` plus `twine check`.
- Later, a PyPI publishing job in `release.yml` (triggered by a manually created GitHub
  Release), depending on its `check-examples` job. It would use `pypa/gh-action-pypi-publish` with PyPI
  trusted publishing (OIDC, so no API token is stored) and a protected `pypi`
  environment, and check that the tag matches the version in `pyproject.toml`.
- Until then, build locally (`python -m build`) and attach the wheel and sdist to
  the GitHub Release by hand.

### 6. Release readiness and cut

- Go through `docs/TODO.md` and the open `tofix` tests (one per test file).
  Decide which ones block the release, and document the remaining known
  limitations in the README (multivariate gaps, quadrature-based cgfs are slow).
- API review: confirm what is public. For example, should `Timer` be exported,
  and `fib` in `util.py`? Underscore or remove anything that should not be a
  compatibility promise.
- Update `CITATION.cff`: version, date and URL.
- Close issues #1 and #2.
- Run the full suite locally (`pytest`, ~16 min plus the examples) and `ruff check .`.
- Build locally, install the wheel into a clean venv, and run the README quick
  start.
- Create the GitHub Release `vYY.MM.N` by hand, with release notes and the built
  artifacts attached. Publishing to PyPI comes later (item 5).

## Out of scope for the first release

- A hosted documentation site (Sphinx/MkDocs, Read the Docs). Revisit if the
  docstrings outgrow the README.
- conda-forge recipe: a follow-up once the PyPI sdist exists.
- New functionality (quotients of random variables and so on), tracked in
  `docs/TODO.md`.
