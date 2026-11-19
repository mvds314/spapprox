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
- CI tests Python 3.11 only. There is no release workflow.
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

## Decisions to make first

1. **Version scheme.** PyPI orders versions by PEP 440, and that order is permanent.
   - Keep CalVer: release `YY.MM.N` as usual (e.g. `26.12.1`). Nothing changes for
     the `version-bump` check. "Official" is then a matter of the README and
     classifiers (`Development Status :: 4 - Beta` or `5 - Production/Stable`).
   - Switch to SemVer `1.0.0`: this can never go back to CalVer, because `26.x`
     sorts above `1.x`. The `version-bump` check would need rewriting.
   - Recommendation: keep CalVer. It is already enforced, and the package has no
     API-compatibility promise yet that SemVer would express.
2. **Does every merge to `main` get published?** Recommendation: no. Publish on a
   GitHub Release (tag `vYY.MM.N`). `main` keeps bumping per PR, and releases are
   cut when worthwhile.
3. **sdist contents.** Recommendation: package, `README.md`, `LICENSE`,
   `CITATION.cff` and `tests/` (downstream packagers such as conda-forge run them).
   Exclude `docs/`, `examples/`, `.claude/`, `.github/` and the agent instruction
   files.
4. **`statsmodels` dependency.** It is heavy for two helpers. Optionally replace
   `PandasWrapper` with a small wrapper in `util.py` and `cov2corr` with three lines
   of numpy, then drop the dependency. This is not a release blocker. It is
   cheap, though, and easier to do before users depend on the install footprint.

## Work packages

Each package is one PR (worktree + version bump, as usual). Items 1-4 are
independent. Item 5 depends on 1 and 2, and item 6 on all the others.

### 1. Packaging and metadata (`pyproject.toml`)

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

### 2. README cleanup

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

### 3. Docstrings

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
  one, has no docstring, so coverage does not regress.
- Optionally, enable ruff's `D` (pydocstyle) rules for `spapprox/` only, with a
  numpydoc convention. Leave this out if it is too noisy.

### 4. Examples

- Port the three `FinDiff` scripts to the `Diff` API, or delete them if they only
  demonstrate internals (`example_gen_lin_opr.py`, `example_einsum.py`).
- Merge `basic_example.py` and `expon_example.py`. Give the remaining scripts a
  short header docstring stating what they show, and whether they need an
  optional extra.
- Add a CI smoke test that runs each example with the `Agg` matplotlib backend
  (a separate job, or a `slow`-marked parametrized test over `examples/*.py`), so
  examples cannot go stale again. Drop the "untested and partly stale" warnings
  from `CLAUDE.md` and the skill once this is in.

### 5. CI and release workflow

- Test matrix: Python 3.11, 3.12 and 3.13 on the fast suite. Optionally add a
  job with all extras (`findiff`, `fastnorm`).
- A `build` job on every PR: `python -m build` plus `twine check`.
- `.github/workflows/release.yml`, triggered on `release: published`: build, then
  publish with `pypa/gh-action-pypi-publish` and PyPI **trusted publishing**
  (OIDC, so no API token is stored). Use a protected `pypi` environment. Optionally
  publish to TestPyPI first, or on `workflow_dispatch`.
- A check that the release tag matches the version in `pyproject.toml`.
- One-time manual setup: create the PyPI (and TestPyPI) project as a pending
  trusted publisher for `mvds314/spapprox`, workflow `release.yml`, environment
  `pypi`.

### 6. Release readiness and cut

- Go through `docs/TODO.md` and the open `tofix` tests (one per test file).
  Decide which ones block the release, and document the remaining known
  limitations in the README (multivariate gaps, quadrature-based cgfs are slow).
- API review: confirm what is public. For example, should `Timer` be exported,
  and `fib` in `util.py`? Underscore or remove anything that should not be a
  compatibility promise.
- Update `CITATION.cff`: version, date and URL.
- Close issues #1 and #2.
- Run the full suite locally (`pytest`, ~16 min) and `ruff check .`.
- Dry run on TestPyPI, then `pip install` from TestPyPI into a clean venv, then
  run the README quick start.
- Create the GitHub Release `vYY.MM.N` with release notes. This publishes to
  PyPI. Afterwards, check the PyPI page rendering and `pip install spapprox`.

## Out of scope for the first release

- A hosted documentation site (Sphinx/MkDocs, Read the Docs). Revisit if the
  docstrings outgrow the README.
- conda-forge recipe: a follow-up once the PyPI sdist exists.
- New functionality (quotients of random variables and so on), tracked in
  `docs/TODO.md`.
