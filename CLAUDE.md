# spapprox -- Agent Instructions

Saddle point approximation library: given a random variable's (or vector's)
cumulant generating function (CGF), approximate its pdf/cdf via the saddle point
method (Butler, 2007, "Saddlepoint Approximations with Applications").

## Architecture

**For anything beyond the commands below -- how the modules fit together, the
CGF/approximation APIs, conventions, and known gotchas -- use the `spapprox`
skill in `.claude/skills/spapprox/`.** It is the maintained map of this codebase
and it goes far deeper than this file. Read `SKILL.md` first; it points to
reference files for the CGF classes, the approximation classes, and the
differentiation/domain internals.

Short version: `util / diff / domain -> cgf_base -> cgfs -> spa`. Build a CGF
(`cgfs.py` factories), do algebra on it (`+` sums independent variables), hand it
to an approximation class (`spa.py`).

## Workflow

All changes go through pull requests into `main`; do not commit to `main` directly.
Do each piece of work on its own branch in a separate **git worktree**, so that
parallel tasks and the main checkout stay untouched:

```bash
git worktree add ../spapprox-<topic> -b feature/<topic> main
cd ../spapprox-<topic>            # work, test, commit here
gh pr create                      # open the PR (draft while in progress)
# after the merge:
git worktree remove ../spapprox-<topic>
git branch -d feature/<topic> && git push origin --delete feature/<topic>
```

Wait for the CI check to pass before merging. The version in `pyproject.toml` is
CalVer (`YY.MM.N`); every PR into `main` must bump it (`N+1` within a month, `YY.MM.1` for a
new month), enforced by the `version-bump` CI check. Bump the version in `CITATION.cff` along
with it (checked by `tests/test_version.py`). Larger features get a
written plan in `docs/` first, and follow-ups are listed in `docs/TODO.md`.

Releases are created by hand with the Release workflow (Actions > Release > Run workflow on
`main`): it requires a successful Examples run on the commit, builds and checks the wheel and
sdist, and creates a draft GitHub release `v<version>` with both files attached. Review and
edit the draft, then publish it: that creates the tag and starts the Publish to PyPI
workflow (`publish.yml`), which uploads the attached files to PyPI with trusted publishing.
A version uploaded to PyPI can never be reused.

## Setup

Editable install (flit backend, no setup.py):

```bash
pip install -e .                                # minimal
pip install -e ".[findiff,numdiff,fastnorm]"    # with all optional extras
```

Optional dependencies, all probed with `try/import` at module load:

| Package | Flag | Defined in | Purpose |
|---|---|---|---|
| `findiff>=0.11` | `_has_findiff` | `diff.py` | fast finite differences (uses the `Diff` API; `FinDiff` is deprecated) |
| `numdifftools` | `has_numdifftools` | `cgf_base.py` | default numerical differentiation backend |
| `fastnorm` | `_has_fastnorm` | `spa.py` | faster bivariate normal cdf |

CI (`.github/workflows/python-app.yml`; runs `pytest -m "not slow"` on Python 3.11-3.14 and `ruff check .` on
pushes to `main`/`dev` and PRs to `main`) installs **only** `numdifftools`
(`pip install -e ".[numdiff]"`), so any code path reachable by default must work
without the other two. This includes test collection: never construct findiff
objects (`Gradient`, `Hessian`, `PartialDerivative`, ...) at module level, e.g.
inside `pytest.param(...)`; wrap them in a lambda so they are built at call time.

## Test and lint

**Default to `pytest -m "not slow"` when running tests.** The full suite takes too
long for routine iteration; run it only when explicitly asked. CI also skips the slow tests, so
run the full suite locally before merging a change that touches slow-marked code paths.

```bash
pytest -m "not slow"                     # default: skip slow tests
pytest                                   # full suite (~16 min), only when asked
pytest tests/test_diff.py                # single file
pytest tests/test_diff.py::test_name     # single test
pytest -k "some_pattern"                 # by name pattern
pytest -m "not tofix"                    # skip known-broken/WIP tests
pytest -m examples                       # run the examples/ scripts (needs matplotlib)
ruff check .                             # lint (line-length 99, see pyproject.toml)
```

Markers `slow` and `tofix` are registered in `pyproject.toml`; `tofix` documents
known-failing behavior rather than serving as a to-do note. Tests that need an
optional backend gate on it, e.g.
`pytest.mark.skipif(not has_findiff, reason="No findiff")`, often combined with
`pytest.mark.slow` inside `pytest.param(..., marks=[...])`. Follow that pattern.

Numerical tests compare against `scipy.stats` references with tolerances. If a
change shifts results, establish whether it is an accuracy regression or a
genuinely better approximation before touching a tolerance.

## Conventions worth knowing up front

- Derivatives are named `dK`, `d2K`, `d3K`; their values at `t=0` (the cumulants)
  are `dK0`, `d2K0`, `d3K0`. Keep this naming.
- A CGF's `K`/`dK`/`d2K`/`d3K` always describe the **standardized** variable;
  `loc`/`scale` are applied by the base class. Never bake them into a
  distribution factory's derivatives -- they would be applied twice.
- `examples/` needs `matplotlib` (not a declared dependency). `tests/test_examples.py`
  runs every script as a smoke test (markers `slow` and `examples`), and the Examples
  workflow runs them after each merge into `main`; a GitHub release checks that this
  run passed on the released commit (`release.yml`). They check that the scripts run,
  not their numbers, so do not treat them as a behavioral reference.

Note: `.github/copilot-instructions.md` is a symlink to this file (`ln -s ../CLAUDE.md
.github/copilot-instructions.md`, gitignored), so GitHub Copilot CLI and Claude Code read
the same content. Edit this file, not the link. Do not use a hard link: git replaces this file
when a pull changes it, which silently breaks a hard link.