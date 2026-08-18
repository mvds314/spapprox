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

CI (`.github/workflows/python-app.yml`, Python 3.11) installs **only**
`numdifftools`, so any code path reachable by default must work without the
other two.

## Test and lint

```bash
pytest                                   # full suite (~11 min)
pytest tests/test_diff.py                # single file
pytest tests/test_diff.py::test_name     # single test
pytest -k "some_pattern"                 # by name pattern
pytest -m "not slow"                     # skip slow tests
pytest -m "not tofix"                    # skip known-broken/WIP tests
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
- `examples/` is untested, needs `matplotlib` (not a declared dependency), and
  is partly stale (some scripts still use the deprecated `FinDiff` API). Do not
  treat it as a behavioral reference.

Note: `.github/copilot-instructions.md` is a hard link to this file, so GitHub
Copilot CLI and Claude Code read the same content. Edit this file, not the link.