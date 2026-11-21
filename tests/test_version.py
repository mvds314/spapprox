import re
import tomllib
from pathlib import Path

import spapprox

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version():
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_version_matches_pyproject():
    expected = _pyproject_version()
    assert spapprox.__version__ == expected, (
        f"spapprox.__version__ is {spapprox.__version__!r}, pyproject.toml says {expected!r}; "
        "reinstall the package (pip install -e .) if the installed metadata is stale"
    )


def test_citation_version_matches_pyproject():
    # The Release workflow requires this too; bump CITATION.cff together with pyproject.toml
    match = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', (ROOT / "CITATION.cff").read_text(), re.M)
    assert match is not None, "CITATION.cff has no version"
    assert match.group(1) == _pyproject_version()
