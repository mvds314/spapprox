import tomllib
from pathlib import Path

import spapprox


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    assert spapprox.__version__ == expected, (
        f"spapprox.__version__ is {spapprox.__version__!r}, pyproject.toml says {expected!r}; "
        "reinstall the package (pip install -e .) if the installed metadata is stale"
    )
