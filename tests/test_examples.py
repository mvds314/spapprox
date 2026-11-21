import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.slow
@pytest.mark.examples
@pytest.mark.parametrize("script", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_runs(script):
    """Smoke test: the example script runs to completion without errors."""
    pytest.importorskip("matplotlib")
    if "statsmodels" in script.read_text():
        pytest.importorskip("statsmodels")
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, script.name],
        cwd=EXAMPLES_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{script.name} failed:\n{result.stderr[-5000:]}"
