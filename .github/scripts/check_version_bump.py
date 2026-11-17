"""Check that the CalVer (YY.MM.N) version in pyproject.toml was bumped relative to a base file.

Usage: check_version_bump.py BASE_PYPROJECT HEAD_PYPROJECT

Rules: the new version must be exactly the next one after the base version, i.e.
`YY.MM.N+1` within the same month, or `YY.MM.1` for a later month.
"""

import re
import sys
import tomllib


def read_version(path):
    with open(path, "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    match = re.fullmatch(r"(\d{2})\.(\d{1,2})\.(\d+)", version)
    if not match:
        sys.exit(f"{path}: version {version!r} does not match CalVer YY.MM.N")
    return version, tuple(int(g) for g in match.groups())


def main(base_path, head_path):
    base, (by, bm, bn) = read_version(base_path)
    head, (hy, hm, hn) = read_version(head_path)
    if (hy, hm) == (by, bm):
        expected = (by, bm, bn + 1)
    elif (hy, hm) > (by, bm):
        expected = (hy, hm, 1)
    else:
        expected = None
    if (hy, hm, hn) != expected:
        hint = f"expected {'.'.join(map(str, expected))}" if expected else "must be later"
        sys.exit(f"Version bump required: base is {base}, this PR has {head} ({hint}).")
    print(f"Version bumped {base} -> {head}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
