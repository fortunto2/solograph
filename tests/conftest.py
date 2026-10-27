"""The Rust binary, for tests where Python writes and Rust reads.

Python keeps only the scrapers; the binary is the one reader of what they write.
A test that cannot find it fails rather than skips: a contract left untested must
not read as a contract that holds. `make test` builds it first.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _find_binary() -> str | None:
    if os.environ.get("SOLOGRAPH_BIN"):
        return os.environ["SOLOGRAPH_BIN"]
    try:
        meta = subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=ROOT / "rust",
            capture_output=True,
            text=True,
            check=True,
        )
        built = Path(json.loads(meta.stdout)["target_directory"]) / "release" / "solograph"
        if built.exists():
            return str(built)
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError):
        pass
    return shutil.which("solograph")


@pytest.fixture(scope="session")
def solograph_bin() -> str:
    found = _find_binary()
    if not found:
        pytest.fail("solograph binary not found: run `make rust-build` or set SOLOGRAPH_BIN")
    return found


@pytest.fixture
def read(solograph_bin, tmp_path):
    """Run `solograph <args> --json` against the sources index in tmp_path."""

    def run(*args: str):
        out = subprocess.run(
            [solograph_bin, *args, "--json"],
            env={**os.environ, "SOURCES_ROOT": str(tmp_path)},
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(out.stdout)

    return run
