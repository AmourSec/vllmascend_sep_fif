from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_resident_policy_and_source_bounds(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native C++ compiler is unavailable")
    source = Path(__file__).parents[1] / "host/test_qli_resident_policy.cpp"
    executable = tmp_path / "qli_resident_policy"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
