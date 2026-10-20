from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_host_contract_rejects_invalid_modes_and_overflow(tmp_path: Path) -> None:
    # Given production mode, geometry, workspace and UB checks without CANN dependencies.
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A host C++ compiler is required for host contract tests")
    source = Path(__file__).resolve().parents[1] / "host/host_contract_test.cpp"
    executable = tmp_path / "host_contract_test"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    # When the native boundary suite runs, then every assertion must pass.
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
