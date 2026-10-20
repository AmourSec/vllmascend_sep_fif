from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_mtp_request_mapping_and_causal_lengths(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native C++ compiler is unavailable")
    source = Path(__file__).parents[1] / "host/query_rows_probe.cpp"
    executable = tmp_path / "query_rows_probe"
    compile_result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    run_result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=10, check=False
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
