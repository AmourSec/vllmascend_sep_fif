from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_resident_simt_core(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native C++ compiler is unavailable")
    source = Path(__file__).parents[1] / "host/test_qli_resident_simt.cpp"
    shim = source.parent / "simt_shim"
    plan_kernel = Path(__file__).parents[3] / "sparse_kv_plan/op_kernel"
    executable = tmp_path / "qli_resident_simt"
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-D__CCE_AICORE__",
            "-I",
            str(shim),
            "-I",
            str(plan_kernel),
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
