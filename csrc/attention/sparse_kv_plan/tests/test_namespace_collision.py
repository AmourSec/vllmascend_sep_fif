from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "csrc/attention/sparse_kv_plan/op_kernel/sparse_kv_plan_config.h"
KERNEL = ROOT / "csrc/attention/sparse_kv_plan/op_kernel/sparse_kv_plan_apt.cpp"
HOST = ROOT / "csrc/attention/sparse_kv_plan/op_host/sparse_kv_plan_tiling.cpp"


def _production_references() -> list[str]:
    kernel = re.findall(r"using\s+([A-Za-z_]\w*::PLAN_[A-Z_]+);", KERNEL.read_text())
    host = re.findall(r"([A-Za-z_]\w*::PLAN_[A-Z_]+)", HOST.read_text())
    assert kernel
    assert host
    references = list(dict.fromkeys(kernel + host))
    assert any(reference.endswith("::PLAN_THREADS") for reference in references)
    return references


def _compile(compiler: str, references: list[str], source_dir: Path) -> subprocess.CompletedProcess[str]:
    declarations = "\n".join(f"constexpr auto k{index} = {name};" for index, name in enumerate(references))
    threads_index = next(index for index, name in enumerate(references) if name.endswith("::PLAN_THREADS"))
    source = source_dir / "namespace_collision.cpp"
    source.write_text(
        f'#include "{CONFIG}"\n'
        "#define sparse_kv_plan sparse_kv_plan_0_tilingkey\n"
        f"{declarations}\n"
        f"static_assert(k{threads_index} == 2048U);\n"
        "int main() { return 0; }\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [compiler, "-std=c++17", "-fsyntax-only", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_generated_tiling_key_macro_does_not_capture_config_namespace(tmp_path: Path) -> None:
    compiler = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the namespace regression")

    result = _compile(compiler, _production_references(), tmp_path)
    assert result.returncode == 0, result.stderr
