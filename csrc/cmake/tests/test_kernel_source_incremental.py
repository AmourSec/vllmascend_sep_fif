from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parents[3]
FUNCTIONS = REPOSITORY / "csrc/cmake/func.cmake"


@pytest.mark.parametrize(
    ("changed_path", "expected_output"),
    (
        ("attention/probe/op_kernel/template.h", "operator-marker-v2"),
        ("attention/helper/op_kernel/helper.h", "dependency-marker-v2"),
        ("generated/dynamic/probe.py", "wrapper-marker-v2"),
        ("attention/probe/extra/added.h", "added-marker-v2"),
    ),
    ids=("operator-header", "dependency-header", "generated-wrapper", "new-source-directory"),
)
def test_kernel_sources_are_rebuilt_when_an_input_changes(
    tmp_path: Path, changed_path: str, expected_output: str
) -> None:
    project = tmp_path / "project"
    _write_project(project)
    build = tmp_path / "build"

    _cmake("-S", str(project), "-B", str(build), "-G", "Unix Makefiles")
    _build(build)
    output = project / "binary" / "ascend950" / "bin" / "probe"
    copied_header = project / "binary" / "ascend950" / "src" / "probe" / "op_kernel" / "template.h"
    before = {index: (output / f"expanded_{index}.txt").read_text(encoding="utf-8") for index in (0, 1)}
    before_wrapper = {index: (output / f"wrapper_{index}.py").read_text(encoding="utf-8") for index in (0, 1)}
    before_mtimes = {path: path.stat().st_mtime_ns for path in output.iterdir()}
    _build(build)
    assert before_mtimes == {path: path.stat().st_mtime_ns for path in output.iterdir()}

    time.sleep(1.05)
    (project / changed_path).parent.mkdir(parents=True, exist_ok=True)
    (project / changed_path).write_text(expected_output + "\n", encoding="utf-8")
    _build(build)

    if changed_path.endswith("template.h"):
        assert copied_header.read_text(encoding="utf-8").strip() == expected_output
        assert all((output / f"expanded_{index}.txt").read_text(encoding="utf-8") != before[index] for index in (0, 1))
    elif changed_path.endswith(".h"):
        assert all(
            expected_output in (output / f"expanded_{index}.txt").read_text(encoding="utf-8") for index in (0, 1)
        )
        assert all((output / f"expanded_{index}.txt").read_text(encoding="utf-8") != before[index] for index in (0, 1))
    else:
        assert all(
            (output / f"wrapper_{index}.py").read_text(encoding="utf-8").strip() == expected_output for index in (0, 1)
        )
        assert all(
            (output / f"wrapper_{index}.py").read_text(encoding="utf-8") != before_wrapper[index] for index in (0, 1)
        )

    after_mtimes = {path: path.stat().st_mtime_ns for path in output.iterdir()}
    _build(build)
    assert after_mtimes == {path: path.stat().st_mtime_ns for path in output.iterdir()}


def _write_project(project: Path) -> None:
    (project / "attention/probe/op_kernel").mkdir(parents=True)
    (project / "attention/helper/op_kernel").mkdir(parents=True)
    (project / "generated/dynamic").mkdir(parents=True)
    (project / "attention/probe/op_kernel/template.h").write_text(
        '#include "../../helper/op_kernel/helper.h"\n'
        '#if __has_include("../extra/added.h")\n#include "../extra/added.h"\n#endif\n'
        "operator-marker-v1\n",
        encoding="utf-8",
    )
    (project / "attention/helper/op_kernel/helper.h").write_text("dependency-marker-v1\n", encoding="utf-8")
    (project / "generated/dynamic/probe.py").write_text("wrapper-marker-v1\n", encoding="utf-8")
    for index in (0, 1):
        script = project / f"binary/ascend950/gen/dynamic-probe-{index}.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            'src=$(dirname "$1")\n'
            "out=$2\n"
            f'c++ -E -P "$src/op_kernel/template.h" > "$out/expanded_{index}.txt"\n'
            f'cp "$1" "$out/wrapper_{index}.py"\n',
            encoding="utf-8",
        )
        script.chmod(0o755)
    cmake_lines = (
        "cmake_minimum_required(VERSION 3.16)",
        "project(kernel_source_incremental C)",
        'set(ASCEND_BINARY_OUT_DIR "${CMAKE_SOURCE_DIR}/binary")',
        'set(ASCEND_IMPL_OUT_DIR "${CMAKE_SOURCE_DIR}/generated")',
        'set(ASCENDC_CMAKE_UTIL_DIR "${CMAKE_SOURCE_DIR}")',
        'set(OPS_ADV_UTILS_KERNEL_INC "${CMAKE_SOURCE_DIR}/missing")',
        "set(BUILD_OPS_RTY_KERNEL OFF)",
        "set(ENABLE_BUILT_IN OFF)",
        "set(BUILD_OPEN_PROJECT ON)",
        "set(ENABLE_OPS_HOST OFF)",
        "set(ENABLE_HOST_TILING OFF)",
        "set(VENDOR_NAME custom_transformer)",
        "set(ASCEND_OP_NAME all)",
        "add_custom_target(ops_transformer_kernel)",
        "add_custom_target(ops_transformer_config)",
        f'include("{FUNCTIONS}")',
        "set(probe_depends attention/helper)",
        'add_bin_compile_target(COMPUTE_UNIT ascend950 OP_INFO "${CMAKE_SOURCE_DIR}/attention/probe")',
    )
    (project / "CMakeLists.txt").write_text("\n".join(cmake_lines) + "\n", encoding="utf-8")


def _cmake(*args: str) -> None:
    result = subprocess.run(("cmake", *args), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _build(build: Path) -> None:
    _cmake("--build", str(build), "--target", "probe_ascend950")
