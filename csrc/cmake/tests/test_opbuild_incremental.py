from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

CMAKE = "cmake"
LANES = {
    "default": ("op_host_aclnn", "empty", "generate_aclnn_srcs"),
    "inner": ("op_host_aclnnInner", "inner", "generate_aclnn_inner_srcs"),
    "exc": ("op_host_aclnnExc", "exc", "generate_exclude_proto_srcs"),
}
PRODUCTION_FILES = (
    Path(__file__).parents[2] / "CMakeLists.txt",
    Path(__file__).parents[1] / "custom_build.cmake",
)


@pytest.mark.parametrize("production_file", PRODUCTION_FILES, ids=lambda path: path.name)
@pytest.mark.parametrize("lane", LANES)
def test_opbuild_regenerates_when_host_library_changes(tmp_path: Path, production_file: Path, lane: str) -> None:
    target, output_dir, output_variable = LANES[lane]
    project = tmp_path / "project"
    project.mkdir()
    source = project / "probe.c"
    source.write_text("volatile int generated_marker = 1;\n", encoding="utf-8")
    block = _production_command(production_file, output_variable)
    (project / "CMakeLists.txt").write_text(_cmake_project(block, target, output_dir), encoding="utf-8")

    build = tmp_path / "build"
    _cmake("-S", str(project), "-B", str(build), "-G", "Unix Makefiles")
    _cmake("--build", str(build), "--target", f"opbuild_gen_{output_dir}")
    generated = build / "generated" / output_dir / "probe.bin"
    before = generated.read_bytes()
    before_mtime = generated.stat().st_mtime_ns

    _cmake("--build", str(build), "--target", f"opbuild_gen_{output_dir}")
    assert generated.stat().st_mtime_ns == before_mtime

    time.sleep(1.05)
    source.write_text("volatile int generated_marker = 2;\n", encoding="utf-8")
    _cmake("--build", str(build), "--target", f"opbuild_gen_{output_dir}")

    host_library = build / "probe.bin"
    assert generated.read_bytes() == host_library.read_bytes()
    assert generated.read_bytes() != before
    after_mtime = generated.stat().st_mtime_ns
    _cmake("--build", str(build), "--target", f"opbuild_gen_{output_dir}")
    assert generated.stat().st_mtime_ns == after_mtime


def _production_command(production_file: Path, output_variable: str) -> str:
    source = production_file.read_text(encoding="utf-8")
    marker = f"add_custom_command(OUTPUT ${{{output_variable}}}"
    start = source.index(marker)
    end = source.index("\n        )", start) + len("\n        )")
    return source[start:end]


def _cmake_project(block: str, target: str, output_dir: str) -> str:
    base_dir = "empty" if output_dir == "empty" else ""
    base_path = f"${{CMAKE_BINARY_DIR}}/generated/{base_dir}".rstrip("/")
    return (
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(opbuild_incremental C)\n\n"
        f"add_library({target} SHARED probe.c)\n"
        f'set_target_properties({target} PROPERTIES OUTPUT_NAME probe PREFIX "" SUFFIX ".bin")\n'
        f'set(base_aclnn_binary_dir "{base_path}")\n'
        f'set(generate_aclnn_srcs "${{CMAKE_BINARY_DIR}}/generated/{output_dir}/probe.bin")\n'
        'set(generate_aclnn_headers "")\n'
        f'set(generate_aclnn_inner_srcs "${{CMAKE_BINARY_DIR}}/generated/{output_dir}/probe.bin")\n'
        f'set(generate_exclude_proto_srcs "${{CMAKE_BINARY_DIR}}/generated/{output_dir}/probe.bin")\n'
        'set(OP_BUILD_TOOL "${CMAKE_COMMAND};-E;copy")\n'
        'set(OPS_PRODUCT_NAME "")\n'
        f"{block}\n"
        f"add_custom_target(opbuild_gen_{output_dir} DEPENDS "
        f"${{CMAKE_BINARY_DIR}}/generated/{output_dir}/probe.bin {target})\n"
    )


def _cmake(*args: str) -> None:
    result = subprocess.run((CMAKE, *args), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
