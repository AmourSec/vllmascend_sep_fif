from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "csrc/attention/sparse_kv_plan/op_kernel/sparse_kv_plan_config.h"
OP_DEFINITION = ROOT / "csrc/attention/sparse_kv_plan/op_host/sparse_kv_plan_def.cpp"
CONSUMERS = (
    ROOT / "csrc/attention/sparse_kv_plan/op_kernel/sparse_kv_plan_shared.h",
    ROOT / "csrc/attention/sparse_kv_plan/op_host/sparse_kv_plan_tiling.cpp",
    ROOT / "csrc/attention/quant_lightning_indexer_sparse_kv/op_host/quant_lightning_indexer_sparse_kv_tiling.cpp",
    ROOT
    / "csrc/attention/quant_lightning_indexer_sparse_kv/vendor/quant_lightning_indexer/op_kernel/arch35"
    / "quant_lightning_indexer_service_vector_arch35.h",
)


def _production_references() -> list[str]:
    references: list[str] = []
    for path in CONSUMERS:
        matches = re.findall(r"\b([A-Za-z_]\w*::PLAN_[A-Z_]+)\b", path.read_text())
        assert matches, f"no planner config references found in {path}"
        references.extend(matches)
    assert any(reference.endswith("::PLAN_THREADS") for reference in references)
    return list(dict.fromkeys(references))


@pytest.mark.parametrize("macro_before_include", [False, True])
def test_generated_tiling_key_macro_does_not_capture_config_namespace(
    tmp_path: Path, macro_before_include: bool
) -> None:
    compiler = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("a host C++ compiler is required for the namespace regression")
    references = _production_references()
    declarations = "\n".join(f"constexpr auto k{index} = {name};" for index, name in enumerate(references))
    threads_index = next(index for index, name in enumerate(references) if name.endswith("::PLAN_THREADS"))
    include = f'#include "{CONFIG}"\n'
    macro = "#define sparse_kv_plan sparse_kv_plan_0_tilingkey\n"
    source = tmp_path / "namespace_collision.cpp"
    source.write_text(
        (macro + include if macro_before_include else include + macro)
        + f"{declarations}\n"
        + f"static_assert(k{threads_index} == 1024U);\n"
        + 'extern "C" void sparse_kv_plan() {}\n'
        + "int main() { sparse_kv_plan(); return 0; }\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [compiler, "-std=c++17", "-Werror", "-fsyntax-only", str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_mutable_plan_inputs_preserve_original_strides_for_validation() -> None:
    source = OP_DEFINITION.read_text()
    mutable_inputs = set(re.findall(r'AddMutableInt(?:32|64)Input\(\*this, "([^"]+)"\)', source))

    assert mutable_inputs == {
        "lastReqIds",
        "slotToToken",
        "lruSlots",
        "currentSlots",
        "missCount",
        "missTokens",
        "missSlots",
        "compactWorkspace",
    }
    assert source.count(".IgnoreContiguous();") == 2
