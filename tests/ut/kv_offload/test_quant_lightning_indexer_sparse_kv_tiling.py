import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = ROOT / "csrc/attention/quant_lightning_indexer_sparse_kv"
HOST_HEADER = OP_ROOT / "op_host/quant_lightning_indexer_sparse_kv_tiling.h"
TEMPLATE_KEY = OP_ROOT / "op_kernel/quant_lightning_indexer_sparse_kv_template_tiling_key.h"

EXPECTED_FIELDS = (
    ("uint32_t", "bSize", 0),
    ("uint32_t", "tSize", 4),
    ("uint32_t", "n2Size", 8),
    ("uint32_t", "gSize", 12),
    ("uint32_t", "s1Size", 16),
    ("uint32_t", "s2Size", 20),
    ("uint32_t", "sparseCount", 24),
    ("uint32_t", "keyStride0", 28),
    ("uint32_t", "keyDequantScaleStride0", 32),
    ("uint32_t", "usedCoreNum", 36),
    ("uint32_t", "blockSize", 40),
    ("uint32_t", "maxBlockNumPerBatch", 44),
    ("uint32_t", "sparseMode", 48),
    ("uint64_t", "hashCapacity", 56),
    ("uint64_t", "workspaceRowElements", 64),
    ("int64_t", "maxRows", 72),
    ("int64_t", "topk", 80),
    ("int64_t", "capacity", 88),
    ("int64_t", "maxToken", 96),
    ("int64_t", "maxRequests", 104),
    ("int64_t", "maxNumBlocks", 112),
    ("int64_t", "hostNumBlocks", 120),
    ("int32_t", "planBlockSize", 128),
    ("uint32_t", "planLocalMemoryBytes", 132),
)


def _cpp_compiler() -> str:
    compiler = next(
        (path for name in ("clang++", "g++", "c++") if (path := shutil.which(name)) is not None),
        None,
    )
    assert compiler is not None, "no C++17 compiler is available"
    return compiler


def _compile_and_run(source: Path, binary: Path, *include_paths: Path) -> None:
    compile_result = subprocess.run(
        [
            _cpp_compiler(),
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            *(argument for path in include_paths for argument in ("-I", str(path))),
            str(source),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
    assert run_result.returncode == 0, run_result.stderr


def test_host_header_has_no_missing_tiling_util_dependency() -> None:
    source = HOST_HEADER.read_text()

    assert '#include "op_host/tiling_util.h"' not in source


def test_device_tiling_data_matches_host_field_order_and_is_visible_to_template(
    tmp_path: Path,
) -> None:
    host_source = HOST_HEADER.read_text()
    template_source = TEMPLATE_KEY.read_text()
    host_fields = tuple(
        (field_type, field_name)
        for field_type, field_name in re.findall(r"TILING_DATA_FIELD_DEF\((u?int(?:32|64)_t),\s*(\w+)\)", host_source)
    )
    expected_fields = tuple((field_type, field_name) for field_type, field_name, _ in EXPECTED_FIELDS)
    assert host_fields == expected_fields
    assert '#include "quant_lightning_indexer_sparse_kv_tiling_data.h"' in template_source

    host_members = "\n".join(f"    {field_type} {field_name};" for field_type, field_name in host_fields)
    offset_checks = "\n".join(
        "\n".join(
            (
                f"static_assert(offsetof(QuantLightningIndexerSparseKvTilingData, {field_name}) == {offset});",
                f"static_assert(offsetof(HostFieldOrderMirror, {field_name}) == {offset});",
            )
        )
        for _, field_name, offset in EXPECTED_FIELDS
    )
    type_checks = "\n".join(
        f"static_assert(std::is_same_v<decltype(QuantLightningIndexerSparseKvTilingData::{field_name}), {field_type}>);"
        for field_type, field_name, _ in EXPECTED_FIELDS
    )
    template_stub = tmp_path / "ascendc/host_api/tiling/template_argument.h"
    template_stub.parent.mkdir(parents=True)
    template_stub.write_text("#define ASCENDC_TPL_ARGS_DECL(...)\n#define ASCENDC_TPL_SEL(...)\n")
    source = tmp_path / "tiling_abi.cpp"
    source.write_text(
        f"""#include <cstddef>
#include <cstdint>
#include <type_traits>
#include \"{TEMPLATE_KEY}\"

struct alignas(8) HostFieldOrderMirror {{
{host_members}
}};

static_assert(sizeof(QuantLightningIndexerSparseKvTilingData) == sizeof(HostFieldOrderMirror));
static_assert(sizeof(QuantLightningIndexerSparseKvTilingData) == 136);
static_assert(alignof(QuantLightningIndexerSparseKvTilingData) == 8);
{offset_checks}
{type_checks}

int main() {{
    QuantLightningIndexerSparseKvTilingData data{{}};
    data.hashCapacity = 4096;
    data.planLocalMemoryBytes = 544;
    return data.hashCapacity == 4096 && data.planLocalMemoryBytes == 544 ? 0 : 1;
}}
"""
    )
    binary = tmp_path / "tiling_abi"
    _compile_and_run(source, binary, tmp_path)

    source.write_text("#define __CCE_AICORE__ 310\n" + source.read_text())
    _compile_and_run(source, tmp_path / "tiling_abi_aicore310", tmp_path)


def test_fixed_contract_formulas_and_workspace_shapes_execute_real_host_code(tmp_path: Path) -> None:
    source = tmp_path / "fixed_contract.cpp"
    source.write_text(
        f"""#define QLI_SPARSE_KV_FIXED_CONTRACT_ONLY
#include \"{HOST_HEADER}\"

using namespace optiling::qli_sparse_kv_contract;

int main() {{
    const PlannerSizing small = ComputePlannerSizing(4096, 64);
    const PlannerSizing large = ComputePlannerSizing(4096, 65);
    if (small.hashCapacity != 4096 || small.workspaceRowElements != 18432 ||
        small.planLocalMemoryBytes != 74016 || small.dynamicUbBytes != 150816) {{
        return 1;
    }}
    if (large.dynamicUbBytes != 187168) {{
        return 2;
    }}
    const PlannerSizing gm = ComputePlannerSizing(14337, 65);
    if (gm.planLocalMemoryBytes != 288 || gm.dynamicUbBytes != 187168) {{
        return 3;
    }}
    const PlannerSizing ubBoundary = ComputePlannerSizing(14336, 65);
    if (ubBoundary.workspaceRowElements != 28672 || ubBoundary.planLocalMemoryBytes != 114976 ||
        ubBoundary.dynamicUbBytes != 212512) {{
        return 9;
    }}
    if (!CompactWorkspaceShapeIsValid(1, 1, small, 64) ||
        !CompactWorkspaceShapeIsValid(64, 18432, small, 64) ||
        CompactWorkspaceShapeIsValid(1, 2, small, 64) ||
        CompactWorkspaceShapeIsValid(63, 18432, small, 64)) {{
        return 4;
    }}
    if (!CompactWorkspaceShapeIsValid(65, gm.workspaceRowElements, gm, 65) ||
        CompactWorkspaceShapeIsValid(65, gm.workspaceRowElements + 1, gm, 65) ||
        CompactWorkspaceShapeIsValid(1, 1, gm, 65)) {{
        return 5;
    }}
    if (ResolveStride(0, 16384) != 0 || ResolveStride(-1, 16384) != 16384 ||
        ResolveStride(7, 16384) != 7) {{
        return 6;
    }}
    if (AdditionalWorkspaceBytes(128, 2) != 2048 || AdditionalWorkspaceBytes(129, 2) != 4096) {{
        return 7;
    }}
    const int64_t indexerBlockTableWidth = 8;
    const int64_t plannerBlockTableWidth = 17;
    if (indexerBlockTableWidth == plannerBlockTableWidth ||
        !PlannerBlockTableShapeIsValid(2, plannerBlockTableWidth, 2) ||
        PlannerBlockTableShapeIsValid(3, plannerBlockTableWidth, 2) ||
        PlannerBlockTableShapeIsValid(2, 0, 2)) {{
        return 8;
    }}
    return 0;
}}
"""
    )
    _compile_and_run(source, tmp_path / "fixed_contract")


def test_template_key_exposes_only_fixed_950_contract() -> None:
    source = TEMPLATE_KEY.read_text()

    assert "QLI_TPL_HIFLOAT8" not in source
    assert "LI_TPL_IN8" not in source
    assert source.count("ASCENDC_TPL_ARGS_SEL(") == 1
    assert "ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 1)" in source
    assert "ASCENDC_TPL_UINT_SEL(Q_LAYOUT_T, ASCENDC_TPL_UI_LIST, QLI_LAYOUT_TND)" in source
    assert "ASCENDC_TPL_UINT_SEL(K_LAYOUT_T, ASCENDC_TPL_UI_LIST, QLI_LAYOUT_PA_BSND)" in source
