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
    compiler = next(
        (path for name in ("clang++", "g++", "c++") if (path := shutil.which(name)) is not None),
        None,
    )
    assert compiler is not None, "no C++17 compiler is available"
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(tmp_path),
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
