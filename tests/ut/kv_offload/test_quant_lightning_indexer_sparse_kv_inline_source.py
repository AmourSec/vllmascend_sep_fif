import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "csrc/attention/quant_lightning_indexer_sparse_kv/op_kernel/quant_lightning_indexer_sparse_kv.cpp"
VECTOR = (
    ROOT
    / "csrc/attention/quant_lightning_indexer_sparse_kv/vendor/quant_lightning_indexer/op_kernel/arch35"
    / "quant_lightning_indexer_service_vector_arch35.h"
)
ARCH = VECTOR.with_name("quant_lightning_indexer_kernel_arch35.h")
TILING = (
    ROOT / "csrc/attention/quant_lightning_indexer_sparse_kv/op_host" / "quant_lightning_indexer_sparse_kv_tiling.cpp"
)
PLAN_CONFIG = ROOT / "csrc/attention/sparse_kv_plan/op_kernel/sparse_kv_plan_config.h"


def _constant(source: str, name: str) -> int:
    match = re.search(rf"constexpr uint32_t {name} = (\d+)U", source)
    assert match is not None, f"missing {name}"
    return int(match.group(1))


def _workspace_row_bytes(capacity: int) -> int:
    topk = 2048
    hash_capacity = 4096
    return (3 * hash_capacity + 2 * capacity + topk) * 4


def test_final_topk_ub_is_consumed_inline_without_global_planner_phase() -> None:
    entry = ENTRY.read_text()
    vector = VECTOR.read_text()
    process_topk = vector[vector.index("QLIVector<QLIT>::ProcessTopK") :]

    assert entry.count("qli.Process();") == 1
    assert "AscendC::SyncAll" not in entry
    assert "pipe.Reset" not in entry
    assert "SparseKvPlanRuntimeVf" not in entry
    assert "LocalTensor<int32_t> topkIndices" in vector
    assert vector.count("reinterpret_cast<__ubuf__ int32_t*>(topkIndices.GetPhyAddr())") == 2
    assert "SparseKvPlanRuntimeRowVf<__ubuf__ int32_t*, __ubuf__ int32_t*>" in vector
    assert "SparseKvPlanRuntimeRowVf<__ubuf__ int32_t*, __gm__ int32_t*>" in vector
    assert process_topk.rfind("ProcessSparseKvPlanRow(") < process_topk.rfind("DataCopyPad(indiceOutGm[")


def test_invalid_rows_are_partitioned_once_and_plan_before_writeback() -> None:
    arch = ARCH.read_text()
    vector = VECTOR.read_text()
    invalid_dispatch = arch[arch.index("QuantLightningIndexerKernel<QLIT>::ProcessInvalid()") :]
    mixed_zero = arch[
        arch.index("QuantLightningIndexerKernel<QLIT>::DealActSeqLenIsZero") : arch.index(
            "QuantLightningIndexerKernel<QLIT>::InitSparseKvPlan"
        )
    ]
    invalid_write = vector[
        vector.index("QLIVector<QLIT>::CleanInvalidOutput") : vector.index("QLIVector<QLIT>::ProcessInvalidRow")
    ]

    assert "if ASCEND_IS_AIC" in mixed_zero
    assert "if ((tmpBlockIdx & 1U) != 0U)" in mixed_zero
    assert "if ASCEND_IS_AIV" in invalid_dispatch
    assert "aivCoreNum = GetBlockNum() * 2" in invalid_dispatch
    assert "row = tmpBlockIdx; row < totalRows; row += aivCoreNum" in invalid_dispatch
    assert invalid_write.count("ProcessSparseKvPlanRow(") == 1
    assert invalid_write.index("ProcessSparseKvPlanRow(") < invalid_write.index("DataCopyPad(indiceOutGm[")


def test_scan_and_dead_topk_slab_resources_cover_ub_and_gm_scratch() -> None:
    vector = VECTOR.read_text()
    tiling = TILING.read_text()
    plan_config = PLAN_CONFIG.read_text()

    threads = _constant(plan_config, "PLAN_THREADS")
    warp_size = _constant(plan_config, "PLAN_WARP_SIZE")
    control_elements = _constant(plan_config, "PLAN_CONTROL_ELEMENTS")
    scan_bytes = ((2 * (threads // warp_size) + control_elements) * 4 + 31) & ~31
    assert scan_bytes == 544
    assert _constant(tiling, "PLAN_SCAN_UB_BYTES") == scan_bytes
    assert "InitBuffer(plannerScanBuf_, sparse_kv_plan::PLAN_SCAN_BYTES)" in vector
    assert "plannerScratchBytes = topkSlabBytes > plannerRowUbBytes_ ? topkSlabBytes : plannerRowUbBytes_" in vector
    assert "reinterpret_cast<__ubuf__ int32_t*>(mrgValueLocal_.GetPhyAddr())" in vector
    assert "plannerContext_.compactWorkspace + static_cast<uint64_t>(row)" in vector
    assert "alignedRowBytes <= sparse_kv_plan::PLAN_ROW_UB_LIMIT_BYTES" in vector


def test_capacity_boundary_and_total_t_resource_counts_match_architecture() -> None:
    tiling = TILING.read_text()
    scan = _constant(tiling, "PLAN_SCAN_UB_BYTES")
    slab = _constant(tiling, "QLI_TOPK_SLAB_BYTES")
    small_t = _constant(tiling, "QLI_UB_SMALL_T_BYTES")
    large_t = _constant(tiling, "QLI_UB_LARGE_T_BYTES")

    assert slab == 89_344
    expected = {
        2048: (151_072, 187_424),
        4096: (151_840, 188_192),
        7168: (176_416, 212_768),
        7169: (151_072, 187_424),
        8192: (151_072, 187_424),
    }
    for capacity, totals in expected.items():
        row_bytes = _workspace_row_bytes(capacity)
        planner_ub = row_bytes if row_bytes <= 112 * 1024 else 0
        extra = max(planner_ub - slab, 0)
        assert (small_t + scan + extra, large_t + scan + extra) == totals

    assert _workspace_row_bytes(7168) == 112 * 1024
    assert _workspace_row_bytes(7169) > 112 * 1024
    assert "tilingInfo->tSize > 64U ? QLI_UB_LARGE_T_BYTES : QLI_UB_SMALL_T_BYTES" in tiling
    assert "plannerRowUbBytes > QLI_TOPK_SLAB_BYTES" in tiling
