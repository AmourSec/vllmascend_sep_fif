# QuantLightningIndexerSparseKv

`QuantLightningIndexerSparseKv` fuses the Ascend 950 quantized lightning
indexer and the sparse KV LRU planner into one mixed AIC/AIV kernel launch.
Each finalized indexer TopK row is passed directly from UB to the planner on
the producing AIV before the unchanged global-memory index output writeback.

The supported target is CANN 9.1.0 on Ascend 950 with TND query
`[T,32,128]`, PA_BSND key `[blocks,128,1,128]`, FP8 E4M3 query/key,
BF16 weights, FP32 scales, one KV head, `sparse_count=2048`,
`sparse_mode=3`, and quant modes zero. The first eight inputs and first ten
attributes retain the stock QuantLightningIndexer order. Planner inputs and
attributes follow them. The generated ACLNN symbol is
`aclnnQuantLightningIndexerSparseKv`.

The planner callback runs after each row is finalized and padded, and before
the existing index output copy. It also covers masked, zero-length, and
globally empty rows. Empty rows are assigned to one AIV only. Planner
`active_rows` is clamped to query `T`, so inactive LRU rows are not modified.

The planner scan uses a dedicated 544-byte UB buffer. Planner row scratch up
to 112 KiB reuses the dead 89,344-byte TopK slab, growing dynamic UB only by
the excess over that slab. Larger rows use the required persistent
`compact_workspace[max_rows, workspace_width]`, where `max_rows >= T`,
`workspace_width >= 3H+2C+2048`, and `H` is the next power of two at least
`2*2048`. The maximum fused dynamic UB is 212,768 bytes and leaves 41,184
bytes of DataCache under the CANN 9.1.0 mixed SIMD/SIMT cache rule.

## Vendored source

The indexer is copied from
`ops-transformer/attention/quant_lightning_indexer` at revision
`2fd074e77b07ecac2fa3f2daf7901b22c4646a1b`. Its v1 kernel, original TopK
chunks, and minimal transitive headers from `lightning_indexer` and
`lightning_indexer_v2` live under `vendor/`. Original CANN Open Software
License headers are preserved. The fused copy adds per-row planner callbacks,
single-owner invalid-row handling, and shared scratch allocation; scoring,
tiling, trunk selection, reduction, and TopK ordering are unchanged. The
upstream license copied from
`ops-transformer/LICENSE` is included at `vendor/LICENSE` and applies to the
vendored source inventory below.

Vendored inventory:

- `vendor/LICENSE`
- `vendor/quant_lightning_indexer/op_kernel/quant_lightning_indexer_common.h`
- `vendor/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_kernel_arch35.h`
- `vendor/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_service_cube_arch35.h`
- `vendor/quant_lightning_indexer/op_kernel/arch35/quant_lightning_indexer_service_vector_arch35.h`
- `vendor/quant_lightning_indexer/op_kernel/arch35/vf/quant_lightning_indexer_topk.h`
- `vendor/quant_lightning_indexer/op_kernel/arch35/vf/quant_lightning_indexer_vector1.h`
- `vendor/lightning_indexer/op_kernel/lightning_indexer_common.h`
- `vendor/lightning_indexer/op_kernel/arch35/common/lightning_indexer_kernel_base_arch35.h`
- `vendor/lightning_indexer/op_kernel/arch35/vf/vf_topk_gather.h`
- `vendor/lightning_indexer/op_kernel/arch35/vf/common/vf_topk_16_gather.h`
- `vendor/lightning_indexer_v2/op_kernel/arch35/common/lightning_indexer_v2_base_arch35.h`
- `vendor/lightning_indexer_v2/op_kernel/arch35/common/lightning_indexer_v2_kernel_base_arch35.h`
- `vendor/lightning_indexer_v2/op_kernel/arch35/common/vf/lightning_indexer_v2_topk_base.h`
- `vendor/lightning_indexer_v2/op_kernel/arch35/common/vf/lightning_indexer_v2_vector1_base.h`
- `vendor/lightning_indexer_v2/op_kernel/arch35/common/vf/vf_topk_base_v2.h`

The planner implementation is shared with the standalone `SparseKvPlan`
operator through
`csrc/attention/sparse_kv_plan/op_kernel/sparse_kv_plan_shared.h`; the
standalone operator keeps its prior launch, UB fast path, GM fallback, and LRU
semantics.

This checkout does not provide a macOS CANN 9.1 compiler or Ascend 950 device.
Resource compilation and device accuracy/performance must therefore be
validated in the target Linux/CANN environment.
