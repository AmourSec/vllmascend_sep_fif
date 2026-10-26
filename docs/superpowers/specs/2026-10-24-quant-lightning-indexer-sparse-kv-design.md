# Quantized indexer and sparse-KV fusion architecture

## Scope

On the Ascend 950 / CANN 9.1.0 GLM-5.2 C8 path, the fused operator performs
quantized lightning-indexer v1 work, consumes each final TopK row in the owning
AIV, and updates SparseKvPlan state without reloading TopK from GM. It retains
the logical TopK GM writeback for IndexCache. Miss KV data is still loaded by
`npu_sparse_kv_transfer` immediately before sparse attention.

The supported route is A5 FP8-indexer decode and speculative decode. Prefill,
mixed-batch debug, non-FP8 indexers, non-A5 paths, and existing fused-overlap
paths retain their prior behavior.

## Device design

Each final row, including short, invalid, and zero-key rows, has exactly one
planner owner. Planning occurs only after final TopK merge and padding, before
that row's GM writeback; intermediate chunks never update planner state. Local
SIMD/SIMT/MTE ordering avoids a global QLI-to-planner barrier. TopK chunking
and tie behavior stay unchanged.

Final merge, score, and temporary storage share a reusable slab after their
SIMD lifetime ends. Final indices remain in their own UB allocation. Planner
scan consumes another 544 bytes. Capacity 2048/4096 uses UB scratch; the
existing 8192 path uses GM scratch. At `T > 64` and capacity 7168, planned
dynamic UB peaks at 212,768 bytes. CANN compiler resource output for static
UB, SIMT stack/spill, and DataCache is mandatory; static accounting is not a
substitute.

## ABI and tensor requirements

`_C_ascend.npu_quant_lightning_indexer_sparse_kv` returns int32 logical indices
`[T, 1, 2048]` and accepts 22 tensor inputs: eight indexer inputs, six planner
metadata inputs, and eight mutable planner-state inputs. Its four scalar
attributes are capacity, max token, block size, and host block count.

- Query: FP8 E4M3FN `[T, 32, 128]`; paged key: FP8 E4M3FN
  `[blocks, 128, 1, 128]`.
- Weights: BF16 `[T, 32]`; query scale: FP32 `[T, 32]`; key scale: FP32
  `[blocks, 128]` or supported trailing-singleton view.
- Request IDs are int64; planner lengths and indices are int32. Mutable state
  preserves storage and PyTorch alias annotations. Validation checks shape,
  dtype, device, and allocation bounds without device-to-host reads.

## Runtime invariants

Planner state is per query-token row; `token_to_req` selects the request and
source page table. It preserves request replacement resets, speculative suffix
invalidation, duplicate-token handling, LRU ordering, and inactive graph-padding
state. The current attention layer transfers a prepared miss list exactly once.
Shared IndexCache layers reuse a prepared plan; MTP layers which reuse TopK after
row compaction run standalone planning. A layer without QLI cannot use fused-plan
state.

## Verification boundary

The differential oracle compares stock QLI plus standalone plan with the fused
result and all mutable state. Coverage includes batch 1/2/4/8/16/32, MTP
1/2/4/6 with ragged rows, contexts near 2048 and through 143360, row counts
near 64/65, capacities 2048/4096/8192 and scratch boundaries, cache
transitions, invalid pages, duplicate selection, graph replay, and invalid
input. BF16-rounded selection tests require a strict post-rounding score gap.

Use the [testing guide](../../quant_lightning_indexer_sparse_kv_testing.md) for
rebuild and suite commands. Current NPU-side changes require a fresh CANN build
and Ascend-950 validation; no historical NPU run is claimed for this checkout.
