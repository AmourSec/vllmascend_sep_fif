# GLM-5.2 C8 indexer and sparse KV planning fusion

## Scope and verified baseline

The target is Ascend 950 with CANN 9.1.0. The user confirms the `clean_simt`
branch runs correctly. Preserve its behavior while moving KV resident-buffer
planning and LRU updates into the quantized lightning indexer kernel. Keep DDR
miss transfers immediately before sparse attention.

Baseline vLLM-Ascend revision: `2804765f8e016574dcbda205c8751e6281be24ef`.
The requested history is `994236084d56e92b102e0192023722e40b1d574a` through
`cd84b83c566e62ac26934758cc563b152292f41e`, inclusive. The only later commit
normalizes line endings.

The relevant runtime is `A5DeviceOperator.indexer_select_post_process` in
`vllm_ascend/device/device_op.py`: `torch_npu.npu_quant_lightning_indexer`,
TND query, PA_BSND key, quantization modes 0, sparse mode 3, TopK 2048.
Use upstream `attention/quant_lightning_indexer`, not QLI v2, GLM5next KPool,
or context-parallel compressed indexers. The source snapshot is the sibling
ops-transformer revision `2fd074e77b07ecac2fa3f2daf7901b22c4646a1b`.

[Official GLM-5.2 configuration](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
confirms indexer heads 32, indexer dimension 128, TopK 2048, MLA latent 512,
RoPE 64, and MTP/indexer sharing. The existing A5 runtime uses FP8 E4M3FN
indexer Q/K, BF16 weights, FP32 dequantization scales, and a BF16 main KV
resident buffer. Its main SFA C8 path remains unsupported by offload.

## Architecture decision

Create `QuantLightningIndexerSparseKv`, with one device kernel launch:

1. Execute the original v1 arch35 QLI computation, including its short-sequence
   and invalid-output paths. Preserve its tile sizes and TopK chunking.
2. After the final TopK merge and padding for each row, invoke the existing
   2048-thread SIMT planner on that same AIV, reading its `__ubuf__` indices.
   A row pointer and the flattened output offset identify the input and state.
3. Retain the original TopK GM writeback for IndexCache. Order SIMD, SIMT and
   MTE operations locally; no global QLI-to-planner barrier or TPipe reset is
   needed. Assign invalid rows to exactly one AIV as well.
4. Return original logical TopK indices and mutate the supplied planner state.
   Do not load main KV payloads in this operator.

The planner must never reload the fused TopK input from GM. Intermediate
TopK chunks do not trigger planning. The short-sequence, nonpositive causal
length, mixed zero-key and all-zero-key paths must each emit a complete final
row and run its active-row planner exactly once. Preserve TopK chunk sizes
because changing them can affect tie ordering.

The QLI pipeline remains live during planning. Its ping-pong result and scale
buffers cannot be reused while AIC/MTE may still access them. Check the CANN
resource report for simultaneous UB allocations, static UB, SIMT stack/spill
and DataCache requirements; static inspection cannot substitute for compilation.

The final row's merge, score and TopK temporary buffers share an 89,344-byte
slab. Once the final SIMD work completes, the planner reuses that slab; final
indices stay in their separate UB allocation. Grow the slab when necessary
to preserve the standalone planner's 112 KiB row-workspace threshold, and
allocate a separate 544-byte scan buffer. Larger planner workspaces retain
the existing GM fallback. The next TopK row initializes its own temporary
contents, so no cross-row TopK history is overwritten. The worst dynamic UB
budget is 212,768 bytes, leaving 41,184 bytes for DataCache before additional
compiler resource use under the CANN 9.1 mixed-mode accounting.

## Operator contract

Python operation: `_C_ascend.npu_quant_lightning_indexer_sparse_kv`.

Inputs, in order:

1. `query`, `key`, `weights`, `query_dequant_scale`, `key_dequant_scale`,
   `actual_seq_lengths_query`, `actual_seq_lengths_key`, `indexer_block_table`.
2. `req_ids`, `stable_prefix_lens`, `visible_seq_lens`, `token_to_req`,
   `kv_block_table`, `active_rows`.
3. Mutable `last_req_ids`, `slot_to_token`, `lru_slots`, `current_slots`,
   `miss_count`, `miss_tokens`, `miss_slots`, `compact_workspace`.
4. Integers `capacity`, `max_token`, `block_size`, `host_num_blocks`.

The output is int32 `[T, 1, 2048]`. Q is FP8 `[T, 32, 128]`, key is paged
FP8 `[blocks, block_size, 1, 128]`, weights and Q scales are `[T, 32]`.
Normalize key scales at the binding boundary as needed by the upstream
operator; preserve supported page-strided cache views.

Request IDs are int64, planner indices/lengths are int32. Mutable tensors
must retain their storage and have correct PyTorch alias annotations. Validate
shape, dtype, device and allocation bounds before launch without reading
device values back to the host. Keep indexer and host-KV block tables distinct.

Expose the existing `SparseKvPlan` as a standalone Python operation for
differential tests; retain its existing combined plan/transfer API.

## Runtime invariants

- Fuse only the supported A5 quantized decode/speculative-decode path.
- Initialize active-row and causal-length metadata before invoking QLI.
- Plan state remains per query-token row, not per request. `token_to_req`
  selects the request and its source block table.
- Preserve request replacement resets, speculative-suffix invalidation,
  duplicate-token behavior, and LRU order: unused evictable slots, new misses,
  then old hits.
- Preserve inactive graph-padding state. Capture/replay must not depend on
  stale Python markers from an earlier forward.
- The current layer's attention path transfers the already prepared miss
  list exactly once, then runs attention.
- Backbone IndexCache layers continue reusing the plan. MTP layers that reuse
  TopK after row compaction retain standalone planning. A layer that did not
  execute a quantized indexer cannot use its fused planner.
- BF16 fallback, mixed prefill debug, non-A5 and fused-overlap paths retain
  their existing behavior. Do not add host synchronization or environment flags.

## Verification and acceptance

Author tests before claiming correctness. Compare all state tensors against
an independent CPU planner and sequential stock QLI plus existing Plan.
Use deterministic inputs with clear score margins for exact TopK checks and
tie-aware score/set checks where the stock operator does not define tie order.

Cover batch 1, 2, 4, 8, 16, 32; ordinary decode and MTP spans 2, 4, 6;
ragged spans; query row counts around 64; sequence lengths below/at/above
2048, TopK chunk boundaries, 16k, 64k, 128k, 140000 and 143360; nonaligned
lengths and graph padding. Include mandatory batch-32 long-context MTP cases.
Test capacities 2048, 4096 and 8192, including both standalone planner scratch
branches and the fused planner's selected scratch allocation.

Stateful scenarios include all miss, all hit, eviction, request replacement,
speculative rollback, row compaction, reused plans, invalid source pages,
duplicate selection/residency, repeated graph replay and invalid input shapes.
Check transfer ordering separately from the fused arithmetic/state result.

The delivery includes a 950 test launcher which fails on missing prerequisites,
records environment details and supports correctness suites and optional
profiling. On macOS perform available static checks and CPU-only validation;
report CANN compilation and NPU execution as unverified until run on 950.
Do not claim latency improvements without hardware measurements.

## Repository and external evidence

Vendor only necessary upstream QLI/helper sources into the vLLM-Ascend tree,
with original license notices and source revision. The pushed checkout must
build without the sibling ops-transformer directory.

Relevant external guidance:

- [CANN 9.1 SetDynUBufSize](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/maintenref/basicdataapi/atlasopapi_07_00786.html)
- [CANN 9.1 SIMD/SIMT programming model](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMT-API/SIMD_SIMT_hybrid_programming_intro/programming_model_140.md)
- [aikg architecture capability skill](https://github.com/AmourSec/aikg/blob/master/skills/npu-arch-capability-check/SKILL.md)
