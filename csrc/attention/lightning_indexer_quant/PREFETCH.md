# Ascend 950 QLI decode and MTP KV prefetch

This implementation extends `LightningIndexerQuant` and exposes
`torch.ops._C_ascend.npu_lightning_indexer_quant_prefetch` and
`torch.ops._C_ascend.npu_lightning_indexer_quant_prefetch_resident`. It does not change
the model, scheduler, offload manager, or attention integration. Build and test
instructions are in [tests/pytest/README.md](tests/pytest/README.md).

## Algorithm and scope

The algorithm adapts the decode intra-query prefetch in
[ECHO, OSDI 2026, section 5.1](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf):

1. Score every visible key. After each 128-key tile, prefetch rows whose score
   is strictly greater than the input predicted threshold. Take the first
   `prefetch_budget` candidates in logical token order.
2. Independently retain exact FP32 TopK through bounded streaming selection.
   Speculative admission does not prune scores or alter TopK selection.
3. In standalone mode, gather the selected rows from staging when present;
   otherwise fetch the original row. In resident mode, copy only candidate
   misses into the caller's K/V cache and update its LRU metadata.
4. Return the next threshold as `ema_decay * old + (1 - ema_decay) * kth_score`.
   A non-finite old threshold is replaced by the actual kth score. A request
   with fewer than K visible keys preserves its old threshold.

TopK order and the choice among equal boundary scores are unspecified. Indices
are logical token positions, never physical block IDs. Unused index entries
are `-1`; unused gathered rows are zero. A NaN threshold admits no candidates,
`+inf` admits no finite score, and `-inf` admits all finite scores until budget
is exhausted. With budget zero, standalone mode executes the same exact
score/TopK/gather path with all valid selected rows counted as fallback copies.
Resident mode still computes TopK and maintains row metadata but performs no
candidate payload transfers.

The standalone call does not implement prefill inter-query prediction or a
persistent LRU cache. Its workspace is private to the call. The caller carries
`next_threshold` to the next invocation for the same request and layer.

## 950 execution

One AIC and its paired AIVs handle each query row, with strided row assignment
when the query count exceeds the physical AIC count. Each MTP query uses its
request's page table and its own causal visible length. The existing local QLI-V2
Cube service supplies QK tiles directly to AIV UB using 950 FIX dual destination
and mode-4 events. AIV0 computes scores and copies KV rows; AIV1 acknowledges
its unused half of the QK tile. The AIV returns the QK-buffer credit before
starting candidate copies, allowing the next Cube tile to overlap those copies.

Scores retain FP32 resolution for selection; they are not converted to BF16.
The INT8 numeric order is INT32 dot, divide by 1024, FP16 RINT round-trip, ReLU,
multiply by FP16-rounded `weight * query_scale`, sequential FP32 head sum,
then multiply by the FP16 key scale converted to FP32. The FP8 E4M3FN path
uses FP32 dot products, ReLU, FP32 `weight * query_scale`, sequential FP32 head
sum, and FP32 key scales, without the INT8 division or FP16 round-trips. Inputs must produce
finite scores. All input heads (32 or 64) participate. TopK uses the existing FP32 radix
selector with ordered float-bit keys and bounded UB buffers.

In standalone mode, candidate IDs record admitted rows while DMA is pending.
The final gather drains both copy banks before reading HBM staging. Candidate IDs are increasing, so
final lookup uses binary search. There is no eviction or cross-core shared slot
state. Request staging strides are padded to 512 bytes. Workspace scales as
`T * align512(max(1,budget) * align32(row_bytes))`, plus runtime workspace.

This is a functional first version, not a demonstrated speedup. Small batches
use only a small number of Cube cores. Scalar candidate scanning, serial row
DMA, TopK compaction, and the extra HBM staging copy can outweigh overlap.
Measure budget zero and enabled budgets on the target server before tuning
core assignment, transfer batching, or integrating with attention.

## Input and lifetime contract

- Q: contiguous `[T,H,128]`, H is 32 or 64, TND. `T` is the total number of
  query rows; `N` is the number of requests. A request may have multiple queries.
- K: contiguous `[physical_blocks,block_size,1,128]`, PA_BSND, with the same dtype as Q.
  Block size is a positive multiple of 16, at most 1024.
- Q/K dtype is INT8 with FP16 weights/scales, or FP8 E4M3FN with FP32 weights/scales.
  Mixed dtype combinations and other FP8 formats are rejected.
- Weights and Q scales: `[T,H]`; K scales: `[physical_blocks,block_size,1]`.
- Query lengths: INT32 nondecreasing cumulative TND ends `[N]`, ending at `T`
  in standalone mode. Equal adjacent ends represent empty requests.
  Key lengths: INT32 per-request final lengths `[N]`, not cumulative.
  For request `b` with `q` queries, local query `j` sees `key_length[b] - q + j + 1`
  tokens. For example, query ends `[2,3]` and key lengths `[5,7]` produce
  visible lengths `[4,5,7]`. A single-query empty-KV row remains supported.
- Block table: INT32 `[N,max_blocks]`. Only pages covering the visible key
  length need valid physical block IDs.
- Addresses: INT64 `[N,max_blocks * block_size]`, stored on the NPU. Each valid
  element points to a contiguous `kv_row_bytes` payload row in HBM or mapped
  host memory. Payload is independent of the indexer's quantized K tensor.
- Thresholds: FLOAT32 `[T]`, stored on the NPU. All tensor inputs are on the
  same NPU. K is 1..2048, budget is 0..2048, row bytes is 1..4096, and EMA decay
  is finite in `[0,1)`.

## Resident cache mode

The resident entry point receives the existing offload manager's K/V buffers
and the same per-layer `last_req_ids`, `slot_to_token` and `lru_slots` tensors.
It mutates them in place. It does not allocate a second persistent cache.
The Python runtime is not wired to this entry point by this change; a caller
or the integration tests must supply these existing buffers.

```python
sparse_indices, candidates, stats, next_threshold = (
    torch.ops._C_ascend.npu_lightning_indexer_quant_prefetch_resident(
        query, key, weights, query_scale, key_scale,
        cumulative_query_lengths, final_key_lengths, indexer_block_table,
        threshold, req_ids, token_to_req, stable_prefix_lens,
        host_block_table, host_cache_bases, resident_k, resident_v,
        last_req_ids, slot_to_token, lru_slots, active_rows,
        host_block_size, host_num_blocks, prefetch_budget, sparse_count, ema_decay,
    )
)
```

- `req_ids` is INT64 `[N]`; `token_to_req` and `stable_prefix_lens` are INT32
  `[T]`. The row-to-request map must agree with cumulative query lengths.
- `active_rows` is an INT32 device tensor `[1]`. The final cumulative query
  endpoint equals its value and may be smaller than the allocated query count.
  Inactive rows keep persistent state unchanged, return `-1` indices and zero
  statistics, and preserve their threshold.
- Resident K/V are contiguous byte tensors `[R,C,Kbytes]` / `[R,C,Vbytes]`,
  or native contiguous `[R,C,1,D]` tensors reinterpreted as byte views without
  copying. `R >= T`; each query row owns its own cache row, including MTP rows
  belonging to the same request. K/V widths may differ and are each 1..4096 bytes.
- `last_req_ids` is INT64 `[R]`; `slot_to_token` and `lru_slots` are INT32
  `[R,C]`. LRU rows are permutations of `[0,C)` ordered oldest to newest.
  Supply the manager's initialized metadata, not uninitialized tensors.
- `host_block_table` is INT32 `[N,host_max_blocks]`, separate from the indexer
  page table. `host_cache_bases` is INT64 `[2]` containing NPU-accessible K/V
  source base addresses. On 950 DDR sources must use mapped device addresses.
  Source layout is contiguous `[host_num_blocks,host_block_size,row_bytes]`
  separately for K and V. The host and indexer block sizes may differ.
- Budget is 0..min(2048,C); TopK is 1..min(2048,C). Resident statistics are
  `[admitted_candidates, transferred_misses]`; standalone statistics retain
  `[admitted_candidates, final_gather_fallbacks]`.

Before prefetch, a changed request identity resets the cache row. Existing
tokens outside the stable prefix are invalidated once, following SparseKvPlan.
Tail tokens can still be candidates and be fetched again in this invocation.
For each score tile, hits are protected before choosing victims for misses.
All admitted slots stay protected through the call, including pending copies.
Victims follow the original LRU order; at completion the order is unused
slots, newly fetched misses, then original hits, preserving order within each
group. A future candidate may already have been evicted by an earlier tile and
need another fetch. Streaming admission cannot know the final candidate set in
advance and is not equivalent to one batch Plan over that entire set.

Both K and V must finish copying before a newly filled slot is published as
valid. Metadata and payload writes finish before the operator returns. Run the
unchanged post-TopK Plan/Transfer on the same stream, or establish an equivalent
dependency. It computes final `current_slots` and fetches remaining misses.
It may invalidate and fetch mutable-tail tokens again; that duplicate work is
intentionally retained. This API does not gather TopK bytes into a separate
output tensor.

Resident workspace is `T * align512(3*C*4)` bytes plus runtime slack. Capacity
dependent policy state stays in workspace rather than consuming capacity-sized
UB allocations.

The GLM-5.2 profile uses 32 indexer heads, head dimension 128, and TopK=2048,
matching the [model configuration](https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json).
The model's 64 attention heads are not its indexer head count. The standalone
operator accepts 32 heads directly; callers do not need to pad to 64.
The standalone interface selects INT8 or ordinary FP8 E4M3FN computation from
the Q/K dtype. FP8 registration is restricted to Ascend 950. The dtype support
does not connect this operator to the model's A5 indexer path, and does not
cover MXFP8 or HiFloat8.

Tests use synthetic per-row quantized inputs with these dimensions, including
per-head Q scales and per-token K scales. Both dtypes cover TopK=2048 with
lengths 2047, 2048, and 4097, two fixed seeds, and zero/enabled budgets.
A 1152-byte payload case represents an unpacked BF16
MLA row with 512 latent and 64 RoPE elements: `(512 + 64) * 2`. The operator
copies opaque bytes; actual cache packing, quantization metadata, and row width
remain the caller's responsibility. These tests do not reproduce trained-model
activation distributions or measure model quality or realistic prefetch hit rates.

Host rows must be allocated and mapped through MemFabric's supported allocator.
Use `offload_get_dva`/`get_dva`, not the numerical value of an arbitrary CPU or
pinned-memory pointer. Keep all source allocations, mappings, and the MemFabric
pool alive until the NPU stream completes, including every graph replay. Do not
mutate sources concurrently with the operator. The address tensor cannot keep
its pointed-to allocations alive automatically.

The host-DVA transfer mechanism follows the official MemFabric release/1.2
[sparse-copy implementation](https://gitcode.com/Ascend/memfabric_hybrid/blob/8bdd71a17eb691391442e3311ae56a296b2dcd55/src/acc_offload/csrc/operators/acc_offload_sparse_copy.h)
and [mapping API](https://gitcode.com/Ascend/memfabric_hybrid/blob/8bdd71a17eb691391442e3311ae56a296b2dcd55/src/acc_offload/include/host/acc_offload.h).
The architecture reference was AIKG's
[950 capability skill](https://github.com/AmourSec/aikg/blob/6d40da6a90b99b20b035527583ffbca2c74952ba/skills/npu-arch-capability-check/SKILL.md).
The unverified scurny prefetch branch was not used as implementation code.

Invalid device query ends, key lengths, or referenced physical block IDs produce
an empty padded result for that query row, `stats=[-1,-1]`, and an unchanged
threshold. Pointer validity remains the caller's responsibility. Inspect stats
when using the raw standalone interface.

## Build compatibility and verification status

Relative to standalone prefetch, the generated ACLNN ABI appends eleven
optional resident inputs and two host-layout attributes. Its five outputs
remain unchanged; the resident Torch API hides the internal dummy gather output
and returns four tensors. Rebuild and install both the custom OPP package and
the PyTorch extension together; an old generated ACLNN library is incompatible.
The original Python operator signature is retained for 910B/910_93 through its
adapter. Both prefetch APIs require Ascend 950.

Development and review were performed on macOS without CANN headers, a device
compiler, or an NPU. Local Python syntax/reference checks and source review do
not establish kernel compilation, DMA correctness, graph support, or performance
on hardware. Run both the HBM and MemFabric host suites on the 950 server before
using this branch in a model.
