# Ascend 950 standalone decode KV prefetch

This implementation extends `LightningIndexerQuant` and exposes
`torch.ops._C_ascend.npu_lightning_indexer_quant_prefetch`. It does not change
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
3. Gather the selected rows from staging when present; otherwise fetch the
   original row. The returned row at position `i` always corresponds to the
   returned index at position `i`.
4. Return the next threshold as `ema_decay * old + (1 - ema_decay) * kth_score`.
   A non-finite old threshold is replaced by the actual kth score. A request
   with fewer than K visible keys preserves its old threshold.

TopK order and the choice among equal boundary scores are unspecified. Indices
are logical token positions, never physical block IDs. Unused index entries
are `-1`; unused gathered rows are zero. A NaN threshold admits no candidates,
`+inf` admits no finite score, and `-inf` admits all finite scores until budget
is exhausted. Budget zero executes the same exact score/TopK/gather path with
all valid selected rows counted as fallback copies.

The standalone call does not implement prefill inter-query prediction or a
persistent LRU cache. Its workspace is private to the call. The caller carries
`next_threshold` to the next invocation for the same request and layer.

## 950 execution

One AIC and its paired AIVs handle each request, with strided request assignment
when the batch exceeds the physical AIC count. The existing local QLI-V2 INT8
Cube service supplies QK tiles directly to AIV UB using 950 FIX dual destination
and mode-4 events. AIV0 computes scores and copies KV rows; AIV1 acknowledges
its unused half of the QK tile. The AIV returns the QK-buffer credit before
starting candidate copies, allowing the next Cube tile to overlap those copies.

Scores retain FP32 resolution for selection; they are not converted to BF16.
The numeric order is INT32 dot, divide by 1024, FP16 RINT round-trip, ReLU,
multiply by FP16-rounded `weight * query_scale`, sequential FP32 head sum,
then multiply by the FP16 key scale converted to FP32. Inputs must produce
finite scores. All 64 heads participate. TopK uses the existing FP32 radix
selector with ordered float-bit keys and bounded UB buffers.

Each speculative row is completely copied from its source through UB into HBM
staging before its candidate ID is published. Candidate IDs are increasing, so
final lookup uses binary search. There is no eviction or cross-core shared slot
state. Request staging strides are padded to 512 bytes. Workspace scales as
`B * align512(max(1,budget) * align32(row_bytes))`, plus runtime workspace.

This is a functional first version, not a demonstrated speedup. Small batches
use only a small number of Cube cores. Scalar candidate scanning, serial row
DMA, TopK compaction, and the extra HBM staging copy can outweigh overlap.
Measure budget zero and enabled budgets on the target server before tuning
core assignment, transfer batching, or integrating with attention.

## Input and lifetime contract

- Q: contiguous INT8 `[B,64,128]`, TND, exactly one decode query per request.
- K: contiguous INT8 `[physical_blocks,block_size,1,128]`, PA_BSND.
  Block size is a positive multiple of 16, at most 1024.
- Weights and Q scales: FP16 `[B,64]`; K scales: FP16
  `[physical_blocks,block_size,1]`.
- Query lengths: INT32 cumulative TND ends `[1,2,...,B]`.
  Key lengths: INT32 per-request visible lengths `[B]`, not cumulative.
- Block table: INT32 `[B,max_blocks]`. Only pages covering the visible key
  length need valid physical block IDs.
- Addresses: INT64 `[B,max_blocks * block_size]`, stored on the NPU. Each valid
  element points to a contiguous `kv_row_bytes` payload row in HBM or mapped
  host memory. Payload is independent of the indexer's quantized K tensor.
- Thresholds: FLOAT32 `[B]`, stored on the NPU. All tensor inputs are on the
  same NPU. K is 1..2048, budget is 0..2048, row bytes is 1..4096, and EMA decay
  is finite in `[0,1)`.

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
an empty padded result for that request, `stats=[-1,-1]`, and an unchanged
threshold. Pointer validity remains the caller's responsibility. Inspect stats
when using the raw standalone interface.

## Build compatibility and verification status

The generated ACLNN ABI has two additional optional inputs, three additional
attributes, and four additional outputs. Rebuild and install both the custom
OPP package and the PyTorch extension together; an old generated ACLNN library
is incompatible. The original Python operator signature is retained for
910B/910_93 through its adapter. The new prefetch API requires Ascend 950.

Development and review were performed on macOS without CANN headers, a device
compiler, or an NPU. Local Python syntax/reference checks and source review do
not establish kernel compilation, DMA correctness, graph support, or performance
on hardware. Run both the HBM and MemFabric host suites on the 950 server before
using this branch in a model.
