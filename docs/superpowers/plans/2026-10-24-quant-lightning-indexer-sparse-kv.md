# Quantized indexer and sparse-KV fusion record

The requested work fused GLM-5.2 C8 final-TopK processing and SparseKvPlan
state updates into one Ascend 950 kernel while retaining the existing DDR
miss-transfer point before attention. The implementation is self-contained in
this repository and targets CANN 9.1.0.

The enduring design, ABI, shape matrix, UB constraints, MTP/reuse invariants,
and acceptance coverage are in the concise
[architecture note](../specs/2026-10-24-quant-lightning-indexer-sparse-kv-design.md).
The executable validation and benchmark commands are in
[testing](../../quant_lightning_indexer_sparse_kv_testing.md) and
[performance measurement](../../quant_lightning_indexer_sparse_kv_performance.md).

This historical record intentionally contains no executable process plan.
Fresh CANN compilation, Ascend-950 correctness, and performance measurements
must be recorded from the current checkout; prior results are not carried
forward after NPU-side changes.
