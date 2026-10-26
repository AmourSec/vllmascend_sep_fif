# QuantLightningIndexerSparseKv performance measurement

Measure the fused operator against stock
`torch_npu.npu_quant_lightning_indexer` plus
`torch.ops._C_ascend.npu_sparse_kv_plan` on Ascend 950 with CANN 9.1.0. The
comparison includes TopK, planner hit/miss/LRU updates, miss metadata, and
IndexCache TopK output. It excludes DDR transfer, attention, other model
layers, and end-to-end throughput.

Rebuild the custom CANN package and PyTorch C++ extension, then complete a
fresh 950 correctness run before timing. The current checkout has NPU-side
changes, so earlier benchmark or correctness numbers are not evidence for it.

## Commands

Run from the repository root in the configured 950 environment. Use a new
output directory for every invocation; add `--device N` or `--cann-root PATH`
when needed.

```bash
# CPU-only benchmark implementation checks; no NPU preflight.
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark-static --output test-results/perf-static

# Eight-config hardware smoke measurement, including batch 32, MTP 6, cold/warm state.
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark --batches 1,32 --seq-lens 140000 --mtps 1,6 \
  --warmup 3 --rounds 2 --iterations 3 --output test-results/perf-check

# Main 140k matrix: every batch 1..32, MTP 1/2/4/6, cold and warm planner state.
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark --seq-lens 140000 --output test-results/perf-140k-eager

# Full eager matrix, additionally covering 4096 and 32768 token contexts.
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark --output test-results/perf-full-eager

# ACLGraph replay matrix. Do not aggregate it with eager results.
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark --seq-lens 140000 --execution graph \
  --output test-results/perf-140k-graph

# Focused operator-stage profile with a separate NPU trace.
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite profile --batch 32 --seq-len 140000 --mtp 6 --capacity 4096 \
  --iterations 20 --trace --output test-results/perf-trace

# Validate the benchmark matrix without importing torch or NPU support.
python tools/run_quant_lightning_indexer_sparse_kv_tests.py \
  --suite benchmark --dry-run --output test-results/perf-plan
```

The default benchmark matrix uses batches `1-32`, sequence lengths
`4096,32768,140000`, MTP `1,2,4,6`, capacity `2048`, cold and warm cache state,
10 warmups, 6 paired rounds, and 20 timed calls per route. Override
`--batches`, `--seq-lens`, `--mtps`, `--capacities`, `--cache-states`,
`--warmup`, `--rounds`, and `--iterations` to target a smaller matrix. Inputs
with batch 32 and MTP 6 have `T=192` query rows.

The benchmark records paired per-call NPU-event times, wall time, medians,
means, percentiles, confidence summaries, and fused/baseline speedup in CSV and
JSON. It alternates route order, restores planner state before each route, and
records a failed configuration rather than publishing a partial timing value.
Review eager and graph output separately, use the same device/input/capacity
for both routes, and inspect traces to confirm one fused mixed kernel versus
the baseline QLI and SparseKvPlan launches. Hardware measurements establish
only this operator-stage comparison; validate model throughput separately.
