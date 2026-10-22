# DDR resident and MTP benchmark

Run from `csrc/attention/lightning_indexer_quant/tests/pytest` on Ascend 950.
The installed OPP must match the checked-out kernel and host tiling. The resident
SIMT change requires rebuilding and installing the OPP, including the updated
workspace sizing. Its Torch/ACLNN arguments are unchanged; an extension already
built for the current MTP/resident ABI can be reused.

## Commands

Run DDR correctness first:

```bash
python3 -m pytest -q --source=host \
  test_qli_mtp_resident.py test_qli_resident_npu_*.py
```

Run the complete performance matrix for INT8 and FP8:

```bash
set -o pipefail
for dtype in int8 fp8; do
  python3 -m pytest -x -q -s test_qli_resident_benchmark.py \
    --benchmark --source=host \
    --resident-bench-profile=full --resident-bench-dtype="$dtype" \
    --resident-bench-budgets=512,2048 \
    --resident-bench-states=cold,partial,warm \
    --resident-bench-warmup=3 --resident-bench-iters=20 \
    --memfabric-pool-mib=2048 \
    --resident-bench-output="qli-resident-${dtype}.jsonl" \
    2>&1 | tee "qli-resident-${dtype}.log" || break
done
```

For a shorter smoke run, use `--resident-bench-profile=smoke`,
`--resident-bench-states=cold`, and `--resident-bench-iters=3`.
The smoke profile includes B=1 decode at 8K and B=4, two queries per request,
at 32K. It retains TopK=2048 and performs the same correctness checks.

JSONL records append after each completed cache-state group, preserving results
if a later case fails. Each invocation has a unique `run_id`. Use a new output
filename to keep separate binary revisions apart; save `git rev-parse HEAD`
alongside the logs.

## Matrix

All shapes use 32 indexer heads, head dimension 128, TopK=2048, resident
capacity=4096 per query row, and 128-token indexer and host pages.

| Batch | Query counts per request | Query rows | Context |
| --- | --- | --- | --- |
| 1 | 1 | 1 | 8192 |
| 4 | 1, 1, 1, 1 | 4 | 32768 |
| 4 | 2, 2, 2, 2 | 8 | 32768 |
| 4 | 4, 4, 4, 4 | 16 | 32768 |
| 8 | 4, 4, 4, 4, 4, 4, 4, 4 | 32 | 131072 |
| 4 | 1, 2, 4, 1 | 8 | 131072 |

Each shape runs three initial states, with budget=0 as the disabled baseline
and each requested enabled budget. One default full invocation produces 54
records; each record contains both QLI and QLI+Plan/Transfer timing.

K/V sources are separate **MemFabric-mapped DDR allocations**, 1024 and 128
bytes per token. This matches BF16 latent-512/RoPE-64 payload sizes used by the
offload manager. Resident buffers are native BF16 HBM views, shared by QLI and
the unchanged SparseKvPlan/Transfer binding. Indexer Q/K, scales, weights and
metadata stay in HBM.

The largest source requires 1152 MiB, independent of MTP query count; multiple
query rows for one request address the same DDR source. The pool defaults to
2048 MiB. Sources are explicitly freed after each shape; partial construction
failures are cleaned up by the fixture. Missing MemFabric or failed mappings
fail on an available NPU. There is no HBM source fallback for these cases.

## Initial state and timing

- `cold`: empty mappings, valid request identity, no resident hits initially.
- `partial`: every other stable token in the baseline TopK is resident.
- `warm`: every stable token in the baseline TopK is resident.

Seeded resident slots are placed at the recent end of LRU. The warm state does
not promise zero misses: false-positive prefetch candidates and mutable MTP
tail tokens can still miss. The actual counts are recorded.

For a request with `q` query tokens and final KV length `L`, row `j` uses
`visible=L-q+j+1` and `stable_prefix=L-q`. Every warmup and measured invocation
restores the same resident K/V, request identities, slot map and LRU snapshot.
Restoration and NPU synchronization happen before the start event.
`cold` describes the resident buffer state; hardware caches are not flushed.

The runner alternates budget order between repetitions. It measures two
separate intervals using NPU events:

1. The resident QLI call alone.
2. The resident QLI call followed on the same stream by the unchanged
   `npu_sparse_kv_plan_transfer`, using that call's TopK and resident state.

The second interval is measured directly; it is not a sum of independently
timed kernels. Buffer seeding, payload checks, snapshot restoration, threshold
calibration, Plan output/workspace allocation and reporting are outside both
intervals. These are eager operator timings, including dispatch gaps visible
to the event span; they are not graph replay or model end-to-end measurements.
Plan still performs its own miss test and tail invalidation, so repeated tail
transfers remain included.

The default threshold mode is `previous_query`: a synthetic preceding query is
formed by rotating each current query's feature vector by one position. Its
budget=0 kth score becomes the fixed threshold for all compared invocations.
It can admit false positives. The original query is restored before validation
and timing. This is a reproducible synthetic workload, not measured model
history or representative application hit rates.

`--resident-bench-threshold-mode=same_input` instead calibrates from the current
input's own kth score. It is an explicitly optimistic diagnostic. The mode and
per-row thresholds are saved in every record. The benchmark does not feed
returned EMA thresholds into subsequent timed calls.

## Checks and output

Before timing every state and budget, the runner checks:

- Prefetch enabled/disabled select the same TopK set and respect per-row MTP
  causal limits.
- Candidate counts are valid, candidate IDs are ordered, and candidate slots
  contain byte-exact DDR K/V after QLI.
- Cold calls must transfer every admitted candidate from DDR; no admitted
  candidate may be reported as a hit in an empty cache.
- After Plan/Transfer, all TopK slots map to the correct tokens and contain
  byte-exact DDR K/V. Request identities, unique mappings and the complete LRU
  permutation are checked.

Long-context checks compare against the disabled QLI result rather than a full
CPU score oracle. Independent numeric-oracle and exact LRU-order checks remain
in the smaller correctness suites.

Console output reports mean milliseconds, speedup and both miss counts.
JSONL adds mean/p50/p95, every measured sample, shape, dtype, initial hit state,
threshold, per-row admitted/prefetch-miss/Transfer-miss counts and DDR bytes.
`prefetch_active` identifies records with actual prefetch DDR traffic. A legal
threshold may admit no candidates; such a record is not evidence of a DDR
prefetch speedup. Warm hits can also legitimately have zero prefetch traffic.
`speedup_qli` and `speedup_chain` are **disabled mean / enabled mean**: greater
than 1 is faster. Miss counts are from untimed validation of the same restored
state; no device-to-CPU read occurs within the measured interval.

This suite has not been executed on an NPU in the development Mac environment.
CPU helper tests verify geometry, state selection, restoration order, warmup
exclusion, JSONL output and allocator cleanup. They do not establish hardware
correctness or performance.
