# QuantLightningIndexerSparseKv validation

`torch.ops._C_ascend.npu_quant_lightning_indexer_sparse_kv` fuses the final
quantized-lightning-indexer TopK row with SparseKvPlan state updates for the
GLM-5.2 C8 decode and speculative-decode path. It keeps the logical TopK
writeback required by IndexCache. `npu_sparse_kv_transfer` remains immediately
before attention and loads only the planner's misses from DDR.

The target is Ascend 950 with CANN 9.1.0. macOS can run CPU/static checks, but
cannot establish CANN compilation, device correctness, or performance. Current
kernel and NPU-side changes require a fresh CANN rebuild and a fresh 950 `full`
run; no earlier NPU result applies to this checkout.

## Contract and invariants

| Item | Required contract |
| --- | --- |
| Output | int32 logical token indices `[T, 1, 2048]` |
| Query/key | FP8 E4M3FN query `[T, 32, 128]`; paged FP8 key `[blocks, 128, 1, 128]` |
| Weights/scales | BF16 weights `[T, 32]`; FP32 query scale `[T, 32]`; key scale `[blocks, 128]` or supported trailing-singleton view |
| Mutable plan state | request IDs, resident-token/slot/LRU state, miss count/tokens/slots, and compact workspace retain their storage and alias annotations |
| ABI | 22 tensor inputs: eight QLI inputs, six planner metadata inputs, eight mutable planner-state inputs; attributes are capacity, max token, block size, and host block count |
| Planner ownership | each final row, including invalid or zero-key rows, is planned exactly once by its owning AIV before TopK GM writeback |
| MTP and reuse | plan state is per query row; `token_to_req` selects request state. Shared IndexCache layers reuse the prepared plan, while MTP row compaction or a layer without QLI uses standalone planning |

Final TopK merge/score/temporary storage uses a reusable slab; final indices
remain in a separate UB allocation. The planner adds a 544-byte scan buffer.
Capacity 2048/4096 uses UB scratch, while 8192 uses the existing GM fallback.
The highest planned dynamic-UB case is 212,768 bytes (`T > 64`, capacity 7168);
the CANN compiler resource report, including static UB, SIMT stack/spill, and
DataCache, is the authority for the final budget.

Correctness covers the stock QLI plus standalone-plan differential, a CPU
planner oracle, repeated cache transitions, duplicate tokens, LRU order,
request replacement, speculative rollback, invalid pages, graph padding and
replay, shape/dtype errors, and unambiguous BF16-rounded TopK boundaries. The
NPU matrix covers batch 1/2/4/8/16/32, MTP 1/2/4/6 including ragged rows,
contexts around 2048 and through 143360, query rows around 64/65, and
capacities 2048/4096/8192 plus the UB/GM-scratch boundary cases.

## Build on an Ascend 950 host

From the repository root, source that host's CANN 9.1.0 environment and rebuild
both the custom CANN package and the PyTorch C++ extension after pulling the
current checkout. `build_aclnn.sh` alone does not rebuild adapter headers. Use
the project's configured extension build, for example:

```bash
set -o pipefail
python -m pip install --no-build-isolation --no-deps -v -e . 2>&1 | tee qli-build.log
```

The build log must include `quant_lightning_indexer_sparse_kv`; retain its
compiler resource report. Do not substitute source inspection or a host C++
compile for this check.

## Test suites

All launcher results are written under `test-results/qli-<timestamp>/` unless
`--output DIR` is supplied. A missing prerequisite, empty collection, or skip
is an error.

```bash
# CPU correctness/static checks only; the default route excludes benchmark tests.
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite static

# CPU tests for benchmark implementation, CLI, and device fakes only.
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite benchmark-static

# NPU correctness subsets (Ascend 950 + CANN 9.1.0 required).
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite smoke
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite inline
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite stress

# Full NPU operator matrix, CPU correctness suite, then Python runtime suite.
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite full

# Re-run only the Python runtime integration coverage.
python tools/run_quant_lightning_indexer_sparse_kv_tests.py --suite runtime
```

`full` retains all operator and runtime coverage, but its CPU phase now runs
only correctness tests. Use `benchmark-static` explicitly when changing the
benchmark implementation. See [performance measurement](quant_lightning_indexer_sparse_kv_performance.md)
for timed benchmark commands and scope.

## Acceptance on 950

Run the rebuild, then `static`, `benchmark-static`, `smoke`, `inline`, and
`full`. Inspect the generated JUnit XML and logs for each phase, preserve the
compiler report, then run the performance matrix only after correctness is
green. Model-level GLM-5.2 regression still needs the existing clean_simt
startup configuration with MTP enabled and disabled and with graph mode.
