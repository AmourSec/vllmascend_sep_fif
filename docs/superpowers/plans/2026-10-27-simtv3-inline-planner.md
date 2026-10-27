# SIMTv3 inline planner implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for implementation and review.

**Goal:** Port the SIMTv3 planner from `chore/simtv3-dependency-audit` into the current fused QLI and standalone shared planner.

**Architecture:** Keep the existing row-local UB/GM TopK pointer interface and persistent GM state. Port the 1024-thread scan, shared hit/evict partition, cached hash buckets, packed owner and cold-row path from `7f37e14ca`; retain the current first-occurrence-only duplicate semantics. Cold rows still build the deduplication hash, but skip resident classification and owner initialization.

**Tech stack:** Ascend C SIMT, C++17, Python/pytest; Ascend 950 and CANN 9.1.0.

**Spec:** User request to update the existing indexer from the optimized branch; existing contract in `csrc/attention/quant_lightning_indexer_sparse_kv/README.md`.

## Global constraints

- Preserve same-AIV consumption of final TopK indices in UB and IndexCache writeback.
- DDR load remains before attention, outside the planner.
- Preserve MTP row ownership, stable-prefix invalidation, request switching, LRU ordering and duplicate semantics from `368b1dbfd`.
- Keep current operator ABI, CANN 9.1.0 and existing dependencies.
- No dependency-branch merge, unrelated renames, or vLLM 0.28 migration.
- Mac validation is static/CPU only; target compilation, correctness and performance require 950 hardware.

## Review focus

- Every scan consumer finishes reads and scatter before its next scan; uniform cold return.
- In-place bucket-to-miss compaction cannot overwrite unread future buckets.
- Duplicate TopK across 1024-thread tiles and duplicate residents retain first-position/last-owner behavior.
- Capacities 14336/14337 select UB/GM correctly; 32768/32769 straddle packed-owner safety.
- Host, adapter, runtime and fixtures use identical workspace dimensions and retain inactive rows.

### Task 1: Pin and port the kernel algorithm

**Files:** `sparse_kv_plan_shared.h`, `sparse_kv_plan_config.h` under `csrc/attention/sparse_kv_plan/op_kernel/`; CPU miss-selection and SIMTv3 tests under `tests/ut/kv_offload/`.

**Interface:** Keep `ProcessRuntimeRow`, `SparseKvPlanRuntimeRowVf` and `SparseKvPlanRuntimeVf` signatures; TopK pointers remain row-local.

- [x] Pin new thread/scan/workspace contracts and observe failures against the old implementation.
- [x] Port the algorithm with 1024 threads, 288-byte scan and `3H+C+TopK` row scratch. Cold and warm paths accept only first TopK occurrences.
- [x] Compile/test real predicates and scan arithmetic on CPU; inspect synchronization and address spaces independently.

### Task 2: Integrate workspace sizing and target tests

**Files:** standalone tiling/validation, fused tiling contract, `sparse_kv_simt_lru.py`, NPU test helpers, tiling/workspace tests and runner capacity choices.

**Interface:** Public operator signatures and tiling-data layout unchanged; workspace width changes to `3H+C+TopK`.

- [x] Update sizing tests first and record expected failures.
- [x] Synchronize production sizing, fixtures and UB budgets; maximum dynamic UB becomes 212512 bytes.
- [x] Add target cases for cold/warm transitions, cross-tile duplicates, packed-owner boundary, stale miss tail, inactive rows and the new UB/GM threshold. Retain batch 32, MTP and 140k coverage.

### Task 3: Verify and deliver

**Files:** operator README and testing/performance guides.

- [x] Run static and benchmark-static suites, Python lint/compile checks, diff checks and required formatter check.
- [x] Review completed diff for algorithm correctness, ABI and CANN compatibility; fix findings.
- [x] Document the optimized standalone benchmark baseline and target rebuild/test commands; report target checks as pending.

## Verification record

- `--suite static`: 43 passed; `--suite benchmark-static`: 37 passed.
- All 17 changed Python files passed AST parsing, Ruff 0.14.0 check and format check.
- `git diff --check` passed. `bash format.sh ci` could not run its hooks because pre-commit is absent; this is not a full CI-format pass.
- Independent production and test/script reviews found no blocking issues after corrections.
- CANN 9.1 compilation, NPU correctness/graph replay and performance remain pending on Ascend 950.
