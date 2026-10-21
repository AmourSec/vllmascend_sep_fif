# Lightning Indexer Quant Decode and MTP Prefetch Tests

These tests exercise standalone prefetch and resident-cache prefetch, including
multiple causal queries per request. They do not start a model or scheduler.
The NumPy oracle is independent of Torch and the custom extension. The complete
resident API and state contract are in [PREFETCH.md](../../PREFETCH.md).

The target path is mapped host DDR to the resident KV buffer in NPU HBM.
`--source=host` is the default. The numerical indexer's Q/K tensors, weights,
scales and cache metadata remain in HBM; only the attention K/V payload source
is in DDR. All MTP/resident device tests use MemFabric DDR allocations.
`--source=hbm` remains an explicit diagnostic option for the older standalone
tests and skips the DDR-only MTP/resident cases.

The pinned release lane uses the vLLM release tag `v0.28.0` and the verified
vLLM commit `84030bbe3d74d99bad477a3d2e37a973ccd8865c` as separate provenance
records. It uses PyTorch `2.10.0` and `torch-npu==2.10.0.post4`. Host-memory
coverage uses MemFabric Hybrid release 1.2 (`VERSION` 1.2.1) from the
[official upstream repository](https://gitcode.com/Ascend/memfabric_hybrid).

The operator is called as:

```text
_C_ascend.npu_lightning_indexer_quant_prefetch(
    query, key, weights, query_dequant_scale, key_dequant_scale,
    actual_seq_lengths_query, actual_seq_lengths_key, block_table,
    kv_addresses, prefetch_threshold, kv_row_bytes, prefetch_budget,
    sparse_count=2048, ema_decay=0.5,
) -> (indices, gathered_kv, prefetch_indices, stats, next_threshold)
```

Decode tests use one query token per request, so `actual_seq_lengths_query` is
the cumulative end vector `[1, 2, ..., B]`; `actual_seq_lengths_key` contains
each request's independent KV length.

The default query shape is `[B,32,128]`, matching GLM-5.2's indexer heads and
head dimension in its
[configuration](https://huggingface.co/zai-org/GLM-5.2/raw/main/config.json).
The GLM-5.2 profile tests use TopK=2048, block size 128, lengths 2047/2048/4097
(including streaming compaction), and a 1152-byte payload representing an unpacked BF16 MLA row
with 512 latent plus 64 RoPE elements. Small K=7 cases remain for boundary
checks. A separate 64-head regression checks equivalence to zero-padded
32-head inputs.

Both INT8 Q/K with FP16 weights/scales and FP8 E4M3FN Q/K with FP32
weights/scales are covered, using seeds 811/1952 and budgets 0/8. Profile inputs
are reproducible synthetic per-row quantizations with varying scales, not
captured GLM-5.2 activations. Passing these tests verifies the shape and
standalone numeric contract; it does not validate model integration, end-to-end model accuracy,
the deployed KV cache's packing, or representative prefetch hit rates.

The expected outputs are `indices:int32[B,1,K]`, `gathered_kv:uint8[B,K,row_bytes]`,
`prefetch_indices:int32[B,max(1,budget)]`, `stats:int32[B,2]` containing
`[candidate_count,fallback_count]`, and `next_threshold:float32[B]`. Every
output is fixed-size; invalid and unused slots are `-1` for index tensors and
zero for gathered bytes.

The INT8 oracle follows the contract's numeric order: INT32 QK dot divided by 1024,
rounded to FP16, converted to FP32, ReLU applied, FP16-rounded
`weights*query_dequant_scale`, FP32 head reduction, then multiplication by the
key scale. The FP8 oracle reads Q/K after E4M3FN quantization, then uses FP32
QK dot, ReLU, FP32 `weights*query_dequant_scale`, sequential FP32 head reduction,
and FP32 key scale, without `/1024` or FP16 rounding. Top-k ties are compared as valid boundary sets, so a tie may appear
in any order. Candidate IDs must be chronological, strictly above the threshold,
and capped by the per-request budget. The selected top-k set must remain valid
when threshold or budget changes. EMA checks use the actual kth score with the
specified cold-start and non-finite-threshold behavior.

## Build and run on an Ascend server

Use the server's compatible CANN/driver/torch-npu installation. CANN 9.2 is the
requested target, but this branch has not been compiled against it locally.
From the repository root, rebuild both the OPP package and the C++ extension:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SOC_VERSION=ascend950pr_9599 COMPILE_CUSTOM_KERNELS=1 \
  python3 -m pip install -e . --no-build-isolation --no-deps
```

Use the actual `SOC_VERSION` reported by the target server if different. This
invokes the repository's custom-op build/install step before compiling the
extension, including the new binding and expanded generated ACLNN signature.
An older extension or OPP library is incompatible with the expanded signature.
The build script's `--soc=ascend950` selector resolves to Ascend950PR_9599.

An `EZ1009` dtype/format mismatch can occur before kernel execution if the
installed common OpDef prototype lacks FP8/FP32 even when the 950 kernel
configuration accepts them. Both declarations must include the dtype pair.
After changing `op_host/lightning_indexer_quant_def.cpp`, rebuild and install
the custom OPP package, including its operator prototype library; rebuilding
only the Torch extension cannot update that prototype. Rerun pytest in a new
Python process after installation.

The resident handoff uses native BF16 K/V buffers. `SparseKvTransfer` must have
BF16 entries in the installed OpDef, matching the current source. An older
locally modified UINT8 OPP can produce `EZ1009` in the resident handoff tests;
rebuild and install the current OPP package, then start a fresh test process.
The kernel's internal byte views do not change the public buffer dtype.

The extension's `build_aclnn.sh` removes `csrc/build` before building the OPP
package. Keep this clean build for the first rebuild after updating the branch:
historical `.done` stamps may be newer than the stale sources they represent.
Kernel source-copy stamps now depend on the files being copied, and kernel
compile stamps depend on those source files, declared helper sources, the
generated Python wrapper, and the compile script. Editing a nested template
header therefore triggers both a fresh copy and kernel compilation during
incremental CMake builds. The `op_build` generation commands also depend on
their host libraries so an OpDef change regenerates ACLNN sources.

To diagnose only the kernel build, start from a clean build directory:

```bash
cd csrc
rm -rf -- build output build_out
bash build.sh --pkg --soc=ascend950 --ops=lightning_indexer_quant -j32
```

Its installer is emitted under `csrc/build/`. This diagnostic alone does not
install the OPP package or rebuild the Torch binding; use the editable-install
command above for the complete build and install.

Run the local test package directly so the repository's broad conftest files
are not discovered:

```bash
cd csrc/attention/lightning_indexer_quant/tests/pytest
python3 -m pytest -q --source=host
```

This command includes the INT8 and FP8 GLM-5.2 cases. For only the profile
and dtype checks, append `test_lightning_indexer_quant_prefetch_glm52.py`.

For score mismatches, first run the deterministic head-mapping regression:

```bash
set -o pipefail
git rev-parse HEAD
python3 -m pytest -q --source=host --tb=short \
  test_lightning_indexer_quant_prefetch_scores.py 2>&1 | tee qli-score-regression.log
```

Its 12 cases cover both dtypes, 32/64 heads, and token positions 0/64/128
(zero-based), including the second QK bank and the next 128-token tile. Each
request isolates one head; one extra request sums all heads. K=1 and a cold
threshold expose that score through `next_threshold`. The expected values are
exact binary fractions for INT8 and exact integers for FP8, checked directly
and against the NumPy oracle. Failures include the full actual/expected score
vectors. Attach this log and the commit ID along with the failing random case
when reporting a numeric regression.

If the 32-head case returns pairwise sums of the expected head scores while
the 64-head case passes, check for an old 64-head kernel before changing the
score calculation. Reading a `[B,32]` weight tensor with a 64-element request
stride combines requests 0/1, 2/3, etc., and then reads beyond the tensor.
Before cleaning, preserve the build log and compare the copied kernel source
with the checked-out source from the repository root:

```bash
git rev-parse HEAD
find csrc/build -name lightning_indexer_quant_kernel_arch35.h \
  -exec grep -nE 'batch \*|actMBaseSize|ComputeScores' {} +
grep -nE 'batch \*|actMBaseSize|ComputeScores' \
  csrc/attention/lightning_indexer_quant/op_kernel/arch35/lightning_indexer_quant_kernel_arch35.h
```

Current code uses `t_->gSize` for request strides and head counts. A copied
header with `batch * 64` is stale. Kernel compilation uses the copy under
`csrc/build/binary/ascend950/src/`; finding a current header under
`csrc/build/_CPack_Packages/` does not prove that the compiled kernel is current.
For FP8 `EZ1009`, also preserve the generated
`csrc/build/autogen/aclnn_lightning_indexer_quant.cpp` and
`csrc/build/autogen/aic-ascend950-ops-info.ini`; inspect the
`LightningIndexerQuant` support entries for FP8 Q/K and FP32 weights/scales.
If a clean build still fails, provide those files, the complete build log, the
individual pytest failure, and the loaded OPP library paths from the same
failing Python process. Source revision alone does not identify the loaded
operator binary.

For `Cannot find tilingKey[50537508]`, check the template selection and rebuilt
kernel metadata. INT8 and FP8 must belong to the same `ASCENDC_TPL_SEL` list;
the CANN compiler reads only the first preprocessed selection list, so a
second unconditional list silently omits its kernel variants.
The host encoder can still produce a key in a release build because it uses
the parameter declaration without checking the selection list. The expected
decode keys are `50528770` for INT8 and `50537508` for FP8.

If a FP8 JSON advertises `dtype=float8_e4m3fn` but contains only the six INT8
keys, compare the template header itself, not just the kernel implementation
header. From the repository root:

```bash
cmp csrc/attention/lightning_indexer_quant/op_kernel/lightning_indexer_quant_template_tiling_key.h \
  csrc/build/binary/ascend950/src/lightning_indexer_quant/op_kernel/lightning_indexer_quant_template_tiling_key.h
```

A mismatch means the compiler's source copy differs from the checkout. Clean
and rebuild OPP before rerunning device tests; do not edit the generated JSON
to add the missing key. The host-only incremental build regression runs the
production CMake copy/compile rules with a host C++ preprocessor:

```bash
python3 -m pytest -q csrc/cmake/tests/test_kernel_source_incremental.py
```

The host-only regression compiles the actual operator header with the installed
CANN template header, checks both keys, and expands the debug selection table
to verify all six existing INT8 combinations plus the FP8 TND/PA combination.
It also checks that kernel preprocessing emits exactly one selection list
containing both dtypes:

```bash
python3 -m pytest -q test_lightning_indexer_quant_tiling_key.py
```

It discovers CANN includes from `ASCEND_HOME_PATH`, `ASCEND_TOOLKIT_HOME`, or the
standard toolkit location. Use `--cann-template-include=/path/to/include` to
select another include root. Without a host C++ compiler or the CANN header,
these three checks are skipped; they never substitute a mock template header.
Passing them does not prove that the NPU binary contains the variants.
The Ascend 950 `build_aclnn.sh` now checks generated kernel metadata before
installation and checks the installed metadata again afterwards. Missing FP8
metadata, a missing `50537508` in a FP8 binary's kernel list, or a missing object
file stops the build. The check reports every JSON path, dtype, compiled key
list, object path, and object SHA-256. It checks normal and available relocatable
variants separately; a valid relocatable variant cannot hide a broken normal
variant. `supportInfo.tilingKey` alone is not accepted as evidence of compilation.

To inspect an existing build without compiling or using an NPU, run from the
repository root (both directories are checked even if the first fails):

```bash
set -o pipefail
python3 csrc/attention/lightning_indexer_quant/check_kernel_metadata.py \
  csrc/build/binary/ascend950/bin \
  vllm_ascend/_cann_ops_custom 2>&1 | tee qli-kernel-metadata.log
```

This is a diagnostic check, not a fix for missing FP8 code. It does not modify
JSON or binaries and does not prove which OPP the runtime loads. If generated
metadata fails, investigate template extraction/compilation. If only installed
metadata fails, investigate packaging/installation. If both pass but the FP8
call still fails, inspect runtime binary selection and loaded OPP paths.
Preserve `qli-kernel-metadata.log` and the complete build log before cleaning.
Once metadata passes, rerun FP8 scores in a fresh process with `-x`, then run the
mixed suite to verify that subsequent INT8 calls also succeed.

To distinguish a standalone INT8 fault from errors following a failed FP8
call, run the six INT8 scores cases together in a fresh Python process:

```bash
python3 -m pytest -q --source=host --tb=long \
  test_lightning_indexer_quant_prefetch_scores.py -k int8
```

If this passes but a mixed run fails, retain the first FP8 error and the first
subsequent INT8 error, including their full tracebacks. Do not infer a tiling
cache or numerical defect from the total failure count alone.

The existing standalone event-timing smoke benchmark is opt-in and reports NPU
event milliseconds for both dtypes without asserting a speedup. It uses DDR
sources by default, 32 heads, context length 4097, TopK=2048, and 1152-byte rows.
It is not an MTP/resident performance benchmark:

```bash
python3 -m pytest -q --source=host --benchmark \
  test_lightning_indexer_quant_prefetch.py::test_prefetch_event_latency -s
```

## MemFabric host source

Host source is selected by default or explicitly with `--source=host`. On an
available NPU, a missing MemFabric allocator or failed mapping must fail the
test; there is no HBM fallback. A valid fixture follows
the MemFabric Hybrid Python API: create `offload.OffloadConfig`, set
`device_id`, equal `reserve_size` and `alloc_size`, set
`flags=offload.OFFLOAD_FLAG_GIANT_PAGE`, set `scene=offload.Scene.LOCAL`, and
call `offload.initialize(config)` before allocating rows. Allocate rows with
`offload.empty` or the corresponding offload allocator, then pass
`offload.get_dva(host_tensor.data_ptr())` to `kv_addresses`; never pass an
arbitrary pinned-CPU pointer. Call `offload.uninitialize()` during teardown.

Initialization order is part of the host fixture contract:
`bootstrap_custom_op_env(include_vendor_lib=True)` runs first, followed by
`offload.initialize(config)`, then `torch.npu.set_device(0)`. Test helpers such
as `_runtime()` must run after this fixture has initialized the host pool.
The user observed all-zero DDR payload reads on Ascend 950 with the earlier
device-first initialization and confirmed that restoring this order resolved
them. The resident fixture delegates directly to the same initialization
path. `test_qli_fixture_initialization.py` checks the call order and teardown
with external-runtime stubs; it does not simulate or validate NPU DMA.

Git attributes require LF for tracked `csrc/**/*.sh` scripts. The host-only
`csrc/cmake/tests/test_shell_scripts.py` checks this policy, line endings and
`bash -n` parsing. The portable Plan configuration uses `sparse_kv_plan_cfg`
because CANN may define `sparse_kv_plan` as a tiling-key macro; the host compiler
regression in `csrc/attention/sparse_kv_plan/tests/test_namespace_collision.py`
checks production namespace references with that macro defined.

The fixture follows the upstream examples
[`kv_exxchange_offload.py`](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/examples/kv_offload/kv_exxchange_offload/kv_exxchange_offload.py)
and
[`local_dram_offload.py`](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/examples/kv_offload/local_dram_offload/local_dram_offload.py).
The older standalone fixture allocates individual payload rows. The MTP and
resident fixtures allocate contiguous paged DDR K/V pools and convert their
base pointers with `get_dva`. Both keep source allocations alive through all
operator calls and graph replays, synchronize the NPU, and then tear down the
LOCAL pool.

## Double-buffered final gather

The Ascend 950 final gather uses two UB banks to overlap one row's MTE3 store
with the next row's MTE2 load.
The extra bank uses `align32(max(kv_row_bytes, 1))` bytes, at most 4096 bytes.
All pending stores complete before zero padding or metadata writeback.

Run the context, payload-width, padding and same-core request reuse checks
after rebuilding the OPP and starting a fresh process:

```bash
python3 -m pytest -q --source=host test_qli_boundaries.py
```

The standalone gather experiment
was tested on Ascend 950 by the user; its reported 46% benefit applies to that
measurement, not every workload. The Mac environment cannot repeat that result.

## Asynchronous candidate prefetch

Candidate prefetch also uses alternating UB banks. An admitted candidate ID
reserves its staging slot immediately; the per-row scalar completion wait is
deferred so scanning and TopK can proceed while DMA is pending. Before final
gather, `Drain()` consumes both outstanding candidate bank events and waits
for MTE3 completion. Only then can the gather phase reuse both UB banks and event
IDs 1/2. Both phases drain their events before the next request starts.

The candidate and gather phases share both UB banks: the candidate stores
finish before gather begins. This keeps the UB allocation identical to the
standalone gather optimization, adding `align32(max(kv_row_bytes, 1))` bytes
over the original baseline, at most 4096 bytes. The public ABI and host
workspace size are unchanged.

After rebuilding the combined OPP, run both regression suites with DDR KV in a
fresh process:

```bash
python3 -m pytest -q --source=host test_qli_boundaries.py test_qli_overlap.py
```

The overlap cases cover INT8/FP8 budgets around bank/tile boundaries, high
budgets, zero-length requests, graph replay with changed thresholds and
lengths, and same-core request reuse with unaligned rows. Standalone variants
03 and 05 were tested by the user; their combined correctness, UB allocation
and performance still require an Ascend 950 run.

## MTP and resident-cache regressions

The [DDR resident/MTP performance runner](RESIDENT_BENCHMARK.md) covers batch
1/4/8, query counts 1/2/4 and mixed MTP, TopK=2048, and contexts through 128K.
It compares disabled/enabled prefetch for QLI alone and QLI+Plan/Transfer,
restoring the same cold/partial/warm cache before every timed invocation.
It reports actual DDR miss traffic and saves timings to JSONL. Python-only test
updates do not require rebuilding an already installed MTP/resident operator.

Rebuild OPP and the Torch extension together before running these tests: the
generated ACLNN signature now includes the optional resident state and host
layout attributes. The standalone Torch signature is unchanged.

From this directory, run the new device suites in a fresh process:

```bash
python3 -m pytest -q --source=host \
  test_qli_mtp_resident.py test_qli_resident_npu_*.py
```

Every case in this matrix uses mapped DDR payloads, including standalone MTP,
warm-hit source poisoning, mixed hits/misses, graph replay and post-TopK
Plan/Transfer. The standalone MTP address table points into its DDR payload
pool; it does not use the numerical indexer K tensor as a payload source.
The resident path writes DDR misses directly into the existing HBM cache.

The MTP cases use mixed query counts per request and check each query against
its own causal visible length. They cover INT8/FP8 with 32/64 heads, separate
host/indexer block geometry, native K/V byte views, warm-cache hits, miss
payloads, LRU state, request identity changes, mutable tails, invalid metadata,
inactive rows, real graph replay, and the unmodified post-TopK Plan/Transfer.
Resident stats contain `[admitted_candidates, transferred_misses]`. This differs
from standalone stats, whose second entry counts final-gather fallbacks.

The native row-mapping and resident-policy tests compile the production portable
helpers with the host C++ compiler. They can run on a Mac; they do not emulate
NPU DMA or establish device correctness. The full command above also collects
the existing standalone regressions when the file filters are omitted.

## Mac limitations

The Mac environment cannot import `torch_npu`, load `_C_ascend`, build CANN
operator binaries, or execute NPU graph/event tests. Static validation may
still run Python AST parsing and the pure NumPy oracle smoke. No test here
silently converts a requested host run into an HBM run, and no benchmark result
is interpreted as a performance gain.

For local CPU checks, select the native and fixture/oracle tests explicitly.
The legacy HBM diagnostic suite can be collected with `--source=hbm`; its
device skips provide no evidence for the DDR path.
