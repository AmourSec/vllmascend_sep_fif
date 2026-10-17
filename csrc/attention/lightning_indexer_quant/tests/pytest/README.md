# Lightning Indexer Quant Decode Prefetch Tests

These tests exercise the standalone decode prefetch operator contract. They do
not start a model, run a scheduler, or depend on model integration. The NumPy
oracle is independent of Torch and the custom extension.

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

The extension's `build_aclnn.sh` now removes `csrc/build` before building the
OPP package. The kernel source-copy and compile `.done` files do not track all
source/header changes, so retaining that directory can repackage old kernels.
This makes OPP rebuilds slower, but prevents mixing new host libraries with old
generated metadata or device binaries. The `op_build` generation commands also
depend on their host libraries so an OpDef change regenerates ACLNN sources.
That dependency fix alone does not make the entire kernel build incremental.

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
python3 -m pytest -q --source=hbm
python3 -m pytest -q --source=host
```

Both commands include the INT8 and FP8 GLM-5.2 cases. For only the new profile
and dtype checks, append `test_lightning_indexer_quant_prefetch_glm52.py`.

For score mismatches, first run the deterministic head-mapping regression:

```bash
set -o pipefail
git rev-parse HEAD
python3 -m pytest -q --source=hbm --tb=short \
  test_lightning_indexer_quant_prefetch_scores.py 2>&1 | tee qli-score-regression.log
```

Its 12 cases cover both dtypes, 32/64 heads, and token positions 0/64/128
(zero-based), including the second QK bank and the next 128-token tile. Each
request isolates one head; one extra request sums all heads. K=1 and a cold
threshold expose that score through `next_threshold`. The expected values are
exact binary fractions for INT8 and exact integers for FP8, checked directly
and against the NumPy oracle. Failures include the full actual/expected score
vectors. Attach this log and the commit ID along with the failing random case
when reporting a numeric regression. These tests can also run with `--source=host`.

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
Passing them does not prove that the NPU binary contains the variants. After
rebuilding OPP, search the generated JSON under
`csrc/build/binary/ascend950/bin/lightning_indexer_quant` for `50537508`, then
rerun the FP8 scores cases.

To distinguish a standalone INT8 fault from errors following a failed FP8
call, run the six INT8 scores cases together in a fresh Python process:

```bash
python3 -m pytest -q --source=hbm --tb=long \
  test_lightning_indexer_quant_prefetch_scores.py -k int8
```

If this passes but a mixed run fails, retain the first FP8 error and the first
subsequent INT8 error, including their full tracebacks. Do not infer a tiling
cache or numerical defect from the total failure count alone.

`--source=hbm` is the default and uses device-resident KV rows. It is the only
source exercised by the default test fixture. The event-timing smoke benchmark
is opt-in and reports NPU event milliseconds for both dtypes without asserting
a speedup. It uses 32 heads, context length 4097, TopK=2048, and 1152-byte rows:

```bash
python3 -m pytest -q --source=hbm --benchmark \
  test_lightning_indexer_quant_prefetch.py::test_prefetch_event_latency -s
```

## MemFabric host source

Host source is selected explicitly with `--source=host`; it must fail loudly
when the official fixture or runtime is unavailable. A valid fixture follows
the MemFabric Hybrid Python API: create `offload.OffloadConfig`, set
`device_id`, equal `reserve_size` and `alloc_size`, set
`flags=offload.OFFLOAD_FLAG_GIANT_PAGE`, set `scene=offload.Scene.LOCAL`, and
call `offload.initialize(config)` before allocating rows. Allocate rows with
`offload.empty` or the corresponding offload allocator, then pass
`offload.get_dva(host_tensor.data_ptr())` to `kv_addresses`; never pass an
arbitrary pinned-CPU pointer. Call `offload.uninitialize()` during teardown.

The fixture follows the upstream examples
[`kv_exxchange_offload.py`](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/examples/kv_offload/kv_exxchange_offload/kv_exxchange_offload.py)
and
[`local_dram_offload.py`](https://gitcode.com/Ascend/memfabric_hybrid/blob/master/examples/kv_offload/local_dram_offload/local_dram_offload.py).
It allocates one offload row per logical token, converts every host pointer
with `get_dva`, keeps the rows alive for the operator call, and tears down the
LOCAL pool after the test.

## Mac limitations

The Mac environment cannot import `torch_npu`, load `_C_ascend`, build CANN
operator binaries, or execute NPU graph/event tests. Static validation may
still run Python AST parsing and the pure NumPy oracle smoke. No test here
silently converts a requested host run into an HBM run, and no benchmark result
is interpreted as a performance gain.
