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

To diagnose only the kernel build, run `bash build.sh --pkg --soc=ascend950
--ops=lightning_indexer_quant -j32` from `csrc/`; its installer is emitted under
`csrc/build/`. This diagnostic alone does not rebuild the Torch binding.

Run the local test package directly so the repository's broad conftest files
are not discovered:

```bash
cd csrc/attention/lightning_indexer_quant/tests/pytest
python3 -m pytest -q --source=hbm
python3 -m pytest -q --source=host
```

Both commands include the INT8 and FP8 GLM-5.2 cases. For only the new profile
and dtype checks, append `test_lightning_indexer_quant_prefetch_glm52.py`.

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
