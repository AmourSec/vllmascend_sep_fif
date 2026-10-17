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

The expected outputs are `indices:int32[B,1,K]`, `gathered_kv:uint8[B,K,row_bytes]`,
`prefetch_indices:int32[B,max(1,budget)]`, `stats:int32[B,2]` containing
`[candidate_count,fallback_count]`, and `next_threshold:float32[B]`. Every
output is fixed-size; invalid and unused slots are `-1` for index tensors and
zero for gathered bytes.

The oracle follows the contract's numeric order: INT32 QK dot divided by 1024,
rounded to FP16, converted to FP32, ReLU applied, FP16-rounded
`weights*query_dequant_scale`, FP32 head reduction, then multiplication by the
key scale. Top-k ties are compared as valid boundary sets, so a tie may appear
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

`--source=hbm` is the default and uses device-resident KV rows. It is the only
source exercised by the default test fixture. The event-timing smoke benchmark
is opt-in and reports NPU event milliseconds without asserting a speedup:

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
