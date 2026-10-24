# Quantized indexer and sparse KV fusion implementation plan

**Goal:** Execute GLM-5.2 C8 index selection and existing KV LRU planning in
one Ascend 950 kernel, retaining DDR transfer before attention.

**Architecture:** Consume each final QLI v1 TopK row directly from UB on its
owning AIV, then write the logical indices required by IndexCache to GM.
Bind mutable state explicitly and pass the prepared plan from the indexer
to the existing transfer path.

**Tech stack:** CANN 9.1.0, AscendC SIMD/SIMT, PyTorch/torch_npu, pytest.

**Spec:** `docs/superpowers/specs/2026-10-24-quant-lightning-indexer-sparse-kv-design.md`.

## Constraints

- Base the implementation on `clean_simt` in `vllmascend_sep_fif` and publish
  the feature on branch `wzj_kv_offload`.
- Preserve the running clean_simt baseline's cache and MTP semantics.
- No second device launch hidden behind the fused API.
- No NPU/CANN runtime is available on this Mac; provide target-machine checks.
- Complete code must be self-contained in this repository.

## Review focus

- Every final row has exactly one planner owner, including invalid rows.
- The planner reads final UB indices before their existing GM writeback.
- Dynamic UB must leave enough SIMT DataCache; validate compiler resources.
- Plan output rows and request lengths differ during MTP.
- Captured graph state must not repeat or omit planning/transfer.
- Shared indexer layers and MTP row compaction have different reuse semantics.

## Status

- [x] Read the requested history, runtime path, planner and upstream v1 source;
  verify the official model dimensions and CANN target.
- [x] Implement the fused kernel, binding, runtime and correctness tests.
- [x] Run static checks and available CPU tests, resolve review findings,
  and deliver the 950 test commands with explicit execution limitations.
- [x] Replace the original GM handoff with inline UB row
  consumption; cover normal, short and zero-key output paths without races.
- [x] Update correctness tests, launcher and resource/source documentation.
- [x] Run fresh static checks and CPU tests; review the inline implementation
  and deliver updated 950 commands with hardware checks still unverified.

Local verification and target-machine commands are recorded in
`docs/quant_lightning_indexer_sparse_kv_testing.md`. CANN compilation and NPU
execution were intentionally deferred to the user's Ascend 950 machine, as
requested; they are not claimed as passed.

## Kernel and host

Files: new `csrc/attention/quant_lightning_indexer_sparse_kv/` host/kernel/vendor
sources; shared planner header under `sparse_kv_plan/op_kernel/`.

- Extract the existing planner unchanged and check the source equivalence.
- Vendor the v1 arch35 dependency closure with licenses and source inventory.
- Register the 22-input, one-output operator and its tiling.
- Preserve QLI computation and chunk ordering; invoke Plan once per final
  row on the AIV that owns its UB indices, before IndexCache writeback.
- Keep SIMD/SIMT and MTE lifetimes explicit without a global phase barrier.
- Validate bounds and calculate simultaneously live UB/workspace needs.
- Verify include closure and registration consistency statically; target compile
  and resource checks are executed by the delivery script on 950.

## PyTorch interface

Files: new operator adapter header; `csrc/torch_binding.cpp`,
`csrc/torch_binding_meta.cpp`, `csrc/build_aclnn.sh`, existing planner adapter.

- Bind the exact spec signature and mutable aliases, with Meta behavior.
- Validate target shape/dtype/device and mutable storage without device reads.
- Normalize readonly input views consistently with the stock operator.
- Expose standalone Plan for the sequential differential oracle.
- Add the fused operator to the A5 build list only.

## Runtime

Files: `attention/indexer.py`, minimal `attention/sfa_v1.py` hook,
`attention/sfa_kv_offload.py`, `sparse_kv_simt_lru.py`,
`sparse_kv_offload_manager.py`, and focused runtime tests.

- Pass typed planner metadata to eligible indexer forwards.
- Prepare per-token causal/stable-prefix lengths and active rows beforehand.
- Mark only the current forward's successfully prepared plan.
- Transfer that plan before attention, preserving all fallback/reuse paths.
- Test prepared-plan routing, fallback, MTP compaction and context reset.

## Correctness and delivery

Files: new operator test directory under
`tests/e2e/nightly/single_node/ops/singlecard_ops/`, CPU oracle tests under
`tests/ut/kv_offload/`, and a 950 launcher under `tools/`.

- Implement an independent CPU planner oracle and deterministic GLM tensors.
- Compare stock QLI + existing Plan and fused output/state, including repeated
  state transitions and exact cases with unambiguous TopK scores.
- Add the shape, batch, sequence, MTP, capacity and graph matrix in the spec.
- Ensure the 950 launcher fails if the extension or NPU is unavailable.
- Run syntax, lint, include/dependency and diff checks locally; report any
  unavailable tool explicitly. Review the finished code with a fresh agent.
- Deliver final test commands and hardware validation limitations.
