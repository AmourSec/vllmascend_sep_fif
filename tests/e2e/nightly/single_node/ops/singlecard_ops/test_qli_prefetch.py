"""Contract and NPU validation tests for staged QLI prefetch.

The NPU assertions validate persisted sortable-score/candidate invariants; the
independent CPU score is not used as a strict oracle because reduction order
can differ. ``benchmark_qli_prefetch.py`` additionally compares both deployed
legacy baselines and reports CPU-reference drift diagnostically.
"""

import pytest
import torch

from benchmarks.benchmark_qli_prefetch import (
    TOP_K,
    candidate_budget,
    compare_topk_membership,
    decode_score_workspace,
    score_workspace_drift,
    reference_scores,
    reference_topk,
    validate_outputs,
)


def _cpu_case(rows: int, seq_len: int, block_size: int = 128):
    torch.manual_seed(123)
    blocks = (seq_len + block_size - 1) // block_size
    query = torch.randn(rows, 64, 128)
    key = torch.randn(blocks, block_size, 1, 128)
    weights = torch.randn(rows, 64)
    qscale = torch.rand(rows, 64)
    kscale = torch.rand(blocks, block_size)
    seq_lens = torch.full((rows,), seq_len, dtype=torch.int32)
    block_table = torch.arange(blocks, dtype=torch.int32).repeat(rows, 1)
    return query, key, weights, qscale, kscale, seq_lens, block_table


def test_reference_scores_handles_short_and_boundary_sequences() -> None:
    for seq_len in (0, 1, 127, 128, 129, 16384):
        args = _cpu_case(2, max(seq_len, 1))
        args = (*args[:-2], args[-2].clamp(max=seq_len), args[-1])
        scores = reference_scores(*args)
        assert scores.shape == (2, max(seq_len, 1))
        assert torch.isfinite(scores[:, :seq_len]).all() if seq_len else torch.isneginf(scores).all()


def test_reference_topk_has_tie_aware_cutoff_and_inf_when_insufficient() -> None:
    scores = torch.tensor([[5.0, 5.0, 3.0, 1.0]])
    seq_lens = torch.tensor([4], dtype=torch.int32)
    indices, kth = reference_topk(scores, seq_lens, torch.tensor([4], dtype=torch.int32))
    assert indices.shape == (1, TOP_K)
    assert set(indices[0, :4].tolist()) == {0, 1, 2, 3}
    assert torch.isinf(kth).all()


def test_candidate_budget_is_ceil_and_capped() -> None:
    assert candidate_budget(0, 32) == 0
    assert candidate_budget(65, 32) == 3
    assert candidate_budget(2048 * 64, 32) == TOP_K


def test_decode_score_workspace_round_trips_sortable_bf16() -> None:
    values = torch.tensor([-4.0, -0.0, 0.0, 1.5, 100.0], dtype=torch.float32).to(torch.bfloat16)
    bits = values.view(torch.int16).to(torch.int32) & 0xFFFF
    sortable = torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000).to(torch.int16)
    torch.testing.assert_close(decode_score_workspace(sortable), values.float(), rtol=0, atol=0)


def test_score_workspace_drift_uses_shared_padded_width() -> None:
    decoded = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    reference = torch.tensor([[1.0, 2.0, 3.0]])
    assert score_workspace_drift(decoded, reference) == 0.0


def test_validate_outputs_rejects_duplicate_or_out_of_range_candidates() -> None:
    candidates = torch.tensor([[1, 1] + [0] * (TOP_K - 2)], dtype=torch.int32)
    counts = torch.tensor([2], dtype=torch.int32)
    exact = torch.full((1, 1, TOP_K), -1, dtype=torch.int32)
    kth = torch.tensor([float("inf")])
    with pytest.raises(AssertionError, match="duplicate"):
        validate_outputs(
            candidates,
            counts,
            exact,
            kth,
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            1,
        )


def test_reference_topk_negative_weights_and_cold_threshold() -> None:
    args = _cpu_case(2, 129)
    query, key, _, qscale, kscale, seq_lens, table = args
    weights = -torch.rand_like(args[2])
    scores = reference_scores(query, key, weights, qscale, kscale, seq_lens, table)
    indices, kth = reference_topk(scores, seq_lens, seq_lens)
    assert indices.shape == (2, TOP_K)
    assert torch.isinf(kth).all()


@pytest.mark.parametrize(
    ("rows", "seq_len"),
    [(rows, seq_len) for seq_len in (0, 1, 127, 128, 129, 257, 2047, 2048, 2049, 16384, 16385, 32769)
     for rows in (1, 2, 5)] + [(65, 129)],
)
def test_qli_prefetch_stage_npu(seq_len: int, rows: int) -> None:
    torch_npu = pytest.importorskip("torch_npu")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401, PLC0415

    if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, "npu_qli_prefetch_stage"):
        pytest.fail("npu_qli_prefetch_stage is not loaded; build/load the Ascend 950 custom op")
    block_size = 128
    blocks = (max(seq_len, 1) + block_size - 1) // block_size
    torch.manual_seed(1000 + rows + seq_len)
    query = torch.randn(rows, 64, 128, device="npu").to(torch.float8_e4m3fn)
    key_storage = torch.randn(blocks * 2, block_size, 1, 128, device="npu").to(torch.float8_e4m3fn)
    key = key_storage[::2]
    weights = torch.randn(rows, 64, device="npu", dtype=torch.bfloat16)
    qscale = torch.rand(rows, 64, device="npu", dtype=torch.float32)
    kscale_storage = torch.rand(blocks * 2, block_size, device="npu", dtype=torch.float32)
    kscale = kscale_storage[::2]
    seq_lens = torch.full((rows,), seq_len, device="npu", dtype=torch.int32)
    permutation = torch.randperm(blocks, device="npu")
    block_table = permutation.to(torch.int32).repeat(rows, 1)
    thresholds = torch.full((rows,), float("inf"), device="npu", dtype=torch.float32)
    stable = torch.full((rows,), max(seq_len, 1), device="npu", dtype=torch.int32)
    active = torch.tensor([rows], device="npu", dtype=torch.int32)
    width = (max(seq_len, 1) + 127) // 128 * 128
    scores = torch.empty((rows, width), device="npu", dtype=torch.int16)
    candidates = torch.empty((rows, TOP_K), device="npu", dtype=torch.int32)
    counts = torch.empty(rows, device="npu", dtype=torch.int32)
    exact = torch.empty((rows, 1, TOP_K), device="npu", dtype=torch.int32)
    kth = torch.empty(rows, device="npu", dtype=torch.float32)
    candidates.fill_(-7)
    exact.fill_(-7)
    op = torch.ops._C_ascend.npu_qli_prefetch_stage
    for _ in range(2):
        for stage in (0, 1):
            op(query, key, weights, qscale, kscale, seq_lens, block_table, thresholds, stable, active,
               scores, candidates, counts, exact, kth, stage, 32)
        torch.npu.synchronize()
        validate_outputs(candidates.cpu(), counts.cpu(), exact.cpu(), kth.cpu(), seq_lens.cpu(), stable.cpu(), 32)
    score_workspace = decode_score_workspace(scores.cpu())
    expected, expected_kth = reference_topk(score_workspace, seq_lens.cpu(), stable.cpu())
    compare_topk_membership(exact.cpu().reshape(rows, -1), expected, score_workspace, seq_lens.cpu())
    torch.testing.assert_close(kth.cpu(), expected_kth, rtol=0, atol=0)
    if seq_len > 0:
        actual_q = torch.arange(1, rows + 1, device="npu", dtype=torch.int32)
        deployed = torch_npu.npu_quant_lightning_indexer(
            query=query, key=key, weights=weights, query_dequant_scale=qscale,
            key_dequant_scale=kscale, actual_seq_lengths_query=actual_q,
            actual_seq_lengths_key=seq_lens, block_table=block_table, query_quant_mode=0,
            key_quant_mode=0, layout_query="TND", layout_key="PA_BSND", sparse_count=TOP_K,
            sparse_mode=3)
        deployed_indices = deployed[0] if isinstance(deployed, tuple) else deployed
        compare_topk_membership(exact.cpu().reshape(rows, -1), deployed_indices.reshape(rows, -1).cpu(),
                                score_workspace, seq_lens.cpu())
    if rows == 1 and seq_len == 128:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
            op(query, key, weights, qscale, kscale, seq_lens, block_table, thresholds, stable, active,
               scores, candidates, counts, exact, kth, 0, 32)
            op(query, key, weights, qscale, kscale, seq_lens, block_table, thresholds, stable, active,
               scores, candidates, counts, exact, kth, 1, 32)
        for _ in range(2):
            graph.replay()
            torch.npu.synchronize()
            validate_outputs(candidates.cpu(), counts.cpu(), exact.cpu(), kth.cpu(), seq_lens.cpu(), stable.cpu(), 32)
    if rows == 2 and seq_len == 128:
        # Reuse contaminated graph workspaces with fewer active rows.  Only
        # row zero is in the active prefix and must be fully rewritten.
        candidates.fill_(-9)
        exact.fill_(-9)
        counts.fill_(TOP_K)
        kth.fill_(float("-inf"))
        active.fill_(1)
        for stage in (0, 1):
            op(query, key, weights, qscale, kscale, seq_lens, block_table, thresholds, stable, active,
               scores, candidates, counts, exact, kth, stage, 32)
        torch.npu.synchronize()
        validate_outputs(candidates[:1].cpu(), counts[:1].cpu(), exact[:1].cpu(), kth[:1].cpu(),
                         seq_lens[:1].cpu(), stable[:1].cpu(), 32)
        assert (candidates[1:] == -1).all(), "inactive candidate rows were not cleared"
        assert (counts[1:] == 0).all(), "inactive candidate counts were not cleared"
        assert (exact[1:] == -1).all(), "inactive exact rows were not cleared"
        assert torch.isinf(kth[1:]).all(), "inactive kth scores were not reset"


def test_qli_prefetch_stage_npu_low_threshold() -> None:
    """A cold low threshold must still produce valid, bounded candidates."""
    torch_npu = pytest.importorskip("torch_npu")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401, PLC0415

    if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, "npu_qli_prefetch_stage"):
        pytest.fail("npu_qli_prefetch_stage is not loaded; build/load the Ascend 950 custom op")
    del torch_npu
    rows, seq_len, block_size = 1, 128, 128
    query = torch.randn(rows, 64, 128, device="npu").to(torch.float8_e4m3fn)
    key = torch.randn(1, block_size, 1, 128, device="npu").to(torch.float8_e4m3fn)
    weights = -torch.rand(rows, 64, device="npu", dtype=torch.bfloat16)
    qscale = torch.ones(rows, 64, device="npu", dtype=torch.float32)
    kscale = torch.ones(1, block_size, device="npu", dtype=torch.float32)
    seq_lens = torch.tensor([seq_len], device="npu", dtype=torch.int32)
    block_table = torch.zeros((rows, 1), device="npu", dtype=torch.int32)
    thresholds = torch.full((rows,), -float("inf"), device="npu", dtype=torch.float32)
    stable = seq_lens.clone()
    active = torch.tensor([rows], device="npu", dtype=torch.int32)
    scores = torch.empty((rows, max(128, seq_len)), device="npu", dtype=torch.int16)
    candidates = torch.empty((rows, TOP_K), device="npu", dtype=torch.int32)
    counts = torch.empty(rows, device="npu", dtype=torch.int32)
    exact = torch.empty((rows, 1, TOP_K), device="npu", dtype=torch.int32)
    kth = torch.empty(rows, device="npu", dtype=torch.float32)
    candidates.fill_(-7)
    exact.fill_(-7)
    op = torch.ops._C_ascend.npu_qli_prefetch_stage
    for stage in (0, 1):
        op(query, key, weights, qscale, kscale, seq_lens, block_table, thresholds, stable, active,
           scores, candidates, counts, exact, kth, stage, 32)
    torch.npu.synchronize()
    validate_outputs(candidates.cpu(), counts.cpu(), exact.cpu(), kth.cpu(), seq_lens.cpu(), stable.cpu(), 32)
    score_workspace = decode_score_workspace(scores.cpu())
    expected, expected_kth = reference_topk(score_workspace, seq_lens.cpu(), stable.cpu())
    compare_topk_membership(exact.cpu().reshape(rows, -1), expected, score_workspace, seq_lens.cpu())
    torch.testing.assert_close(kth.cpu(), expected_kth, rtol=0, atol=0)
