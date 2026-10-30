"""Validate and benchmark the Ascend 950 staged QLI prefetch operator.

``--verify`` runs the independent torch CPU score/top-k contract and needs no
NPU or CANN installation.  The NPU mode compares staged outputs with the
deployed ``torch_npu.npu_quant_lightning_indexer`` baseline.
The timing intentionally measures compute only; it does not claim Plan/Transfer
overlap because that requires a user memfabric setup.
"""

from __future__ import annotations

import argparse
import json
from typing import Final

import torch

TOP_K: Final = 2048
HEADS: Final = 64
HEAD_DIM: Final = 128
SCORE_TILE: Final = 128


def aligned_width(seq_len: int) -> int:
    """Return the int16 score workspace width."""
    return max(SCORE_TILE, (seq_len + SCORE_TILE - 1) // SCORE_TILE * SCORE_TILE)


def candidate_budget(seq_len: int, divisor: int = 32) -> int:
    """Return the bounded ceil(seq_len / divisor) candidate budget."""
    if divisor <= 0:
        raise ValueError("budget divisor must be positive")
    return min(TOP_K, (max(seq_len, 0) + divisor - 1) // divisor)


def decode_score_workspace(scores: torch.Tensor) -> torch.Tensor:
    """Decode sortable BF16 keys persisted by the staged kernel to FP32."""
    sortable = scores.to(torch.int32) & 0xFFFF
    bits = torch.where((sortable & 0x8000) != 0, sortable ^ 0x8000, (~sortable) & 0xFFFF)
    return (bits << 16).view(torch.float32)


def score_workspace_drift(decoded: torch.Tensor, reference: torch.Tensor) -> float:
    """Report finite CPU-reference drift over the shared score width."""
    width = min(decoded.shape[1], reference.shape[1])
    decoded_shared = decoded[:, :width]
    reference_shared = reference[:, :width]
    valid = torch.isfinite(decoded_shared) & torch.isfinite(reference_shared)
    drift = (decoded_shared - reference_shared).abs()[valid]
    return float(drift.max()) if drift.numel() else 0.0


def reference_scores(query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
                     qscale: torch.Tensor, kscale: torch.Tensor, seq_lens: torch.Tensor,
                     block_table: torch.Tensor) -> torch.Tensor:
    """Compute weighted per-token dot products independently on CPU/torch."""
    rows, _, width = query.shape
    block_size = key.shape[1]
    max_tokens = max(1, min(key.shape[0] * block_size, max(int(value) for value in seq_lens)))
    result = torch.full((rows, max_tokens), float("-inf"), dtype=torch.float32)
    for row in range(rows):
        visible = min(int(seq_lens[row]), max_tokens)
        if visible <= 0:
            continue
        for start in range(0, visible, 1024):
            stop = min(start + 1024, visible)
            logical = torch.arange(start, stop, dtype=torch.long)
            physical = block_table[row, logical // block_size].to(torch.long)
            values = key[physical, logical % block_size, 0, :width].float()
            dots = torch.sum(query[row].float().unsqueeze(0) * values.unsqueeze(1), dim=-1)
            scaled = torch.relu(dots) * qscale[row].float().unsqueeze(0) * kscale[physical, logical % block_size].float().unsqueeze(-1)
            # The deployed vector path rounds weighted scores to BF16 before
            # its sortable representation; matching that boundary avoids false
            # failures when neighboring values collapse to the same BF16 value.
            result[row, start:stop] = torch.sum(scaled * weights[row].float().unsqueeze(0), dim=-1).to(torch.bfloat16).float()
    return result


def reference_topk(scores: torch.Tensor, seq_lens: torch.Tensor,
                   stable_prefix_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact indices and kth cutoffs, preserving inf for short rows."""
    rows = scores.shape[0]
    indices = torch.full((rows, TOP_K), -1, dtype=torch.int32)
    kth = torch.full((rows,), float("inf"), dtype=torch.float32)
    for row in range(rows):
        visible = min(int(seq_lens[row]), scores.shape[1])
        if visible <= 0:
            continue
        order = torch.argsort(scores[row, :visible], descending=True, stable=True)
        count = min(TOP_K, visible)
        indices[row, :count] = order[:count].to(torch.int32)
        if visible >= TOP_K:
            kth[row] = scores[row, order[TOP_K - 1]]
    return indices, kth


def validate_outputs(candidates: torch.Tensor, counts: torch.Tensor, exact_indices: torch.Tensor,
                     kth_scores: torch.Tensor, seq_lens: torch.Tensor,
                     stable_prefix_lens: torch.Tensor, divisor: int = 32) -> None:
    """Check output invariants without imposing an order on equal-score top-k."""
    rows = seq_lens.numel()
    assert candidates.shape == (rows, TOP_K)
    assert exact_indices.shape == (rows, 1, TOP_K)
    for row in range(rows):
        seq_len = max(int(seq_lens[row]), 0)
        stable = max(min(int(stable_prefix_lens[row]), seq_len), 0)
        count = int(counts[row])
        assert 0 <= count <= min(candidate_budget(seq_len, divisor), stable)
        selected = candidates[row, :count].tolist()
        assert len(selected) == len(set(selected)), f"duplicate candidate at row {row}"
        assert all(0 <= index < stable for index in selected), f"invalid candidate at row {row}"
        exact_count = min(TOP_K, seq_len)
        exact = exact_indices[row, 0, :exact_count].tolist()
        assert len(exact) == len(set(exact)), f"duplicate exact index at row {row}"
        assert all(0 <= index < seq_len for index in exact), f"invalid exact index at row {row}"
        assert (candidates[row, count:] == -1).all(), f"candidate tail is not -1 at row {row}"
        assert (exact_indices[row, 0, exact_count:] == -1).all(), f"exact tail is not -1 at row {row}"
        if seq_len < TOP_K:
            assert torch.isinf(kth_scores[row]), f"short row {row} needs infinite kth score"
        else:
            assert torch.isfinite(kth_scores[row]), f"full row {row} needs finite kth score"


def compare_topk_membership(staged: torch.Tensor, baseline: torch.Tensor,
                            scores: torch.Tensor, seq_lens: torch.Tensor) -> None:
    """Compare exact results while allowing arbitrary ordering at the cutoff tie."""
    for row in range(seq_lens.numel()):
        visible = min(max(int(seq_lens[row]), 0), scores.shape[1])
        count = min(TOP_K, visible)
        if count == 0:
            continue
        reference = torch.topk(scores[row, :visible], count, largest=True, sorted=False).values
        cutoff = reference.min()
        strict = {i for i, value in enumerate(scores[row, :visible].tolist()) if value > cutoff}
        staged_set = set(staged[row, :count].tolist())
        baseline_set = set(baseline[row, :count].tolist())
        assert strict <= staged_set and strict <= baseline_set, f"lost strict top-k index at row {row}"
        assert all(scores[row, i] >= cutoff for i in staged_set), f"staged below cutoff at row {row}"
        assert all(scores[row, i] >= cutoff for i in baseline_set), f"baseline below cutoff at row {row}"


def _npu_event_latency(fn, repeats: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


def _run_npu(args: argparse.Namespace) -> None:
    import torch_npu  # noqa: F401
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    import vllm_ascend.vllm_ascend_C  # noqa: F401, PLC0415

    if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, "npu_qli_prefetch_stage"):
        raise RuntimeError("npu_qli_prefetch_stage is unavailable; build/load the 950 custom op")
    torch.npu.set_device(0)
    rows, seq_len, block_size = args.rows, args.seq_len, 128
    blocks = (max(seq_len, 1) + block_size - 1) // block_size
    torch.manual_seed(args.seed)
    query = torch.randn(rows, HEADS, HEAD_DIM, device="npu").to(torch.float8_e4m3fn)
    key = torch.randn(blocks, block_size, 1, HEAD_DIM, device="npu").to(torch.float8_e4m3fn)
    weights = torch.randn(rows, HEADS, device="npu", dtype=torch.bfloat16)
    qscale = torch.rand(rows, HEADS, device="npu", dtype=torch.float32)
    kscale = torch.rand(blocks, block_size, device="npu", dtype=torch.float32)
    seq_lens = torch.full((rows,), seq_len, device="npu", dtype=torch.int32)
    block_table = torch.arange(blocks, device="npu", dtype=torch.int32).repeat(rows, 1)
    thresholds = torch.full((rows,), float("inf"), device="npu", dtype=torch.float32)
    stable = torch.full((rows,), seq_len, device="npu", dtype=torch.int32)
    active = torch.tensor([rows], device="npu", dtype=torch.int32)
    scores = torch.empty((rows, aligned_width(seq_len)), device="npu", dtype=torch.int16)
    candidates = torch.empty((rows, TOP_K), device="npu", dtype=torch.int32)
    counts = torch.empty(rows, device="npu", dtype=torch.int32)
    exact = torch.empty((rows, 1, TOP_K), device="npu", dtype=torch.int32)
    kth = torch.empty(rows, device="npu", dtype=torch.float32)
    candidates.fill_(-7)
    exact.fill_(-7)
    op = torch.ops._C_ascend.npu_qli_prefetch_stage
    run_stage = lambda: (op(query, key, weights, qscale, kscale, seq_lens, block_table, thresholds, stable, active,
                             scores, candidates, counts, exact, kth, 0, args.budget),
                         op(query, key, weights, qscale, kscale, seq_lens, block_table, thresholds, stable, active,
                            scores, candidates, counts, exact, kth, 1, args.budget))
    run_stage()
    torch.npu.synchronize()
    validate_outputs(candidates.cpu(), counts.cpu(), exact.cpu(), kth.cpu(), seq_lens.cpu(), stable.cpu(), args.budget)
    assert (counts.cpu() == 0).all(), "infinite cold thresholds must not emit candidates"
    prior_kth = kth.clone()
    if args.threshold_mode == "warm":
        thresholds.copy_(prior_kth)
    elif args.threshold_mode == "low":
        thresholds.fill_(-float("inf"))
    staged_us = _npu_event_latency(run_stage)
    torch.npu.synchronize()
    validate_outputs(candidates.cpu(), counts.cpu(), exact.cpu(), kth.cpu(), seq_lens.cpu(), stable.cpu(), args.budget)
    import torch_npu
    actual_q = torch.arange(1, rows + 1, device="npu", dtype=torch.int32)
    deployed = torch_npu.npu_quant_lightning_indexer(
        query=query, key=key, weights=weights, query_dequant_scale=qscale,
        key_dequant_scale=kscale, actual_seq_lengths_query=actual_q,
        actual_seq_lengths_key=seq_lens, block_table=block_table, query_quant_mode=0,
        key_quant_mode=0, layout_query="TND", layout_key="PA_BSND", sparse_count=TOP_K,
        sparse_mode=3)
    deployed_indices = deployed[0] if isinstance(deployed, tuple) else deployed
    deployed_indices = deployed_indices.reshape(rows, -1).cpu()
    score_workspace = decode_score_workspace(scores.cpu())
    expected, expected_kth = reference_topk(score_workspace, seq_lens.cpu(), stable.cpu())
    compare_topk_membership(exact.cpu().reshape(rows, -1), expected, score_workspace, seq_lens.cpu())
    compare_topk_membership(exact.cpu().reshape(rows, -1), deployed_indices, score_workspace, seq_lens.cpu())
    torch.testing.assert_close(kth.cpu(), expected_kth, rtol=0, atol=0)
    score_ref = reference_scores(query.cpu(), key.cpu(), weights.cpu(), qscale.cpu(), kscale.cpu(),
                                 seq_lens.cpu(), block_table.cpu())
    max_drift = score_workspace_drift(score_workspace, score_ref)
    baseline_us = _npu_event_latency(lambda: torch_npu.npu_quant_lightning_indexer(
        query=query, key=key, weights=weights, query_dequant_scale=qscale,
        key_dequant_scale=kscale, actual_seq_lengths_query=actual_q,
        actual_seq_lengths_key=seq_lens, block_table=block_table, query_quant_mode=0,
        key_quant_mode=0, layout_query="TND", layout_key="PA_BSND", sparse_count=TOP_K,
        sparse_mode=3))
    result = {"rows": rows, "seq_len": seq_len, "staged_compute_us": staged_us,
              "deployed_compute_us": baseline_us, "compute_overhead_ratio": staged_us / baseline_us,
              "threshold_mode": args.threshold_mode,
              "candidate_counts": counts.cpu().tolist(),
              "cpu_reference_max_abs_drift": max_drift,
              "note": "compute-only; Plan/Transfer overlap requires user memfabric setup"}
    print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="run the independent CPU contract")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--threshold-mode", choices=("cold", "warm", "low"), default="warm")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    if args.verify:
        query = torch.randn(args.rows, HEADS, HEAD_DIM)
        key = torch.randn((max(args.seq_len, 1) + 127) // 128, 128, 1, HEAD_DIM)
        weights = torch.randn(args.rows, HEADS)
        qscale = torch.rand(args.rows, HEADS)
        kscale = torch.rand(key.shape[0], 128)
        seq_lens = torch.full((args.rows,), args.seq_len, dtype=torch.int32)
        table = torch.arange(key.shape[0], dtype=torch.int32).repeat(args.rows, 1)
        scores = reference_scores(query, key, weights, qscale, kscale, seq_lens, table)
        exact, kth = reference_topk(scores, seq_lens, seq_lens)
        counts = torch.minimum(seq_lens, torch.full_like(seq_lens, candidate_budget(args.seq_len, args.budget)))
        candidates = torch.full_like(exact, -1)
        for row in range(args.rows):
            candidates[row, : int(counts[row])] = exact[row, : int(counts[row])]
        validate_outputs(candidates, counts, exact.unsqueeze(1), kth, seq_lens, seq_lens, args.budget)
        print(json.dumps({"verify": "ok", "rows": args.rows, "seq_len": args.seq_len}), flush=True)
        return
    _run_npu(args)


if __name__ == "__main__":
    main()
