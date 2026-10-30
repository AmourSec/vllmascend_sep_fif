"""Optional two-stage ECHO prefetch for the standard GLM indexer.

The module deliberately contains no model-specific scheduling logic.  It owns
stable device workspaces and the custom-op boundary; callers can fall back to
the regular indexer when :func:`is_supported` is false or the op is absent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, TypeVar

import torch

PREFETCH_TOPK: Final = 2048
PREFETCH_HEADS: Final = 64
PREFETCH_HEAD_DIM: Final = 128
SCORE_TILE: Final = 128
PrefetchEvent = TypeVar("PrefetchEvent")


@dataclass(frozen=True, slots=True)
class IndexerPrefetchConfigError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class IndexerPrefetchRuntimeError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def aligned_score_width(max_seq_len: int) -> int:
    """Return the score row width required by the prefetch kernel."""
    if max_seq_len <= 0:
        raise IndexerPrefetchConfigError("max_seq_len must be positive")
    return (max_seq_len + SCORE_TILE - 1) // SCORE_TILE * SCORE_TILE


@dataclass(frozen=True, slots=True)
class IndexerPrefetchScratch:
    """Global staged-kernel scratch allocated before graph capture."""

    scores: torch.Tensor
    candidates: torch.Tensor
    candidate_counts: torch.Tensor
    exact_indices: torch.Tensor
    kth_scores: torch.Tensor
    active_rows: torch.Tensor

    @classmethod
    def allocate(cls, max_rows: int, max_seq_len: int, device: torch.device) -> "IndexerPrefetchScratch":
        """Allocate fixed-address buffers for all supported decode rows."""
        if max_rows <= 0:
            raise IndexerPrefetchConfigError("max_rows must be positive")
        score_width = aligned_score_width(max_seq_len)
        return cls(
            scores=torch.empty((max_rows, score_width), dtype=torch.int16, device=device),
            candidates=torch.empty((max_rows, PREFETCH_TOPK), dtype=torch.int32, device=device),
            candidate_counts=torch.empty(max_rows, dtype=torch.int32, device=device),
            exact_indices=torch.empty((max_rows, 1, PREFETCH_TOPK), dtype=torch.int32, device=device),
            kth_scores=torch.empty(max_rows, dtype=torch.float32, device=device),
            active_rows=torch.zeros(1, dtype=torch.int32, device=device),
        )


@dataclass(frozen=True, slots=True)
class IndexerPrefetchLayerState:
    """Request-aware threshold history owned by one SFA layer."""

    thresholds: torch.Tensor
    request_ids: torch.Tensor
    previous_thresholds: torch.Tensor
    previous_request_ids: torch.Tensor

    @classmethod
    def allocate(cls, max_rows: int, device: torch.device) -> "IndexerPrefetchLayerState":
        if max_rows <= 0:
            raise IndexerPrefetchConfigError("max_rows must be positive")
        return cls(
            thresholds=torch.full((max_rows,), float("inf"), dtype=torch.float32, device=device),
            request_ids=torch.full((max_rows,), -1, dtype=torch.int64, device=device),
            previous_thresholds=torch.empty((max_rows,), dtype=torch.float32, device=device),
            previous_request_ids=torch.empty((max_rows,), dtype=torch.int64, device=device),
        )


def is_supported(
    *,
    sparse_offload: bool,
    rank_local_simt: bool,
    fused_overlap: bool,
    head_count: int,
    head_dim: int,
    topk: int,
    is_decode: bool,
    uniform_queries: bool,
    mtp: bool,
) -> bool:
    """Check the narrow GLM standard-C8 runtime contract."""
    return (
        sparse_offload
        and rank_local_simt
        and not fused_overlap
        and is_decode
        and uniform_queries
        and not mtp
        and head_count == PREFETCH_HEADS
        and head_dim == PREFETCH_HEAD_DIM
        and topk == PREFETCH_TOPK
    )


def normalize_key_scale(key_scale: torch.Tensor) -> torch.Tensor:
    """View paged [P,BS,1[,1]] scale caches as the kernel's [P,BS]."""
    if key_scale.ndim in (3, 4) and key_scale.shape[2:] in ((1,), (1, 1)):
        return key_scale.view(key_scale.shape[0], key_scale.shape[1])
    return key_scale


def normalize_index_weights(
    weights: torch.Tensor,
    *,
    rows: int,
    heads: int,
) -> torch.Tensor | None:
    """Return the staged op's contiguous BF16 weight view or signal fallback."""
    if (
        weights.dtype != torch.bfloat16
        or weights.ndim != 2
        or weights.shape[0] < rows
        or weights.shape[1] != heads
    ):
        return None
    return weights[:rows].contiguous()


def run_prefetch_stage(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    thresholds: torch.Tensor,
    stable_prefix_lens: torch.Tensor,
    active_rows: torch.Tensor,
    scratch: IndexerPrefetchScratch,
    stage: int,
    budget_divisor: int = 32,
) -> bool:
    """Dispatch one ECHO stage, returning false when the op is unavailable."""
    if stage not in (0, 1):
        raise IndexerPrefetchConfigError("prefetch stage must be 0 or 1")
    if budget_divisor <= 0:
        raise IndexerPrefetchConfigError("budget_divisor must be positive")
    op = getattr(torch.ops._C_ascend, "npu_qli_prefetch_stage", None)
    if op is None:
        return False
    rows = query.shape[0]
    op(
        query,
        key,
        weights,
        qscale,
        kscale,
        seq_lens,
        block_table,
        thresholds,
        stable_prefix_lens,
        active_rows,
        scratch.scores[:rows],
        scratch.candidates[:rows],
        scratch.candidate_counts[:rows],
        scratch.exact_indices[:rows],
        scratch.kth_scores[:rows],
        stage,
        budget_divisor,
    )
    return True


def execute_prefetch_overlap(
    stage_zero: Callable[[], bool],
    launch_transfer: Callable[[], PrefetchEvent],
    stage_one: Callable[[], bool],
    update_history: Callable[[], None],
    wait_transfer: Callable[[PrefetchEvent], None],
) -> bool:
    """Enforce stage0 -> overlapped transfer/stage1 -> wait ordering."""
    if not stage_zero():
        return False
    prefetch_done = launch_transfer()
    if not stage_one():
        raise IndexerPrefetchRuntimeError("npu_qli_prefetch_stage disappeared after stage 0")
    update_history()
    wait_transfer(prefetch_done)
    return True


def prepare_threshold_history(
    state: IndexerPrefetchLayerState,
    req_ids: torch.Tensor,
    active_rows: torch.Tensor,
) -> None:
    """Map prior thresholds to current rows by request id before stage 0."""
    capacity = state.thresholds.shape[0]
    current_ids = req_ids[:capacity]
    rows = current_ids.shape[0]
    state.previous_request_ids.copy_(state.request_ids)
    state.previous_thresholds.copy_(state.thresholds)
    old_ids = state.previous_request_ids
    old_thresholds = state.previous_thresholds
    request_matches = (current_ids.view(-1, 1) == old_ids.view(1, -1)) & (old_ids.view(1, -1) >= 0)
    has_history = request_matches.any(dim=1)
    history_rows = torch.argmax(request_matches.to(dtype=torch.int32), dim=1)
    previous = torch.index_select(old_thresholds, 0, history_rows)
    selected = torch.where(has_history, previous, torch.full_like(previous, float("inf")))
    valid = torch.arange(rows, device=current_ids.device) < active_rows.reshape(1).to(dtype=torch.int64)
    state.thresholds[:rows].copy_(torch.where(valid, selected, torch.full_like(selected, float("inf"))))
    state.request_ids[:rows].copy_(torch.where(valid, current_ids, torch.full_like(current_ids, -1)))
    if rows < capacity:
        state.thresholds[rows:].fill_(float("inf"))
        state.request_ids[rows:].fill_(-1)


def update_threshold_ema(
    state: IndexerPrefetchLayerState,
    req_ids: torch.Tensor,
    kth_scores: torch.Tensor,
    active_rows: torch.Tensor,
    alpha: float = 0.5,
) -> None:
    """Update request-aware thresholds without host synchronization."""
    if not 0.0 < alpha <= 1.0:
        raise IndexerPrefetchConfigError("alpha must be in (0, 1]")
    prepare_threshold_history(state, req_ids, active_rows)
    rows = active_rows.reshape(1).to(dtype=torch.int64)
    current_ids = req_ids[: state.thresholds.shape[0]]
    previous = state.thresholds[: current_ids.shape[0]]
    finite_kth = torch.isfinite(kth_scores)
    finite_previous = torch.isfinite(previous)
    same_request = state.request_ids[: current_ids.shape[0]] == current_ids
    can_average = same_request & finite_previous
    values = torch.where(can_average, previous * (1.0 - alpha) + kth_scores * alpha, kth_scores)
    fallback = torch.where(same_request, previous, torch.full_like(previous, float("inf")))
    values = torch.where(finite_kth, values, fallback)
    valid = torch.arange(current_ids.shape[0], device=current_ids.device) < rows
    state.thresholds[: current_ids.shape[0]].copy_(
        torch.where(valid, values, torch.full_like(values, float("inf")))
    )
    state.request_ids[: current_ids.shape[0]].copy_(
        torch.where(valid, current_ids, torch.full_like(current_ids, -1))
    )
