from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest
from prefetch_test_cases import Case, DType, _make_case, _run, _runtime

if TYPE_CHECKING:
    from torch import Tensor

pytestmark = pytest.mark.npu


_SCORE_STEPS: Final[tuple[tuple[float, float], ...]] = (
    (1.0, 2.0),
    (2.0, 1.0),
    (1.0, 3.0),
    (3.0, 2.0),
)
_LENGTHS: Final = (2048, 2048, 2047, 0)
_INITIAL_THRESHOLDS: Final = (-np.inf, -np.inf, 1.5, 7.0)


def _make_deterministic_case(torch: ModuleType, dtype: DType, offload: ModuleType | None) -> Case:
    case = _make_case(
        torch,
        lengths=_LENGTHS,
        physical=16,
        block_size=128,
        sparse_count=2048,
        budget=4,
        row_bytes=128,
        dtype=dtype,
        quantized_inputs=dtype == "int8",
        seed=1952,
        offload=offload,
    )
    query_fp = torch.zeros(case.query.shape, dtype=torch.float32, device="npu")
    key_fp = torch.zeros(case.key.shape, dtype=torch.float32, device="npu")
    query_fp[..., 0] = 1
    key_fp[..., 0] = 1
    case.query.copy_(query_fp.to(case.query.dtype))
    case.key.copy_(key_fp.to(case.key.dtype))
    case.query_scale.fill_(1024 if dtype == "int8" else 1)
    case.key_scale.fill_(1)
    return case


def _set_scores(case: Case, scores: tuple[float, float]) -> None:
    case.weights.zero_()
    for row, score in enumerate((*scores, *scores)):
        case.weights[row, 0] = score


def _assert_step(
    actual: tuple[Tensor, ...],
    old_threshold: np.ndarray,
    scores: tuple[float, float],
) -> np.ndarray:
    next_threshold = old_threshold.copy()
    for row, score in enumerate((*scores, *scores)):
        if _LENGTHS[row] >= 2048:
            next_threshold[row] = score if not np.isfinite(old_threshold[row]) else 0.5 * (old_threshold[row] + score)
        candidates = actual[2][row].cpu().numpy()
        expected_candidates = (
            np.arange(4, dtype=np.int32)
            if _LENGTHS[row] > 0 and score > old_threshold[row]
            else np.full(4, -1, dtype=np.int32)
        )
        np.testing.assert_array_equal(candidates, expected_candidates)
        count = int(np.count_nonzero(expected_candidates >= 0))
        np.testing.assert_array_equal(actual[3][row].cpu().numpy(), [count, min(_LENGTHS[row], 2048) - count])
    np.testing.assert_array_equal(actual[4].cpu().numpy(), next_threshold)
    return next_threshold


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
def test_prefetch_ema_feedback_across_decode_steps(dtype: DType, memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_deterministic_case(torch, dtype, memfabric_offload)
    threshold = torch.tensor(_INITIAL_THRESHOLDS, dtype=torch.float32, device="npu")
    old_threshold = np.array(_INITIAL_THRESHOLDS, dtype=np.float32)

    for scores in _SCORE_STEPS:
        _set_scores(case, scores)
        before = threshold.clone()
        actual = _run(torch, case, threshold=threshold, budget=4)
        torch.testing.assert_close(threshold, before)
        old_threshold = _assert_step(actual, old_threshold, scores)
        threshold = actual[4]


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
def test_prefetch_ema_feedback_is_captured_in_npu_graph(dtype: DType, memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_deterministic_case(torch, dtype, memfabric_offload)
    threshold = torch.tensor(_INITIAL_THRESHOLDS, dtype=torch.float32, device="npu")
    threshold_ptr = threshold.data_ptr()
    _set_scores(case, _SCORE_STEPS[0])
    _run(torch, case, threshold=threshold, budget=4)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = _run(torch, case, threshold=threshold, budget=4)
        threshold.copy_(captured[4])

    threshold.copy_(torch.tensor(_INITIAL_THRESHOLDS, dtype=torch.float32, device="npu"))
    torch.npu.synchronize()
    old_threshold = np.array(_INITIAL_THRESHOLDS, dtype=np.float32)
    for scores in _SCORE_STEPS:
        _set_scores(case, scores)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        old_threshold = _assert_step(captured, old_threshold, scores)
        np.testing.assert_array_equal(threshold.cpu().numpy(), old_threshold)
        assert threshold.data_ptr() == threshold_ptr
