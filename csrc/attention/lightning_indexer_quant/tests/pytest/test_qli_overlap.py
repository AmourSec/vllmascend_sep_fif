from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from lightning_indexer_quant_prefetch_golden import compute_expected
from prefetch_test_assertions import _assert_selected_and_rows
from prefetch_test_cases import DType, _make_case, _run, _runtime

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
@pytest.mark.parametrize(
    "budget,lengths",
    [
        (1, (257, 129, 0)),
        (2, (257, 129, 0)),
        (3, (257, 129, 0)),
        (127, (257, 129, 0)),
        (128, (257, 129, 0)),
        (129, (257, 129, 0)),
        (2048, (2177, 129, 0)),
    ],
)
def test_qli_unprotected_budget_boundaries(
    budget: int, lengths: tuple[int, ...], dtype: DType, memfabric_offload: ModuleType | None
) -> None:
    torch = _runtime()
    case = _make_case(
        torch,
        lengths=lengths,
        physical=22 if budget == 2048 else 8,
        block_size=128,
        sparse_count=2048,
        row_bytes=1152,
        budget=budget,
        threshold=-np.inf,
        offload=memfabric_offload,
        quantized_inputs=dtype == "int8",
        dtype=dtype,
    )
    actual = _run(torch, case, threshold=-np.inf, budget=budget)
    _assert_selected_and_rows(actual, case, compute_expected(case.golden))


def test_qli_graph_replay_updates_threshold_lengths_and_all_outputs(
    memfabric_offload: ModuleType | None,
) -> None:
    torch = _runtime()
    case = _make_case(
        torch,
        lengths=(257,),
        physical=8,
        block_size=128,
        sparse_count=2048,
        row_bytes=127,
        budget=3,
        threshold=-np.inf,
        offload=memfabric_offload,
        quantized_inputs=True,
        dtype="int8",
    )
    threshold = torch.full((1,), -np.inf, dtype=torch.float32, device="npu")
    _run(torch, case, threshold=threshold, budget=3)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = _run(torch, case, threshold=threshold, budget=3)

    replay_inputs = ((-np.inf, 257), (np.inf, 129), (np.nan, 0), (0.0, 257))
    for threshold_value, length in replay_inputs:
        threshold.fill_(threshold_value)
        case.key_lengths.fill_(length)
        case = case._replace(
            golden=case.golden._replace(
                actual_seq_lengths_key=np.asarray((length,), dtype=np.int32),
                prefetch_threshold=np.asarray((threshold_value,), dtype=np.float32),
            )
        )
        graph.replay()
        torch.npu.synchronize()
        _assert_selected_and_rows(captured, case, compute_expected(case.golden))


@pytest.mark.parametrize("row_bytes", [1, 127])
def test_qli_batch_reuse_same_core_mixed_tile_lengths(
    row_bytes: int, memfabric_offload: ModuleType | None
) -> None:
    torch = _runtime()
    pattern = (0, 1, 2, 127, 128, 129)
    lengths = tuple(pattern[index % len(pattern)] for index in range(65))
    case = _make_case(
        torch,
        lengths=lengths,
        physical=8,
        block_size=128,
        sparse_count=127,
        row_bytes=row_bytes,
        budget=3,
        threshold=-np.inf,
        offload=memfabric_offload,
        quantized_inputs=True,
        dtype="int8",
    )
    actual = _run(torch, case, threshold=-np.inf, budget=3)
    _assert_selected_and_rows(actual, case, compute_expected(case.golden))
