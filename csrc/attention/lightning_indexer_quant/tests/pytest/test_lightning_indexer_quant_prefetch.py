from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from lightning_indexer_quant_prefetch_golden import assert_contract, compute_expected
from prefetch_test_assertions import _assert_id_sets_equal, _assert_selected_and_rows, _assert_stats
from prefetch_test_cases import _make_case, _run, _runtime

pytestmark = pytest.mark.npu


def test_prefetch_contract_and_exact_rows(memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(torch, threshold=-np.inf, budget=4, offload=memfabric_offload)
    before = tuple(
        value.clone() for value in (case.query, case.key, case.weights, case.query_scale, case.key_scale, case.payload)
    )
    actual = _run(torch, case, threshold=-np.inf, budget=4)
    expected = compute_expected(case.golden)
    assert_contract(expected, topk=7, row_bytes=128, budget=4)
    _assert_selected_and_rows(actual, case, expected)
    np.testing.assert_array_equal(actual[2].cpu().numpy(), expected.prefetch_indices)
    np.testing.assert_array_equal(actual[3].cpu().numpy(), expected.stats)
    np.testing.assert_allclose(actual[4].cpu().numpy(), expected.next_threshold, rtol=2e-3, atol=2e-3)
    for old, current in zip(
        before, (case.query, case.key, case.weights, case.query_scale, case.key_scale, case.payload), strict=True
    ):
        torch.testing.assert_close(old, current)


@pytest.mark.parametrize("budget", [0, 1, 4])
@pytest.mark.parametrize("threshold", [np.inf, -np.inf, 1.0e-6])
def test_prefetch_budget_threshold_does_not_change_exact_topk(
    threshold: float, budget: int, memfabric_offload: ModuleType | None
) -> None:
    torch = _runtime()
    case = _make_case(torch, threshold=threshold, budget=budget, offload=memfabric_offload)
    actual = _run(torch, case, threshold=threshold, budget=budget)
    baseline = _run(torch, case, threshold=-np.inf, budget=0)
    expected = compute_expected(case.golden)
    _assert_selected_and_rows(actual, case, expected)
    _assert_stats(actual, expected)
    np.testing.assert_array_equal(actual[2].cpu().numpy(), expected.prefetch_indices)
    np.testing.assert_allclose(actual[4].cpu().numpy(), expected.next_threshold, rtol=2e-3, atol=2e-3)
    baseline_expected = compute_expected(
        case.golden._replace(prefetch_budget=0, prefetch_threshold=np.full(3, -np.inf, dtype=np.float32))
    )
    _assert_selected_and_rows(baseline, case, baseline_expected)
    _assert_id_sets_equal(actual[0].cpu().numpy(), baseline[0].cpu().numpy())


def test_prefetch_graph_replay_uses_current_inputs(memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(torch, threshold=-np.inf, budget=4, offload=memfabric_offload)
    threshold = torch.full((case.query.size(0),), -np.inf, dtype=torch.float32, device="npu")
    _run(torch, case, threshold=threshold, budget=4)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = _run(torch, case, threshold=threshold, budget=4)
    graph.replay()
    first = tuple(output.clone() for output in captured)
    torch.npu.synchronize()
    case.payload.fill_(17)
    for host_row in case.host_rows:
        host_row.fill_(17)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    assert not torch.equal(first[1], captured[1])
    assert torch.all(captured[0] >= -1)
    for row, length in enumerate(case.golden.actual_seq_lengths_key):
        count = min(int(length), case.golden.sparse_count)
        assert torch.all(captured[1][row, :count] == 17)
        assert torch.all(captured[1][row, count:] == 0)
    threshold.fill_(np.inf)
    graph.replay()
    torch.npu.synchronize()
    assert torch.all(captured[2] == -1)
    assert torch.all(captured[3][:, 0] == 0)
    for row, length in enumerate(case.golden.actual_seq_lengths_key):
        assert captured[3][row, 1].item() == min(int(length), case.golden.sparse_count)
    case.key_lengths.zero_()
    graph.replay()
    torch.npu.synchronize()
    assert torch.all(captured[0] == -1)
    assert torch.all(captured[1] == 0)
    assert torch.all(captured[3] == 0)


@pytest.mark.parametrize("decay", [0.0, 0.25, 0.9])
def test_prefetch_ema_decay(decay: float, memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(torch, threshold=1.0, offload=memfabric_offload)
    case = case._replace(golden=case.golden._replace(ema_decay=decay))
    actual = _run(torch, case, threshold=1.0, budget=4)
    _assert_selected_and_rows(actual, case, compute_expected(case.golden))


def test_prefetch_host_rows(memfabric_offload: ModuleType | None) -> None:
    if memfabric_offload is None:
        pytest.skip("pass --source=host to exercise MemFabric DVA rows")
    torch = _runtime()
    case = _make_case(torch, threshold=-np.inf, budget=4, offload=memfabric_offload)
    actual = _run(torch, case, threshold=-np.inf, budget=4)
    expected = compute_expected(case.golden)
    _assert_selected_and_rows(actual, case, expected)
    np.testing.assert_array_equal(actual[3].cpu().numpy(), expected.stats)


def test_prefetch_threshold_just_above_topk_boundary(memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(torch, threshold=-np.inf, budget=1, offload=memfabric_offload)
    unfiltered = compute_expected(case.golden)
    boundary = np.partition(unfiltered.scores[0], unfiltered.scores[0].size - 7)[-7]
    threshold = float(np.nextafter(np.float32(boundary), np.float32(np.inf)))
    case = case._replace(golden=case.golden._replace(prefetch_threshold=np.full(3, threshold, dtype=np.float32)))
    actual = _run(torch, case, threshold=threshold, budget=1)
    expected = compute_expected(case.golden)
    _assert_selected_and_rows(actual, case, expected)
    assert expected.stats[0, 1] > 0


def test_prefetch_all_negative_scores_with_radix_tail(memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(
        torch,
        threshold=-0.125,
        budget=4,
        lengths=(129,),
        physical=12,
        block_size=16,
        sparse_count=7,
        offload=memfabric_offload,
    )
    case.query.fill_(1)
    case.key.fill_(1)
    case.weights.fill_(-1)
    case = case._replace(
        golden=case.golden._replace(
            query=case.query.cpu().numpy(),
            key=case.key.cpu().numpy(),
            weights=case.weights.cpu().numpy(),
        )
    )
    actual = _run(torch, case, threshold=-0.125, budget=4)
    expected = compute_expected(case.golden)
    _assert_selected_and_rows(actual, case, expected)
    assert np.all(expected.scores[0] == np.float32(-0.25))
    assert tuple(expected.stats[0]) == (0, 7)


@pytest.mark.parametrize("row_bytes", [1, 127, 128, 4096])
def test_prefetch_row_byte_boundaries(row_bytes: int, memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(torch, threshold=-np.inf, budget=4, row_bytes=row_bytes, offload=memfabric_offload)
    actual = _run(torch, case, threshold=-np.inf, budget=4)
    expected = compute_expected(case.golden)
    assert actual[1].shape == (3, 7, row_bytes)
    _assert_selected_and_rows(actual, case, expected)


@pytest.mark.parametrize("invalid", ["negative_length", "over_capacity", "invalid_block"])
def test_prefetch_invalid_metadata_is_padded(invalid: str, memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(torch, threshold=0.75, budget=4, offload=memfabric_offload)
    query_lengths = case.query_lengths
    key_lengths = case.key_lengths
    block_table = case.block_table
    if invalid == "negative_length":
        key_lengths = key_lengths.clone()
        key_lengths[0] = -1
    elif invalid == "over_capacity":
        key_lengths = key_lengths.clone()
        key_lengths[0] = case.addresses.shape[1] + 1
    else:
        block_table = block_table.clone()
        block_table[0, 0] = 99
    actual = _run(
        torch,
        case,
        threshold=0.75,
        budget=4,
        query_lengths=query_lengths,
        key_lengths=key_lengths,
        block_table=block_table,
    )
    assert torch.all(actual[0][0] == -1)
    assert torch.all(actual[1][0] == 0)
    assert torch.all(actual[2][0] == -1)
    assert torch.all(actual[3][0] == -1)
    torch.testing.assert_close(actual[4][0], torch.tensor(0.75, dtype=torch.float32, device="npu"))


@pytest.mark.parametrize("tied_scores", [False, True])
def test_prefetch_streaming_tail_negative_weights_and_ties(
    tied_scores: bool, memfabric_offload: ModuleType | None
) -> None:
    torch = _runtime()
    case = _make_case(
        torch,
        threshold=float("nan"),
        budget=1,
        lengths=(6145,),
        physical=2048,
        block_size=16,
        sparse_count=2048,
        negative_weights=True,
        tied_scores=tied_scores,
        offload=memfabric_offload,
    )
    actual = _run(torch, case, threshold=float("nan"), budget=1)
    expected = compute_expected(case.golden)
    _assert_selected_and_rows(actual, case, expected)
    assert actual[2].shape == (1, 1)


@pytest.mark.parametrize("block_size", [16, 48, 128, 1024])
def test_prefetch_paged_key_tiles(block_size: int, memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(torch, lengths=(257, 129, 0), block_size=block_size, offload=memfabric_offload)
    actual = _run(torch, case, threshold=-np.inf, budget=4)
    _assert_selected_and_rows(actual, case, compute_expected(case.golden))


def test_prefetch_batch_reuse_and_exact_k(memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(
        torch, lengths=(7,), physical=2, block_size=16, sparse_count=7, budget=0, offload=memfabric_offload
    )
    actual = _run(torch, case, threshold=-np.inf, budget=0)
    expected = compute_expected(case.golden)
    _assert_selected_and_rows(actual, case, expected)


def test_prefetch_batch_larger_than_one_core(memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(
        torch, lengths=(1,) * 65, physical=4, block_size=16, sparse_count=1, budget=1, offload=memfabric_offload
    )
    actual = _run(torch, case, threshold=-np.inf, budget=1)
    expected = compute_expected(case.golden)
    _assert_selected_and_rows(actual, case, expected)


def test_prefetch_event_latency(pytestconfig: pytest.Config, memfabric_offload: ModuleType | None) -> None:
    if not pytestconfig.getoption("--benchmark"):
        pytest.skip("pass --benchmark to collect NPU event timings")
    torch = _runtime()
    case = _make_case(torch, threshold=-np.inf, budget=4, offload=memfabric_offload)
    threshold_tensor = torch.full((case.query.size(0),), -np.inf, dtype=torch.float32, device="npu")
    for budget in (0, 4):
        for _ in range(5):
            _run(torch, case, threshold=threshold_tensor, budget=budget)
        torch.npu.synchronize()
        samples = []
        for _ in range(20):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            _run(torch, case, threshold=threshold_tensor, budget=budget)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        print(f"prefetch_event_ms budget={budget} min={min(samples):.4f} avg={sum(samples) / len(samples):.4f}")
