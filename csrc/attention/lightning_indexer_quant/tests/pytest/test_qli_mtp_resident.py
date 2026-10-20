from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from mtp_resident_cases import make_case, runtime
from mtp_resident_oracle import DType, Geometry, INDEX_BLOCK, causal_scores, topk

pytestmark = pytest.mark.npu


def assert_topk(actual: np.ndarray, expected: np.ndarray) -> None:
    for row in range(actual.shape[0]):
        np.testing.assert_array_equal(np.sort(actual[row, 0]), np.sort(expected[row, 0]))


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("geometry", [
    Geometry(), Geometry((1, 3, 2), (5, 35, 17)), Geometry((0, 2, 1), (0, 37, 21)),
])
def test_qli_mtp_mixed_query_causal_topk(dtype: DType, heads: int, geometry: Geometry) -> None:
    # Given mixed query counts and independent paged numerical scores.
    torch = runtime()
    case = make_case(torch, geometry, dtype=dtype, heads=heads)
    expected = topk(causal_scores(case.golden), 7)
    address_values = np.empty((len(geometry.query_counts), case.table.shape[1] * INDEX_BLOCK), dtype=np.int64)
    table = case.table.cpu().numpy()
    for request in range(address_values.shape[0]):
        for token in range(address_values.shape[1]):
            page = int(table[request, token // INDEX_BLOCK])
            address_values[request, token] = case.key[page, token % INDEX_BLOCK].data_ptr()
    addresses = torch.tensor(address_values, dtype=torch.int64, device="npu")
    # When the standalone MTP path scores and gathers each query row.
    sparse, gathered, candidates, stats, _next = torch.ops._C_ascend.npu_lightning_indexer_quant_prefetch(
        case.query, case.key, case.weights, case.query_scale, case.key_scale,
        case.query_ends, case.key_lengths, case.table, addresses, case.threshold, 128, 4, 7, 0.5,
    )
    # Then TopK is causal, and gathering uses the request's address row.
    assert_topk(sparse.cpu().numpy(), expected)
    for row, visible in enumerate(geometry.visible):
        np.testing.assert_array_equal(candidates[row].cpu().numpy(), np.arange(min(4, visible)))
        assert int(stats[row, 0].cpu()) == min(4, visible)
        for position, token in enumerate(sparse[row, 0].cpu().tolist()):
            if token >= 0:
                request = int(geometry.requests[row])
                page = int(table[request, token // INDEX_BLOCK])
                expected_bytes = case.key[page, token % INDEX_BLOCK].view(torch.uint8).reshape(-1)
                torch.testing.assert_close(gathered[row, position], expected_bytes, rtol=0, atol=0)
            else:
                assert torch.all(gathered[row, position] == 0)


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
@pytest.mark.parametrize("budget", [0, 1, 4, 129])
@pytest.mark.parametrize("typed_cache", [False, True])
def test_qli_resident_causal_topk_and_exact_kv(dtype: DType, budget: int, typed_cache: bool) -> None:
    # Given host blocks of 24 tokens, index blocks of 16, and multiple score tiles.
    case = make_case(runtime(), Geometry((2, 1), (137, 129), capacity=max(8, budget)),
                     dtype=dtype, typed_cache=typed_cache)
    scores = causal_scores(case.golden)
    # When resident prefetch admits the first budget candidates.
    sparse, candidates, stats, next_threshold = case.run(budget)
    # Then all output rows match the independent score oracle and both byte streams.
    assert_topk(sparse.cpu().numpy(), topk(scores, 7))
    assert candidates.shape == (3, max(1, budget))
    for row, values in enumerate(scores):
        count = min(budget, values.size)
        expected_candidates = np.full(max(1, budget), -1, dtype=np.int32)
        expected_candidates[:count] = np.arange(count)
        np.testing.assert_array_equal(candidates[row].cpu().numpy(), expected_candidates)
        np.testing.assert_array_equal(stats[row].cpu().numpy(), [count, count])
        np.testing.assert_array_equal(np.sort(case.lru[row].cpu().numpy()), np.arange(case.geometry.capacity))
        for token in range(count):
            case.assert_payload(row, token)
        boundary = np.sort(values)[-7]
        np.testing.assert_allclose(next_threshold[row].cpu().numpy(), boundary, rtol=2e-3, atol=2e-3)


def test_qli_resident_inactive_padded_row_preserves_all_state() -> None:
    # Given T=3 allocated rows but two live queries and a matching cumulative vector.
    case = make_case(runtime())
    case.query_ends.copy_(case.torch.tensor([2, 2], dtype=case.torch.int32, device="npu"))
    case.active.fill_(2)
    persistent = (case.resident_k, case.resident_v, case.last_req_ids, case.slots, case.lru)
    before = tuple(tensor[2].clone() for tensor in persistent)
    # When the padded row is dispatched alongside active rows.
    sparse, candidates, stats, next_threshold = case.run()
    # Then padding is deterministic and its complete cache state is preserved.
    assert case.torch.all(sparse[2] == -1)
    assert case.torch.all(candidates[2] == -1)
    assert case.torch.all(stats[2] == 0)
    case.torch.testing.assert_close(next_threshold[2], case.threshold[2])
    for expected, tensor in zip(before, persistent, strict=True):
        case.torch.testing.assert_close(tensor[2], expected, rtol=0, atol=0)
    np.testing.assert_array_equal(stats[:2].cpu().numpy(), [[4, 4], [4, 4]])


def test_qli_resident_memfabric_mapped_host_bases(memfabric_offload: ModuleType | None) -> None:
    # Given the existing LOCAL MemFabric fixture and contiguous mapped K/V pools.
    if memfabric_offload is None:
        pytest.skip("pass --source=host to exercise MemFabric mapped resident K/V pools")
    case = make_case(runtime(), offload=memfabric_offload)
    # When QLI copies candidates using the pools' DVAs.
    _sparse, candidates, stats, _next = case.run()
    # Then host-backed K and V are byte exact in every admitted resident slot.
    np.testing.assert_array_equal(stats.cpu().numpy(), [[4, 4]] * 3)
    for row in range(3):
        for token in candidates[row].cpu().tolist():
            case.assert_payload(row, token)


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
def test_qli_resident_threshold_filters_candidates_and_updates_ema(dtype: DType) -> None:
    # Given per-row finite thresholds derived independently from causal numeric scores.
    case = make_case(runtime(), dtype=dtype)
    scores = causal_scores(case.golden)
    thresholds = np.asarray([
        np.mean(np.sort(values)[values.size // 2 - 1:values.size // 2 + 1]) for values in scores
    ], dtype=np.float32)
    case.threshold.copy_(case.torch.tensor(thresholds, dtype=case.torch.float32, device="npu"))
    # When the threshold admits only qualifying tokens in logical-token order.
    _sparse, candidates, stats, next_threshold = case.run()
    # Then admitted tokens, real misses, exact bytes, and the TopK-boundary EMA agree.
    for row, values in enumerate(scores):
        tokens = np.flatnonzero(values > thresholds[row])[:4]
        expected = np.full(4, -1, dtype=np.int32)
        expected[:tokens.size] = tokens
        np.testing.assert_array_equal(candidates[row].cpu().numpy(), expected)
        np.testing.assert_array_equal(stats[row].cpu().numpy(), [tokens.size, tokens.size])
        for token in tokens:
            case.assert_payload(row, int(token))
        boundary = np.sort(values)[-7]
        np.testing.assert_allclose(next_threshold[row].cpu().numpy(),
                                   0.5 * thresholds[row] + 0.5 * boundary, rtol=2e-3, atol=2e-3)
