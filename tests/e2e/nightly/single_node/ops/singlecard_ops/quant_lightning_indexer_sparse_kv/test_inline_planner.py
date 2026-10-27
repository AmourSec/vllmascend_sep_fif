import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.utils import bootstrap_custom_op_env  # noqa: E402

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

from .assertions import assert_state_matches, assert_states_equal, oracle_after  # noqa: E402
from .helpers import (  # noqa: E402
    TOPK,
    make_case,
    make_state,
    require_ops,
    run_fused,
    run_indexer,
    run_plan,
)


@pytest.fixture(scope="module", autouse=True)
def _ops_available() -> None:
    require_ops()


def _assert_exact_stock_differential(
    seq_lens: list[int],
    spans: list[int],
    *,
    capacity: int = TOPK,
    seed: int = 19,
) -> None:
    case = make_case(seq_lens, spans, capacity=capacity, seed=seed)
    before = make_state(case)
    fused_state = before.clone()
    baseline_state = before.clone()
    fused_indices = run_fused(case, fused_state)
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_state, baseline_indices)
    torch_npu.npu.synchronize()
    torch.testing.assert_close(fused_indices, baseline_indices, rtol=0, atol=0)
    assert_states_equal(fused_state, baseline_state)
    assert_state_matches(fused_state, oracle_after(case, before, fused_indices))


@pytest.mark.npu_inline
@pytest.mark.parametrize("total_rows", [64, 65])
@pytest.mark.parametrize("capacity", [2048, 7168, 7169, 14336, 14337])
def test_inline_row_scheduler_boundary_matches_stock_route(total_rows: int, capacity: int) -> None:
    # Given: total T crosses the scheduler boundary that changes s1Base from two to four rows.
    spans = [2] * 32
    if total_rows == 65:
        spans[-1] = 3
    seq_lens = [TOPK + 129 + request for request in range(32)]

    # When/Then: every final TopK row is planned exactly where the stock route plans it.
    _assert_exact_stock_differential(seq_lens, spans, capacity=capacity)


@pytest.mark.npu_inline
@pytest.mark.parametrize("seq_lens", [[0, TOPK + 129], [0, 0]])
def test_zero_key_requests_match_stock(seq_lens: list[int]) -> None:
    # Given: either one or every request has no visible key token.
    case = make_case(seq_lens, [1] * len(seq_lens))
    before = make_state(case)
    baseline_state = before.clone()
    fused_state = before.clone()

    # When: stock QLI and the fused path process the valid zero-key representation.
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_state, baseline_indices)
    fused_indices = run_fused(case, fused_state)
    torch_npu.npu.synchronize()

    # Then: the inline consumer preserves every raw index and planner mutation.
    torch.testing.assert_close(fused_indices, baseline_indices, rtol=0, atol=0)
    assert_states_equal(fused_state, baseline_state)
    assert_state_matches(fused_state, oracle_after(case, before, fused_indices))


@pytest.mark.npu_inline
@pytest.mark.npu_stress
@pytest.mark.parametrize("capacity", [2048, 4096, 14337])
def test_inline_multiple_rows_share_a_long_multitrunk_request(capacity: int) -> None:
    # Given/When/Then: six final TopK rows from one 140k request each consume their own UB indices once.
    _assert_exact_stock_differential([140_000], [6], capacity=capacity, seed=601_400)


@pytest.mark.npu_inline
@pytest.mark.npu_stress
def test_inline_cap4096_batch32_multitrunk_rows() -> None:
    # Given/When/Then: T=96 schedules multiple final rows per AIV while three trunks reuse the TopK slab.
    _assert_exact_stock_differential([32_769] * 32, [3] * 32, capacity=4096, seed=4096_32_769)


@pytest.mark.npu_inline
def test_invalid_pages_plan_once_and_preserve_inactive_padding() -> None:
    # Given: only the planner table has invalid pages and three state rows are inactive padding.
    case = make_case([TOPK + 257], [2], invalid_planner_pages=True)
    assert bool((case.indexer_block_table >= 0).all())
    before = make_state(case, extra_rows=3, fill=-77)
    fused_state = before.clone()
    baseline_state = before.clone()

    # When: fused planning and exactly one stock Plan consume the same raw indices.
    fused_indices = run_fused(case, fused_state)
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_state, baseline_indices)
    torch_npu.npu.synchronize()

    # Then: active and inactive state matches once, and a second Plan changes the fixture.
    torch.testing.assert_close(fused_indices, baseline_indices, rtol=0, atol=0)
    assert_states_equal(fused_state, baseline_state)
    assert_state_matches(fused_state, oracle_after(case, before, fused_indices))
    for actual, original in zip(fused_state.tensors()[:-1], before.tensors()[:-1]):
        torch.testing.assert_close(actual[2:], original[2:], rtol=0, atol=0)
    twice_planned = baseline_state.clone()
    run_plan(case, twice_planned, baseline_indices)
    torch_npu.npu.synchronize()
    assert any(
        not torch.equal(once, twice) for once, twice in zip(baseline_state.tensors()[:-1], twice_planned.tensors()[:-1])
    )
