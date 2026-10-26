from dataclasses import replace

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.utils import bootstrap_custom_op_env  # noqa: E402

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

from .assertions import (  # noqa: E402
    assert_state_matches,
    assert_states_equal,
    oracle_after,
)
from .helpers import (  # noqa: E402
    QUERY_HEADS,
    TOPK,
    FusedCase,
    PlannerStateTensors,
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


def _run_recurrent_differential(
    case: FusedCase,
    fused_state: PlannerStateTensors,
    baseline_state: PlannerStateTensors,
) -> torch.Tensor:
    before = fused_state.clone()
    fused_indices = run_fused(case, fused_state)
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_state, baseline_indices)
    torch_npu.npu.synchronize()
    torch.testing.assert_close(fused_indices, baseline_indices, rtol=0, atol=0)
    assert_state_matches(fused_state, oracle_after(case, before, fused_indices))
    assert_states_equal(fused_state, baseline_state)
    return fused_indices


@pytest.mark.parametrize("seq_len", [16_384, 65_536, 131_072, 140_000, 143_360])
@pytest.mark.npu_stress
def test_long_contexts_match_independent_planner_oracle(seq_len: int) -> None:
    # Given: one GLM-5.2 decode row at each production long-context length.
    case = make_case([seq_len], [1], seed=seq_len)
    before = make_state(case)
    state = before.clone()
    baseline_state = before.clone()

    # When: fused index selection and planning run together.
    indices = run_fused(case, state)
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_state, baseline_indices)
    torch_npu.npu.synchronize()

    # Then: selection, standalone planning, and all seven fused planner fields agree.
    torch.testing.assert_close(indices, baseline_indices, rtol=0, atol=0)
    assert_state_matches(state, oracle_after(case, before, indices))
    assert_states_equal(state, baseline_state)


@pytest.mark.npu_stress
@pytest.mark.npu_inline
def test_batch32_140k_ragged_mtp_stress() -> None:
    # Given: the mandatory maximum batch, 140k context, and ragged 1/2/4/6 MTP spans.
    batch = 32
    spans = [1, 2, 4, 6] * (batch // 4)
    case = make_case([140_000] * batch, spans, capacity=8192, seed=320_140)
    before = make_state(case)
    state = before.clone()
    baseline_state = before.clone()

    # When: all query rows share one graph-sized invocation.
    indices = run_fused(case, state)
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_state, baseline_indices)
    torch_npu.npu.synchronize()

    # Then: long-context selection and the large GM planner branch match both baselines.
    assert indices.shape == (sum(spans), TOPK)
    torch.testing.assert_close(indices, baseline_indices, rtol=0, atol=0)
    assert_state_matches(state, oracle_after(case, before, indices))
    assert_states_equal(state, baseline_state)


def test_ties_are_checked_against_each_produced_index_set() -> None:
    # Given: zero scores create a deliberate TopK tie wider than 2048 tokens.
    case = make_case([TOPK + 257], [1])
    case.query.zero_()
    case.key.zero_()
    fused_before = make_state(case)
    baseline_before = fused_before.clone()
    fused_state = fused_before.clone()
    baseline_state = baseline_before.clone()

    # When: fused and standalone indexers independently resolve the tie.
    fused_indices = run_fused(case, fused_state)
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_state, baseline_indices)
    torch_npu.npu.synchronize()

    # Then: each planner is checked against its own legal result without assuming tie order.
    for indices in (fused_indices, baseline_indices):
        selected = indices[0].cpu().tolist()
        assert len(set(selected)) == TOPK
        assert all(0 <= token < TOPK + 257 for token in selected)
    assert_state_matches(fused_state, oracle_after(case, fused_before, fused_indices))
    assert_state_matches(baseline_state, oracle_after(case, baseline_before, baseline_indices))


@pytest.mark.npu_inline
def test_recurrent_mtp_rollback_row_compaction_and_request_reuse() -> None:
    # Given: four requests with ragged MTP rows warm persistent planner state.
    case = make_case([4097, 4111, 4129, 4153], [1, 2, 4, 6])
    initial = make_state(case)
    state = initial.clone()
    baseline_state = initial.clone()
    _run_recurrent_differential(case, state, baseline_state)

    # When: an identical recurrent step reuses resident slots.
    _run_recurrent_differential(case, state, baseline_state)

    # Then: recurrent hits still match the state machine.
    assert_states_equal(state, baseline_state)

    # Given: MTP rolls the stable and visible boundaries back for the same requests.
    case.stable_prefix_lens.sub_(2)
    case.visible_seq_lens.sub_(2)
    # When: suffix residents are reconsidered without a request-id reset.
    _run_recurrent_differential(case, state, baseline_state)

    # Then: stable-prefix invalidation follows the independent state machine.
    assert_states_equal(state, baseline_state)

    # Given: rows compact and are reused for new request ids.
    compact_rows = case.query.shape[0] - 3
    case.req_ids.add_(10_000)
    case.active_rows.fill_(compact_rows)
    before_reuse = state.clone()

    # When: the same IndexCache-like tensor addresses serve the compacted batch.
    _run_recurrent_differential(case, state, baseline_state)

    # Then: active rows reset for request identity and padding rows remain untouched.
    assert_states_equal(state, baseline_state)
    for actual, original in zip(state.tensors()[:-1], before_reuse.tensors()[:-1]):
        torch.testing.assert_close(actual[compact_rows:], original[compact_rows:], rtol=0, atol=0)


@pytest.mark.parametrize("active_rows", [0, 99])
@pytest.mark.npu_inline
def test_active_rows_clamps_to_query_row_bounds(active_rows: int) -> None:
    # Given: activeRows is either empty or larger than the two-row query tensor.
    case = make_case([TOPK + 1], [2])
    case.active_rows.fill_(active_rows)
    before = make_state(case, fill=-63)
    state = before.clone()
    baseline_state = before.clone()

    # When: selection runs for the query while planning clamps its row count.
    indices = run_fused(case, state)
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_state, baseline_indices)
    torch_npu.npu.synchronize()

    # Then: output is complete and planner state follows the clamped oracle exactly.
    torch.testing.assert_close(indices, baseline_indices, rtol=0, atol=0)
    assert_states_equal(state, baseline_state)
    assert_state_matches(state, oracle_after(case, before, indices))
    if active_rows == 0:
        assert_states_equal(state, before)


@pytest.mark.parametrize("capacity", [2048, 4096])
@pytest.mark.npu_inline
def test_acl_graph_replay_preserves_addresses_and_eager_results(capacity: int) -> None:
    # Given: fixed graph tensors include two active and two padded query rows.
    case = make_case([TOPK + 257], [4], capacity=capacity)
    case.active_rows.fill_(2)
    initial = make_state(case, fill=-55)
    eager = initial.clone()
    baseline_state = initial.clone()
    eager_indices = run_fused(case, eager)
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_state, baseline_indices)
    torch_npu.npu.synchronize()
    torch.testing.assert_close(eager_indices, baseline_indices, rtol=0, atol=0)
    assert_states_equal(eager, baseline_state)
    assert_state_matches(eager, oracle_after(case, initial, eager_indices))
    for actual, original in zip(eager.tensors()[:-1], initial.tensors()[:-1]):
        torch.testing.assert_close(actual[2:], original[2:], rtol=0, atol=0)
    graph_state = initial.clone()
    addresses = tuple(tensor.data_ptr() for tensor in graph_state.tensors())
    graph = torch.npu.NPUGraph()

    # When: capture completes, state is restored, and the graph is replayed.
    try:
        with torch.npu.graph(graph):
            graph_indices = run_fused(case, graph_state)
        for target, source in zip(graph_state.tensors(), initial.tensors()):
            target.copy_(source)
        graph.replay()
        torch_npu.npu.synchronize()

        # Then: replay uses stable addresses and produces the eager indices and state.
        assert tuple(tensor.data_ptr() for tensor in graph_state.tensors()) == addresses
        torch.testing.assert_close(graph_indices, eager_indices, rtol=0, atol=0)
        assert_states_equal(graph_state, eager)
    finally:
        graph.reset()
        torch_npu.npu.synchronize()


@pytest.mark.parametrize("invalid_input", ["query_heads", "key_dtype", "weights_dtype", "visible_dtype"])
def test_invalid_shape_and_dtype_fail_before_state_mutation(invalid_input: str) -> None:
    # Given: one malformed fused input and a sentinel-filled planner state.
    case = make_case([TOPK + 1], [1])
    invalid_cases = {
        "query_heads": replace(case, query=case.query[:, : QUERY_HEADS - 1]),
        "key_dtype": replace(case, key=case.key.to(torch.bfloat16)),
        "weights_dtype": replace(case, weights=case.weights.to(torch.float32)),
        "visible_dtype": replace(case, visible_seq_lens=case.visible_seq_lens.to(torch.int64)),
    }
    case = invalid_cases[invalid_input]
    state = make_state(case, fill=-71)
    before = state.clone()

    # When/Then: validation fails synchronously and no planner field is modified.
    with pytest.raises(RuntimeError):
        run_fused(case, state)
        torch_npu.npu.synchronize()
    assert_states_equal(state, before)
