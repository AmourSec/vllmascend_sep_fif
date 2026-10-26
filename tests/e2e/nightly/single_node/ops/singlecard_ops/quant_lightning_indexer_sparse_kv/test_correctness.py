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
    HEAD_DIM,
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


def _assert_fused_differential(
    seq_lens: list[int],
    spans: list[int],
    *,
    capacity: int = TOPK,
    disjoint_permuted_blocks: bool = False,
    planner_extra_blocks: int = 0,
) -> None:
    case = make_case(
        seq_lens,
        spans,
        capacity=capacity,
        disjoint_permuted_blocks=disjoint_permuted_blocks,
    )
    if planner_extra_blocks:
        padding = case.kv_block_table.new_full((len(seq_lens), planner_extra_blocks), -1)
        case = replace(case, kv_block_table=torch.cat((case.kv_block_table, padding), dim=1))
    before = make_state(case)
    fused_state = before.clone()
    if before.compact_workspace.shape[1] * before.compact_workspace.element_size() <= 112 * 1024:
        # Exercise the runtime's minimal UB placeholder against the legacy full-workspace baseline.
        fused_state = replace(fused_state, compact_workspace=before.compact_workspace.new_empty((1, 1)))
    baseline_state = before.clone()

    fused_indices = run_fused(case, fused_state)
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_state, baseline_indices)
    torch_npu.npu.synchronize()

    torch.testing.assert_close(fused_indices, baseline_indices, rtol=0, atol=0)
    expected = oracle_after(case, before, fused_indices)
    assert_state_matches(fused_state, expected)
    assert_state_matches(baseline_state, expected)
    assert_states_equal(fused_state, baseline_state)


@pytest.mark.npu_smoke
def test_smoke_fused_matches_qli_then_old_plan() -> None:
    # Given/When/Then: one real unaligned MTP invocation covers selection and all planner outputs.
    _assert_fused_differential([TOPK + 1], [2])


@pytest.mark.parametrize(
    ("batch_size", "spans"),
    [
        (1, [1]),
        (2, [1, 2]),
        (4, [1, 2, 4, 6]),
        (8, [1, 2, 4, 6, 1, 2, 4, 6]),
        (16, [1, 2, 4, 6] * 4),
    ],
)
def test_random_fused_matches_qli_then_old_plan_for_batch_and_ragged_mtp(
    batch_size: int,
    spans: list[int],
) -> None:
    # Given: GLM-5.2 shaped random inputs with ragged MTP query rows.
    seq_lens = [TOPK + 137 + request * 17 for request in range(batch_size)]

    # When/Then: fused selection and every planner mutation match the two old operations.
    _assert_fused_differential(seq_lens, spans)


def test_multi_request_mtp_uses_disjoint_permuted_physical_keys() -> None:
    # Given/When/Then: distinct request key regions expose request-0 aliasing or logical/physical block confusion.
    _assert_fused_differential(
        [TOPK + 129, TOPK + 257, TOPK + 385, TOPK + 513],
        [1, 2, 4, 6],
        disjoint_permuted_blocks=True,
    )


def test_planner_and_indexer_block_tables_can_have_different_widths() -> None:
    # Distinct requests expose an incorrect planner row stride when its table is wider.
    _assert_fused_differential(
        [TOPK + 129, TOPK + 257],
        [1, 2],
        disjoint_permuted_blocks=True,
        planner_extra_blocks=3,
    )


@pytest.mark.parametrize("seq_len", [1999, TOPK, TOPK + 1, TOPK + 129])
def test_sequence_boundaries_match_baseline(seq_len: int) -> None:
    # Given/When/Then: below, at, above, and unaligned TopK boundaries are exact.
    _assert_fused_differential([seq_len], [1])


@pytest.mark.parametrize("capacity", [2048, 4096, 7168, 7169, 8192])
@pytest.mark.npu_inline
def test_ub_and_gm_workspace_capacity_branches_match_oracle(capacity: int) -> None:
    # Given/When/Then: planner scratch crosses its UB/GM branch without changing state semantics.
    _assert_fused_differential([TOPK + 257], [2], capacity=capacity)


def test_key_scale_binding_normalizes_rank3_exactly_like_rank2() -> None:
    # Given: the binding-supported [blocks, 128] and [blocks, 128, 1] key scales.
    rank2_case = make_case([TOPK + 1], [1])
    rank3_case = replace(rank2_case, key_scale=rank2_case.key_scale.unsqueeze(2))
    before = make_state(rank2_case)
    rank2_state = before.clone()
    rank3_state = before.clone()

    # When: fused selection consumes both representations of the same scales.
    rank2_indices = run_fused(rank2_case, rank2_state)
    rank3_indices = run_fused(rank3_case, rank3_state)
    torch_npu.npu.synchronize()

    # Then: indices and every planner mutation are exact, not merely valid independently.
    torch.testing.assert_close(rank3_indices, rank2_indices, rtol=0, atol=0)
    assert_states_equal(rank3_state, rank2_state)
    assert_state_matches(rank3_state, oracle_after(rank3_case, before, rank3_indices))


def test_axis0_strided_key_and_scale_match_contiguous_and_baseline() -> None:
    # Given: physical padding separates every key and scale block along axis 0.
    contiguous_case = make_case([TOPK + 257], [2])
    blocks = contiguous_case.key.shape[0]
    key_storage = torch.zeros(
        (blocks * 2, 128, 1, HEAD_DIM),
        dtype=contiguous_case.key.dtype,
        device="npu",
    )
    scale_storage = torch.zeros((blocks * 2, 128), dtype=torch.float32, device="npu")
    strided_key = key_storage[::2]
    strided_scale = scale_storage[::2]
    strided_key.copy_(contiguous_case.key)
    strided_scale.copy_(contiguous_case.key_scale)
    assert strided_key.stride(0) == 2 * contiguous_case.key.stride(0)
    assert strided_scale.stride(0) == 2 * contiguous_case.key_scale.stride(0)
    strided_case = replace(contiguous_case, key=strided_key, key_scale=strided_scale)
    before = make_state(contiguous_case)
    contiguous_state = before.clone()
    strided_state = before.clone()
    baseline_state = before.clone()

    # When: fused contiguous, fused strided, and standalone strided routes execute.
    contiguous_indices = run_fused(contiguous_case, contiguous_state)
    strided_indices = run_fused(strided_case, strided_state)
    baseline_indices = run_indexer(strided_case)
    run_plan(strided_case, baseline_state, baseline_indices)
    torch_npu.npu.synchronize()

    # Then: real axis-0 strides do not affect selection or any planner state.
    torch.testing.assert_close(strided_indices, contiguous_indices, rtol=0, atol=0)
    torch.testing.assert_close(strided_indices, baseline_indices, rtol=0, atol=0)
    assert_states_equal(strided_state, contiguous_state)
    assert_states_equal(strided_state, baseline_state)


def test_duplicate_semantics_match_oracle_for_standalone_planner() -> None:
    # Given: duplicate TopK tokens and duplicate residents in different LRU positions.
    case = make_case([32], [1], capacity=4)
    before = make_state(case)
    before.last_req_ids.fill_(1000)
    before.slot_to_token[0].copy_(torch.tensor([3, 8, 3, 12], dtype=torch.int32, device="npu"))
    before.lru_slots[0].copy_(torch.tensor([2, 1, 0, 3], dtype=torch.int32, device="npu"))
    indices = torch.full((1, TOPK), -1, dtype=torch.int32, device="npu")
    indices[0, :4] = torch.tensor([3, 3, 9, 10], dtype=torch.int32, device="npu")
    state = before.clone()

    # When: the old planner processes the crafted state contract directly.
    run_plan(case, state, indices)
    torch_npu.npu.synchronize()

    # Then: first TopK occurrence and last resident LRU owner match the independent oracle.
    assert_state_matches(state, oracle_after(case, before, indices))
