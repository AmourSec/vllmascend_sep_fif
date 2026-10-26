from dataclasses import dataclass, replace

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.utils import bootstrap_custom_op_env  # noqa: E402

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401


@dataclass(frozen=True, slots=True)
class PlanCase:
    req_ids: torch.Tensor
    topk_indices: torch.Tensor
    stable_prefix_lens: torch.Tensor
    visible_seq_lens: torch.Tensor
    token_to_req: torch.Tensor
    block_table: torch.Tensor
    active_rows: torch.Tensor
    last_req_ids: torch.Tensor
    slot_to_token: torch.Tensor
    lru_slots: torch.Tensor
    current_slots: torch.Tensor
    miss_count: torch.Tensor
    miss_tokens: torch.Tensor
    miss_slots: torch.Tensor
    compact_workspace: torch.Tensor
    topk: int
    capacity: int


def _workspace_elements(topk: int, capacity: int) -> int:
    hash_capacity = 32
    while hash_capacity < 2 * topk:
        hash_capacity <<= 1
    return 3 * hash_capacity + 2 * capacity + topk


def _make_case(*, topk: int = 4, capacity: int = 4, rows: int = 1) -> PlanCase:
    device = torch.device("npu")
    topk_indices = torch.arange(topk, dtype=torch.int32, device=device).repeat(rows, 1)
    return PlanCase(
        req_ids=torch.tensor([100], dtype=torch.int64, device=device),
        topk_indices=topk_indices,
        stable_prefix_lens=torch.zeros(rows, dtype=torch.int32, device=device),
        visible_seq_lens=torch.full((rows,), 16, dtype=torch.int32, device=device),
        token_to_req=torch.zeros(rows, dtype=torch.int32, device=device),
        block_table=torch.arange(4, dtype=torch.int32, device=device).reshape(1, 4),
        active_rows=torch.tensor([rows], dtype=torch.int32, device=device),
        last_req_ids=torch.full((rows,), -1, dtype=torch.int64, device=device),
        slot_to_token=torch.full((rows, capacity), -1, dtype=torch.int32, device=device),
        lru_slots=torch.arange(capacity, dtype=torch.int32, device=device).repeat(rows, 1),
        current_slots=torch.full((rows, topk), -1, dtype=torch.int32, device=device),
        miss_count=torch.zeros(rows, dtype=torch.int32, device=device),
        miss_tokens=torch.full((rows, topk), -1, dtype=torch.int32, device=device),
        miss_slots=torch.full((rows, topk), -1, dtype=torch.int32, device=device),
        compact_workspace=torch.empty((1, 1), dtype=torch.int32, device=device),
        topk=topk,
        capacity=capacity,
    )


def _run_plan(case: PlanCase) -> None:
    torch.ops._C_ascend.npu_sparse_kv_plan(
        case.req_ids,
        case.topk_indices,
        case.stable_prefix_lens,
        case.visible_seq_lens,
        case.token_to_req,
        case.block_table,
        case.active_rows,
        case.last_req_ids,
        case.slot_to_token,
        case.lru_slots,
        case.current_slots,
        case.miss_count,
        case.miss_tokens,
        case.miss_slots,
        case.compact_workspace,
        case.topk,
        case.capacity,
        16,
        4,
        4,
    )


@pytest.mark.npu_inline
def test_plan_supports_capacity_four_with_duplicate_topk() -> None:
    # Given: the public planner's 2048 selections contain three valid unique tokens for a four-slot cache.
    case = _make_case(topk=2048, capacity=4)
    case.topk_indices.fill_(3)
    case.topk_indices[0, :4] = torch.tensor([3, 3, 1, 2], dtype=torch.int32, device="npu")

    # When: standalone Plan consumes the UB placeholder workspace.
    _run_plan(case)
    torch_npu.npu.synchronize()

    # Then: only the first duplicate is assigned and all unique misses fit.
    torch.testing.assert_close(case.current_slots[0, :4].cpu(), torch.tensor([0, -1, 1, 2], dtype=torch.int32))
    torch.testing.assert_close(case.miss_count.cpu(), torch.tensor([3], dtype=torch.int32))


@pytest.mark.npu_inline
@pytest.mark.parametrize("capacity", [4, 16384])
def test_plan_accepts_legacy_flat_workspace_for_ub_and_gm(capacity: int) -> None:
    # Given: the legacy standalone caller supplies a flat UB placeholder or complete GM storage.
    case = _make_case(capacity=capacity)
    elements = 1 if capacity == 4 else _workspace_elements(case.topk, capacity)
    case = replace(case, compact_workspace=torch.empty(elements, dtype=torch.int32, device="npu"))

    # When: standalone Plan validates and consumes the legacy workspace form.
    _run_plan(case)
    torch_npu.npu.synchronize()

    # Then: planning runs and records ownership for the active row.
    torch.testing.assert_close(case.last_req_ids.cpu(), torch.tensor([100], dtype=torch.int64))


@pytest.mark.npu_inline
def test_plan_accepts_scalar_active_rows() -> None:
    # Given: a legacy caller represents the one planner count as a scalar tensor.
    case = _make_case()
    case = replace(case, active_rows=torch.tensor(1, dtype=torch.int32, device="npu"))

    # When: standalone Plan consumes the scalar count accepted by the host tiler.
    _run_plan(case)
    torch_npu.npu.synchronize()

    # Then: the active row is planned normally.
    torch.testing.assert_close(case.last_req_ids.cpu(), torch.tensor([100], dtype=torch.int64))


@pytest.mark.npu_inline
@pytest.mark.parametrize("capacity", [4, 16384])
def test_plan_rejects_short_current_slots_before_mutation(capacity: int) -> None:
    # Given: UB and GM planner cases whose current_slots row cannot hold TopK.
    case = _make_case(capacity=capacity)
    if capacity == 16384:
        width = _workspace_elements(case.topk, capacity)
        case = replace(case, compact_workspace=torch.empty((1, width), dtype=torch.int32, device="npu"))
    case = replace(case, current_slots=torch.full((1, 1), -7, dtype=torch.int32, device="npu"))
    last_req_ids_before = case.last_req_ids.clone()

    # When/Then: validation rejects the underrun before any persistent state changes.
    with pytest.raises(RuntimeError, match="current_slots"):
        _run_plan(case)
    torch.testing.assert_close(case.last_req_ids, last_req_ids_before)


@pytest.mark.npu_inline
def test_plan_rejects_wrong_mutable_dtype_before_mutation() -> None:
    # Given: a planner state row with a floating-point mutable tensor.
    case = _make_case()
    case = replace(case, current_slots=case.current_slots.to(torch.float32))
    last_req_ids_before = case.last_req_ids.clone()

    # When/Then: adapter validation fails before dispatch mutates request ownership.
    with pytest.raises(RuntimeError, match="current_slots"):
        _run_plan(case)
    torch.testing.assert_close(case.last_req_ids, last_req_ids_before)


@pytest.mark.npu_inline
def test_plan_rejects_noncontiguous_mutable_before_mutation() -> None:
    # Given: correct dimensions backed by a transposed mutable view.
    case = _make_case(rows=2)
    current_slots = torch.full((case.topk, 2), -1, dtype=torch.int32, device="npu").transpose(0, 1)
    case = replace(case, current_slots=current_slots)
    last_req_ids_before = case.last_req_ids.clone()

    # When/Then: noncontiguous mutation is rejected rather than redirected to a copy.
    with pytest.raises(RuntimeError, match="current_slots"):
        _run_plan(case)
    torch.testing.assert_close(case.last_req_ids, last_req_ids_before)


@pytest.mark.npu_inline
def test_plan_rejects_mixed_device_state_before_mutation() -> None:
    # Given: current_slots is the only planner tensor not resident on the dispatch device.
    case = _make_case()
    case = replace(case, current_slots=torch.full((1, case.topk), -1, dtype=torch.int32))
    last_req_ids_before = case.last_req_ids.clone()

    # When/Then: mixed-device state fails before the first persistent field is mutated.
    with pytest.raises(RuntimeError, match="current_slots"):
        _run_plan(case)
    torch.testing.assert_close(case.last_req_ids, last_req_ids_before)


@pytest.mark.npu_inline
def test_plan_rejects_gm_workspace_with_wrong_row_stride_before_mutation() -> None:
    # Given: a GM-path workspace whose second row starts at a wider stride than the kernel uses.
    case = _make_case(capacity=16384, rows=2)
    width = _workspace_elements(case.topk, case.capacity)
    case = replace(case, compact_workspace=torch.empty((2, width + 1), dtype=torch.int32, device="npu"))
    last_req_ids_before = case.last_req_ids.clone()

    # When/Then: the unsafe stride is rejected before planner state mutation.
    with pytest.raises(RuntimeError, match="compact_workspace"):
        _run_plan(case)
    torch.testing.assert_close(case.last_req_ids, last_req_ids_before)
