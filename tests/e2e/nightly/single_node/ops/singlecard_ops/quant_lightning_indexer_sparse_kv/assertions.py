import torch

from tests.e2e.nightly.single_node.ops.singlecard_ops.quant_lightning_indexer_sparse_kv.helpers import (
    BLOCK_SIZE,
    FusedCase,
    PlannerStateTensors,
)
from tests.ut.kv_offload.sparse_kv_plan_oracle import PlanInputs, PlanState, sparse_kv_plan_reference


def oracle_after(case: FusedCase, before: PlannerStateTensors, indices: torch.Tensor) -> PlanState:
    planner_inputs = PlanInputs(
        req_ids=case.req_ids.cpu().tolist(),
        topk_indices=indices.cpu().tolist(),
        stable_prefix_lens=case.stable_prefix_lens.cpu().tolist(),
        visible_seq_lens=case.visible_seq_lens.cpu().tolist(),
        token_to_req=case.token_to_req.cpu().tolist(),
        block_table=case.kv_block_table.cpu().tolist(),
        active_rows=int(case.active_rows.cpu()[0]),
        max_token=case.max_token,
        block_size=BLOCK_SIZE,
        host_num_blocks=case.host_num_blocks,
    )
    prior = PlanState(
        last_req_ids=before.last_req_ids.cpu().tolist(),
        slot_to_token=before.slot_to_token.cpu().tolist(),
        lru_slots=before.lru_slots.cpu().tolist(),
        current_slots=before.current_slots.cpu().tolist(),
        miss_count=before.miss_count.cpu().tolist(),
        miss_tokens=before.miss_tokens.cpu().tolist(),
        miss_slots=before.miss_slots.cpu().tolist(),
    )
    return sparse_kv_plan_reference(planner_inputs, prior)


def assert_state_matches(actual: PlannerStateTensors, expected: PlanState) -> None:
    expected_fields = (
        expected.last_req_ids,
        expected.slot_to_token,
        expected.lru_slots,
        expected.current_slots,
        expected.miss_count,
        expected.miss_tokens,
        expected.miss_slots,
    )
    for tensor, values in zip(actual.tensors()[:-1], expected_fields):
        torch.testing.assert_close(tensor.cpu(), torch.tensor(values, dtype=tensor.dtype), rtol=0, atol=0)


def assert_states_equal(left: PlannerStateTensors, right: PlannerStateTensors) -> None:
    for left_tensor, right_tensor in zip(left.tensors()[:-1], right.tensors()[:-1]):
        torch.testing.assert_close(left_tensor, right_tensor, rtol=0, atol=0)
