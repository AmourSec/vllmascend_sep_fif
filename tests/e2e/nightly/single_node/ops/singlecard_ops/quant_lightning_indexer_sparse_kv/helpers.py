from dataclasses import dataclass
from typing import Final

import torch
import torch_npu

TOPK: Final = 2048
BLOCK_SIZE: Final = 128
QUERY_HEADS: Final = 32
HEAD_DIM: Final = 128


@dataclass(frozen=True)
class FusedCase:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor
    indexer_block_table: torch.Tensor
    req_ids: torch.Tensor
    stable_prefix_lens: torch.Tensor
    visible_seq_lens: torch.Tensor
    token_to_req: torch.Tensor
    kv_block_table: torch.Tensor
    active_rows: torch.Tensor
    capacity: int
    max_token: int
    host_num_blocks: int


@dataclass(frozen=True)
class PlannerStateTensors:
    last_req_ids: torch.Tensor
    slot_to_token: torch.Tensor
    lru_slots: torch.Tensor
    current_slots: torch.Tensor
    miss_count: torch.Tensor
    miss_tokens: torch.Tensor
    miss_slots: torch.Tensor
    compact_workspace: torch.Tensor

    def clone(self) -> "PlannerStateTensors":
        return PlannerStateTensors(*(tensor.clone() for tensor in self.tensors()))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.last_req_ids,
            self.slot_to_token,
            self.lru_slots,
            self.current_slots,
            self.miss_count,
            self.miss_tokens,
            self.miss_slots,
            self.compact_workspace,
        )


def require_ops() -> torch._ops._OpNamespace:
    required = ("npu_quant_lightning_indexer_sparse_kv", "npu_sparse_kv_plan")
    missing = [name for name in required if not hasattr(torch.ops._C_ascend, name)]
    if missing:
        raise AssertionError(f"required custom operators are missing: {missing}")
    return torch.ops._C_ascend


def make_case(
    seq_lens: list[int],
    query_spans: list[int],
    *,
    capacity: int = TOPK,
    seed: int = 19,
    invalid_planner_pages: bool = False,
    disjoint_permuted_blocks: bool = False,
) -> FusedCase:
    if len(seq_lens) != len(query_spans):
        raise AssertionError("each request needs one query span")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    block_counts = [(seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE for seq_len in seq_lens]
    mapped_blocks = sum(block_counts) if disjoint_permuted_blocks else max(block_counts)
    physical_blocks = max(1, mapped_blocks)
    rows = sum(query_spans)
    query_i8 = torch.randint(-12, 13, (rows, QUERY_HEADS, HEAD_DIM), dtype=torch.int8, generator=generator)
    key_i8 = torch.randint(
        -12,
        13,
        (physical_blocks, BLOCK_SIZE, 1, HEAD_DIM),
        dtype=torch.int8,
        generator=generator,
    )
    query = query_i8.to(torch.float8_e4m3fn).npu()
    key = key_i8.to(torch.float8_e4m3fn).npu()
    weights = torch.rand((rows, QUERY_HEADS), dtype=torch.bfloat16, generator=generator).npu()
    query_scale = torch.rand((rows, QUERY_HEADS), dtype=torch.float32, generator=generator).add_(0.25).npu()
    key_scale = torch.rand((physical_blocks, BLOCK_SIZE), dtype=torch.float32, generator=generator).add_(0.25).npu()
    cumulative_query = torch.tensor(query_spans, dtype=torch.int32).cumsum(0, dtype=torch.int32).npu()
    actual_key = torch.tensor(seq_lens, dtype=torch.int32, device="npu")
    table_width = max(1, max(block_counts))
    indexer_table = torch.full((len(seq_lens), table_width), -1, dtype=torch.int32)
    physical_offset = 0
    for request, block_count in enumerate(block_counts):
        block_ids = torch.arange(block_count, dtype=torch.int32)
        if disjoint_permuted_blocks:
            block_ids = (block_ids.flip(0) + physical_offset).contiguous()
            physical_offset += block_count
        indexer_table[request, :block_count] = block_ids
    kv_table = indexer_table.clone()
    if invalid_planner_pages:
        kv_table[0, 1] = -1
        if table_width > 3:
            kv_table[0, 3] = physical_blocks
    visible: list[int] = []
    token_to_req: list[int] = []
    stable: list[int] = []
    for request, (seq_len, span) in enumerate(zip(seq_lens, query_spans)):
        visible.extend(range(seq_len - span + 1, seq_len + 1))
        stable.extend([seq_len - span] * span)
        token_to_req.extend([request] * span)
    max_token = max(1, max(seq_lens))
    return FusedCase(
        query=query,
        key=key,
        weights=weights,
        query_scale=query_scale,
        key_scale=key_scale,
        actual_seq_lengths_query=cumulative_query,
        actual_seq_lengths_key=actual_key,
        indexer_block_table=indexer_table.npu(),
        req_ids=torch.arange(1000, 1000 + len(seq_lens), dtype=torch.int64, device="npu"),
        stable_prefix_lens=torch.tensor(stable, dtype=torch.int32, device="npu"),
        visible_seq_lens=torch.tensor(visible, dtype=torch.int32, device="npu"),
        token_to_req=torch.tensor(token_to_req, dtype=torch.int32, device="npu"),
        kv_block_table=kv_table.npu(),
        active_rows=torch.tensor([rows], dtype=torch.int32, device="npu"),
        capacity=capacity,
        max_token=max_token,
        host_num_blocks=physical_blocks,
    )


def make_state(case: FusedCase, *, extra_rows: int = 0, fill: int = -1) -> PlannerStateTensors:
    rows = case.query.shape[0] + extra_rows
    hash_capacity = 1 << (2 * TOPK - 1).bit_length()
    workspace_row_elements = 3 * hash_capacity + 2 * case.capacity + TOPK
    return PlannerStateTensors(
        last_req_ids=torch.full((rows,), fill, dtype=torch.int64, device="npu"),
        slot_to_token=torch.full((rows, case.capacity), fill, dtype=torch.int32, device="npu"),
        lru_slots=torch.arange(case.capacity, dtype=torch.int32, device="npu").repeat(rows, 1),
        current_slots=torch.full((rows, TOPK), fill, dtype=torch.int32, device="npu"),
        miss_count=torch.full((rows,), fill, dtype=torch.int32, device="npu"),
        miss_tokens=torch.full((rows, TOPK), fill, dtype=torch.int32, device="npu"),
        miss_slots=torch.full((rows, TOPK), fill, dtype=torch.int32, device="npu"),
        compact_workspace=torch.empty((rows, workspace_row_elements), dtype=torch.int32, device="npu"),
    )


def run_fused(case: FusedCase, state: PlannerStateTensors) -> torch.Tensor:
    op = require_ops().npu_quant_lightning_indexer_sparse_kv
    result = op(
        case.query,
        case.key,
        case.weights,
        case.query_scale,
        case.key_scale,
        case.actual_seq_lengths_query,
        case.actual_seq_lengths_key,
        case.indexer_block_table,
        case.req_ids,
        case.stable_prefix_lens,
        case.visible_seq_lens,
        case.token_to_req,
        case.kv_block_table,
        case.active_rows,
        state.last_req_ids,
        state.slot_to_token,
        state.lru_slots,
        state.current_slots,
        state.miss_count,
        state.miss_tokens,
        state.miss_slots,
        state.compact_workspace,
        case.capacity,
        case.max_token,
        BLOCK_SIZE,
        case.host_num_blocks,
    )
    return result.reshape(case.query.shape[0], TOPK)


def run_indexer(case: FusedCase) -> torch.Tensor:
    result = torch_npu.npu_quant_lightning_indexer(
        query=case.query,
        key=case.key,
        weights=case.weights,
        query_dequant_scale=case.query_scale,
        key_dequant_scale=case.key_scale,
        actual_seq_lengths_query=case.actual_seq_lengths_query,
        actual_seq_lengths_key=case.actual_seq_lengths_key,
        block_table=case.indexer_block_table,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=3,
    )
    return result.reshape(case.query.shape[0], TOPK)


def run_plan(case: FusedCase, state: PlannerStateTensors, indices: torch.Tensor) -> None:
    require_ops().npu_sparse_kv_plan(
        case.req_ids,
        indices,
        case.stable_prefix_lens,
        case.visible_seq_lens,
        case.token_to_req,
        case.kv_block_table,
        case.active_rows,
        state.last_req_ids,
        state.slot_to_token,
        state.lru_slots,
        state.current_slots,
        state.miss_count,
        state.miss_tokens,
        state.miss_slots,
        state.compact_workspace,
        TOPK,
        case.capacity,
        case.max_token,
        BLOCK_SIZE,
        case.host_num_blocks,
    )
