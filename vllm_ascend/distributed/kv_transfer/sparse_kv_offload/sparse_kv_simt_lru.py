# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_indexer import (
    SparseKVIndexerPlanContext,
)

_MIN_HASH_CAPACITY = 32
_INT32_BYTES = 4
_PLAN_ROW_UB_LIMIT_BYTES = 112 * 1024


def sparse_kv_plan_hash_capacity(topk: int) -> int:
    """Return the power-of-two hash capacity used by SparseKvPlan."""
    target = 2 * topk
    capacity = _MIN_HASH_CAPACITY
    while capacity < target:
        capacity <<= 1
    return capacity


def sparse_kv_plan_workspace_elements(topk: int, capacity: int) -> int:
    hash_capacity = sparse_kv_plan_hash_capacity(topk)
    return 3 * hash_capacity + capacity + topk


@dataclass
class _SparseKVSimtLayerState:
    last_req_ids: torch.Tensor
    slot_to_token: torch.Tensor
    lru_slots: torch.Tensor
    miss_count: torch.Tensor
    miss_tokens: torch.Tensor
    miss_slots: torch.Tensor
    host_cache_bases: torch.Tensor


class SparseKVSimtLru:
    """Persistent A5 device state for sparse-KV Plan and Transfer.

    All tensors are allocated while KV caches are registered.  The decode hot
    path only selects existing views and invokes custom operators, which keeps
    their addresses stable for ACL graph capture and replay.
    """

    def __init__(
        self,
        *,
        max_rows: int,
        topk: int,
        capacity: int,
        max_token: int,
        block_size: int,
        host_num_blocks: int,
        token_size_bytes_k: int,
        token_size_bytes_v: int,
        host_k_bases: list[int],
        host_v_bases: list[int],
        current_slots: torch.Tensor,
        device: torch.device,
    ) -> None:
        if max_rows <= 0 or topk <= 0 or topk > capacity:
            raise ValueError(f"Invalid sparse KV SIMT shape: rows={max_rows}, topk={topk}, capacity={capacity}")
        self.max_rows = max_rows
        self.topk = topk
        self.capacity = capacity
        self.max_token = max_token
        self.block_size = block_size
        self.host_num_blocks = host_num_blocks
        self.token_size_bytes_k = token_size_bytes_k
        self.token_size_bytes_v = token_size_bytes_v
        self.current_slots = current_slots
        self.active_rows = torch.zeros(1, dtype=torch.int32, device=device)
        self.plan_owner_layer_id: int | None = None

        row_elements = sparse_kv_plan_workspace_elements(topk, capacity)
        workspace_shape = (
            (1, 1) if row_elements * _INT32_BYTES <= _PLAN_ROW_UB_LIMIT_BYTES else (max_rows, row_elements)
        )
        self.compact_workspace = torch.empty(workspace_shape, dtype=torch.int32, device=device)
        initial_lru = torch.arange(capacity, dtype=torch.int32, device=device).view(1, capacity)
        self.layers = [
            _SparseKVSimtLayerState(
                last_req_ids=torch.full((max_rows,), -1, dtype=torch.int64, device=device),
                slot_to_token=torch.full((max_rows, capacity), -1, dtype=torch.int32, device=device),
                lru_slots=initial_lru.repeat(max_rows, 1),
                miss_count=torch.zeros(max_rows, dtype=torch.int32, device=device),
                miss_tokens=torch.full((max_rows, topk), -1, dtype=torch.int32, device=device),
                miss_slots=torch.full((max_rows, topk), -1, dtype=torch.int32, device=device),
                host_cache_bases=torch.tensor(
                    [host_k_bases[layer_id], host_v_bases[layer_id]],
                    dtype=torch.int64,
                    device=device,
                ),
            )
            for layer_id in range(len(host_k_bases))
        ]

        self._plan_transfer_op = torch.ops._C_ascend.npu_sparse_kv_plan_transfer
        self._transfer_op = torch.ops._C_ascend.npu_sparse_kv_transfer
        self._fused_indexer_op = torch.ops._C_ascend.npu_quant_lightning_indexer_sparse_kv

    def set_active_rows(self, num_rows: int) -> None:
        if num_rows <= 0 or num_rows > self.max_rows:
            raise ValueError(f"Sparse KV SIMT active rows must be in [1, {self.max_rows}], got {num_rows}")
        self.active_rows.fill_(num_rows)

    def plan_and_transfer(
        self,
        *,
        layer_id: int,
        req_ids: torch.Tensor,
        topk_indices: torch.Tensor,
        stable_prefix_lens: torch.Tensor,
        visible_seq_lens: torch.Tensor,
        token_to_req: torch.Tensor,
        block_table: torch.Tensor,
        resident_k: torch.Tensor,
        resident_v: torch.Tensor,
    ) -> None:
        state = self.layers[layer_id]
        self._plan_transfer_op(
            req_ids,
            topk_indices,
            stable_prefix_lens,
            visible_seq_lens,
            token_to_req,
            block_table,
            self.active_rows,
            state.last_req_ids,
            state.slot_to_token,
            state.lru_slots,
            self.current_slots,
            state.miss_count,
            state.miss_tokens,
            state.miss_slots,
            self.compact_workspace,
            state.host_cache_bases,
            resident_k,
            resident_v,
            self.topk,
            self.capacity,
            self.max_token,
            self.block_size,
            self.host_num_blocks,
            self.token_size_bytes_k,
            self.token_size_bytes_v,
        )
        self.plan_owner_layer_id = layer_id

    def select_and_plan(
        self,
        context: SparseKVIndexerPlanContext,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_dequant_scale: torch.Tensor,
        key_dequant_scale: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        indexer_block_table: torch.Tensor,
    ) -> torch.Tensor:
        state = self.layers[context.layer_id]
        result = self._fused_indexer_op(
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            indexer_block_table,
            context.req_ids,
            context.stable_prefix_lens,
            context.visible_seq_lens,
            context.token_to_req,
            context.kv_block_table,
            self.active_rows,
            state.last_req_ids,
            state.slot_to_token,
            state.lru_slots,
            self.current_slots,
            state.miss_count,
            state.miss_tokens,
            state.miss_slots,
            self.compact_workspace,
            self.capacity,
            self.max_token,
            self.block_size,
            self.host_num_blocks,
        )
        self.plan_owner_layer_id = context.layer_id
        return result

    def transfer_prepared_plan(
        self,
        *,
        layer_id: int,
        token_to_req: torch.Tensor,
        block_table: torch.Tensor,
        resident_k: torch.Tensor,
        resident_v: torch.Tensor,
    ) -> None:
        if self.plan_owner_layer_id != layer_id:
            raise RuntimeError(
                f"Sparse KV SIMT prepared Plan owner mismatch: expected={layer_id}, actual={self.plan_owner_layer_id}"
            )
        state = self.layers[layer_id]
        self._transfer_op(
            state.miss_count,
            state.miss_tokens,
            state.miss_slots,
            token_to_req,
            block_table,
            state.host_cache_bases,
            self.active_rows,
            resident_k,
            resident_v,
            self.topk,
            self.capacity,
            self.block_size,
            self.host_num_blocks,
            self.token_size_bytes_k,
            self.token_size_bytes_v,
        )

    def transfer_reused_plan(
        self,
        *,
        layer_id: int,
        token_to_req: torch.Tensor,
        block_table: torch.Tensor,
        resident_k: torch.Tensor,
        resident_v: torch.Tensor,
    ) -> None:
        if self.plan_owner_layer_id is None:
            raise RuntimeError("Sparse KV SIMT cannot reuse TopK before a Plan has run")
        plan_state = self.layers[self.plan_owner_layer_id]
        target_state = self.layers[layer_id]
        self._transfer_op(
            plan_state.miss_count,
            plan_state.miss_tokens,
            plan_state.miss_slots,
            token_to_req,
            block_table,
            target_state.host_cache_bases,
            self.active_rows,
            resident_k,
            resident_v,
            self.topk,
            self.capacity,
            self.block_size,
            self.host_num_blocks,
            self.token_size_bytes_k,
            self.token_size_bytes_v,
        )
