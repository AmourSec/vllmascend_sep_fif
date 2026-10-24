from dataclasses import dataclass
from typing import Protocol

import torch


class SparseKVIndexerPlanner(Protocol):
    def select_and_plan(
        self,
        context: "SparseKVIndexerPlanContext",
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_dequant_scale: torch.Tensor,
        key_dequant_scale: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        indexer_block_table: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(slots=True)
class SparseKVIndexerPlanContext:
    planner: SparseKVIndexerPlanner
    layer_id: int
    active_rows: int
    num_indexer_tokens: int
    req_ids: torch.Tensor
    stable_prefix_lens: torch.Tensor
    visible_seq_lens: torch.Tensor
    token_to_req: torch.Tensor
    kv_block_table: torch.Tensor
    prepared: bool = False

    def select_and_plan(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_dequant_scale: torch.Tensor,
        key_dequant_scale: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        indexer_block_table: torch.Tensor,
    ) -> torch.Tensor:
        result = self.planner.select_and_plan(
            self,
            query,
            key,
            weights,
            query_dequant_scale,
            key_dequant_scale,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            indexer_block_table,
        )
        self.prepared = True
        return result
