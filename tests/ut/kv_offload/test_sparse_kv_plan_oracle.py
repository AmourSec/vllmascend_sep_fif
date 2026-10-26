from dataclasses import replace

from tests.ut.kv_offload.sparse_kv_plan_oracle import PlanInputs, PlanState, sparse_kv_plan_reference


def _state(rows: int = 2, capacity: int = 4, topk: int = 4, fill: int = -9) -> PlanState:
    return PlanState(
        last_req_ids=[fill] * rows,
        slot_to_token=[[fill] * capacity for _ in range(rows)],
        lru_slots=[list(range(capacity)) for _ in range(rows)],
        current_slots=[[fill] * topk for _ in range(rows)],
        miss_count=[fill] * rows,
        miss_tokens=[[fill] * topk for _ in range(rows)],
        miss_slots=[[fill] * topk for _ in range(rows)],
    )


def _inputs(topk: list[list[int]], *, active_rows: int = 2) -> PlanInputs:
    return PlanInputs(
        req_ids=[100, 200],
        topk_indices=topk,
        stable_prefix_lens=[32, 32],
        visible_seq_lens=[32, 32],
        token_to_req=[0, 1],
        block_table=[[0, 1, 2, 3], [4, 5, 6, 7]],
        active_rows=active_rows,
        max_token=32,
        block_size=8,
        host_num_blocks=8,
    )


def test_duplicate_topk_uses_first_position_and_last_resident_owner() -> None:
    # Given: token 3 occupies two slots, with slot 0 later in old LRU order.
    before = replace(
        _state(rows=1),
        last_req_ids=[100],
        slot_to_token=[[3, 8, 3, 12]],
        lru_slots=[[2, 1, 0, 3]],
    )
    inputs = replace(
        _inputs([[3, 3, 9, 10]], active_rows=1),
        req_ids=[100],
        token_to_req=[0],
        block_table=[[0, 1, 2, 3]],
        stable_prefix_lens=[32],
        visible_seq_lens=[32],
    )

    # When: the planner handles duplicate selected and resident tokens.
    after = sparse_kv_plan_reference(inputs, before)

    # Then: only the first selection is a hit and the last LRU owner supplies it.
    assert after.current_slots == [[0, 1, 3, -1]]
    assert after.miss_count == [2]
    assert after.miss_tokens == [[3, 9, -1, -1]]
    assert after.lru_slots == [[1, 3, 2, 0]]


def test_suffix_invalidation_preserves_eviction_and_hit_order() -> None:
    # Given: two speculative suffix residents surround one stable hit.
    before = replace(
        _state(rows=1, capacity=5),
        last_req_ids=[100],
        slot_to_token=[[2, 11, 4, 19, -1]],
        lru_slots=[[4, 1, 0, 3, 2]],
    )
    inputs = replace(
        _inputs([[2, 6, 7, 8]], active_rows=1),
        req_ids=[100],
        token_to_req=[0],
        block_table=[[0, 1, 2, 3]],
        stable_prefix_lens=[10],
        visible_seq_lens=[32],
    )

    # When: suffix invalidation precedes resident hit classification.
    after = sparse_kv_plan_reference(inputs, before)

    # Then: old evictions feed misses in order, followed by the old stable hit.
    assert after.slot_to_token == [[2, 7, 4, 8, 6]]
    assert after.current_slots == [[0, 4, 1, 3]]
    assert after.miss_slots == [[4, 1, 3, -1]]
    assert after.lru_slots == [[2, 4, 1, 3, 0]]


def test_invalid_planner_pages_are_filtered_and_inactive_rows_untouched() -> None:
    # Given: the planner source table has invalid pages while row 1 is padding.
    before = _state()
    inputs = replace(
        _inputs([[0, 8, 16, 24], [4, 5, 6, 7]], active_rows=1),
        block_table=[[0, -1, 99, 3], [4, 5, 6, 7]],
    )

    # When: only one row is active.
    after = sparse_kv_plan_reference(inputs, before)

    # Then: invalid pages never become misses and every padded field is bitwise preserved.
    assert after.current_slots[0] == [0, -1, -1, 1]
    assert after.miss_count[0] == 2
    assert after.miss_tokens[0] == [0, 24, -1, -1]
    assert after.last_req_ids[1] == before.last_req_ids[1]
    assert after.slot_to_token[1] == before.slot_to_token[1]
    assert after.lru_slots[1] == before.lru_slots[1]
    assert after.current_slots[1] == before.current_slots[1]
    assert after.miss_count[1] == before.miss_count[1]
    assert after.miss_tokens[1] == before.miss_tokens[1]
    assert after.miss_slots[1] == before.miss_slots[1]


def test_recurrent_row_compaction_reset_and_rollback() -> None:
    # Given: two rows warm the cache for distinct request identities.
    first_inputs = _inputs([[0, 1, 2, 3], [8, 9, 10, 11]])
    first = sparse_kv_plan_reference(first_inputs, _state(fill=-1))
    assert first.miss_count == [4, 4]

    # When: compacted rows swap requests, as an IndexCache-style row reuse does.
    compacted = replace(
        first_inputs,
        topk_indices=[[8, 9, 10, 11], [0, 1, 2, 3]],
        token_to_req=[1, 0],
    )
    second = sparse_kv_plan_reference(compacted, first)

    # Then: req-id mismatch resets each physical row rather than leaking residency.
    assert second.last_req_ids == [200, 100]
    assert second.miss_count == [4, 4]
    assert second.current_slots == [[0, 1, 2, 3], [0, 1, 2, 3]]

    # Given/When: MTP rolls visible length back and reuses only row 0.
    rollback = replace(
        compacted,
        topk_indices=[[8, 9, 10, 11], [20, 21, 22, 23]],
        visible_seq_lens=[10, 32],
        active_rows=1,
    )
    third = sparse_kv_plan_reference(rollback, second)

    # Then: future tokens are rejected and the compacted padding row is unchanged.
    assert third.current_slots[0] == [0, 1, -1, -1]
    assert third.miss_count[0] == 0
    assert third.slot_to_token[1] == second.slot_to_token[1]
    assert third.current_slots[1] == second.current_slots[1]
