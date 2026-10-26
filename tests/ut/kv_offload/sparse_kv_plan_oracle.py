from dataclasses import dataclass
from typing import Final

INVALID_TOKEN: Final = -1


@dataclass(frozen=True)
class PlanInputs:
    req_ids: list[int]
    topk_indices: list[list[int]]
    stable_prefix_lens: list[int]
    visible_seq_lens: list[int]
    token_to_req: list[int]
    block_table: list[list[int]]
    active_rows: int
    max_token: int
    block_size: int
    host_num_blocks: int


@dataclass(frozen=True)
class PlanState:
    last_req_ids: list[int]
    slot_to_token: list[list[int]]
    lru_slots: list[list[int]]
    current_slots: list[list[int]]
    miss_count: list[int]
    miss_tokens: list[list[int]]
    miss_slots: list[list[int]]


def _source_is_valid(inputs: PlanInputs, row: int, token: int) -> bool:
    request = inputs.token_to_req[row]
    if not 0 <= token < min(inputs.max_token, inputs.visible_seq_lens[row]):
        return False
    if not 0 <= request < len(inputs.req_ids) or inputs.block_size <= 0:
        return False
    block_id = token // inputs.block_size
    if not 0 <= block_id < len(inputs.block_table[request]):
        return False
    physical_block = inputs.block_table[request][block_id]
    return 0 <= physical_block < inputs.host_num_blocks


def sparse_kv_plan_reference(inputs: PlanInputs, before: PlanState) -> PlanState:
    last_req_ids = before.last_req_ids.copy()
    slot_to_token = [row.copy() for row in before.slot_to_token]
    lru_slots = [row.copy() for row in before.lru_slots]
    current_slots = [row.copy() for row in before.current_slots]
    miss_count = before.miss_count.copy()
    miss_tokens = [row.copy() for row in before.miss_tokens]
    miss_slots = [row.copy() for row in before.miss_slots]
    max_rows = len(inputs.topk_indices)
    num_rows = min(max(inputs.active_rows, 0), max_rows)

    for row in range(num_rows):
        topk_row = inputs.topk_indices[row]
        capacity = len(slot_to_token[row])
        current_slots[row] = [INVALID_TOKEN] * len(topk_row)
        miss_tokens[row] = [INVALID_TOKEN] * len(topk_row)
        miss_slots[row] = [INVALID_TOKEN] * len(topk_row)
        miss_count[row] = 0

        request = inputs.token_to_req[row]
        current_req_id = inputs.req_ids[request] if 0 <= request < len(inputs.req_ids) else INVALID_TOKEN
        if last_req_ids[row] != current_req_id:
            last_req_ids[row] = current_req_id
            slot_to_token[row] = [INVALID_TOKEN] * capacity
            lru_slots[row] = list(range(capacity))

        first_position: dict[int, int] = {}
        for position, token in enumerate(topk_row):
            if _source_is_valid(inputs, row, token) and token not in first_position:
                first_position[token] = position

        stable_prefix = min(max(inputs.stable_prefix_lens[row], 0), inputs.max_token)
        evictable: list[int] = []
        hits: list[int] = []
        owner: dict[int, int] = {}
        for slot in lru_slots[row]:
            if not 0 <= slot < capacity:
                continue
            token = slot_to_token[row][slot]
            if 0 <= token < inputs.max_token and token >= stable_prefix:
                slot_to_token[row][slot] = INVALID_TOKEN
                token = INVALID_TOKEN
            if 0 <= token < inputs.max_token and token in first_position:
                hits.append(slot)
                owner[token] = slot
            else:
                evictable.append(slot)

        for token, slot in owner.items():
            current_slots[row][first_position[token]] = slot

        misses = [position for position in first_position.values() if current_slots[row][position] < 0]
        assign_count = min(len(misses), len(evictable))
        for miss_index in range(assign_count):
            slot = evictable[miss_index]
            position = misses[miss_index]
            token = topk_row[position]
            slot_to_token[row][slot] = token
            current_slots[row][position] = slot
            miss_tokens[row][miss_index] = token
            miss_slots[row][miss_index] = slot

        unassigned = evictable[assign_count:]
        rebuilt_lru = unassigned + evictable[:assign_count] + hits
        lru_slots[row][: len(rebuilt_lru)] = rebuilt_lru
        miss_count[row] = assign_count

    return PlanState(
        last_req_ids=last_req_ids,
        slot_to_token=slot_to_token,
        lru_slots=lru_slots,
        current_slots=current_slots,
        miss_count=miss_count,
        miss_tokens=miss_tokens,
        miss_slots=miss_slots,
    )
