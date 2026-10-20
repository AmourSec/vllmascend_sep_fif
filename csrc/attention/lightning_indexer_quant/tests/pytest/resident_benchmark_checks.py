from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import assert_never

from resident_benchmark import CAPACITY, HOST_BLOCK, KEY_BYTES, TOPK, VALUE_BYTES, BenchmarkCase, ResidentSnapshot
from resident_benchmark_measure import CacheState

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True, slots=True)
class Traffic:
    admitted: tuple[int, ...]
    prefetch_misses: tuple[int, ...]
    transfer_misses: tuple[int, ...]


def expected_payload(case: BenchmarkCase, row: int, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    request = int(case.spec.requests[row])
    table = case.spec.table(HOST_BLOCK)
    pages = table[request, tokens // HOST_BLOCK]
    offsets = tokens % HOST_BLOCK
    return case.source_k.numpy()[pages, offsets], case.source_v.numpy()[pages, offsets]


def seeded_tokens(selected: np.ndarray, stable: int, state: CacheState) -> np.ndarray:
    tokens = selected[(selected >= 0) & (selected < stable)]
    match state:
        case "cold":
            return tokens[:0]
        case "partial":
            return tokens[::2]
        case "warm":
            return tokens
        case unreachable:
            assert_never(unreachable)


def seed_cache(case: BenchmarkCase, selected: np.ndarray, state: CacheState) -> ResidentSnapshot:
    torch = case.torch
    case.resident_k.view(torch.uint8).fill_(173)
    case.resident_v.view(torch.uint8).fill_(211)
    case.slots.fill_(-1)
    case.lru.copy_(torch.arange(CAPACITY, dtype=torch.int32, device="npu").repeat(case.spec.rows, 1))
    case.last_req_ids.copy_(case.req_ids[case.token_to_req.long()])
    resident_k = case.resident_k.view(torch.uint8).reshape(case.spec.rows, CAPACITY, KEY_BYTES)
    resident_v = case.resident_v.view(torch.uint8).reshape(case.spec.rows, CAPACITY, VALUE_BYTES)
    for row in range(case.spec.rows):
        tokens = seeded_tokens(selected[row], int(case.spec.stable[row]), state)
        count = len(tokens)
        if count:
            key, value = expected_payload(case, row, tokens)
            case.slots[row, :count].copy_(torch.from_numpy(tokens).to("npu"))
            resident_k[row, :count].copy_(torch.from_numpy(key).to("npu"))
            resident_v[row, :count].copy_(torch.from_numpy(value).to("npu"))
            order = np.concatenate((np.arange(count, CAPACITY), np.arange(count))).astype(np.int32)
            case.lru[row].copy_(torch.from_numpy(order).to("npu"))
    return case.snapshot()


def check_metadata(case: BenchmarkCase) -> np.ndarray:
    slots = case.slots.cpu().numpy()
    lru = case.lru.cpu().numpy()
    identities = case.last_req_ids.cpu().numpy()
    np.testing.assert_array_equal(identities, case.req_ids.cpu().numpy()[case.spec.requests])
    for row in range(case.spec.rows):
        valid = slots[row][slots[row] >= 0]
        assert len(np.unique(valid)) == len(valid)
        assert np.all(valid < case.spec.visible[row])
        np.testing.assert_array_equal(np.sort(lru[row]), np.arange(CAPACITY))
    return slots


def check_payloads(case: BenchmarkCase, tokens: np.ndarray, slots: np.ndarray) -> None:
    resident_k = case.resident_k.view(case.torch.uint8).reshape(case.spec.rows, CAPACITY, KEY_BYTES)
    resident_v = case.resident_v.view(case.torch.uint8).reshape(case.spec.rows, CAPACITY, VALUE_BYTES)
    for row in range(case.spec.rows):
        valid = tokens[row] >= 0
        if not np.any(valid):
            continue
        target_slots = slots[row, valid]
        assert np.all((target_slots >= 0) & (target_slots < CAPACITY))
        expected_k, expected_v = expected_payload(case, row, tokens[row, valid])
        indices = case.torch.tensor(target_slots, dtype=case.torch.int64, device="npu")
        np.testing.assert_array_equal(resident_k[row, indices].cpu().numpy(), expected_k)
        np.testing.assert_array_equal(resident_v[row, indices].cpu().numpy(), expected_v)


def check_qli_result(case: BenchmarkCase, result: tuple[Tensor, ...], expected: np.ndarray) -> None:
    selected = result[0].cpu().numpy()[:, 0]
    np.testing.assert_array_equal(np.sort(selected, axis=1), np.sort(expected, axis=1))
    for row in range(case.spec.rows):
        assert len(np.unique(selected[row])) == TOPK
        assert np.all((selected[row] >= 0) & (selected[row] < case.spec.visible[row]))
    mapping = check_metadata(case)
    candidates = result[1].cpu().numpy()
    stats = result[2].cpu().numpy()
    assert np.all(stats >= 0), "invalid resident row"
    assert np.all(stats[:, 1] <= stats[:, 0])
    np.testing.assert_array_equal(np.count_nonzero(candidates >= 0, axis=1), stats[:, 0])
    candidate_slots = np.full(candidates.shape, -1, dtype=np.int32)
    for row in range(case.spec.rows):
        admitted = candidates[row, :int(stats[row, 0])]
        assert np.all(np.diff(admitted) > 0)
        assert np.all(admitted < case.spec.visible[row])
        lookup = {int(token): slot for slot, token in enumerate(mapping[row]) if token >= 0}
        candidate_slots[row, :len(admitted)] = [lookup[int(token)] for token in admitted]
    check_payloads(case, candidates, candidate_slots)


def validate_chain(
    case: BenchmarkCase, result: tuple[Tensor, ...], expected: np.ndarray,
) -> Traffic:
    selected = result[0].cpu().numpy()[:, 0]
    np.testing.assert_array_equal(np.sort(selected, axis=1), np.sort(expected, axis=1))
    mapping = check_metadata(case)
    current_slots = case.current_slots.cpu().numpy()
    assert np.all((current_slots >= 0) & (current_slots < CAPACITY))
    np.testing.assert_array_equal(np.take_along_axis(mapping, current_slots, axis=1), selected)
    check_payloads(case, selected, current_slots)
    stats = result[2].cpu().numpy()
    counts = case.miss_count.cpu().numpy()
    assert np.all((counts >= 0) & (counts <= TOPK))
    assert np.all(stats >= 0)
    return Traffic(tuple(int(x) for x in stats[:, 0]), tuple(int(x) for x in stats[:, 1]),
                   tuple(int(x) for x in counts))
