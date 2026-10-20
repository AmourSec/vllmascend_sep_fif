from __future__ import annotations

from types import ModuleType
from typing import Literal

import numpy as np
import pytest
from typing_extensions import assert_never

from mtp_resident_cases import make_case, runtime
from mtp_resident_oracle import Geometry

pytestmark = pytest.mark.npu


def test_qli_resident_warm_hits_do_not_copy_poisoned_sources(resident_offload: ModuleType) -> None:
    # Given a warm row whose source pools are subsequently poisoned.
    case = make_case(runtime(), Geometry((1,), (37,)), offload=resident_offload)
    case.run()
    case.torch.npu.synchronize()
    before_k = case.resident_k.clone()
    before_v = case.resident_v.clone()
    case.source_k.fill_(17)
    case.source_v.fill_(29)
    # When identical candidates hit resident slots.
    _sparse, candidates, stats, _next = case.run()
    # Then hits consume budget without DMA; original resident bytes survive.
    np.testing.assert_array_equal(candidates.cpu().numpy(), [[0, 1, 2, 3]])
    np.testing.assert_array_equal(stats.cpu().numpy(), [[4, 0]])
    case.torch.testing.assert_close(case.resident_k, before_k, rtol=0, atol=0)
    case.torch.testing.assert_close(case.resident_v, before_v, rtol=0, atol=0)


def test_qli_resident_partial_hits_protect_slots_and_rebuild_lru(resident_offload: ModuleType) -> None:
    # Given two hits interleaved with oldest victims in original LRU order.
    case = make_case(runtime(), Geometry((1,), (37,)), offload=resident_offload)
    case.last_req_ids.copy_(case.req_ids)
    case.slots.copy_(case.torch.tensor([[1, 20, 2, 21, 22, 23, 24, 25]], dtype=case.torch.int32, device="npu"))
    # When candidates 0..3 hit slots 0/2 and allocate misses in slots 1/3.
    _sparse, _candidates, stats, _next = case.run()
    # Then all tile hits are protected before victim selection and LRU groups are stable.
    np.testing.assert_array_equal(stats.cpu().numpy(), [[4, 2]])
    np.testing.assert_array_equal(case.slots.cpu().numpy(), [[1, 0, 2, 3, 22, 23, 24, 25]])
    np.testing.assert_array_equal(case.lru.cpu().numpy(), [[4, 5, 6, 7, 1, 3, 0, 2]])
    case.assert_payload(0, 0)
    case.assert_payload(0, 3)
    assert case.torch.all(case.resident_k[0, [0, 2]] == 173)
    assert case.torch.all(case.resident_v[0, [0, 2]] == 211)


def test_qli_resident_request_identity_reset_refills_poisoned_sources(resident_offload: ModuleType) -> None:
    # Given warm mappings belonging to a previous request identity.
    case = make_case(runtime(), Geometry((1,), (37,)), offload=resident_offload)
    case.run()
    case.torch.npu.synchronize()
    case.source_k.fill_(17)
    case.source_v.fill_(29)
    case.req_ids.add_(100)
    # When the same physical row serves a new request.
    _sparse, _candidates, stats, _next = case.run()
    # Then all candidates miss and state belongs to the new identity.
    np.testing.assert_array_equal(stats.cpu().numpy(), [[4, 4]])
    case.torch.testing.assert_close(case.last_req_ids, case.req_ids)
    np.testing.assert_array_equal(case.slots.cpu().numpy(), [[0, 1, 2, 3, -1, -1, -1, -1]])
    for token in range(4):
        case.assert_payload(0, token)


def test_qli_resident_stable_prefix_invalidates_tail_but_admits_tail_candidates(resident_offload: ModuleType) -> None:
    # Given cached tokens 0..3 but only 0..1 remain stable between invocations.
    case = make_case(runtime(), Geometry((1,), (37,)), offload=resident_offload)
    case.run()
    case.torch.npu.synchronize()
    case.stable.fill_(2)
    case.source_k[:, 2:4].fill_(17)
    case.source_v[:, 2:4].fill_(29)
    # When prefetch processes both stable hits and tail candidates.
    _sparse, candidates, stats, _next = case.run()
    # Then tail tokens are invalidated once and copied back during this invocation.
    np.testing.assert_array_equal(candidates.cpu().numpy(), [[0, 1, 2, 3]])
    np.testing.assert_array_equal(stats.cpu().numpy(), [[4, 2]])
    for token in (2, 3):
        case.assert_payload(0, token)


@pytest.mark.parametrize("invalid", ["mapping", "cumulative", "host_page"])
def test_qli_resident_invalid_row_metadata_preserves_cache(
    invalid: Literal["mapping", "cumulative", "host_page"], resident_offload: ModuleType,
) -> None:
    # Given malformed row metadata and sentinel persistent state.
    case = make_case(runtime(), offload=resident_offload)
    match invalid:
        case "mapping":
            case.token_to_req[0] = 1
        case "cumulative":
            case.query_ends[0] = 4
        case "host_page":
            case.host_table[0, 0] = case.source_k.shape[0]
        case unreachable:
            assert_never(unreachable)
    state = (case.resident_k, case.resident_v, case.last_req_ids, case.slots, case.lru)
    before = tuple(tensor[0].clone() for tensor in state)
    # When the invalid row is evaluated.
    sparse, candidates, stats, next_threshold = case.run()
    # Then invalid outputs are deterministic and the row's cache is untouched.
    assert case.torch.all(sparse[0] == -1)
    assert case.torch.all(candidates[0] == -1)
    assert case.torch.all(stats[0] == -1)
    case.torch.testing.assert_close(next_threshold[0], case.threshold[0])
    for expected, tensor in zip(before, state, strict=True):
        case.torch.testing.assert_close(tensor[0], expected, rtol=0, atol=0)


def test_qli_resident_capacity_above_2048_uses_dynamic_workspace(resident_offload: ModuleType) -> None:
    # Given a legal cache capacity larger than the candidate/TopK count limit.
    capacity = 2053
    case = make_case(runtime(), Geometry((1,), (37,), capacity=capacity), offload=resident_offload)
    # When one candidate is admitted using capacity-sized policy state.
    _sparse, candidates, stats, _next = case.run(budget=1)
    # Then the whole LRU remains a permutation and the new slot is most recent.
    np.testing.assert_array_equal(candidates.cpu().numpy(), [[0]])
    np.testing.assert_array_equal(stats.cpu().numpy(), [[1, 1]])
    np.testing.assert_array_equal(case.lru[0].cpu().numpy(), np.r_[np.arange(1, capacity), 0])
    np.testing.assert_array_equal(case.slots[0, 1:].cpu().numpy(), np.full(capacity - 1, -1))
    case.assert_payload(0, 0)
