from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from mtp_resident_cases import make_case, runtime
from mtp_resident_oracle import Geometry

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("budget", [0, 127, 128, 129, 512, 2048])
def test_simt_resident_mtp_multiple_tiles_unaligned_capacity(
    budget: int, resident_offload: ModuleType,
) -> None:
    geometry = Geometry((1, 3), (2200, 2100), host_block=24, capacity=2053)
    case = make_case(runtime(), geometry, offload=resident_offload)
    sparse, candidates, stats, _next_threshold = case.run(budget=budget, count=2048)
    assert sparse.shape == (geometry.rows, 1, 2048)
    assert candidates.shape == (geometry.rows, max(1, budget))
    for row, visible in enumerate(geometry.visible):
        count = min(budget, int(visible))
        expected = np.full(max(1, budget), -1, dtype=np.int32)
        expected[:count] = np.arange(count, dtype=np.int32)
        np.testing.assert_array_equal(candidates[row].cpu().numpy(), expected)
        np.testing.assert_array_equal(stats[row].cpu().numpy(), [count, count])
        np.testing.assert_array_equal(
            np.sort(case.lru[row].cpu().numpy()), np.arange(geometry.capacity),
        )
        if count:
            for token in sorted({0, count // 2, count - 1}):
                case.assert_payload(row, token)


def test_simt_resident_capacity_4099_keeps_mtp_rows_independent(
    resident_offload: ModuleType,
) -> None:
    geometry = Geometry((2, 1), (2200, 2100), host_block=24, capacity=4099)
    case = make_case(runtime(), geometry, offload=resident_offload)
    _sparse, candidates, stats, _next_threshold = case.run(budget=2048, count=2048)
    for row, visible in enumerate(geometry.visible):
        count = min(2048, int(visible))
        expected = np.arange(count, dtype=np.int32)
        np.testing.assert_array_equal(candidates[row].cpu().numpy()[:count], expected)
        np.testing.assert_array_equal(stats[row].cpu().numpy(), [count, count])
        np.testing.assert_array_equal(
            np.sort(case.lru[row].cpu().numpy()), np.arange(geometry.capacity),
        )
        for token in (0, count // 2, count - 1):
            case.assert_payload(row, token)


def test_simt_resident_invalid_lru_preserves_large_row_state(
    resident_offload: ModuleType,
) -> None:
    geometry = Geometry((2, 1), (2200, 2100), host_block=24, capacity=2053)
    case = make_case(runtime(), geometry, offload=resident_offload)
    case.last_req_ids.copy_(case.req_ids.index_select(0, case.token_to_req.long()))
    case.lru[0, 1] = case.lru[0, 0]
    before = tuple(tensor[0].clone() for tensor in (case.resident_k, case.resident_v, case.last_req_ids,
                                                     case.slots, case.lru))
    sparse, candidates, stats, next_threshold = case.run(budget=128, count=2048)
    assert case.torch.all(sparse[0] == -1)
    assert case.torch.all(candidates[0] == -1)
    assert case.torch.all(stats[0] == -1)
    case.torch.testing.assert_close(next_threshold[0], case.threshold[0])
    for expected, tensor in zip(before, (case.resident_k, case.resident_v, case.last_req_ids,
                                         case.slots, case.lru), strict=True):
        case.torch.testing.assert_close(tensor[0], expected, rtol=0, atol=0)



def test_simt_resident_mtp_protects_hits_and_misses_across_tiles(
    resident_offload: ModuleType,
) -> None:
    geometry = Geometry((2, 1), (2200, 2100), host_block=24, capacity=2053)
    case = make_case(runtime(), geometry, offload=resident_offload)
    case.last_req_ids.copy_(case.req_ids.index_select(0, case.token_to_req.long()))
    case.slots[:, :2] = 0
    case.slots[:, 2] = 127
    case.slots[:, 3] = 128
    case.stable.fill_(128)
    before_k = case.resident_k[:, :3].clone()
    before_v = case.resident_v[:, :3].clone()
    _sparse, candidates, stats, _next = case.run(budget=512, count=2048)
    expected_mapping = np.full(geometry.capacity, -1, dtype=np.int32)
    expected_mapping[:3] = [0, 0, 127]
    expected_mapping[3:129] = np.arange(1, 127)
    expected_mapping[129:513] = np.arange(128, 512)
    expected_lru = np.r_[np.arange(513, geometry.capacity), np.arange(3, 513), [0, 1, 2]]
    for row in range(geometry.rows):
        np.testing.assert_array_equal(candidates[row].cpu().numpy(), np.arange(512))
        np.testing.assert_array_equal(stats[row].cpu().numpy(), [512, 510])
        np.testing.assert_array_equal(case.slots[row].cpu().numpy(), expected_mapping)
        np.testing.assert_array_equal(case.lru[row].cpu().numpy(), expected_lru)
        for token in (1, 126, 128, 255, 511):
            case.assert_payload(row, token)
    case.torch.testing.assert_close(case.resident_k[:, :3], before_k, rtol=0, atol=0)
    case.torch.testing.assert_close(case.resident_v[:, :3], before_v, rtol=0, atol=0)
