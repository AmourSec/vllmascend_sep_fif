from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mtp_resident_cases import make_case, runtime
from mtp_resident_oracle import (
    INDEX_BLOCK,
    KEY_BYTES,
    VALUE_BYTES,
    Geometry,
    causal_scores,
    plan_workspace_elements,
    topk,
)

pytestmark = pytest.mark.npu


def test_qli_resident_graph_replay_refreshes_identity_lengths_mapping_and_active_rows() -> None:
    # Given an actual captured NPUGraph with persistent input allocations.
    torch = runtime()
    case = make_case(torch)
    case.run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = case.run()
    # When replay sees changed identity and poisoned K/V sources at the same addresses.
    case.source_k.fill_(17)
    case.source_v.fill_(29)
    case.req_ids.add_(100)
    graph.replay()
    torch.npu.synchronize()
    sparse, candidates, stats, _next = captured
    np.testing.assert_array_equal(stats.cpu().numpy(), [[4, 4]] * 3)
    np.testing.assert_array_equal(candidates.cpu().numpy(), [[0, 1, 2, 3]] * 3)
    np.testing.assert_array_equal(case.last_req_ids.cpu().numpy(), [141, 141, 142])
    for row in range(3):
        for token in range(4):
            case.assert_payload(row, token)
    expected = topk(causal_scores(case.golden), 7)
    np.testing.assert_array_equal(np.sort(sparse.cpu().numpy(), axis=-1), np.sort(expected, axis=-1))

    # When another replay changes query boundaries, request mapping, lengths, and threshold.
    changed = Geometry((1, 2), (17, 21))
    case.query_ends.copy_(torch.tensor(changed.query_ends, dtype=torch.int32, device="npu"))
    case.token_to_req.copy_(torch.tensor(changed.requests, dtype=torch.int32, device="npu"))
    case.key_lengths.copy_(torch.tensor(changed.key_lengths, dtype=torch.int32, device="npu"))
    case.stable.copy_(torch.tensor(changed.visible, dtype=torch.int32, device="npu"))
    case.threshold.fill_(np.inf)
    graph.replay()
    torch.npu.synchronize()
    # Then output contents reflect current metadata, without recapture.
    expected = topk(causal_scores(replace(case.golden, geometry=changed)), 7)
    np.testing.assert_array_equal(np.sort(sparse.cpu().numpy(), axis=-1), np.sort(expected, axis=-1))
    assert torch.all(candidates == -1)
    assert torch.all(stats == 0)
    np.testing.assert_array_equal(case.last_req_ids.cpu().numpy(), [141, 142, 142])

    # When the live prefix shrinks while T stays fixed.
    case.query_ends.copy_(torch.tensor([1, 2], dtype=torch.int32, device="npu"))
    case.active.fill_(2)
    persistent = (case.resident_k, case.resident_v, case.last_req_ids, case.slots, case.lru)
    before = tuple(tensor[2].clone() for tensor in persistent)
    graph.replay()
    torch.npu.synchronize()
    # Then the inactive output and its resident state follow the padded-row contract.
    assert torch.all(sparse[2] == -1)
    assert torch.all(candidates[2] == -1)
    assert torch.all(stats[2] == 0)
    for expected_row, tensor in zip(before, persistent, strict=True):
        torch.testing.assert_close(tensor[2], expected_row, rtol=0, atol=0)


@pytest.mark.parametrize("host_block", [24, 32])
def test_qli_resident_handoff_to_unmodified_sparse_plan_transfer(host_block: int) -> None:
    # Given QLI's actual TopK and prefilled resident slots, with an unstable tail.
    torch = runtime()
    assert hasattr(torch.ops._C_ascend, "npu_sparse_kv_plan_transfer"), (
        "installed extension lacks SparseKvPlan/Transfer"
    )
    case = make_case(torch, Geometry(host_block=host_block), typed_cache=True)
    assert case.resident_k.dtype == case.resident_v.dtype == torch.bfloat16
    case.query.fill_(1)
    case.weights.fill_(1)
    case.key.fill_(1)
    for request in range(len(case.geometry.query_counts)):
        page = int(case.table[request, 0].cpu())
        for token in range(4):
            case.key[page, token % INDEX_BLOCK].fill_(100 - token)
    sparse, _candidates, _stats, _next = case.run()
    case.stable.fill_(2)
    # Poison source bytes so Plan's required tail invalidation must cause a real refetch.
    case.source_k.fill_(31)
    case.source_v.fill_(47)
    rows, count, capacity = case.geometry.rows, 7, case.geometry.capacity
    selected = sparse[:, 0].contiguous()
    selected_cpu = selected.cpu().numpy()
    for selected_row in selected_cpu:
        assert set(range(4)).issubset(selected_row.tolist())
    before_slots = case.slots.cpu().numpy()
    old_k = case.cache_bytes(case.resident_k).cpu().numpy().copy()
    old_v = case.cache_bytes(case.resident_v).cpu().numpy().copy()
    current_slots = torch.full((rows, count), -1, dtype=torch.int32, device="npu")
    miss_count = torch.zeros(rows, dtype=torch.int32, device="npu")
    miss_tokens = torch.full_like(current_slots, -1)
    miss_slots = torch.full_like(current_slots, -1)
    workspace = torch.empty((rows, plan_workspace_elements(count, capacity)), dtype=torch.int32, device="npu")
    visible = torch.tensor(case.geometry.visible, dtype=torch.int32, device="npu")
    # When the existing operator consumes all T rows and exact per-query visible lengths.
    torch.ops._C_ascend.npu_sparse_kv_plan_transfer(
        case.req_ids, selected, case.stable, visible, case.token_to_req, case.host_table,
        case.active, case.last_req_ids, case.slots, case.lru, current_slots,
        miss_count, miss_tokens, miss_slots, workspace, case.bases,
        case.resident_k, case.resident_v, count, capacity,
        case.host_table.shape[1] * host_block, host_block, case.source_k.shape[0], KEY_BYTES, VALUE_BYTES,
    )
    torch.npu.synchronize()
    # Then every selected token has a usable slot and required K/V bytes are present.
    mapped = case.slots.cpu().numpy()
    current = current_slots.cpu().numpy()
    for row in range(rows):
        retained = {int(token): slot for slot, token in enumerate(before_slots[row]) if 0 <= token < 2}
        expected_misses = [int(token) for token in selected_cpu[row] if token >= 0 and token not in retained]
        assert int(miss_count[row].cpu()) == len(expected_misses)
        np.testing.assert_array_equal(miss_tokens[row, :len(expected_misses)].cpu().numpy(), expected_misses)
        np.testing.assert_array_equal(np.sort(case.lru[row].cpu().numpy()), np.arange(capacity))
        for position, token in enumerate(selected_cpu[row]):
            slot = int(current[row, position])
            if token < 0:
                assert slot == -1
                continue
            assert slot >= 0 and mapped[row, slot] == token
            if token in retained:
                np.testing.assert_array_equal(
                    case.cache_bytes(case.resident_k)[row, slot].cpu().numpy(), old_k[row, retained[token]],
                )
                np.testing.assert_array_equal(
                    case.cache_bytes(case.resident_v)[row, slot].cpu().numpy(), old_v[row, retained[token]],
                )
            else:
                case.assert_payload(row, int(token))
