from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from lightning_indexer_quant_prefetch_golden import assert_contract, compute_expected
from prefetch_test_assertions import _assert_selected_and_rows
from prefetch_test_cases import Case, DType, _make_case, _run, _runtime

pytestmark = pytest.mark.npu


def _deterministic_case(torch: ModuleType, heads: int, position: int, dtype: DType, offload: ModuleType | None) -> Case:
    length = position + 1
    case = _make_case(
        torch,
        threshold=-np.inf,
        budget=1,
        offload=offload,
        lengths=(length,) * (heads + 1),
        physical=3,
        block_size=128,
        sparse_count=1,
        row_bytes=128,
        heads=heads,
        dtype=dtype,
    )
    batch = heads + 1
    width = 128
    query_cpu = np.zeros((batch, heads, width), dtype=np.float32)
    query_cpu[:, np.arange(heads), np.arange(heads)] = 1.0 + np.arange(heads, dtype=np.float32) % 8
    key_cpu = np.zeros((3, 128, 1, width), dtype=np.float32)
    head_values = 1.0 + np.arange(heads, dtype=np.float32) // 8
    block_table = case.block_table.cpu().numpy()
    for row in range(batch):
        page = int(block_table[row, position // 128])
        offset = position % 128
        key_cpu[page, offset, 0, :heads] = head_values
    weights_cpu = np.zeros((batch, heads), dtype=np.float32)
    weights_cpu[np.arange(heads), np.arange(heads)] = 1.0
    weights_cpu[-1] = 1.0
    query_scale_cpu = np.ones((batch, heads), dtype=np.float32)
    key_scale_cpu = np.ones((3, 128, 1), dtype=np.float32)

    query_npu = torch.from_numpy(query_cpu).to(device="npu", dtype=case.query.dtype)
    key_npu = torch.from_numpy(key_cpu).to(device="npu", dtype=case.key.dtype)
    weights_npu = torch.from_numpy(weights_cpu).to(device="npu", dtype=case.weights.dtype)
    query_scale_npu = torch.from_numpy(query_scale_cpu).to(device="npu", dtype=case.query_scale.dtype)
    key_scale_npu = torch.from_numpy(key_scale_cpu).to(device="npu", dtype=case.key_scale.dtype)
    case.query.copy_(query_npu)
    case.key.copy_(key_npu)
    case.weights.copy_(weights_npu)
    case.query_scale.copy_(query_scale_npu)
    case.key_scale.copy_(key_scale_npu)
    golden_query = query_npu.float().cpu().numpy() if dtype == "fp8" else query_cpu.astype(np.int8)
    golden_key = key_npu.float().cpu().numpy() if dtype == "fp8" else key_cpu.astype(np.int8)
    golden = case.golden._replace(
        query=golden_query,
        key=golden_key,
        weights=weights_npu.cpu().numpy(),
        query_dequant_scale=query_scale_npu.cpu().numpy(),
        key_dequant_scale=key_scale_npu.cpu().numpy(),
    )
    return case._replace(golden=golden)


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("position", [0, 64, 128])
def test_prefetch_deterministic_head_mapping(
    dtype: DType, heads: int, position: int, memfabric_offload: ModuleType | None
) -> None:
    torch = _runtime()
    case = _deterministic_case(torch, heads, position, dtype, memfabric_offload)
    actual = _run(torch, case, threshold=-np.inf, budget=1)
    expected = compute_expected(case.golden)
    assert_contract(expected, topk=1, row_bytes=128, budget=1)
    head_ids = np.arange(heads, dtype=np.float32)
    head_scores = (1.0 + head_ids % 8) * (1.0 + head_ids // 8)
    if dtype == "int8":
        head_scores /= 1024.0
    analytical = np.concatenate((head_scores, [head_scores.sum()])).astype(np.float32)
    np.testing.assert_array_equal(expected.next_threshold, analytical)
    observed = actual[4].cpu().numpy()
    np.testing.assert_array_equal(
        observed,
        analytical,
        err_msg=(
            f"dtype={dtype}, heads={heads}, position={position}; "
            f"actual={observed.tolist()}, expected={analytical.tolist()}"
        ),
    )
    np.testing.assert_array_equal(actual[0].cpu().numpy(), np.full((heads + 1, 1, 1), position, dtype=np.int32))
    _assert_selected_and_rows(actual, case, expected)
