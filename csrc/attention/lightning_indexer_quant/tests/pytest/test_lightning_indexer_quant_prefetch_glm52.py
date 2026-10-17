from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from lightning_indexer_quant_prefetch_golden import compute_expected
from prefetch_test_assertions import _assert_selected_and_rows, _assert_stats
from prefetch_test_cases import DType, _make_case, _run, _runtime

pytestmark = pytest.mark.npu


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
@pytest.mark.parametrize("budget", [0, 8])
@pytest.mark.parametrize("seed", [811, 1952])
def test_prefetch_glm52_production_shape(
    seed: int, budget: int, dtype: DType, memfabric_offload: ModuleType | None
) -> None:
    torch = _runtime()
    case = _make_case(
        torch,
        lengths=(2047, 2048, 4097),
        physical=37,
        block_size=128,
        sparse_count=2048,
        budget=budget,
        row_bytes=1152,
        quantized_inputs=dtype == "int8",
        dtype=dtype,
        seed=seed,
        negative_weights=dtype == "fp8",
        offload=memfabric_offload,
    )
    assert case.query.shape == (3, 32, 128)
    assert case.key.shape == (37, 128, 1, 128)
    assert case.weights.shape == (3, 32)
    assert case.golden.sparse_count == 2048
    assert case.query.dtype == (torch.int8 if dtype == "int8" else torch.float8_e4m3fn)
    assert case.weights.dtype == (torch.float16 if dtype == "int8" else torch.float32)
    if dtype == "int8":
        assert int(case.query.min()) <= -120
        assert int(case.query.max()) >= 120
        assert int(case.key.min()) <= -120
        assert int(case.key.max()) >= 120
    actual = _run(torch, case, threshold=-np.inf, budget=budget)
    expected = compute_expected(case.golden)
    assert actual[1].shape == (3, 2048, 1152)
    _assert_selected_and_rows(actual, case, expected)
    _assert_stats(actual, expected)


@pytest.mark.parametrize("dtype", ["int8", "fp8"])
def test_prefetch_glm52_zero_padded_64_head_consistency(dtype: DType, memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case32 = _make_case(
        torch,
        lengths=(257,),
        physical=4,
        block_size=128,
        sparse_count=128,
        budget=8,
        row_bytes=1152,
        offload=memfabric_offload,
        seed=532,
        heads=32,
        dtype=dtype,
        quantized_inputs=dtype == "int8",
    )
    zero = torch.zeros((1, 32, 128), dtype=torch.float32, device="npu")
    zero_scale = torch.zeros((1, 32), dtype=case32.weights.dtype, device="npu")
    query64 = torch.cat((case32.query.float(), zero), dim=1).to(case32.query.dtype)
    weights64 = torch.cat((case32.weights, zero_scale), dim=1)
    query_scale64 = torch.cat((case32.query_scale, zero_scale), dim=1)
    golden_query64 = np.concatenate(
        (case32.golden.query, np.zeros((1, 32, 128), dtype=case32.golden.query.dtype)), axis=1
    )
    golden_weights64 = np.concatenate(
        (case32.golden.weights, np.zeros((1, 32), dtype=case32.golden.weights.dtype)), axis=1
    )
    golden_query_scale64 = np.concatenate(
        (case32.golden.query_dequant_scale, np.zeros((1, 32), dtype=case32.golden.query_dequant_scale.dtype)),
        axis=1,
    )
    case64 = case32._replace(
        query=query64,
        weights=weights64,
        query_scale=query_scale64,
        golden=case32.golden._replace(
            query=golden_query64,
            weights=golden_weights64,
            query_dequant_scale=golden_query_scale64,
        ),
    )
    actual32 = _run(torch, case32, threshold=-np.inf, budget=8)
    actual64 = _run(torch, case64, threshold=-np.inf, budget=8)
    _assert_selected_and_rows(actual32, case32, compute_expected(case32.golden))
    _assert_selected_and_rows(actual64, case64, compute_expected(case64.golden))
    np.testing.assert_array_equal(actual32[2].cpu().numpy(), actual64[2].cpu().numpy())
    np.testing.assert_allclose(actual32[4].cpu().numpy(), actual64[4].cpu().numpy(), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("threshold", [-np.inf, np.inf, np.nan, 0.0])
def test_prefetch_glm52_fp8_thresholds(threshold: float, memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(
        torch,
        threshold=threshold,
        lengths=(257,),
        physical=4,
        block_size=128,
        sparse_count=128,
        budget=4,
        row_bytes=1152,
        dtype="fp8",
        negative_weights=True,
        offload=memfabric_offload,
    )
    actual = _run(torch, case, threshold=threshold, budget=4)
    _assert_selected_and_rows(actual, case, compute_expected(case.golden))


def test_prefetch_rejects_mixed_query_key_dtype(memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(torch, lengths=(1,), physical=1, block_size=16, dtype="fp8", offload=memfabric_offload)
    mixed_key = case.key.float().to(torch.int8)
    with pytest.raises(RuntimeError, match="query and key must have the same dtype"):
        _run(torch, case._replace(key=mixed_key), threshold=-np.inf, budget=0)


@pytest.mark.parametrize("auxiliary", ["weights", "query_scale", "key_scale"])
def test_prefetch_rejects_fp8_wrong_auxiliary_dtype(auxiliary: str, memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(torch, lengths=(1,), physical=1, block_size=16, dtype="fp8", offload=memfabric_offload)
    updates = {auxiliary: getattr(case, auxiliary).to(torch.float16)}
    with pytest.raises(RuntimeError, match="FLOAT8_E4M3FN query/key require FP32"):
        _run(torch, case._replace(**updates), threshold=-np.inf, budget=0)


def test_prefetch_rejects_unsupported_16_head_query(memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _make_case(torch, lengths=(1,), physical=1, block_size=16, heads=16, offload=memfabric_offload)
    with pytest.raises(RuntimeError, match="H must be 32 or 64"):
        _run(torch, case, threshold=-np.inf, budget=0)
