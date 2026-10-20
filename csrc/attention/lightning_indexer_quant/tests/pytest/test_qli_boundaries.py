from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pytest

from lightning_indexer_quant_prefetch_golden import compute_expected
from prefetch_test_assertions import _assert_selected_and_rows
from prefetch_test_cases import _make_case, _run, _runtime

if TYPE_CHECKING:
    from prefetch_test_cases import Case

pytestmark = pytest.mark.npu


def _boundary_case(
    torch: ModuleType, lengths: tuple[int, ...], row_bytes: int, offload: ModuleType | None
) -> Case:
    largest = max(lengths, default=0)
    physical = max(1, (max(largest, 1) + 127) // 128 + 2)
    if physical % 5 == 0:
        physical += 1
    return _make_case(
        torch,
        lengths=lengths,
        physical=physical,
        block_size=128,
        sparse_count=2048,
        row_bytes=row_bytes,
        budget=128,
        threshold=-np.inf,
        offload=offload,
        quantized_inputs=True,
        dtype="int8",
    )


@pytest.mark.parametrize("length", [0, 1, 127, 6144, 6145, 10241])
def test_qli_context_boundaries(length: int, memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _boundary_case(torch, (length,), 1152, memfabric_offload)
    actual = _run(torch, case, threshold=-np.inf, budget=128)
    _assert_selected_and_rows(actual, case, compute_expected(case.golden))


@pytest.mark.parametrize("row_bytes", [1, 127, 4096])
def test_qli_payload_row_boundaries(row_bytes: int, memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _boundary_case(torch, (127,), row_bytes, memfabric_offload)
    actual = _run(torch, case, threshold=-np.inf, budget=128)
    _assert_selected_and_rows(actual, case, compute_expected(case.golden))


def test_qli_batch_crosses_core_group(memfabric_offload: ModuleType | None) -> None:
    torch = _runtime()
    case = _boundary_case(torch, (1,) * 65, 1152, memfabric_offload)
    actual = _run(torch, case, threshold=-np.inf, budget=128)
    _assert_selected_and_rows(actual, case, compute_expected(case.golden))
