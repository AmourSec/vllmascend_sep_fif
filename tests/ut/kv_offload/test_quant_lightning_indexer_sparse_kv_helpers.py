from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

HELPERS_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests/e2e/nightly/single_node/ops/singlecard_ops/quant_lightning_indexer_sparse_kv/helpers.py"
)


class _TorchNpuStub(ModuleType):
    npu_quant_lightning_indexer: Callable[..., torch.Tensor]


def _load_helpers(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, list[torch.Tensor]]:
    captured_scales: list[torch.Tensor] = []
    torch_npu_stub = _TorchNpuStub("torch_npu")

    def indexer_stub(**kwargs: torch.Tensor | int | str) -> torch.Tensor:
        scale = kwargs["key_dequant_scale"]
        query = kwargs["query"]
        assert isinstance(scale, torch.Tensor)
        assert isinstance(query, torch.Tensor)
        captured_scales.append(scale)
        return torch.zeros((query.shape[0], 1, 2048), dtype=torch.int32)

    torch_npu_stub.npu_quant_lightning_indexer = indexer_stub
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_stub)
    module_name = "_quant_lightning_indexer_sparse_kv_helpers_under_test"
    spec = importlib.util.spec_from_file_location(module_name, HELPERS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, captured_scales


def _case(key_scale: torch.Tensor) -> SimpleNamespace:
    rows = 2
    return SimpleNamespace(
        query=torch.empty((rows, 32, 128)),
        key=torch.empty((3, 128, 1, 128)),
        weights=torch.empty((rows, 32)),
        query_scale=torch.empty((rows, 32)),
        key_scale=key_scale,
        actual_seq_lengths_query=torch.tensor([rows], dtype=torch.int32),
        actual_seq_lengths_key=torch.tensor([384], dtype=torch.int32),
        indexer_block_table=torch.tensor([[0, 1, 2]], dtype=torch.int32),
    )


def test_run_indexer_normalizes_rank2_scale_and_preserves_rank3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers, captured = _load_helpers(monkeypatch)
    rank2 = torch.empty((3, 128))
    rank3 = torch.empty((3, 128, 1))

    rank2_result = helpers.run_indexer(_case(rank2))
    rank3_result = helpers.run_indexer(_case(rank3))

    assert rank2_result.shape == (2, 2048)
    assert rank3_result.shape == (2, 2048)
    assert captured[0].shape == (3, 128, 1)
    assert captured[0].stride() == (128, 1, 1)
    assert captured[1] is rank3


def test_run_indexer_rank2_normalization_preserves_axis0_stride(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helpers, captured = _load_helpers(monkeypatch)
    strided_rank2 = torch.empty((6, 128))[::2]

    helpers.run_indexer(_case(strided_rank2))

    assert strided_rank2.stride() == (256, 1)
    assert captured[0].shape == (3, 128, 1)
    assert captured[0].stride() == (256, 1, 1)
