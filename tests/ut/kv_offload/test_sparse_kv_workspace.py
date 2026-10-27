import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MODULE_NAME = "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_simt_lru"


def _load_simt_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    for name in (
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.kv_transfer",
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload",
    ):
        package = types.ModuleType(name)
        package.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, package)

    indexer_name = "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_indexer"
    indexer = types.ModuleType(indexer_name)
    indexer.SparseKVIndexerPlanContext = type("SparseKVIndexerPlanContext", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, indexer_name, indexer)

    spec = importlib.util.spec_from_file_location(
        MODULE_NAME,
        ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_simt_lru.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("capacity", "row_elements", "expected_shape"),
    [
        (2048, 16384, (1, 1)),
        (4096, 18432, (1, 1)),
        (7168, 21504, (1, 1)),
        (14336, 28672, (1, 1)),
        (14337, 28673, (192, 28673)),
    ],
)
def test_constructor_allocates_workspace_only_for_gm_rows(
    capacity: int,
    row_elements: int,
    expected_shape: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a real CPU constructor at, below, and above the 112-KiB UB boundary.
    module = _load_simt_module(monkeypatch)
    namespace = torch.ops._C_ascend
    plan_transfer = MagicMock()
    monkeypatch.setattr(namespace, "npu_sparse_kv_plan_transfer", plan_transfer, raising=False)
    monkeypatch.setattr(namespace, "npu_sparse_kv_transfer", MagicMock(), raising=False)
    monkeypatch.setattr(namespace, "npu_quant_lightning_indexer_sparse_kv", MagicMock(), raising=False)
    planner = module.SparseKVSimtLru(
        max_rows=192,
        topk=2048,
        capacity=capacity,
        max_token=32768,
        block_size=128,
        host_num_blocks=256,
        token_size_bytes_k=128,
        token_size_bytes_v=128,
        host_k_bases=[100],
        host_v_bases=[200],
        current_slots=torch.full((192, 2048), -1, dtype=torch.int32),
        device=torch.device("cpu"),
    )
    workspace_pointer = planner.compact_workspace.data_ptr()

    # When: Plan consumes the persistent workspace allocated by the constructor.
    planner.set_active_rows(1)
    planner.plan_and_transfer(
        layer_id=0,
        req_ids=torch.tensor([100], dtype=torch.int64),
        topk_indices=torch.arange(2048, dtype=torch.int32).reshape(1, 2048),
        stable_prefix_lens=torch.zeros(1, dtype=torch.int32),
        visible_seq_lens=torch.full((1,), 2048, dtype=torch.int32),
        token_to_req=torch.zeros(1, dtype=torch.int32),
        block_table=torch.arange(16, dtype=torch.int32).reshape(1, 16),
        resident_k=torch.empty(1),
        resident_v=torch.empty(1),
    )

    # Then: UB uses one placeholder, GM uses exact row width, and the pointer stays stable.
    assert module.sparse_kv_plan_workspace_elements(2048, capacity) == row_elements
    assert tuple(planner.compact_workspace.shape) == expected_shape
    assert planner.compact_workspace.dtype == torch.int32
    assert planner.compact_workspace.is_contiguous()
    assert planner.compact_workspace.data_ptr() == workspace_pointer
    assert plan_transfer.call_args.args[14] is planner.compact_workspace
