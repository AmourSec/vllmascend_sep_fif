from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("memfabric_hybrid")

from vllm_ascend.attention.attention_v1 import AscendAttentionState  # noqa: E402
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend  # noqa: E402
from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl  # noqa: E402
from vllm_ascend.attention.sfa_v1 import SFAForwardContext  # noqa: E402
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_indexer import (  # noqa: E402
    SparseKVIndexerPlanContext,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (  # noqa: E402
    SparseKVOffloadManager,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_simt_lru import (  # noqa: E402
    SparseKVSimtLru,
    sparse_kv_plan_workspace_elements,
)


def _tensor(value: int, *, dtype=torch.int32) -> torch.Tensor:
    return torch.tensor([value], dtype=dtype)


def _make_plan_context(planner: SparseKVSimtLru) -> SparseKVIndexerPlanContext:
    return SparseKVIndexerPlanContext(
        planner=planner,
        layer_id=0,
        active_rows=17,
        num_indexer_tokens=32,
        req_ids=torch.arange(100, 132, dtype=torch.int64),
        stable_prefix_lens=torch.full((32,), 10, dtype=torch.int32),
        visible_seq_lens=torch.full((32,), 11, dtype=torch.int32),
        token_to_req=torch.arange(32, dtype=torch.int32),
        kv_block_table=torch.arange(64, dtype=torch.int32).reshape(32, 2),
    )


def test_simt_fused_plan_uses_full_persistent_workspace_and_original_state():
    planner = SparseKVSimtLru.__new__(SparseKVSimtLru)
    planner.max_rows = 32
    planner.topk = 4
    planner.capacity = 8
    planner.max_token = 64
    planner.block_size = 16
    planner.host_num_blocks = 32
    planner.current_slots = torch.arange(256, dtype=torch.int32).reshape(32, 8)
    planner.active_rows = _tensor(17)
    planner.compact_workspace = torch.empty(
        (planner.max_rows, sparse_kv_plan_workspace_elements(planner.topk, planner.capacity)), dtype=torch.int32
    )
    state = SimpleNamespace(
        last_req_ids=torch.full((32,), -1, dtype=torch.int64),
        slot_to_token=torch.full((32, 8), -1, dtype=torch.int32),
        lru_slots=torch.arange(8, dtype=torch.int32).repeat(32, 1),
        miss_count=torch.zeros(32, dtype=torch.int32),
        miss_tokens=torch.full((32, 4), -1, dtype=torch.int32),
        miss_slots=torch.full((32, 4), -1, dtype=torch.int32),
    )
    planner.layers = [state]
    planner.plan_owner_layer_id = None
    expected_indices = torch.arange(128, dtype=torch.int32).reshape(32, 1, 4)
    planner._fused_indexer_op = MagicMock(return_value=expected_indices)
    context = _make_plan_context(planner)
    query = torch.zeros((32, 32, 128))
    key = torch.zeros((2, 16, 1, 128))
    weights = torch.zeros((32, 32))
    query_scale = torch.ones((32, 32))
    key_scale = torch.ones((2, 16))
    query_lens = torch.arange(1, 33, dtype=torch.int32)
    key_lens = torch.arange(11, 43, dtype=torch.int32)
    indexer_table = torch.arange(64, dtype=torch.int32).reshape(32, 2)

    result = context.select_and_plan(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        actual_seq_lengths_query=query_lens,
        actual_seq_lengths_key=key_lens,
        indexer_block_table=indexer_table,
    )

    assert result.shape == (32, 1, 4)
    torch.testing.assert_close(result, expected_indices)
    assert context.prepared is True
    assert planner.compact_workspace.numel() == 32 * sparse_kv_plan_workspace_elements(4, 8)
    planner._fused_indexer_op.assert_called_once_with(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        query_lens,
        key_lens,
        indexer_table,
        context.req_ids,
        context.stable_prefix_lens,
        context.visible_seq_lens,
        context.token_to_req,
        context.kv_block_table,
        planner.active_rows,
        state.last_req_ids,
        state.slot_to_token,
        state.lru_slots,
        planner.current_slots,
        state.miss_count,
        state.miss_tokens,
        state.miss_slots,
        planner.compact_workspace,
        planner.capacity,
        planner.max_token,
        planner.block_size,
        planner.host_num_blocks,
    )
    assert planner.plan_owner_layer_id == 0


def test_simt_constructor_allocates_workspace_for_every_row():
    op_namespace = torch.ops._C_ascend
    with (
        patch.object(op_namespace, "npu_sparse_kv_plan_transfer", MagicMock(), create=True),
        patch.object(op_namespace, "npu_sparse_kv_transfer", MagicMock(), create=True),
        patch.object(op_namespace, "npu_quant_lightning_indexer_sparse_kv", MagicMock(), create=True),
    ):
        planner = SparseKVSimtLru(
            max_rows=3,
            topk=4,
            capacity=8,
            max_token=64,
            block_size=16,
            host_num_blocks=32,
            token_size_bytes_k=1024,
            token_size_bytes_v=128,
            host_k_bases=[100],
            host_v_bases=[200],
            current_slots=torch.zeros((3, 4), dtype=torch.int32),
            device=torch.device("cpu"),
        )

    assert planner.compact_workspace.shape == (3, sparse_kv_plan_workspace_elements(4, 8))


def test_indexer_fused_adapter_preserves_padded_query_and_logical_output():
    backend = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    key = torch.zeros((2, 16, 1, 128))
    key_scale = torch.ones((2, 16, 1))
    backend.k_cache = SimpleNamespace(kv_cache=(key, key_scale))
    context = MagicMock(spec=SparseKVIndexerPlanContext)
    context.num_indexer_tokens = 32
    expected = torch.arange(32 * 2048, dtype=torch.int32).reshape(32, 1, 2048)
    context.select_and_plan.return_value = expected
    q_li = torch.zeros((32 * 32, 128), dtype=torch.float8_e4m3fn)
    q_scale = torch.ones((32 * 32,), dtype=torch.float32)
    weights = torch.zeros((32, 32), dtype=torch.bfloat16)
    query_lens = torch.arange(1, 33, dtype=torch.int32)
    key_lens = torch.arange(11, 43, dtype=torch.int32)
    block_table = torch.arange(64, dtype=torch.int32).reshape(32, 2)
    metadata = SimpleNamespace(
        actual_seq_lengths_query=query_lens,
        actual_seq_lengths_key=key_lens,
        block_table=block_table,
    )

    result = backend._run_fused_sparse_kv_indexer(
        q_li,
        q_scale,
        torch.Size((32, 32, 128)),
        weights,
        metadata,
        context,
    )

    assert result is expected
    call = context.select_and_plan.call_args.kwargs
    assert call["query"].shape == (32, 32, 128)
    assert call["weights"] is weights
    assert call["query_dequant_scale"].shape == (32, 32)
    assert call["actual_seq_lengths_query"] is query_lens
    assert call["actual_seq_lengths_key"] is key_lens
    assert call["indexer_block_table"] is block_table


def test_prepared_plan_runs_transfer_without_running_plan_again():
    planner = SparseKVSimtLru.__new__(SparseKVSimtLru)
    planner.topk = 4
    planner.capacity = 8
    planner.block_size = 16
    planner.host_num_blocks = 32
    planner.token_size_bytes_k = 1024
    planner.token_size_bytes_v = 128
    planner.active_rows = _tensor(1)
    planner.plan_owner_layer_id = 0
    planner.layers = [
        SimpleNamespace(
            miss_count=torch.zeros(3, dtype=torch.int32),
            miss_tokens=torch.full((3, 4), -1, dtype=torch.int32),
            miss_slots=torch.full((3, 4), -1, dtype=torch.int32),
            host_cache_bases=torch.tensor([100, 200], dtype=torch.int64),
        )
    ]
    planner._transfer_op = MagicMock()
    planner._plan_transfer_op = MagicMock()

    planner.transfer_prepared_plan(
        layer_id=0,
        token_to_req=_tensor(0),
        block_table=torch.tensor([[3]], dtype=torch.int32),
        resident_k=torch.zeros((1, 16, 1, 4)),
        resident_v=torch.zeros((1, 16, 1, 2)),
    )

    planner._transfer_op.assert_called_once()
    planner._plan_transfer_op.assert_not_called()


def test_manager_sets_active_rows_before_returning_fused_context():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.layer_name_to_offload_id = {"layer.0": 0}
    manager.simt_lru = MagicMock(spec=SparseKVSimtLru)
    manager.simt_lru.max_rows = 4

    context = manager.prepare_fused_indexer_plan(
        layer_name="layer.0",
        active_rows=3,
        num_indexer_tokens=4,
        req_ids=_tensor(101, dtype=torch.int64),
        stable_prefix_lens=_tensor(10),
        visible_seq_lens=_tensor(11),
        token_to_req=_tensor(0),
        kv_block_table=torch.tensor([[3]], dtype=torch.int32),
    )

    manager.simt_lru.set_active_rows.assert_called_once_with(3)
    assert context.planner is manager.simt_lru
    assert context.layer_id == 0
    assert context.active_rows == 3
    assert context.num_indexer_tokens == 4


def test_mtp_skip_topk_still_runs_plan_for_compacted_rows():
    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager.layer_name_to_offload_id = {"mtp": 4}
    manager.mtp_layer_id = 4
    manager.simt_lru = MagicMock(spec=SparseKVSimtLru)
    manager.topk_buffers_k = [None, None, None, None, torch.zeros(1)]
    manager.topk_buffers_v = [None, None, None, None, torch.zeros(1)]

    manager._onload_topk_kv_simt(
        layer_name="mtp",
        num_tokens=2,
        num_reqs=1,
        block_table=torch.tensor([[3]], dtype=torch.int32),
        topk_indices_npu=torch.zeros((2, 2048), dtype=torch.int32),
        current_slots_npu=torch.zeros((2, 2048), dtype=torch.int32),
        req_ids_npu=_tensor(101, dtype=torch.int64),
        stable_prefix_lens_npu=torch.zeros(2, dtype=torch.int32),
        visible_seq_lens_npu=torch.ones(2, dtype=torch.int32),
        token_to_req_npu=torch.zeros(2, dtype=torch.int32),
        skip_topk=True,
    )

    manager.simt_lru.set_active_rows.assert_called_once_with(2)
    manager.simt_lru.plan_and_transfer.assert_called_once()
    manager.simt_lru.transfer_reused_plan.assert_not_called()


def test_offload_hook_sets_active_rows_before_exposing_context_and_clears_on_fallback():
    impl = AscendSFAKVOffloadImpl.__new__(AscendSFAKVOffloadImpl)
    impl.has_indexer = True
    impl.skip_topk = False
    impl.enable_sparse_li_c8 = True
    impl.use_fused_overlap = False
    impl.indexer = SimpleNamespace(
        use_torch_npu_lightning_indexer=True,
        n_head=32,
        head_dim=128,
        topk_tokens=2048,
        c8_k_cache_dtype=torch.float8_e4m3fn,
    )
    impl.layer_name = "model.layers.0.self_attn"
    impl._current_layer_name = None
    manager = MagicMock()
    manager.uses_simt_lru = True
    manager.topk = 2048
    manager.block_size = 128
    context = MagicMock(spec=SparseKVIndexerPlanContext)
    manager.prepare_fused_indexer_plan.return_value = context
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        num_prefills=0,
        num_decodes=32,
        num_decode_tokens=17,
        token_to_req=torch.arange(32, dtype=torch.int32),
        req_ids_tensor=torch.arange(100, 132, dtype=torch.int64),
        block_table=torch.arange(64, dtype=torch.int32).reshape(32, 2),
        sparse_kv_plan_context=SimpleNamespace(prepared=True),
    )
    indexer_metadata = SimpleNamespace(
        block_size=128,
        sparse_kv_plan_context=SimpleNamespace(prepared=True),
    )
    parallel = SFAForwardContext(
        actual_seq_lengths_query=torch.arange(1, 33, dtype=torch.int32),
        actual_seq_lengths_key=torch.arange(11, 43, dtype=torch.int32),
        kv_slot_mapping=_tensor(0),
        topk_num_tokens=32,
    )

    with patch(
        "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
        return_value=manager,
    ):
        impl._prepare_indexer_runtime_context(indexer_metadata, metadata, parallel)

    assert metadata.sparse_kv_plan_context is context
    assert indexer_metadata.sparse_kv_plan_context is context
    manager.prepare_fused_indexer_plan.assert_called_once()
    call = manager.prepare_fused_indexer_plan.call_args.kwargs
    assert call["active_rows"] == 17
    assert call["num_indexer_tokens"] == 32
    assert call["token_to_req"].shape == (32,)
    assert call["stable_prefix_lens"].shape == (32,)
    assert call["visible_seq_lens"].shape == (32,)
    assert call["req_ids"].shape == (32,)
    assert call["kv_block_table"].shape == (32, 2)

    impl.skip_topk = True
    impl._prepare_indexer_runtime_context(indexer_metadata, metadata, parallel)
    assert metadata.sparse_kv_plan_context is None
    assert indexer_metadata.sparse_kv_plan_context is None

    impl.skip_topk = False
    impl.indexer.c8_k_cache_dtype = torch.bfloat16
    metadata.sparse_kv_plan_context = context
    indexer_metadata.sparse_kv_plan_context = context
    impl._prepare_indexer_runtime_context(indexer_metadata, metadata, parallel)
    assert metadata.sparse_kv_plan_context is None
    assert indexer_metadata.sparse_kv_plan_context is None

    impl.indexer.c8_k_cache_dtype = torch.float8_e4m3fn
    metadata.num_prefills = 1
    metadata.sparse_kv_plan_context = context
    indexer_metadata.sparse_kv_plan_context = context
    impl._prepare_indexer_runtime_context(indexer_metadata, metadata, parallel)
    assert metadata.sparse_kv_plan_context is None
    assert indexer_metadata.sparse_kv_plan_context is None
    manager.prepare_fused_indexer_plan.assert_called_once()

    metadata.num_prefills = 0
    manager.block_size = 16
    metadata.sparse_kv_plan_context = context
    indexer_metadata.sparse_kv_plan_context = context
    with patch(
        "vllm_ascend.attention.sfa_kv_offload.get_sparse_kv_offload_manager",
        return_value=manager,
    ):
        impl._prepare_indexer_runtime_context(indexer_metadata, metadata, parallel)
    assert metadata.sparse_kv_plan_context is None
    assert indexer_metadata.sparse_kv_plan_context is None

    manager.block_size = 128
    indexer_metadata.block_size = 16
    metadata.sparse_kv_plan_context = context
    indexer_metadata.sparse_kv_plan_context = context
    impl._prepare_indexer_runtime_context(indexer_metadata, metadata, parallel)
    assert metadata.sparse_kv_plan_context is None
    assert indexer_metadata.sparse_kv_plan_context is None
    manager.prepare_fused_indexer_plan.assert_called_once()
