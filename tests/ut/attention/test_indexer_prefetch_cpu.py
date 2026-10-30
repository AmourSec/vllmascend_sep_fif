"""CPU tensor behavior tests for ECHO indexer prefetch helpers."""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_prefetch_module():
    module_name = "_indexer_prefetch_cpu_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "vllm_ascend/attention/indexer_prefetch.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    original_dataclass = dataclasses.dataclass

    def _python39_dataclass(*args, **kwargs):
        kwargs.pop("slots", None)
        return original_dataclass(*args, **kwargs)

    dataclasses.dataclass = _python39_dataclass
    try:
        spec.loader.exec_module(module)
    finally:
        dataclasses.dataclass = original_dataclass
    return module


def _load_reuse_simt_plan():
    source = (REPO_ROOT / "vllm_ascend/attention/sfa_kv_offload.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_reuse_simt_plan"
    )
    standalone = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(standalone)
    namespace = {}
    exec(compile(standalone, "sfa_kv_offload.py", "exec"), namespace)
    return namespace["_reuse_simt_plan"]


class TestIndexerPrefetchCpuBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prefetch = _load_prefetch_module()

    def test_threshold_initializes_from_infinite_history(self) -> None:
        torch = self.prefetch.torch
        state = self.prefetch.IndexerPrefetchLayerState.allocate(2, torch.device("cpu"))

        self.prefetch.update_threshold_ema(
            state,
            torch.tensor([10, 20]),
            torch.tensor([8.0, 4.0]),
            torch.tensor([2], dtype=torch.int32),
        )

        self.assertTrue(torch.equal(state.thresholds, torch.tensor([8.0, 4.0])))

    def test_threshold_history_follows_request_reorder(self) -> None:
        torch = self.prefetch.torch
        state = self.prefetch.IndexerPrefetchLayerState.allocate(2, torch.device("cpu"))
        self.prefetch.update_threshold_ema(
            state, torch.tensor([10, 20]), torch.tensor([8.0, 4.0]), torch.tensor([2], dtype=torch.int32)
        )

        self.prefetch.update_threshold_ema(
            state, torch.tensor([20, 10]), torch.tensor([6.0, 10.0]), torch.tensor([2], dtype=torch.int32)
        )

        self.assertTrue(torch.equal(state.thresholds, torch.tensor([5.0, 9.0])))
        self.assertTrue(torch.equal(state.request_ids, torch.tensor([20, 10])))

    def test_stage_zero_receives_request_matched_thresholds(self) -> None:
        torch = self.prefetch.torch
        state = self.prefetch.IndexerPrefetchLayerState.allocate(2, torch.device("cpu"))
        state.thresholds.copy_(torch.tensor([8.0, 4.0]))
        state.request_ids.copy_(torch.tensor([10, 20]))

        self.prefetch.prepare_threshold_history(
            state, torch.tensor([20, 10]), torch.tensor([2], dtype=torch.int32)
        )

        self.assertTrue(torch.equal(state.thresholds, torch.tensor([4.0, 8.0])))

    def test_threshold_lookup_searches_full_history_before_batch_shrink(self) -> None:
        torch = self.prefetch.torch
        state = self.prefetch.IndexerPrefetchLayerState.allocate(3, torch.device("cpu"))
        self.prefetch.update_threshold_ema(
            state,
            torch.tensor([10, 20, 30]),
            torch.tensor([2.0, 4.0, 8.0]),
            torch.tensor([3], dtype=torch.int32),
        )

        self.prefetch.update_threshold_ema(
            state, torch.tensor([30]), torch.tensor([10.0]), torch.tensor([1], dtype=torch.int32)
        )

        self.assertEqual(state.thresholds.tolist(), [9.0, float("inf"), float("inf")])
        self.assertEqual(state.request_ids.tolist(), [30, -1, -1])

    def test_threshold_history_is_independent_per_layer(self) -> None:
        torch = self.prefetch.torch
        first = self.prefetch.IndexerPrefetchLayerState.allocate(1, torch.device("cpu"))
        second = self.prefetch.IndexerPrefetchLayerState.allocate(1, torch.device("cpu"))

        self.prefetch.update_threshold_ema(
            first, torch.tensor([7]), torch.tensor([3.0]), torch.tensor([1], dtype=torch.int32)
        )

        self.assertEqual(second.thresholds.untyped_storage().tolist(), [0, 0, 128, 127])
        self.assertEqual(second.request_ids.tolist(), [-1])

    def test_overlap_orders_transfer_between_stages_and_waits_last(self) -> None:
        events: list[str] = []

        executed = self.prefetch.execute_prefetch_overlap(
            stage_zero=lambda: events.append("stage0") is None,
            launch_transfer=lambda: events.append("launch") or "event",
            stage_one=lambda: events.append("stage1") is None,
            update_history=lambda: events.append("ema"),
            wait_transfer=lambda event: events.append(f"wait:{event}"),
        )

        self.assertTrue(executed)
        self.assertEqual(events, ["stage0", "launch", "stage1", "ema", "wait:event"])

    def test_missing_stage_zero_never_mutates_transfer_planner(self) -> None:
        events: list[str] = []

        executed = self.prefetch.execute_prefetch_overlap(
            stage_zero=lambda: False,
            launch_transfer=lambda: events.append("launch"),
            stage_one=lambda: events.append("stage1") is None,
            update_history=lambda: events.append("ema"),
            wait_transfer=lambda event: events.append("wait"),
        )

        self.assertFalse(executed)
        self.assertEqual(events, [])

    def test_stage_slices_persistent_scratch_to_query_rows(self) -> None:
        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape

            def __getitem__(self, row_slice):
                return FakeTensor((row_slice.stop, *self.shape[1:]))

        captured_shapes: list[tuple[int, ...]] = []

        def fake_op(*args):
            captured_shapes.extend(tensor.shape for tensor in args[10:15])

        real_torch = self.prefetch.torch
        self.prefetch.torch = SimpleNamespace(
            ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_qli_prefetch_stage=fake_op))
        )
        try:
            scratch = SimpleNamespace(
                scores=FakeTensor((8, 256)),
                candidates=FakeTensor((8, 2048)),
                candidate_counts=FakeTensor((8,)),
                exact_indices=FakeTensor((8, 1, 2048)),
                kth_scores=FakeTensor((8,)),
            )
            inputs = [FakeTensor((3, 64, 128))] + [FakeTensor((3,)) for _ in range(9)]
            executed = self.prefetch.run_prefetch_stage(*inputs, scratch, 0, 32)
        finally:
            self.prefetch.torch = real_torch

        self.assertTrue(executed)
        self.assertEqual(captured_shapes, [(3, 256), (3, 2048), (3,), (3, 1, 2048), (3,)])

    def test_production_four_dimensional_key_scale_normalizes_to_kernel_shape(self) -> None:
        torch = self.prefetch.torch
        key_scale = torch.empty((7, 128, 1, 1), dtype=torch.float32)

        normalized = self.prefetch.normalize_key_scale(key_scale)

        self.assertEqual(normalized.shape, (7, 128))
        self.assertEqual(normalized.data_ptr(), key_scale.data_ptr())
        self.assertEqual(normalized.stride(1), 1)

    def test_noncontiguous_bfloat16_weights_preserve_values_and_become_contiguous(self) -> None:
        torch = self.prefetch.torch
        backing = torch.arange(3 * 128, dtype=torch.bfloat16).view(3, 128)
        weights = backing[:, ::2]
        self.assertFalse(weights.is_contiguous())

        normalized = self.prefetch.normalize_index_weights(weights, rows=3, heads=64)

        self.assertIsNotNone(normalized)
        self.assertEqual(normalized.dtype, torch.bfloat16)
        self.assertTrue(normalized.is_contiguous())
        self.assertTrue(torch.equal(normalized, weights))

    def test_non_bfloat16_weights_keep_baseline_path(self) -> None:
        torch = self.prefetch.torch
        weights = torch.ones((2, 64), dtype=torch.float32)

        normalized = self.prefetch.normalize_index_weights(weights, rows=2, heads=64)

        self.assertIsNone(normalized)

    def test_prefetch_requires_independent_shared_layer_plan(self) -> None:
        reuse_plan = _load_reuse_simt_plan()

        self.assertFalse(reuse_plan(True, True, True, False))
        self.assertTrue(reuse_plan(True, True, False, False))
        self.assertFalse(reuse_plan(True, True, False, True))


if __name__ == "__main__":
    unittest.main()
