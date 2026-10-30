"""Mac-runnable structural tests for the optional ECHO indexer prefetch."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _module(relative_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)


def _function(container: ast.Module | ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(node for node in container.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _assigned_names(container: ast.Module | ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for node in container.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _called_attributes(function: ast.FunctionDef) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


class TestIndexerPrefetchStaticContracts(unittest.TestCase):
    def test_config_is_owned_by_sparse_offload_config(self) -> None:
        """Moving the fields back to AscendConfig must fail this test."""
        module = _module("vllm_ascend/ascend_config.py")
        ascend_fields = _assigned_names(_class(module, "AscendConfig"))
        sparse_fields = _assigned_names(_class(module, "SparseKVOffloadConfig"))

        self.assertNotIn("indexer_prefetch", ascend_fields)
        self.assertNotIn("indexer_prefetch_budget_divisor", ascend_fields)
        self.assertIn("indexer_prefetch", sparse_fields)
        self.assertIn("indexer_prefetch_budget_divisor", sparse_fields)

    def test_manager_allocates_per_layer_history_and_global_scratch(self) -> None:
        """Collapsing request history into the shared scratch must fail."""
        module = _module(
            "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/"
            "sparse_kv_offload_manager.py"
        )
        register = _function(_class(module, "SparseKVOffloadManager"), "register_kv_caches")
        assigned_attributes = {
            node.targets[0].attr
            for node in ast.walk(register)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
        }

        self.assertIn("indexer_prefetch_scratch", assigned_attributes)
        self.assertIn("indexer_prefetch_layer_states", assigned_attributes)

    def test_indexer_executes_both_stages_and_waits_for_prefetch(self) -> None:
        """Dropping the real staged invocation from forward must fail."""
        module = _module("vllm_ascend/attention/indexer.py")
        staged = _function(_class(module, "AscendSFAIndexerBackend"), "_run_indexer_prefetch")
        stages = {
            node.args[-2].value
            for node in ast.walk(staged)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_prefetch_stage"
            and len(node.args) >= 2
            and isinstance(node.args[-2], ast.Constant)
        }

        self.assertEqual(stages, {0, 1})
        self.assertIn("launch_indexer_candidate_prefetch", _called_attributes(staged))
        referenced_attributes = {
            node.attr for node in ast.walk(staged) if isinstance(node, ast.Attribute)
        }
        self.assertIn("wait_indexer_candidate_prefetch", referenced_attributes)

    def test_sfa_handoff_runs_before_indexer_forward(self) -> None:
        """Removing request/layer metadata propagation must fail."""
        module = _module("vllm_ascend/attention/sfa_v1.py")
        forward = _function(_class(module, "AscendSFAImpl"), "forward")
        calls = [
            node.func.attr
            for node in ast.walk(forward)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]

        self.assertIn("_prepare_indexer_prefetch_metadata", calls)

    def test_threshold_update_uses_finite_prior_and_request_lookup(self) -> None:
        """Reintroducing inf EMA or row-position-only history must fail."""
        module = _module("vllm_ascend/attention/indexer_prefetch.py")
        update = _function(module, "update_threshold_ema")
        prepare = _function(module, "prepare_threshold_history")
        calls = _called_attributes(update) | _called_attributes(prepare)
        torch_calls = {
            node.func.attr
            for node in [*ast.walk(prepare), *ast.walk(update)]
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
        }

        self.assertIn("isfinite", torch_calls)
        self.assertIn("argmax", torch_calls)
        self.assertIn("copy_", calls)


if __name__ == "__main__":
    unittest.main()
