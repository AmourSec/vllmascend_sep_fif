from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

from .benchmark_config import BenchmarkCase, TimingConfig

if TYPE_CHECKING:
    import torch

    from .helpers import FusedCase, PlannerStateTensors


class _KernelCalls(NamedTuple):
    fused: Callable[[PlannerStateTensors], torch.Tensor]
    baseline: Callable[[PlannerStateTensors], torch.Tensor]


def _bind_kernel_calls(case: FusedCase, torch_npu, topk: int, block_size: int, ops) -> _KernelCalls:
    fused_op = ops.npu_quant_lightning_indexer_sparse_kv
    plan_op = ops.npu_sparse_kv_plan
    indexer_op = torch_npu.npu_quant_lightning_indexer
    key_scale = case.key_scale.unsqueeze(-1) if case.key_scale.dim() == 2 else case.key_scale
    rows = case.query.shape[0]

    def fused(state: PlannerStateTensors) -> torch.Tensor:
        result = fused_op(
            case.query,
            case.key,
            case.weights,
            case.query_scale,
            case.key_scale,
            case.actual_seq_lengths_query,
            case.actual_seq_lengths_key,
            case.indexer_block_table,
            case.req_ids,
            case.stable_prefix_lens,
            case.visible_seq_lens,
            case.token_to_req,
            case.kv_block_table,
            case.active_rows,
            state.last_req_ids,
            state.slot_to_token,
            state.lru_slots,
            state.current_slots,
            state.miss_count,
            state.miss_tokens,
            state.miss_slots,
            state.compact_workspace,
            case.capacity,
            case.max_token,
            block_size,
            case.host_num_blocks,
        )
        return result.reshape(rows, topk)

    def baseline(state: PlannerStateTensors) -> torch.Tensor:
        indices = indexer_op(
            query=case.query,
            key=case.key,
            weights=case.weights,
            query_dequant_scale=case.query_scale,
            key_dequant_scale=key_scale,
            actual_seq_lengths_query=case.actual_seq_lengths_query,
            actual_seq_lengths_key=case.actual_seq_lengths_key,
            block_table=case.indexer_block_table,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=topk,
            sparse_mode=3,
        ).reshape(rows, topk)
        plan_op(
            case.req_ids,
            indices,
            case.stable_prefix_lens,
            case.visible_seq_lens,
            case.token_to_req,
            case.kv_block_table,
            case.active_rows,
            state.last_req_ids,
            state.slot_to_token,
            state.lru_slots,
            state.current_slots,
            state.miss_count,
            state.miss_tokens,
            state.miss_slots,
            state.compact_workspace,
            topk,
            case.capacity,
            case.max_token,
            block_size,
            case.host_num_blocks,
        )
        return indices

    return _KernelCalls(fused=fused, baseline=baseline)


def measure_case(spec: BenchmarkCase, config: TimingConfig) -> dict[str, object]:
    import torch
    import torch_npu

    from .assertions import assert_state_matches, assert_states_equal, oracle_after
    from .benchmark_state import invoke_from_snapshot, make_route, measure_route, prime_events
    from .helpers import BLOCK_SIZE, TOPK, make_case, make_state, require_ops

    case = make_case(
        [spec.seq_len] * spec.batch,
        [spec.mtp] * spec.batch,
        capacity=spec.capacity,
        seed=config.seed,
        disjoint_permuted_blocks=True,
    )
    calls = _bind_kernel_calls(case, torch_npu, TOPK, BLOCK_SIZE, require_ops())
    initial = make_state(case)
    snapshot = initial
    if spec.cache_state == "warm":
        snapshot = initial.clone()
        calls.baseline(snapshot)
        torch_npu.npu.synchronize()
    if spec.cache_state not in ("cold", "warm"):
        raise AssertionError(f"unsupported cache state: {spec.cache_state}")

    fused_expected = snapshot.clone()
    baseline_expected = snapshot.clone()
    fused_indices = calls.fused(fused_expected)
    baseline_indices = calls.baseline(baseline_expected)
    torch_npu.npu.synchronize()
    torch.testing.assert_close(fused_indices, baseline_indices, rtol=0, atol=0)
    expected = oracle_after(case, snapshot, fused_indices)
    assert_state_matches(fused_expected, expected)
    assert_state_matches(baseline_expected, expected)
    assert_states_equal(fused_expected, baseline_expected)

    routes = {}
    try:
        routes["fused"] = make_route("fused", spec.execution, calls.fused, calls.baseline, snapshot, torch)
        routes["baseline"] = make_route("baseline", spec.execution, calls.fused, calls.baseline, snapshot, torch)
        for name, expected_indices, expected_state in (
            ("fused", fused_indices, fused_expected),
            ("baseline", baseline_indices, baseline_expected),
        ):
            actual = invoke_from_snapshot(routes[name], snapshot, torch_npu)
            torch_npu.npu.synchronize()
            torch.testing.assert_close(actual, expected_indices, rtol=0, atol=0)
            assert_states_equal(routes[name].state, expected_state)
        for _ in range(config.warmup):
            invoke_from_snapshot(routes["fused"], snapshot, torch_npu)
            invoke_from_snapshot(routes["baseline"], snapshot, torch_npu)
        torch_npu.npu.synchronize()

        events = {name: prime_events(torch) for name in routes}
        samples = {name: [] for name in routes}
        walls = {name: [] for name in routes}
        per_call_samples = {name: [] for name in routes}
        per_call_walls = {name: [] for name in routes}
        round_order: list[str] = []
        for round_index in range(config.rounds):
            order = ("fused", "baseline") if round_index % 2 == 0 else ("baseline", "fused")
            round_order.append(",".join(order))
            for name in order:
                measurement = measure_route(routes[name], snapshot, config.iterations, events[name], torch_npu)
                samples[name].append(measurement.device_mean_ms)
                walls[name].append(measurement.wall_mean_ms)
                per_call_samples[name].append(measurement.device_per_call_ms)
                per_call_walls[name].append(measurement.wall_per_call_ms)
    finally:
        torch_npu.npu.synchronize()
        for route in routes.values():
            route.close()
        torch_npu.npu.synchronize()

    selected_count = int((fused_indices >= 0).sum().item())
    miss_count = int(fused_expected.miss_count.sum().item())
    return {
        "fused_ms": samples["fused"],
        "baseline_ms": samples["baseline"],
        "fused_wall_ms": walls["fused"],
        "baseline_wall_ms": walls["baseline"],
        "fused_per_call_ms": per_call_samples["fused"],
        "baseline_per_call_ms": per_call_samples["baseline"],
        "fused_wall_per_call_ms": per_call_walls["fused"],
        "baseline_wall_per_call_ms": per_call_walls["baseline"],
        "miss_count_total": miss_count,
        "selected_count_total": selected_count,
        "hit_rate": 1.0 - miss_count / selected_count,
        "query_rows": int(case.query.shape[0]),
        "key_blocks": int(case.key.shape[0]),
        "key_bytes": int(case.key.numel() * case.key.element_size()),
        "timing_scope": "operator_stage",
        "wall_includes_event_sync": True,
        "round_order": round_order,
        "fusion_first_rounds": (config.rounds + 1) // 2,
        "baseline_first_rounds": config.rounds // 2,
        "correctness_validated": True,
    }
