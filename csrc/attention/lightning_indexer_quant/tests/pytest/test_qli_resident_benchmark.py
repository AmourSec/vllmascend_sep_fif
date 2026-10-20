from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from types import ModuleType
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest

from mtp_resident_cases import runtime
from mtp_resident_oracle import plan_workspace_elements
from resident_benchmark import (
    CAPACITY, HOST_BLOCK, INDEX_BLOCK, KEY_BYTES, TOPK, VALUE_BYTES,
    BenchmarkCase, make_case, specs,
)
from resident_benchmark_checks import Traffic, check_qli_result, seed_cache, validate_chain
from resident_benchmark_measure import (
    CacheState, JsonValue, Sampling, paired_samples, parse_budgets, parse_states, write_jsonl,
)

if TYPE_CHECKING:
    from torch import Tensor

pytestmark = pytest.mark.npu


class EventTimer:
    def __init__(self, case: BenchmarkCase, workspace: Tensor) -> None:
        self.case = case
        self.workspace = workspace
        self.start = case.torch.npu.Event(enable_timing=True)
        self.end = case.torch.npu.Event(enable_timing=True)

    def measure(self, budget: int, chain: bool) -> float:
        self.case.torch.npu.synchronize()
        self.start.record()
        result = self.case.chain(budget, self.workspace) if chain else self.case.qli(budget)
        self.end.record()
        self.end.synchronize()
        elapsed = float(self.start.elapsed_time(self.end))
        assert np.isfinite(elapsed) and elapsed > 0, 'NPU event returned an invalid duration'
        del result
        return elapsed


def _run_state(case: BenchmarkCase, selected: np.ndarray, state: CacheState,
               sampling: Sampling, workspace: Tensor) -> list[dict[str, JsonValue]]:
    snapshot = seed_cache(case, selected, state)
    initial_counts = np.count_nonzero(snapshot.slots.cpu().numpy() >= 0, axis=1)
    traffic: dict[int, Traffic] = {}
    for budget in (0, *sampling.budgets):
        case.restore(snapshot)
        result = case.qli(budget)
        check_qli_result(case, result, selected)
        stats = result[2].cpu().numpy()
        if budget > 0:
            if state == 'cold':
                np.testing.assert_array_equal(stats[:, 1], stats[:, 0])
        else:
            np.testing.assert_array_equal(stats, np.zeros_like(stats))
        case.restore(snapshot)
        traffic[budget] = validate_chain(case, case.chain(budget, workspace), selected)
        if budget == 0 and state == 'cold':
            assert all(count == TOPK for count in traffic[budget].transfer_misses)
    timer = EventTimer(case, workspace)
    timings = paired_samples(lambda: case.restore(snapshot), timer.measure, sampling)
    baseline = timings[0]
    records: list[dict[str, JsonValue]] = []
    for measured in timings:
        counts = traffic[measured.budget]
        prefetch = sum(counts.prefetch_misses)
        transfer = sum(counts.transfer_misses)
        records.append({
            'batch': case.spec.batch, 'query_counts': list(case.spec.query_counts),
            'rows': case.spec.rows, 'context': case.spec.context, 'dtype': case.spec.dtype,
            'state': state, 'initial_resident_tokens_per_row': initial_counts.tolist(),
            'topk': TOPK, 'capacity': CAPACITY, 'host_block': HOST_BLOCK,
            'index_block': INDEX_BLOCK, 'key_bytes': KEY_BYTES, 'value_bytes': VALUE_BYTES,
            'source': 'memfabric_ddr', 'budget': measured.budget,
            'prefetch_active': prefetch > 0,
            'threshold_per_row': case.threshold.cpu().tolist(),
            'stable_prefix_per_row': case.spec.stable.tolist(),
            'visible_lengths_per_row': case.spec.visible.tolist(),
            'admitted_per_row': list(counts.admitted),
            'prefetch_misses_per_row': list(counts.prefetch_misses),
            'transfer_misses_per_row': list(counts.transfer_misses),
            'prefetch_bytes': prefetch * (KEY_BYTES + VALUE_BYTES),
            'transfer_bytes': transfer * (KEY_BYTES + VALUE_BYTES),
            'total_ddr_bytes': (prefetch + transfer) * (KEY_BYTES + VALUE_BYTES),
            'qli': asdict(measured.qli), 'chain': asdict(measured.chain),
            'speedup_qli': baseline.qli.mean_ms / measured.qli.mean_ms,
            'speedup_chain': baseline.chain.mean_ms / measured.chain.mean_ms,
            'warmup': sampling.warmup, 'iterations': sampling.iterations,
            'timing': 'eager_npu_events', 'correctness': 'topk_metadata_payload_checked',
        })
        print(f'resident B={case.spec.batch} Q={case.spec.query_counts} L={case.spec.context} '
              f'{state=} budget={measured.budget} '
              f'QLI={measured.qli.mean_ms:.4f}ms '
              f'chain={measured.chain.mean_ms:.4f}ms '
              f'prefetch_miss={prefetch} transfer_miss={transfer} '
              f'speedup_QLI={baseline.qli.mean_ms / measured.qli.mean_ms:.3f}x '
              f'speedup_chain={baseline.chain.mean_ms / measured.chain.mean_ms:.3f}x', flush=True)
    return records


def _run_spec(case: BenchmarkCase, states: tuple[CacheState, ...], sampling: Sampling,
              output: str | None, run_id: str, threshold_mode: str) -> None:
    workspace = case.torch.empty(
        (case.spec.rows, plan_workspace_elements(TOPK, CAPACITY)),
        dtype=case.torch.int32, device='npu',
    )
    baseline = case.qli(0)
    selected = baseline[0].cpu().numpy()[:, 0]
    threshold = baseline[3]
    if threshold_mode == 'previous_query':
        original_query = case.query.clone()
        try:
            case.query.copy_(original_query.float().roll(1, dims=-1).to(original_query.dtype))
            threshold = case.qli(0)[3]
        finally:
            case.query.copy_(original_query)
    assert bool(case.torch.isfinite(threshold).all()), 'calibration returned a non-finite threshold'
    case.threshold.copy_(threshold)
    for state in states:
        records = _run_state(case, selected, state, sampling, workspace)
        for record in records:
            record['run_id'] = run_id
            record['threshold_mode'] = threshold_mode
            record['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
            record['torch_version'] = str(case.torch.__version__)
            record['device_name'] = str(case.torch.npu.get_device_name(0))
            write_jsonl(output, record)


def test_qli_resident_benchmark(request: pytest.FixtureRequest, pytestconfig: pytest.Config) -> None:
    if not pytestconfig.getoption('--benchmark'):
        pytest.skip('pass --benchmark to run DDR-backed QLI resident timings')
    profile = pytestconfig.getoption('--resident-bench-profile')
    dtype = pytestconfig.getoption('--resident-bench-dtype')
    cases = specs(profile, dtype)
    sampling = Sampling(parse_budgets(pytestconfig.getoption('--resident-bench-budgets')),
                        pytestconfig.getoption('--resident-bench-warmup'),
                        pytestconfig.getoption('--resident-bench-iters'))
    states = parse_states(pytestconfig.getoption('--resident-bench-states'))
    output = pytestconfig.getoption('--resident-bench-output')
    threshold_mode = pytestconfig.getoption('--resident-bench-threshold-mode')
    pool_mib = pytestconfig.getoption('--memfabric-pool-mib')
    source_bytes = max(spec.table(HOST_BLOCK).size * HOST_BLOCK * (KEY_BYTES + VALUE_BYTES)
                       for spec in cases)
    if (pool_mib << 20) < source_bytes + (64 << 20):
        raise pytest.UsageError('mapped DDR pool too small; increase --memfabric-pool-mib (full default: 2048)')
    offload: ModuleType = request.getfixturevalue('resident_offload')
    torch = runtime()
    assert hasattr(torch.ops._C_ascend, 'npu_sparse_kv_plan_transfer'), 'SparseKvPlan/Transfer is required'
    run_id = str(uuid4())
    for spec in cases:
        print(f'prepare B={spec.batch} Q={spec.query_counts} L={spec.context} source=DDR', flush=True)
        case = make_case(torch, offload, spec)
        try:
            _run_spec(case, states, sampling, output, run_id, threshold_mode)
        finally:
            case.close()
