import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch as cpu_torch

PACKAGE = "tests.e2e.nightly.single_node.ops.singlecard_ops.quant_lightning_indexer_sparse_kv"


@dataclass(frozen=True)
class FakeCase:
    query: cpu_torch.Tensor
    key: cpu_torch.Tensor
    weights: cpu_torch.Tensor
    query_scale: cpu_torch.Tensor
    key_scale: cpu_torch.Tensor
    actual_seq_lengths_query: cpu_torch.Tensor
    actual_seq_lengths_key: cpu_torch.Tensor
    indexer_block_table: cpu_torch.Tensor
    req_ids: cpu_torch.Tensor
    stable_prefix_lens: cpu_torch.Tensor
    visible_seq_lens: cpu_torch.Tensor
    token_to_req: cpu_torch.Tensor
    kv_block_table: cpu_torch.Tensor
    active_rows: cpu_torch.Tensor
    capacity: int
    max_token: int
    host_num_blocks: int


class FakeState:
    def __init__(self, rows: int, fill: int = 0) -> None:
        self._tensors = tuple(cpu_torch.full((rows,), fill, dtype=cpu_torch.int32) for _ in range(8))
        (
            self.last_req_ids,
            self.slot_to_token,
            self.lru_slots,
            self.current_slots,
            self.miss_count,
            self.miss_tokens,
            self.miss_slots,
            self.compact_workspace,
        ) = self._tensors

    def clone(self) -> "FakeState":
        clone = FakeState(len(self.miss_count))
        for target, source in zip(clone.tensors(), self.tensors()):
            target.copy_(source)
        return clone

    def tensors(self) -> tuple[cpu_torch.Tensor, ...]:
        return self._tensors


class FakeEvent:
    def __init__(self, clock: list[float], log: list[str]) -> None:
        self._clock = clock
        self._log = log
        self._recorded = 0.0

    def record(self) -> None:
        self._recorded = self._clock[0]

    def elapsed_time(self, end: "FakeEvent") -> float:
        return end._recorded - self._recorded

    def synchronize(self) -> None:
        self._log.append("event_sync")


class FakeNpu:
    def __init__(self, clock: list[float], log: list[str]) -> None:
        self._clock = clock
        self._log = log

    def Event(self, *, enable_timing: bool) -> FakeEvent:  # noqa: N802
        assert enable_timing
        return FakeEvent(self._clock, self._log)

    def synchronize(self) -> None:
        self._log.append("sync")


class FakeGraph:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[], None]] = []
        self.reset_called = False

    def replay(self) -> None:
        for callback in self.callbacks:
            callback()

    def reset(self) -> None:
        self.reset_called = True


class FakeGraphNpu(FakeNpu):
    def __init__(self, clock: list[float], log: list[str]) -> None:
        super().__init__(clock, log)
        self.capturing: FakeGraph | None = None
        self.last_graph: FakeGraph | None = None

    def NPUGraph(self) -> FakeGraph:  # noqa: N802
        self.last_graph = FakeGraph()
        return self.last_graph

    def graph(self, graph: FakeGraph):
        owner = self

        class Capture:
            def __enter__(self) -> None:
                owner.capturing = graph

            def __exit__(self, *_args) -> None:
                owner.capturing = None

        return Capture()


@pytest.mark.parametrize(("cache_state", "restored_value"), [("cold", 0), ("warm", 1)])
def test_measure_case_restores_identical_state_and_alternates_round_order(
    monkeypatch: pytest.MonkeyPatch, cache_state: str, restored_value: int
) -> None:
    # A missing restore, batched event, or fixed route order must break this test.
    rows = 2
    clock = [0.0]
    log: list[str] = []
    entries: list[tuple[str, int]] = []
    indices = cpu_torch.arange(2048, dtype=cpu_torch.int32).repeat(rows, 1)

    def make_case(*_args, **_kwargs) -> FakeCase:
        scalar = cpu_torch.zeros(rows, dtype=cpu_torch.int32)
        return FakeCase(
            query=cpu_torch.empty((rows, 32, 128), dtype=cpu_torch.int8),
            key=cpu_torch.empty((4, 128, 1, 128), dtype=cpu_torch.uint8),
            weights=scalar,
            query_scale=scalar,
            key_scale=cpu_torch.zeros((4, 128), dtype=cpu_torch.float32),
            actual_seq_lengths_query=scalar,
            actual_seq_lengths_key=scalar,
            indexer_block_table=scalar,
            req_ids=scalar,
            stable_prefix_lens=scalar,
            visible_seq_lens=scalar,
            token_to_req=scalar,
            kv_block_table=scalar,
            active_rows=scalar,
            capacity=2048,
            max_token=256,
            host_num_blocks=4,
        )

    def mutate(route: str, tensors: tuple[cpu_torch.Tensor, ...], duration: float) -> None:
        entries.append((route, int(tensors[0][0])))
        for tensor in tensors[:-1]:
            tensor.add_(1)
        tensors[4].fill_(1)
        clock[0] += duration

    class FakeOps:
        def npu_quant_lightning_indexer_sparse_kv(self, *_args):
            mutate("fused", _args[14:22], 1.0)
            return indices.flatten().clone()

        def npu_sparse_kv_plan(self, *_args) -> None:
            mutate("baseline", _args[7:15], 0.25)

    def run_indexer(**_kwargs) -> cpu_torch.Tensor:
        clock[0] += 0.75
        return indices.flatten().clone()

    helpers = ModuleType(f"{PACKAGE}.helpers")
    helpers.make_case = make_case
    helpers.make_state = lambda _case: FakeState(rows)
    helpers.require_ops = FakeOps
    helpers.TOPK = 2048
    helpers.BLOCK_SIZE = 128
    assertions = ModuleType(f"{PACKAGE}.assertions")
    assertions.oracle_after = lambda *_args: SimpleNamespace()
    assertions.assert_state_matches = lambda *_args: None
    assertions.assert_states_equal = lambda *_args: None
    fake_npu = FakeNpu(clock, log)
    fake_torch_npu = ModuleType("torch_npu")
    fake_torch_npu.npu = fake_npu
    fake_torch_npu.npu_quant_lightning_indexer = run_indexer
    monkeypatch.setattr(cpu_torch, "npu", fake_npu, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setitem(sys.modules, f"{PACKAGE}.helpers", helpers)
    monkeypatch.setitem(sys.modules, f"{PACKAGE}.assertions", assertions)
    monkeypatch.delitem(sys.modules, f"{PACKAGE}.benchmark_device", raising=False)

    module = importlib.import_module(f"{PACKAGE}.benchmark_device")
    spec = SimpleNamespace(
        batch=1,
        seq_len=256,
        mtp=2,
        capacity=2048,
        cache_state=cache_state,
        execution="eager",
    )
    config = SimpleNamespace(warmup=2, iterations=2, rounds=2, seed=2026)
    result = module.measure_case(spec, config)

    assert (result["fused_ms"], result["baseline_ms"]) == ([1.0, 1.0], [1.0, 1.0])
    assert result["fused_per_call_ms"] == [[1.0, 1.0], [1.0, 1.0]]
    assert result["baseline_per_call_ms"] == [[1.0, 1.0], [1.0, 1.0]]
    assert all(len(round_samples) == 2 for round_samples in result["fused_wall_per_call_ms"])
    assert all(len(round_samples) == 2 for round_samples in result["baseline_wall_per_call_ms"])
    assert result["round_order"] == ["fused,baseline", "baseline,fused"]
    assert (result["fusion_first_rounds"], result["baseline_first_rounds"]) == (1, 1)
    assert result["timing_scope"] == "operator_stage"
    assert result["correctness_validated"] is result["wall_includes_event_sync"] is True
    assert result["miss_count_total"] == 2
    assert result["selected_count_total"] == rows * 2048
    expected_routes = [
        "fused",
        "baseline",
        "fused",
        "baseline",
        "fused",
        "baseline",
        "fused",
        "baseline",
        "fused",
        "fused",
        "baseline",
        "baseline",
        "baseline",
        "baseline",
        "fused",
        "fused",
    ]
    if cache_state == "warm":
        expected_routes.insert(0, "baseline")
    assert [route for route, _ in entries] == expected_routes
    expected_values = ([0] if cache_state == "warm" else []) + [restored_value] * 16
    assert [value for _, value in entries] == expected_values
    assert log.count("event_sync") == 10


@pytest.mark.parametrize("route_name", ["fused", "baseline"])
def test_graph_route_replays_captured_work_on_fixed_state(route_name: str) -> None:
    module = importlib.import_module(f"{PACKAGE}.benchmark_state")
    fake_npu = FakeGraphNpu([0.0], [])
    torch_module = SimpleNamespace(npu=fake_npu)
    torch_npu = SimpleNamespace(npu=fake_npu)
    snapshot = FakeState(rows=1)
    output = cpu_torch.tensor([7])

    def captured_call(state: FakeState) -> cpu_torch.Tensor:
        def mutate() -> None:
            for tensor in state.tensors()[:-1]:
                tensor.add_(1)

        if fake_npu.capturing is not None:
            fake_npu.capturing.callbacks.append(mutate)
        mutate()
        return output

    route = module.make_route(route_name, "graph", captured_call, captured_call, snapshot, torch_module)
    assert int(route.state.last_req_ids[0]) == 1

    actual = module.invoke_from_snapshot(route, snapshot, torch_npu)

    assert actual is output
    assert int(route.state.last_req_ids[0]) == 1
    route.close()
    assert fake_npu.last_graph is not None and fake_npu.last_graph.reset_called


def test_graph_capture_failure_resets_graph_without_eager_fallback() -> None:
    module = importlib.import_module(f"{PACKAGE}.benchmark_state")
    fake_npu = FakeGraphNpu([0.0], [])

    class CaptureError(RuntimeError):
        pass

    def fail(_state: FakeState) -> cpu_torch.Tensor:
        raise CaptureError("capture failed")

    with pytest.raises(CaptureError, match="capture failed"):
        module.make_route("fused", "graph", fail, fail, FakeState(rows=1), SimpleNamespace(npu=fake_npu))

    assert fake_npu.last_graph is not None and fake_npu.last_graph.reset_called


def test_measure_route_releases_each_eager_output_after_synchronization() -> None:
    module = importlib.import_module(f"{PACKAGE}.benchmark_state")
    live = [0, 0]

    class Output:
        def __init__(self) -> None:
            live[0] += 1
            live[1] = max(live)

        def __del__(self) -> None:
            live[0] -= 1

    fake_npu = FakeNpu([0.0], [])
    route = module.Route(FakeState(1), Output, lambda: None)
    events = (FakeEvent([0.0], []), FakeEvent([0.0], []))

    module.measure_route(route, FakeState(1), 4, events, SimpleNamespace(npu=fake_npu))

    assert live == [0, 1]
