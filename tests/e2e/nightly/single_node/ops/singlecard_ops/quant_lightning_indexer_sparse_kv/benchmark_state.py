from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from statistics import fmean
from time import perf_counter
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import torch

    from .helpers import PlannerStateTensors


class Route(NamedTuple):
    state: PlannerStateTensors
    invoke: Callable[[], torch.Tensor]
    close: Callable[[], None]


class RouteMeasurement(NamedTuple):
    device_mean_ms: float
    wall_mean_ms: float
    device_per_call_ms: list[float]
    wall_per_call_ms: list[float]


def restore_persistent(target: PlannerStateTensors, snapshot: PlannerStateTensors) -> None:
    for target_tensor, source_tensor in zip(target.tensors()[:-1], snapshot.tensors()[:-1]):
        target_tensor.copy_(source_tensor)


def make_route(
    name: str,
    execution: str,
    fused: Callable[[PlannerStateTensors], torch.Tensor],
    baseline: Callable[[PlannerStateTensors], torch.Tensor],
    snapshot: PlannerStateTensors,
    torch_module,
) -> Route:
    state = snapshot.clone()
    if name == "fused":
        eager = lambda: fused(state)
    elif name == "baseline":
        eager = lambda: baseline(state)
    else:
        raise AssertionError(f"unsupported benchmark route: {name}")
    if execution == "eager":
        return Route(state=state, invoke=eager, close=lambda: None)
    if execution != "graph":
        raise AssertionError(f"unsupported benchmark execution: {execution}")
    graph = torch_module.npu.NPUGraph()
    with ExitStack() as cleanup:
        cleanup.callback(graph.reset)
        with torch_module.npu.graph(graph):
            captured_output = eager()
        cleanup.pop_all()

    def replay() -> torch.Tensor:
        graph.replay()
        return captured_output

    return Route(state=state, invoke=replay, close=graph.reset)


def invoke_from_snapshot(route: Route, snapshot: PlannerStateTensors, torch_npu) -> torch.Tensor:
    restore_persistent(route.state, snapshot)
    torch_npu.npu.synchronize()
    return route.invoke()


def prime_events(torch_module) -> tuple[torch.npu.Event, torch.npu.Event]:
    start = torch_module.npu.Event(enable_timing=True)
    end = torch_module.npu.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    start.elapsed_time(end)
    return start, end


def measure_route(
    route: Route,
    snapshot: PlannerStateTensors,
    iterations: int,
    events: tuple[torch.npu.Event, torch.npu.Event],
    torch_npu,
) -> RouteMeasurement:
    device_samples: list[float] = []
    wall_samples: list[float] = []
    start, end = events
    for _ in range(iterations):
        restore_persistent(route.state, snapshot)
        torch_npu.npu.synchronize()
        wall_start = perf_counter()
        start.record()
        output = route.invoke()
        end.record()
        end.synchronize()
        wall_samples.append((perf_counter() - wall_start) * 1000.0)
        device_samples.append(start.elapsed_time(end))
        del output
    return RouteMeasurement(
        device_mean_ms=fmean(device_samples),
        wall_mean_ms=fmean(wall_samples),
        device_per_call_ms=device_samples,
        wall_per_call_ms=wall_samples,
    )
