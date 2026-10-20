from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pytest

CacheState: TypeAlias = Literal["cold", "partial", "warm"]
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class Sampling:
    budgets: tuple[int, ...]
    warmup: int
    iterations: int

    def __post_init__(self) -> None:
        if self.warmup < 0 or self.iterations < 1:
            raise pytest.UsageError("resident benchmark warmup must be >=0 and iters must be >0")


@dataclass(frozen=True, slots=True)
class Timing:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    samples_ms: tuple[float, ...]

    @classmethod
    def from_samples(cls, values: list[float]) -> Timing:
        samples = np.asarray(values, dtype=np.float64)
        return cls(float(samples.mean()), float(np.percentile(samples, 50)),
                   float(np.percentile(samples, 95)), tuple(values))


@dataclass(frozen=True, slots=True)
class TimedBudget:
    budget: int
    qli: Timing
    chain: Timing


def parse_budgets(value: str) -> tuple[int, ...]:
    try:
        result = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except ValueError as exc:
        raise pytest.UsageError("--resident-bench-budgets requires comma-separated integers") from exc
    if any(item < 1 or item > 2048 for item in result):
        raise pytest.UsageError("--resident-bench-budgets values must be in [1, 2048]")
    return result


def parse_states(value: str) -> tuple[CacheState, ...]:
    result: list[CacheState] = []
    for part in value.split(","):
        name = part.strip()
        if name not in ("cold", "partial", "warm"):
            raise pytest.UsageError("--resident-bench-states must use cold,partial,warm")
        if name not in result:
            result.append(name)
    return tuple(result)


def paired_samples(
    restore: Callable[[], None], measure: Callable[[int, bool], float], sampling: Sampling,
) -> tuple[TimedBudget, ...]:
    budgets = (0, *sampling.budgets)
    samples: dict[tuple[int, bool], list[float]] = {
        (budget, chain): [] for budget in budgets for chain in (False, True)
    }
    for chain in (False, True):
        for iteration in range(sampling.warmup + sampling.iterations):
            order = budgets if iteration % 2 == 0 else budgets[::-1]
            for budget in order:
                restore()
                elapsed = measure(budget, chain)
                if iteration >= sampling.warmup:
                    samples[budget, chain].append(elapsed)
    return tuple(TimedBudget(budget, Timing.from_samples(samples[budget, False]),
                             Timing.from_samples(samples[budget, True])) for budget in budgets)


def write_jsonl(path: str | None, record: dict[str, JsonValue]) -> None:
    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
