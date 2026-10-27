from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Final

CAPACITIES: Final = frozenset({2048, 4096, 7168, 7169, 8192, 14336, 14337, 16384})
CACHE_STATES: Final = frozenset({"cold", "warm"})
EXECUTIONS: Final = frozenset({"eager", "graph"})
MAX_EXPANDED_VALUES: Final = 4096


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark matrix value violates the CLI contract."""


@dataclass(frozen=True)
class BenchmarkCase:
    batch: int
    seq_len: int
    mtp: int
    capacity: int
    cache_state: str = "warm"
    execution: str = "eager"

    def __post_init__(self) -> None:
        if not 1 <= self.batch <= 32:
            raise BenchmarkConfigError(f"batch must be between 1 and 32: {self.batch}")
        if self.seq_len < 2048:
            raise BenchmarkConfigError(f"seq_len must be at least 2048: {self.seq_len}")
        if not 1 <= self.mtp <= self.seq_len:
            raise BenchmarkConfigError(f"mtp must be between 1 and seq_len: {self.mtp}")
        if self.capacity not in CAPACITIES:
            raise BenchmarkConfigError(f"unsupported capacity: {self.capacity}")
        if self.cache_state not in CACHE_STATES:
            raise BenchmarkConfigError(f"unsupported cache state: {self.cache_state}")
        if self.execution not in EXECUTIONS:
            raise BenchmarkConfigError(f"unsupported execution mode: {self.execution}")

    @property
    def case_id(self) -> str:
        return f"b{self.batch}-s{self.seq_len}-m{self.mtp}-c{self.capacity}-{self.cache_state}-{self.execution}"


@dataclass(frozen=True)
class TimingConfig:
    warmup: int = 10
    iterations: int = 10
    rounds: int = 6
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.warmup < 0:
            raise BenchmarkConfigError(f"warmup must be nonnegative: {self.warmup}")
        if self.iterations < 1:
            raise BenchmarkConfigError(f"iterations must be positive: {self.iterations}")
        if self.rounds < 1:
            raise BenchmarkConfigError(f"rounds must be positive: {self.rounds}")
        if not 0 <= self.seed < 2**63:
            raise BenchmarkConfigError(f"seed must be between 0 and 2**63 - 1: {self.seed}")


def _ordered_unique(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def _parse_integers(raw: str, name: str) -> list[int]:
    parsed: list[int] = []
    expanded_count = 0
    for component in raw.split(","):
        token = component.strip()
        if not token:
            raise BenchmarkConfigError(f"{name} contains an empty value")
        pieces = [piece.strip() for piece in token.split("-")]
        if len(pieces) == 1 and pieces[0].isdigit():
            expanded_count += 1
            if expanded_count > MAX_EXPANDED_VALUES:
                raise BenchmarkConfigError(f"{name} accepts at most {MAX_EXPANDED_VALUES} expanded values")
            parsed.append(int(pieces[0]))
        elif len(pieces) == 2 and all(piece.isdigit() for piece in pieces):
            start, end = (int(piece) for piece in pieces)
            if start > end:
                raise BenchmarkConfigError(f"{name} range must be ascending: {token}")
            expanded_count += end - start + 1
            if expanded_count > MAX_EXPANDED_VALUES:
                raise BenchmarkConfigError(f"{name} accepts at most {MAX_EXPANDED_VALUES} expanded values")
            parsed.extend(range(start, end + 1))
        else:
            raise BenchmarkConfigError(f"invalid {name} value: {token}")
    return _ordered_unique(parsed)


def _parse_cache_states(raw: str) -> list[str]:
    states = list(dict.fromkeys(component.strip() for component in raw.split(",")))
    if not states or any(state not in CACHE_STATES for state in states):
        raise BenchmarkConfigError(f"cache_states must contain only cold or warm: {raw}")
    return states


def build_cases(
    batches: str,
    seq_lens: str,
    mtps: str,
    capacities: str,
    cache_states: str,
    execution: str,
) -> list[BenchmarkCase]:
    batch_values = _parse_integers(batches, "batches")
    seq_len_values = _parse_integers(seq_lens, "seq_lens")
    mtp_values = _parse_integers(mtps, "mtps")
    capacity_values = _parse_integers(capacities, "capacities")
    if execution not in EXECUTIONS:
        raise BenchmarkConfigError(f"execution must be eager or graph: {execution}")
    return [
        BenchmarkCase(batch, seq_len, mtp, capacity, cache_state, execution)
        for batch, seq_len, mtp, capacity, cache_state in product(
            batch_values,
            seq_len_values,
            mtp_values,
            capacity_values,
            _parse_cache_states(cache_states),
        )
    ]
