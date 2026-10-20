from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from resident_benchmark_checks import seeded_tokens
from resident_benchmark_measure import Sampling, paired_samples, parse_budgets, parse_states, write_jsonl


def test_each_sample_and_warmup_restores_state_and_alternates_budget_order() -> None:
    # Given a destructive operation and a known initial cache state.
    cache = [17]
    calls: list[tuple[int, bool]] = []
    resets: list[int] = []

    def restore() -> None:
        cache[0] = 17
        resets.append(len(calls))

    def measure(budget: int, chain: bool) -> float:
        assert cache == [17]
        cache[0] = -1
        calls.append((budget, chain))
        return float(budget + 1 + 10 * chain)

    # When both paths are sampled with one warmup and two measured repetitions.
    results = paired_samples(restore, measure, Sampling((512, 2048), 1, 2))

    # Then every invocation starts identically, and both orders are exercised.
    assert resets == list(range(18))
    assert calls[:9] == [(b, False) for b in (0, 512, 2048, 2048, 512, 0, 0, 512, 2048)]
    assert calls[9:] == [(b, True) for b in (0, 512, 2048, 2048, 512, 0, 0, 512, 2048)]
    assert [item.budget for item in results] == [0, 512, 2048]
    assert results[0].qli.mean_ms == 1
    assert results[0].chain.mean_ms == 11
    assert all(len(item.qli.samples_ms) == 2 for item in results)


def test_warmup_outliers_are_excluded_from_statistics(tmp_path: Path) -> None:
    # Given one large warmup followed by stable samples for each path.
    calls = [0]

    def measure(_budget: int, _chain: bool) -> float:
        position = calls[0] % 6
        calls[0] += 1
        return 1000.0 if position < 2 else 2.0

    # When results are serialized incrementally.
    results = paired_samples(lambda: None, measure, Sampling((512,), 1, 2))
    output = tmp_path / "results.jsonl"
    write_jsonl(str(output), {"result": results[0].qli.mean_ms})
    write_jsonl(str(output), {"result": results[1].chain.mean_ms})

    # Then warmup does not affect the timing and previous records are preserved.
    assert results[0].qli.mean_ms == results[1].chain.p95_ms == 2
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"result": 2.0}, {"result": 2.0},
    ]


@pytest.mark.parametrize("value", ["", "0", "-1", "2049", "2,,4", "abc"])
def test_invalid_budgets_fail_before_device_initialization(value: str) -> None:
    with pytest.raises(pytest.UsageError):
        parse_budgets(value)


def test_states_and_budgets_are_deduplicated() -> None:
    assert parse_budgets("512,2048,512") == (512, 2048)
    assert parse_states("cold,warm,cold") == ("cold", "warm")
    with pytest.raises(pytest.UsageError):
        parse_states("cold,unknown")


@pytest.mark.parametrize("warmup,iterations", [(-1, 2), (0, 0)])
def test_invalid_sampling_is_rejected(warmup: int, iterations: int) -> None:
    with pytest.raises(pytest.UsageError):
        Sampling((512,), warmup, iterations)


def test_cache_states_seed_only_stable_topk_tokens() -> None:
    selected = np.asarray([9, 3, 6, 1, 7, 4], dtype=np.int32)
    assert seeded_tokens(selected, 7, "cold").tolist() == []
    assert seeded_tokens(selected, 7, "partial").tolist() == [3, 1]
    assert seeded_tokens(selected, 7, "warm").tolist() == [3, 6, 1, 4]
