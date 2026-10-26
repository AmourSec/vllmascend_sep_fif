from __future__ import annotations

import csv
import json
import math

import pytest

from tests.e2e.nightly.single_node.ops.singlecard_ops.quant_lightning_indexer_sparse_kv.benchmark_config import (
    BenchmarkCase,
    TimingConfig,
    build_cases,
)
from tests.e2e.nightly.single_node.ops.singlecard_ops.quant_lightning_indexer_sparse_kv.benchmark_report import (
    summarize_case,
    write_results,
)


def test_case_identity_and_timing_defaults_are_stable() -> None:
    case = BenchmarkCase(batch=2, seq_len=140_000, mtp=4, capacity=7168, cache_state="cold", execution="graph")

    assert case.case_id == "b2-s140000-m4-c7168-cold-graph"
    assert TimingConfig() == TimingConfig(warmup=10, iterations=10, rounds=6, seed=2026)


def test_build_cases_expands_ranges_and_deduplicates_in_input_order() -> None:
    cases = build_cases(
        batches="2,1-3,2",
        seq_lens="2048,4096",
        mtps="1,2",
        capacities="4096,2048,4096",
        cache_states="warm,cold,warm",
        execution="eager",
    )

    assert len(cases) == 3 * 2 * 2 * 2 * 2
    assert cases[0] == BenchmarkCase(2, 2048, 1, 4096, "warm", "eager")
    assert cases[1] == BenchmarkCase(2, 2048, 1, 4096, "cold", "eager")
    assert cases[2] == BenchmarkCase(2, 2048, 1, 2048, "warm", "eager")
    assert cases[-1] == BenchmarkCase(3, 4096, 2, 2048, "cold", "eager")
    assert len({case.case_id for case in cases}) == len(cases)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batches", "0"),
        ("batches", "33"),
        ("batches", "3-1"),
        ("batches", "1,,2"),
        ("seq_lens", "2047"),
        ("mtps", "0"),
        ("capacities", "7167"),
        ("cache_states", "warm,hot"),
        ("execution", "compiled"),
    ],
)
def test_build_cases_rejects_invalid_cli_values(field: str, value: str) -> None:
    values = {
        "batches": "1",
        "seq_lens": "2048",
        "mtps": "1",
        "capacities": "2048",
        "cache_states": "warm",
        "execution": "eager",
    }
    values[field] = value

    with pytest.raises(ValueError):
        build_cases(**values)


def test_case_and_timing_config_reject_cross_field_and_nonpositive_values() -> None:
    with pytest.raises(ValueError):
        BenchmarkCase(batch=1, seq_len=2048, mtp=2049, capacity=2048)
    with pytest.raises(ValueError):
        TimingConfig(warmup=-1)
    with pytest.raises(ValueError):
        TimingConfig(iterations=0)
    with pytest.raises(ValueError):
        TimingConfig(rounds=0)
    with pytest.raises(ValueError):
        TimingConfig(seed=-1)
    with pytest.raises(ValueError):
        TimingConfig(seed=2**63)


def test_build_cases_rejects_accidentally_huge_ranges_before_expansion() -> None:
    with pytest.raises(ValueError, match="at most 4096 expanded values"):
        build_cases("1-100000000", "2048", "1", "2048", "warm", "eager")


def _measurement() -> dict[str, object]:
    return {
        "fused_ms": [1.0, 2.0, 3.0, 4.0, 5.0],
        "baseline_ms": [2.0, 4.0, 6.0, 8.0, 10.0],
        "fused_wall_ms": [2.0, 2.0, 2.0, 2.0, 2.0],
        "baseline_wall_ms": [6.0, 6.0, 6.0, 6.0, 6.0],
        "miss_count_total": 7,
        "selected_count_total": 20,
        "hit_rate": 0.65,
        "query_rows": 4,
        "key_blocks": 32,
        "key_bytes": 1_048_576,
        "correctness_validated": True,
    }


def test_summarize_case_reports_round_mean_statistics_and_paired_ratios() -> None:
    case = BenchmarkCase(2, 4096, 2, 7169, "warm", "graph")

    row = summarize_case(case, _measurement())

    assert row["case_id"] == case.case_id
    assert row["status"] == "ok"
    assert row["aggregation"] == "per_round_mean"
    assert row["fused_round_mean_median_ms"] == 3.0
    assert row["fused_round_mean_p90_ms"] == pytest.approx(4.6)
    assert row["fused_round_mean_min_ms"] == 1.0
    assert row["fused_round_mean_max_ms"] == 5.0
    assert row["fused_round_mean_stdev_ms"] == pytest.approx(math.sqrt(2.5))
    assert row["event_speedup"] == 2.0
    assert row["wall_speedup"] == 3.0
    assert row["paired_speedup_median"] == 2.0
    assert row["paired_speedup_min"] == 2.0
    assert row["paired_speedup_max"] == 2.0
    assert row["paired_speedups"] == [2.0] * 5
    assert row["rounds_count"] == 5
    assert row["fused_ms"] == _measurement()["fused_ms"]
    assert row["miss_count_total"] == 7


def test_summarize_case_preserves_backend_metadata_without_allowing_spoofed_results() -> None:
    measurement = _measurement()
    measurement.update(
        {
            "correctness_validated": True,
            "round_order": ["fused,baseline", "baseline,fused"] * 2 + ["fused,baseline"],
            "fused_per_call_ms": [0.9, 1.1],
            "timing_scope": "operator_stage",
            "wall_includes_event_sync": True,
            "backend_revision": "abc123",
            "case_id": "spoofed",
            "status": "failed",
            "event_speedup": -1,
        }
    )
    case = BenchmarkCase(1, 2048, 1, 2048)

    row = summarize_case(case, measurement)

    assert row["correctness_validated"] is True
    assert row["round_order"] == ["fused,baseline", "baseline,fused"] * 2 + ["fused,baseline"]
    assert row["fused_per_call_ms"] == [0.9, 1.1]
    assert row["timing_scope"] == "operator_stage"
    assert row["wall_includes_event_sync"] is True
    assert row["backend_revision"] == "abc123"
    assert row["case_id"] == case.case_id
    assert row["status"] == "ok"
    assert row["event_speedup"] == 2.0


@pytest.mark.parametrize("correctness", [None, False])
def test_summarize_case_requires_explicit_correctness_validation(correctness: bool | None) -> None:
    measurement = _measurement()
    if correctness is None:
        measurement.pop("correctness_validated")
    else:
        measurement["correctness_validated"] = correctness

    with pytest.raises(ValueError, match="correctness_validated must be true"):
        summarize_case(BenchmarkCase(1, 2048, 1, 2048), measurement)


@pytest.mark.parametrize(
    "replacement",
    [
        {"fused_ms": []},
        {"fused_ms": [1.0], "baseline_ms": [2.0, 3.0]},
        {"fused_ms": [0.0, 1.0]},
        {"fused_ms": [float("nan"), 1.0]},
        {"fused_wall_ms": [float("inf"), 1.0]},
    ],
)
def test_summarize_case_rejects_invalid_round_timings(replacement: dict[str, object]) -> None:
    measurement = _measurement()
    measurement.update(replacement)

    with pytest.raises(ValueError):
        summarize_case(BenchmarkCase(1, 2048, 1, 2048), measurement)


def test_write_results_keeps_raw_json_and_flattens_csv(tmp_path) -> None:
    measurement = _measurement()
    measurement.update(
        {
            "correctness_validated": True,
            "round_order": ["fused,baseline", "baseline,fused"] * 2 + ["fused,baseline"],
            "fused_per_call_ms": [0.9, 1.1],
            "timing_scope": "operator_stage",
            "wall_includes_event_sync": True,
        }
    )
    success = summarize_case(BenchmarkCase(1, 2048, 1, 2048), measurement)
    failure = {
        "case_id": "b2-s4096-m2-c4096-cold-graph",
        "batch": 2,
        "seq_len": 4096,
        "mtp": 2,
        "capacity": 4096,
        "cache_state": "cold",
        "execution": "graph",
        "status": "failed",
        "error": "device timeout",
    }

    write_results(tmp_path, [success, failure])

    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload == {"results": [success, failure]}
    with (tmp_path / "results.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2
    assert csv_rows[0]["status"] == "ok"
    assert csv_rows[0]["correctness_validated"] == "True"
    assert csv_rows[0]["timing_scope"] == "operator_stage"
    assert csv_rows[0]["wall_includes_event_sync"] == "True"
    assert csv_rows[1]["status"] == "failed"
    assert csv_rows[1]["error"] == "device timeout"
    assert "fused_ms" not in csv_rows[0]
    assert "round_order" not in csv_rows[0]
    assert "fused_per_call_ms" not in csv_rows[0]
    summary = (tmp_path / "summary.md").read_text()
    assert success["case_id"] in summary
    assert failure["case_id"] in summary
    assert "Failures: 1" in summary
    assert "geomean" not in summary.lower()
    assert "Fused event median (ms)" in summary
    assert "Baseline event median (ms)" in summary
    assert "3.000" in summary
    assert "6.000" in summary
