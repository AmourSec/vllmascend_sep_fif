from __future__ import annotations

import csv
import io
import json
import math
import statistics
from pathlib import Path

from .benchmark_config import BenchmarkCase

TIMING_KEYS = ("fused_ms", "baseline_ms", "fused_wall_ms", "baseline_wall_ms")
MEASUREMENT_KEYS = (
    "miss_count_total",
    "selected_count_total",
    "hit_rate",
    "query_rows",
    "key_blocks",
    "key_bytes",
)


class BenchmarkMeasurementError(ValueError):
    """Raised when device timing output cannot be summarized safely."""


def _samples(measurement: dict[str, object], key: str) -> list[float]:
    raw = measurement.get(key)
    if not isinstance(raw, list) or not raw:
        raise BenchmarkMeasurementError(f"{key} must be a nonempty list")
    samples: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BenchmarkMeasurementError(f"{key} contains a nonnumeric value")
        sample = float(value)
        if not math.isfinite(sample) or sample <= 0:
            raise BenchmarkMeasurementError(f"{key} timings must be positive and finite")
        samples.append(sample)
    return samples


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] + (ordered[lower + 1] - ordered[lower]) * fraction


def _statistics(samples: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(samples),
        "p90": _percentile(samples, 0.9),
        "min": min(samples),
        "max": max(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _case_fields(spec: BenchmarkCase) -> dict[str, object]:
    return {
        "case_id": spec.case_id,
        "batch": spec.batch,
        "seq_len": spec.seq_len,
        "mtp": spec.mtp,
        "capacity": spec.capacity,
        "cache_state": spec.cache_state,
        "execution": spec.execution,
    }


def summarize_case(spec: BenchmarkCase, measurement: dict[str, object]) -> dict[str, object]:
    if measurement.get("correctness_validated") is not True:
        raise BenchmarkMeasurementError("correctness_validated must be true before reporting success")
    timings = {key: _samples(measurement, key) for key in TIMING_KEYS}
    lengths = {len(samples) for samples in timings.values()}
    if len(lengths) != 1:
        raise BenchmarkMeasurementError("timing lists must have equal lengths")

    row = dict(measurement)
    row.update(_case_fields(spec))
    row.update({"status": "ok", "aggregation": "per_round_mean"})
    for key, samples in timings.items():
        row[key] = samples
        route = key.removesuffix("_ms")
        for statistic, value in _statistics(samples).items():
            row[f"{route}_round_mean_{statistic}_ms"] = value
    for key in MEASUREMENT_KEYS:
        if key not in measurement:
            raise BenchmarkMeasurementError(f"missing measurement: {key}")
        row[key] = measurement[key]

    fused_median = row["fused_round_mean_median_ms"]
    baseline_median = row["baseline_round_mean_median_ms"]
    fused_wall_median = row["fused_wall_round_mean_median_ms"]
    baseline_wall_median = row["baseline_wall_round_mean_median_ms"]
    assert isinstance(fused_median, float)
    assert isinstance(baseline_median, float)
    assert isinstance(fused_wall_median, float)
    assert isinstance(baseline_wall_median, float)
    row["event_speedup"] = baseline_median / fused_median
    row["wall_speedup"] = baseline_wall_median / fused_wall_median
    paired = [baseline / fused for baseline, fused in zip(timings["baseline_ms"], timings["fused_ms"])]
    row["rounds_count"] = len(paired)
    row["paired_speedups"] = paired
    row["paired_speedup_median"] = statistics.median(paired)
    row["paired_speedup_min"] = min(paired)
    row["paired_speedup_max"] = max(paired)
    return row


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _csv_content(rows: list[dict[str, object]]) -> str:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if not isinstance(value, (list, dict, tuple)) and key not in fields:
                fields.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _display(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_content(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Benchmark summary",
        "",
        "Timings are statistics across per-round means; speedups are baseline divided by fused.",
        "",
        "| Case | Execution | Cache | Status | Fused event median (ms) | "
        "Baseline event median (ms) | Event speedup | Wall speedup | Error |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        values = (
            row.get("case_id"),
            row.get("execution"),
            row.get("cache_state"),
            row.get("status"),
            row.get("fused_round_mean_median_ms"),
            row.get("baseline_round_mean_median_ms"),
            row.get("event_speedup"),
            row.get("wall_speedup"),
            row.get("error"),
        )
        lines.append("| " + " | ".join(_display(value) for value in values) + " |")
    failures = sum(row.get("status") != "ok" for row in rows)
    lines.extend(("", f"Failures: {failures}", ""))
    return "\n".join(lines)


def write_results(output: Path, rows: list[dict[str, object]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        output / "results.json", json.dumps({"results": rows}, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _atomic_write(output / "results.csv", _csv_content(rows))
    _atomic_write(output / "summary.md", _markdown_content(rows))
