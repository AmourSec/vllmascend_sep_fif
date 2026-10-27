"""Matrix orchestration; importing this module does not require torch or an NPU."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict

from .benchmark_config import CAPACITIES, TimingConfig, build_cases
from .benchmark_report import summarize_case, write_results


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("benchmark matrix (comma-separated values or inclusive integer ranges)")
    group.add_argument("--batches", default="1-32", help="Benchmark: request counts, default every integer 1..32")
    group.add_argument("--seq-lens", default="4096,32768,140000", help="Benchmark: context lengths in tokens")
    group.add_argument("--mtps", default="1,2,4,6", help="Benchmark: query rows per request; 1 is normal decode")
    group.add_argument("--capacities", default="2048", help=f"Benchmark: {','.join(map(str, sorted(CAPACITIES)))}")
    group.add_argument("--cache-states", default="warm,cold", help="Benchmark: warm,cold planner residency")
    group.add_argument("--execution", choices=("eager", "graph"), default="eager", help="Performance execution mode")
    group.add_argument("--warmup", type=int, default=10, help="Untimed calls per route and case")
    group.add_argument("--rounds", type=int, default=6, help="Paired timing rounds, alternating route order")
    group.add_argument("--seed", type=int, default=2026, help="Deterministic synthetic input seed")
    group.add_argument("--dry-run", action="store_true", help="Benchmark: write plan.json without loading torch/NPU")


def run_benchmark(args: argparse.Namespace) -> int:
    """Checkpoint completed cases; fail promptly on correctness, device, or timing errors."""
    output = args.output.resolve()
    rows: list[dict[str, object]] = []
    status: dict[str, object] | None = None
    try:
        if args.trace:
            raise ValueError("Use --suite profile --trace for a representative case; tracing a matrix is disabled")
        timing = TimingConfig(args.warmup, args.iterations, args.rounds, args.seed)
        cases = build_cases(args.batches, args.seq_lens, args.mtps, args.capacities, args.cache_states, args.execution)
        if (output / "results.json").exists():
            raise ValueError(f"Results already exist under {output}; choose a fresh --output directory")
        output.mkdir(parents=True, exist_ok=True)
        plan = {
            "case_count": len(cases),
            "timing": asdict(timing),
            "cases": [asdict(case) for case in cases],
            "scope": "QLI + SparseKvPlan only; DDR transfer and attention excluded",
            "cache_definition": "Planner residency; hardware caches are not flushed",
            "event_timing": "NPU event interval per call, including any host launch gaps; averaged within rounds",
            "wall_timing": "Synchronized instrumented host duration, including event recording and end wait",
            "reset": "Identical persistent planner state restored and synchronized before every call, outside timing",
            "physical_keys": "Distinct paged key blocks per request",
        }
        (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        status = {"status": "planned", "planned_cases": len(cases), "completed_cases": 0}
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(
            f"Planned {len(cases)} cases; {args.rounds} rounds x {args.iterations} calls per route; {output}",
            flush=True,
        )
        if args.dry_run:
            return 0

        # The runner also exposes preflight; defer its import to keep dry-run CPU-only.
        from tools.run_quant_lightning_indexer_sparse_kv_tests import preflight

        metadata = preflight(args.device, args.cann_root)
        (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        status["status"] = "running"
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")

        from .benchmark_device import measure_case

        for number, case in enumerate(cases, 1):
            print(f"[{number}/{len(cases)}] {case.case_id}", flush=True)
            status["current_case"] = case.case_id
            try:
                result = summarize_case(case, measure_case(case, timing))
            except (AssertionError, RuntimeError, ValueError) as error:
                rows.append({**asdict(case), "case_id": case.case_id, "status": "failed", "error": str(error)})
                write_results(output, rows)
                raise
            result.update(asdict(timing))
            rows.append(result)
            write_results(output, rows)
            status["completed_cases"] = number
            (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
            print(
                f"  fused={result['fused_round_mean_median_ms']:.6f} ms "
                f"baseline={result['baseline_round_mean_median_ms']:.6f} ms "
                f"speedup={result['event_speedup']:.3f}x hit_rate={result['hit_rate']:.4f}",
                flush=True,
            )
        status["status"] = "complete"
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(f"Completed {len(rows)} cases; results: {output / 'results.csv'}", flush=True)
        return 0
    except (AssertionError, ImportError, OSError, RuntimeError, ValueError) as error:
        if status is not None:
            status.update(status="failed", error=str(error))
            (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
