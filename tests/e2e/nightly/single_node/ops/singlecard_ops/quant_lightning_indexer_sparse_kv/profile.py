"""Single-case compatibility entry point using the matrix benchmark's timing engine."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from pathlib import Path

from .benchmark_config import BenchmarkCase, TimingConfig
from .benchmark_report import summarize_case, write_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=140_000)
    parser.add_argument("--mtp", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.add_argument("--capacity", type=int, choices=(2048, 4096, 7168, 7169, 8192), default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def export_traces(spec: BenchmarkCase, seed: int, output: Path) -> None:
    """Collect separate eager traces, never mixed into benchmark measurements."""
    import torch
    import torch_npu

    from .helpers import make_case, make_state, run_fused, run_indexer, run_plan

    case = make_case(
        [spec.seq_len] * spec.batch,
        [spec.mtp] * spec.batch,
        capacity=spec.capacity,
        seed=seed,
        disjoint_permuted_blocks=True,
    )
    snapshot = make_state(case)
    run_plan(case, snapshot, run_indexer(case))
    torch.npu.synchronize()
    activities = [torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
    routes = (
        ("fused_inline_ub_qli_plan", lambda state: run_fused(case, state)),
        ("standalone_qli_then_plan", lambda state: run_plan(case, state, run_indexer(case))),
    )
    for label, route in routes:
        trace_state = snapshot.clone()
        torch.npu.synchronize()
        handler = torch_npu.profiler.tensorboard_trace_handler(str(output / label), worker_name=label)
        with torch_npu.profiler.profile(activities=activities, on_trace_ready=handler) as profiler:
            with torch.autograd.profiler.record_function(label):
                route(trace_state)
            profiler.step()
        torch.npu.synchronize()


def main() -> None:
    args = parse_args()
    spec = BenchmarkCase(args.batch, args.seq_len, args.mtp, args.capacity, "warm", args.execution)
    timing = TimingConfig(args.warmup, args.iterations, args.rounds, args.seed)

    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    importlib.import_module("vllm_ascend.vllm_ascend_C")

    from .benchmark_device import measure_case

    result = summarize_case(spec, measure_case(spec, timing))
    result.update(
        source_revision=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[7],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        iterations=args.iterations,
        rounds=args.rounds,
        warmup=args.warmup,
        seed=args.seed,
        trace_enabled=args.trace,
        trace_execution="eager" if args.trace else None,
    )
    write_results(args.output, [result])
    # Preserve the old scalar profile.json names; results.json retains the raw round means.
    legacy = dict(result)
    legacy.update(
        fused_ms=result["fused_round_mean_median_ms"],
        standalone_qli_then_plan_ms=result["baseline_round_mean_median_ms"],
        standalone_over_fused_ratio=result["event_speedup"],
    )
    (args.output / "profile.json").write_text(json.dumps(legacy, indent=2) + "\n")
    print(json.dumps(legacy, sort_keys=True), flush=True)
    if args.trace:
        export_traces(spec, args.seed, args.output / "traces")


if __name__ == "__main__":
    main()
