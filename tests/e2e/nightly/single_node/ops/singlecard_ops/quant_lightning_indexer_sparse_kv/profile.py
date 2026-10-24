import argparse
import importlib
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile inline per-row fused QLI+Plan against the stock two-op route")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=140_000)
    parser.add_argument("--mtp", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--capacity", type=int, choices=(2048, 4096, 8192), default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch, args.seq_len, args.mtp, args.iterations) <= 0:
        raise SystemExit("batch, seq-len, mtp, and iterations must be positive")

    import torch
    import torch_npu

    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    importlib.import_module("vllm_ascend.vllm_ascend_C")

    from .assertions import assert_state_matches, assert_states_equal, oracle_after
    from .helpers import make_case, make_state, require_ops, run_fused, run_indexer, run_plan

    require_ops()
    args.output.mkdir(parents=True, exist_ok=True)
    case = make_case(
        [args.seq_len] * args.batch,
        [args.mtp] * args.batch,
        capacity=args.capacity,
        seed=2026,
    )
    initial = make_state(case)
    fused_warm = initial.clone()
    baseline_warm = initial.clone()
    fused_indices = run_fused(case, fused_warm)
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_warm, baseline_indices)
    torch_npu.npu.synchronize()
    torch.testing.assert_close(fused_indices, baseline_indices, rtol=0, atol=0)
    assert_state_matches(fused_warm, oracle_after(case, initial, fused_indices))
    assert_states_equal(fused_warm, baseline_warm)

    fused_state = fused_warm.clone()
    fused_start = torch.npu.Event(enable_timing=True)
    fused_end = torch.npu.Event(enable_timing=True)
    fused_start.record()
    for _ in range(args.iterations):
        run_fused(case, fused_state)
    fused_end.record()
    torch_npu.npu.synchronize()
    fused_ms = fused_start.elapsed_time(fused_end) / args.iterations

    baseline_state = baseline_warm.clone()
    baseline_start = torch.npu.Event(enable_timing=True)
    baseline_end = torch.npu.Event(enable_timing=True)
    baseline_start.record()
    for _ in range(args.iterations):
        indices = run_indexer(case)
        run_plan(case, baseline_state, indices)
    baseline_end.record()
    torch_npu.npu.synchronize()
    baseline_ms = baseline_start.elapsed_time(baseline_end) / args.iterations

    if args.trace:
        activities = [
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ]
        trace_root = args.output / "traces"
        trace_routes = (
            ("fused_inline_ub_qli_plan", fused_warm.clone(), lambda state: run_fused(case, state)),
            (
                "standalone_qli_then_plan",
                baseline_warm.clone(),
                lambda state: run_plan(case, state, run_indexer(case)),
            ),
        )
        for label, trace_state, route in trace_routes:
            handler = torch_npu.profiler.tensorboard_trace_handler(str(trace_root / label), worker_name=label)
            with torch_npu.profiler.profile(activities=activities, on_trace_ready=handler) as profiler:
                with torch.autograd.profiler.record_function(label):
                    route(trace_state)
                profiler.step()
            torch_npu.npu.synchronize()

    result = {
        "source_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[7],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "batch": args.batch,
        "seq_len": args.seq_len,
        "mtp": args.mtp,
        "capacity": args.capacity,
        "iterations": args.iterations,
        "fused_ms": fused_ms,
        "standalone_qli_then_plan_ms": baseline_ms,
        "standalone_over_fused_ratio": baseline_ms / fused_ms,
        "trace_enabled": args.trace,
    }
    output_path = args.output / "profile.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
