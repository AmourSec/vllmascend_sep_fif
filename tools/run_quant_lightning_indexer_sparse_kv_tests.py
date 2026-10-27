#!/usr/bin/env python3
"""Run inline per-row fused QLI+Plan checks on Ascend 950 with CANN 9.1.0."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[0] = str(REPO_ROOT)
OP_TESTS = Path("tests/e2e/nightly/single_node/ops/singlecard_ops/quant_lightning_indexer_sparse_kv")
CPU_TEST = Path("tests/ut/kv_offload/test_sparse_kv_plan_oracle.py")
INLINE_SOURCE_TEST = Path("tests/ut/kv_offload/test_quant_lightning_indexer_sparse_kv_inline_source.py")
TILING_TEST = Path("tests/ut/kv_offload/test_quant_lightning_indexer_sparse_kv_tiling.py")
NAMESPACE_TEST = Path("csrc/attention/sparse_kv_plan/tests/test_namespace_collision.py")
HELPERS_TEST = Path("tests/ut/kv_offload/test_quant_lightning_indexer_sparse_kv_helpers.py")
SCORES_TEST = Path("tests/ut/kv_offload/test_quant_lightning_indexer_sparse_kv_scores.py")
IMPORT_TEST = Path("tests/ut/device/test_device_op_import.py")
RUNNER_TEST = Path("tests/ut/kv_offload/test_quant_lightning_indexer_sparse_kv_runner.py")
WORKSPACE_TEST = Path("tests/ut/kv_offload/test_sparse_kv_workspace.py")
MISS_SELECTION_TEST = Path("tests/ut/kv_offload/test_sparse_kv_plan_miss_selection.py")
SCAN_TEST = Path("tests/ut/kv_offload/test_sparse_kv_plan_scan.py")
RUNTIME_TEST = Path("tests/ut/kv_offload/test_sparse_kv_indexer_fusion.py")
STATIC_TESTS = (
    CPU_TEST,
    INLINE_SOURCE_TEST,
    TILING_TEST,
    NAMESPACE_TEST,
    HELPERS_TEST,
    SCORES_TEST,
    IMPORT_TEST,
    RUNNER_TEST,
    WORKSPACE_TEST,
    MISS_SELECTION_TEST,
    SCAN_TEST,
)
BENCHMARK_STATIC_TESTS = (
    Path("tests/ut/kv_offload/test_quant_lightning_indexer_sparse_kv_benchmark.py"),
    Path("tests/ut/kv_offload/test_quant_lightning_indexer_sparse_kv_benchmark_cli.py"),
    Path("tests/ut/kv_offload/test_quant_lightning_indexer_sparse_kv_benchmark_device.py"),
)


def parse_args() -> argparse.Namespace:
    from tests.e2e.nightly.single_node.ops.singlecard_ops.quant_lightning_indexer_sparse_kv.benchmark import (
        add_arguments,
    )
    from tests.e2e.nightly.single_node.ops.singlecard_ops.quant_lightning_indexer_sparse_kv.benchmark_config import (
        CAPACITIES,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=(
            "static",
            "benchmark-static",
            "smoke",
            "inline",
            "runtime",
            "full",
            "stress",
            "profile",
            "benchmark",
        ),
        default="smoke",
        help=(
            "static runs CPU correctness checks; benchmark-static runs CPU benchmark checks; "
            "inline tests UB handoff; runtime runs Python fusion integration only"
        ),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("test-results") / datetime.now().strftime("qli-%Y%m%d-%H%M%S")
    )
    parser.add_argument(
        "--cann-root", type=Path, default=Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    )
    parser.add_argument("--batch", type=int, default=1, help="Profile only: request batch size")
    parser.add_argument("--seq-len", type=int, default=140_000, help="Profile only: context length")
    parser.add_argument("--mtp", type=int, default=1, help="Profile only: query rows per request")
    parser.add_argument("--iterations", type=int, default=20, help="Performance: timed calls per route per round")
    parser.add_argument(
        "--capacity",
        type=int,
        choices=sorted(CAPACITIES),
        default=2048,
        help="Profile only: resident planner capacity",
    )
    parser.add_argument("--trace", action="store_true", help="Profile only: also export separate NPU traces")
    add_arguments(parser)
    return parser.parse_args()


def verify_cann(cann_root: Path) -> dict[str, str]:
    version_files: dict[str, str] = {}
    versions: set[str] = set()
    for relative in (
        "version.cfg",
        "version.info",
        "aarch64-linux/ascend_toolkit_install.info",
        "x86_64-linux/ascend_toolkit_install.info",
    ):
        path = cann_root / relative
        if path.is_file():
            content = path.read_text(errors="replace")
            version_files[str(path)] = content
            for line in content.splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip().lower() == "version":
                    versions.add(value.strip().strip("\"'"))
    if versions != {"9.1.0"}:
        raise RuntimeError(
            f"Expected CANN 9.1.0, found {sorted(versions) or 'no version field'} under {cann_root}; "
            "source the CANN 9.1.0 environment and pass --cann-root if needed"
        )
    return version_files


def preflight(device: int, cann_root: Path) -> dict[str, object]:
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        raise RuntimeError("An available Ascend 950 NPU is required; refusing a skipped test run")
    torch.npu.set_device(device)
    name = torch.npu.get_device_name(device)
    if "950" not in name.lower():
        raise RuntimeError(f"This suite targets Ascend 950, but device {device} reports {name!r}")
    utils = importlib.import_module("vllm_ascend.utils")
    utils.bootstrap_custom_op_env(include_vendor_lib=True)
    extension = importlib.import_module("vllm_ascend.vllm_ascend_C")
    for op in ("npu_quant_lightning_indexer_sparse_kv", "npu_sparse_kv_plan", "npu_sparse_kv_transfer"):
        if not hasattr(torch.ops._C_ascend, op):
            raise RuntimeError(f"Missing {op}; rebuild this checkout's custom operators and extension")
    if not hasattr(torch_npu, "npu_quant_lightning_indexer"):
        raise RuntimeError("torch_npu is missing the standalone quant_lightning_indexer baseline")
    version_files = verify_cann(cann_root)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=REPO_ROOT, capture_output=True, check=True
    ).stdout
    return {
        "git_head": head,
        "git_dirty": bool(status),
        "git_status": status,
        "git_tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "extension": extension.__file__,
        "device": name,
        "device_index": device,
        "expected_cann": "9.1.0",
        "cann_version_files": version_files,
    }


def run_logged(command: list[str], log_path: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    with (
        log_path.open("w") as log,
        subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as process,
    ):
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"Command exited {code}; see {log_path}")


def module_command(module: str, device: int) -> list[str]:
    launcher = (
        "import runpy,sys,torch,torch_npu; "
        f"torch.npu.set_device({device}); "
        "sys.argv=sys.argv[1:]; runpy.run_module(sys.argv[0],run_name='__main__')"
    )
    return [sys.executable, "-c", launcher, module]


def run_pytest(
    paths: list[Path],
    name: str,
    output: Path,
    device: int | None,
    marker: str | None = None,
) -> None:
    report = output / f"{name}.xml"
    confcutdir = paths[0] if paths[0].is_dir() else paths[0].parent
    command = (module_command("pytest", device) if device is not None else [sys.executable, "-m", "pytest"]) + [
        "-v",
        "--strict-markers",
        f"--confcutdir={confcutdir}",
    ]
    command.extend(str(path) for path in paths)
    command.append(f"--junitxml={report}")
    if marker:
        command.extend(["-m", marker])
    run_logged(command, output / f"{name}.log")
    cases = list(ET.parse(report).getroot().iter("testcase"))
    skipped = [case.attrib.get("name", "?") for case in cases if case.find("skipped") is not None]
    if not cases or skipped:
        raise RuntimeError(f"{name} did not fully execute: {len(cases)} tests; skipped={skipped}")


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.suite == "benchmark":
            from tests.e2e.nightly.single_node.ops.singlecard_ops.quant_lightning_indexer_sparse_kv.benchmark import (
                run_benchmark,
            )

            return run_benchmark(args)
        if args.dry_run:
            raise RuntimeError("--dry-run requires --suite benchmark")
        if args.suite == "static":
            run_pytest(list(STATIC_TESTS), "static", output, None)
            print(f"Completed static; results: {output}")
            return 0
        if args.suite == "benchmark-static":
            run_pytest(list(BENCHMARK_STATIC_TESTS), "benchmark-static", output, None)
            print(f"Completed benchmark-static; results: {output}")
            return 0
        metadata = preflight(args.device, args.cann_root)
        (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2), flush=True)
        if args.suite == "profile":
            module = ".".join(OP_TESTS.parts) + ".profile"
            run_logged(
                module_command(module, args.device)
                + [
                    "--output",
                    str(output / "profile"),
                    "--batch",
                    str(args.batch),
                    "--seq-len",
                    str(args.seq_len),
                    "--mtp",
                    str(args.mtp),
                    "--iterations",
                    str(args.iterations),
                    "--capacity",
                    str(args.capacity),
                    "--warmup",
                    str(args.warmup),
                    "--rounds",
                    str(args.rounds),
                    "--seed",
                    str(args.seed),
                    "--execution",
                    args.execution,
                ]
                + (["--trace"] if args.trace else []),
                output / "profile.log",
            )
        elif args.suite == "runtime":
            run_pytest([RUNTIME_TEST], "runtime", output, args.device)
        else:
            marker = None if args.suite == "full" else f"npu_{args.suite}"
            run_pytest([OP_TESTS], "operator", output, args.device, marker)
            if args.suite == "full":
                run_pytest(list(STATIC_TESTS), "cpu", output, None)
                run_pytest([RUNTIME_TEST], "runtime", output, args.device)
        print(f"Completed {args.suite}; results: {output}")
        return 0
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
