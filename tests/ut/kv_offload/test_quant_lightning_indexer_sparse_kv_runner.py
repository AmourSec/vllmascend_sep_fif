from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[3] / "tools/run_quant_lightning_indexer_sparse_kv_tests.py"


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("_qli_runner_under_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "preflight", lambda *_: {})
    return module


@pytest.mark.parametrize("runtime_fails", [False, True])
def test_full_isolates_cpu_checks_from_runtime_collection(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runtime_fails: bool
) -> None:
    calls: list[tuple[list[Path], str, int | None, str | None]] = []

    def record_run(paths: list[Path], name: str, output: Path, device: int | None, marker: str | None = None) -> None:
        assert output == tmp_path
        calls.append((paths, name, device, marker))
        if runtime_fails and runner.RUNTIME_TEST in paths:
            raise RuntimeError("runtime collection failed")

    monkeypatch.setattr(sys, "argv", [str(RUNNER_PATH), "--suite", "full", "--output", str(tmp_path)])
    monkeypatch.setattr(runner, "run_pytest", record_run)

    assert runner.main() == int(runtime_fails)
    assert [call[1] for call in calls] == ["operator", "cpu", "runtime"]
    assert calls[0] == ([runner.OP_TESTS], "operator", 0, None)
    assert runner.CPU_TEST in calls[1][0]
    assert runner.RUNTIME_TEST not in calls[1][0]
    assert calls[1][2:] == (None, None)
    assert calls[2] == ([runner.RUNTIME_TEST], "runtime", 0, None)


def test_runtime_suite_runs_only_runtime_tests(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[Path], str, int | None, str | None]] = []

    def record_run(paths: list[Path], name: str, output: Path, device: int | None, marker: str | None = None) -> None:
        assert output == tmp_path
        calls.append((paths, name, device, marker))

    monkeypatch.setattr(
        sys,
        "argv",
        [str(RUNNER_PATH), "--suite", "runtime", "--device", "2", "--output", str(tmp_path)],
    )
    monkeypatch.setattr(runner, "run_pytest", record_run)

    assert runner.main() == 0
    assert calls == [([runner.RUNTIME_TEST], "runtime", 2, None)]
