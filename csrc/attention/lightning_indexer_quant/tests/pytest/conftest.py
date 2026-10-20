from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from memfabric_test_pool import tracked_allocations

if TYPE_CHECKING:
    from _pytest.config.argparsing import Parser
    from _pytest.fixtures import FixtureRequest


def pytest_addoption(parser: Parser) -> None:
    parser.addoption("--source", choices=("hbm", "host"), default="host")
    parser.addoption("--benchmark", action="store_true", default=False)
    parser.addoption("--memfabric-pool-mib", type=int, default=2048,
                     help="LOCAL mapped DDR pool size in MiB (default: 2048)")
    benchmark = parser.getgroup("QLI resident benchmark")
    benchmark.addoption("--resident-bench-profile", choices=("smoke", "full"), default="full")
    benchmark.addoption("--resident-bench-dtype", choices=("int8", "fp8"), default="int8")
    benchmark.addoption("--resident-bench-budgets", default="512,2048",
                        help="Comma-separated enabled budgets; each is compared with budget=0")
    benchmark.addoption("--resident-bench-states", default="cold,partial,warm")
    benchmark.addoption("--resident-bench-threshold-mode", choices=("previous_query", "same_input"),
                        default="previous_query",
                        help="Synthetic previous-query threshold, or optimistic same-input calibration")
    benchmark.addoption("--resident-bench-warmup", type=int, default=3)
    benchmark.addoption("--resident-bench-iters", type=int, default=10)
    benchmark.addoption("--resident-bench-output", help="Append benchmark results to this JSONL file")
    parser.addoption(
        "--cann-template-include", help="Include root containing ascendc/host_api/tiling/template_argument.h"
    )


@pytest.fixture
def memfabric_offload(request: FixtureRequest) -> Iterator[ModuleType | None]:
    if request.config.getoption("--source") != "host":
        yield None
        return
    torch = pytest.importorskip("torch", reason="QLI NPU tests require PyTorch")
    pytest.importorskip("torch_npu", reason="QLI NPU tests require Ascend torch_npu")
    if not torch.npu.is_available():
        pytest.skip("QLI NPU tests require an available Ascend NPU")
    try:
        from memfabric_hybrid import offload
    except ImportError as exc:
        pytest.fail(f"--source=host requires MemFabric Hybrid release 1.2: {exc}")
    pool_mib = request.config.getoption("--memfabric-pool-mib")
    if pool_mib < 1:
        raise pytest.UsageError("--memfabric-pool-mib must be positive")
    torch.npu.set_device(0)
    config = offload.OffloadConfig()
    config.device_id = 0
    config.reserve_size = pool_mib << 20
    config.alloc_size = pool_mib << 20
    config.flags = offload.OFFLOAD_FLAG_GIANT_PAGE
    config.scene = offload.Scene.LOCAL
    if offload.initialize(config) != 0:
        pytest.fail("MemFabric offload.initialize failed for LOCAL GIANT_PAGE pool")
    try:
        with tracked_allocations(offload):
            try:
                yield offload
            finally:
                torch.npu.synchronize()
    finally:
        offload.uninitialize()


@pytest.fixture
def resident_offload(request: FixtureRequest) -> ModuleType:
    if request.config.getoption("--source") != "host":
        pytest.skip("QLI MTP/resident tests require --source=host for DDR-to-HBM validation")
    torch = pytest.importorskip("torch", reason="QLI NPU tests require PyTorch")
    pytest.importorskip("torch_npu", reason="QLI NPU tests require Ascend torch_npu")
    if not torch.npu.is_available():
        pytest.skip("QLI NPU tests require an available Ascend NPU")
    offload = request.getfixturevalue("memfabric_offload")
    if offload is None:
        pytest.fail("resident tests require an initialized MemFabric host pool")
    return offload
