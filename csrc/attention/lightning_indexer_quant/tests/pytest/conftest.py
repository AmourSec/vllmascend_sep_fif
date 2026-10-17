from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.config.argparsing import Parser
    from _pytest.fixtures import FixtureRequest


def pytest_addoption(parser: Parser) -> None:
    parser.addoption("--source", choices=("hbm", "host"), default="hbm")
    parser.addoption("--benchmark", action="store_true", default=False)
    parser.addoption(
        "--cann-template-include", help="Include root containing ascendc/host_api/tiling/template_argument.h"
    )


@pytest.fixture
def memfabric_offload(request: FixtureRequest) -> Iterator[ModuleType | None]:
    if request.config.getoption("--source") != "host":
        yield None
        return
    try:
        from memfabric_hybrid import offload
    except ImportError as exc:
        pytest.fail(f"--source=host requires MemFabric Hybrid release 1.2: {exc}")
    try:
        import torch

        __import__("torch_npu")
    except ImportError as exc:
        pytest.fail(f"--source=host requires torch_npu on an Ascend server: {exc}")
    torch.npu.set_device(0)
    config = offload.OffloadConfig()
    config.device_id = 0
    config.reserve_size = 1 << 30
    config.alloc_size = 1 << 30
    config.flags = offload.OFFLOAD_FLAG_GIANT_PAGE
    config.scene = offload.Scene.LOCAL
    if offload.initialize(config) != 0:
        pytest.fail("MemFabric offload.initialize failed for LOCAL GIANT_PAGE pool")
    try:
        yield offload
    finally:
        torch.npu.synchronize()
        offload.uninitialize()
