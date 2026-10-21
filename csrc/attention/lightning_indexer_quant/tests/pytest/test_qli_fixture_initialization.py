from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from conftest import memfabric_offload, resident_offload


class DeviceSelectionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FakeConfig:
    source: str = 'host'

    def getoption(self, name: str) -> str | int:
        if name == '--source':
            return self.source
        if name == '--memfabric-pool-mib':
            return 2048
        raise AssertionError(name)


@dataclass(frozen=True, slots=True)
class FakeRequest:
    offload: ModuleType
    events: list[str]
    config: FakeConfig = FakeConfig()

    def getfixturevalue(self, name: str) -> ModuleType:
        assert name == 'memfabric_offload'
        self.events.append('resolve_memfabric_fixture')
        return self.offload


def fake_runtime(monkeypatch: pytest.MonkeyPatch, *, init_status: int = 0,
                 device_failure: bool = False) -> FakeRequest:
    events: list[str] = []

    def bootstrap(*, include_vendor_lib: bool) -> None:
        assert include_vendor_lib
        events.append('bootstrap')

    def available() -> bool:
        events.append('is_available')
        return True

    def initialize(config: SimpleNamespace) -> int:
        assert config.device_id == 0
        assert config.reserve_size == config.alloc_size == 2048 << 20
        events.append('initialize')
        return init_status

    def set_device(device: int) -> None:
        assert device == 0
        events.append('set_device')
        if device_failure:
            raise DeviceSelectionError('synthetic set_device failure')

    def synchronize() -> None:
        events.append('synchronize')

    def uninitialize() -> None:
        events.append('uninitialize')

    def empty() -> None:
        pytest.fail('initialization test must not allocate payloads')

    def free(_pointer: int, _flags: int = 0) -> None:
        pytest.fail('initialization test has no allocations to free')

    torch = ModuleType('torch')
    torch.npu = SimpleNamespace(is_available=available, set_device=set_device, synchronize=synchronize)
    utils = ModuleType('vllm_ascend.utils')
    utils.bootstrap_custom_op_env = bootstrap
    offload = ModuleType('offload')
    offload.OffloadConfig = SimpleNamespace
    offload.OFFLOAD_FLAG_GIANT_PAGE = 1
    offload.Scene = SimpleNamespace(LOCAL=0)
    offload.initialize = initialize
    offload.uninitialize = uninitialize
    offload.empty = empty
    offload.free = free
    package = ModuleType('memfabric_hybrid')
    package.offload = offload
    for module in (torch, ModuleType('torch_npu'), utils, package):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return FakeRequest(offload, events)


def test_host_fixture_bootstraps_then_initializes_memfabric_before_device(monkeypatch: pytest.MonkeyPatch) -> None:
    request = fake_runtime(monkeypatch)
    fixture = memfabric_offload.__wrapped__(request)
    try:
        assert next(fixture) is request.offload
        assert request.events == ['bootstrap', 'is_available', 'initialize', 'set_device']
    finally:
        fixture.close()
    assert request.events[-2:] == ['synchronize', 'uninitialize']


def test_failed_memfabric_initialization_does_not_select_device(monkeypatch: pytest.MonkeyPatch) -> None:
    request = fake_runtime(monkeypatch, init_status=1)
    fixture = memfabric_offload.__wrapped__(request)
    with pytest.raises(pytest.fail.Exception, match='offload.initialize failed'):
        next(fixture)
    assert request.events == ['bootstrap', 'is_available', 'initialize']


def test_device_selection_failure_releases_initialized_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    request = fake_runtime(monkeypatch, device_failure=True)
    fixture = memfabric_offload.__wrapped__(request)
    with pytest.raises(DeviceSelectionError):
        next(fixture)
    assert request.events == ['bootstrap', 'is_available', 'initialize', 'set_device', 'uninitialize']


def test_resident_fixture_delegates_before_touching_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    request = fake_runtime(monkeypatch)
    assert resident_offload.__wrapped__(request) is request.offload
    assert request.events == ['resolve_memfabric_fixture']


def test_hbm_source_does_not_initialize_memfabric(monkeypatch: pytest.MonkeyPatch) -> None:
    request = fake_runtime(monkeypatch)
    hbm = FakeRequest(request.offload, request.events, FakeConfig('hbm'))
    fixture = memfabric_offload.__wrapped__(hbm)
    assert next(fixture) is None
    fixture.close()
    assert request.events == []
