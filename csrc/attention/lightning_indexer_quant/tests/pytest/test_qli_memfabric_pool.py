from __future__ import annotations

from types import ModuleType
from typing import TypeVar

import pytest

from memfabric_test_pool import tracked_allocations

_DTypeT = TypeVar("_DTypeT")


class SyntheticAllocationError(RuntimeError):
    pass


class _FakeTensor:
    def __init__(self, pointer: int) -> None:
        self._pointer = pointer

    def data_ptr(self) -> int:
        return self._pointer


@pytest.fixture
def fake_offload() -> tuple[ModuleType, list[tuple[str, int, int]]]:
    module = ModuleType("fake_offload")
    calls: list[tuple[str, int, int]] = []
    next_pointer = 100

    def empty(sizes: list[int], dtype: _DTypeT | None = None, pin_memory: bool = False) -> _FakeTensor:
        nonlocal next_pointer
        del sizes, dtype, pin_memory
        pointer = next_pointer
        next_pointer += 1
        calls.append(("alloc", pointer, 0))
        return _FakeTensor(pointer)

    def free(pointer: int, flags: int = 0) -> None:
        calls.append(("free", pointer, flags))

    module.empty = empty
    module.free = free
    return module, calls


def test_tracked_allocations_frees_owners_and_restores_api(
    fake_offload: tuple[ModuleType, list[tuple[str, int, int]]],
) -> None:
    module, calls = fake_offload
    original_empty = module.empty
    original_free = module.free

    with tracked_allocations(module):
        module.empty([4], dtype="bf16", pin_memory=True)
        module.empty([8])

    assert calls == [("alloc", 100, 0), ("alloc", 101, 0), ("free", 100, 0), ("free", 101, 0)]
    assert module.empty is original_empty
    assert module.free is original_free


def test_explicit_close_is_idempotent_and_free_preserves_flags(
    fake_offload: tuple[ModuleType, list[tuple[str, int, int]]],
) -> None:
    module, calls = fake_offload

    with tracked_allocations(module) as tracker:
        module.empty([4])
        module.free(100, 7)
        tracker.close()
        module.empty([8])
        tracker.close()

    assert calls == [("alloc", 100, 0), ("free", 100, 7), ("alloc", 101, 0), ("free", 101, 0)]


def test_partial_construction_failure_closes_prior_allocations(
    fake_offload: tuple[ModuleType, list[tuple[str, int, int]]],
) -> None:
    module, calls = fake_offload
    original_empty = module.empty

    def failing_empty(sizes: list[int], dtype: _DTypeT | None = None, pin_memory: bool = False) -> _FakeTensor:
        del sizes, dtype, pin_memory
        raise SyntheticAllocationError("synthetic allocation failure")

    with pytest.raises(RuntimeError, match="synthetic allocation failure"):
        with tracked_allocations(module):
            module.empty([4])
            module.empty = failing_empty
            module.empty([8])

    assert calls == [("alloc", 100, 0), ("free", 100, 0)]
    assert module.empty is original_empty
