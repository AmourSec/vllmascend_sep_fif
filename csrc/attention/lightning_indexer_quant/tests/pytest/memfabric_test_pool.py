from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from torch import Tensor
    from torch import dtype as TorchDType


class _TensorOwner(Protocol):
    def data_ptr(self) -> int: ...


Owner = TypeVar("Owner", bound=_TensorOwner)


class _FreeFn(Protocol):
    def __call__(self, pointer: int, flags: int = 0) -> None: ...


class NullAllocationPointer(ValueError):
    pass


class AllocationTracker:
    def __init__(self, free_fn: _FreeFn) -> None:
        self._free_fn = free_fn
        self._owners: dict[int, _TensorOwner] = {}

    def track(self, tensor: Owner) -> Owner:
        pointer = tensor.data_ptr()
        if pointer == 0:
            raise NullAllocationPointer("MemFabric returned a null allocation pointer")
        self._owners[pointer] = tensor
        return tensor

    def forget(self, pointer: int) -> None:
        self._owners.pop(pointer, None)

    def close(self) -> None:
        for pointer in tuple(self._owners):
            self._free_fn(pointer, 0)
            del self._owners[pointer]


@contextmanager
def tracked_allocations(offload: ModuleType) -> Iterator[AllocationTracker]:
    original_empty = offload.empty
    original_free = offload.free
    tracker = AllocationTracker(original_free)

    def tracked_empty(
        sizes: Sequence[int],
        dtype: TorchDType | None = None,
        pin_memory: bool = False,
    ) -> Tensor:
        tensor = original_empty(sizes, dtype, pin_memory)
        return tracker.track(tensor)

    def tracked_free(pointer: int, flags: int = 0) -> None:
        original_free(pointer, flags)
        tracker.forget(pointer)

    offload.empty = tracked_empty
    offload.free = tracked_free
    try:
        yield tracker
    finally:
        try:
            tracker.close()
        finally:
            offload.empty = original_empty
            offload.free = original_free
