from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from typing_extensions import assert_never

if TYPE_CHECKING:
    from torch import Tensor

DType = Literal["int8", "fp8"]
INDEX_BLOCK: Final = 128
HOST_BLOCK: Final = 128
KEY_BYTES: Final = 1024
VALUE_BYTES: Final = 128
TOPK: Final = 2048
CAPACITY: Final = 4096
HEADS: Final = 32
DIM: Final = 128


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    query_counts: tuple[int, ...]
    context: int
    dtype: DType

    @property
    def batch(self) -> int:
        return len(self.query_counts)

    @property
    def rows(self) -> int:
        return sum(self.query_counts)

    @property
    def query_ends(self) -> np.ndarray:
        return np.cumsum(self.query_counts, dtype=np.int32)

    @property
    def requests(self) -> np.ndarray:
        return np.repeat(np.arange(self.batch, dtype=np.int32), self.query_counts)

    @property
    def visible(self) -> np.ndarray:
        return np.asarray([
            self.context - count + offset + 1
            for count in self.query_counts for offset in range(count)
        ], dtype=np.int32)

    @property
    def stable(self) -> np.ndarray:
        return np.repeat(
            np.asarray(self.context - np.asarray(self.query_counts), dtype=np.int32),
            self.query_counts,
        )

    def table(self, block_size: int) -> np.ndarray:
        columns = max(1, (self.context + block_size - 1) // block_size)
        return np.arange(self.batch * columns, dtype=np.int32).reshape(self.batch, columns)


@dataclass(frozen=True, slots=True)
class ResidentSnapshot:
    resident_k: Tensor
    resident_v: Tensor
    last_req_ids: Tensor
    slots: Tensor
    lru: Tensor


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    torch: ModuleType
    offload: ModuleType
    spec: BenchmarkSpec
    query: Tensor
    key: Tensor
    weights: Tensor
    query_scale: Tensor
    key_scale: Tensor
    query_ends: Tensor
    key_lengths: Tensor
    block_table: Tensor
    threshold: Tensor
    req_ids: Tensor
    token_to_req: Tensor
    stable: Tensor
    visible: Tensor
    host_table: Tensor
    bases: Tensor
    resident_k: Tensor
    resident_v: Tensor
    last_req_ids: Tensor
    slots: Tensor
    lru: Tensor
    active: Tensor
    source_k: Tensor
    source_v: Tensor
    current_slots: Tensor
    miss_count: Tensor
    miss_tokens: Tensor
    miss_slots: Tensor

    def snapshot(self) -> ResidentSnapshot:
        return ResidentSnapshot(*(value.clone() for value in (
            self.resident_k, self.resident_v, self.last_req_ids, self.slots, self.lru
        )))

    def restore(self, snapshot: ResidentSnapshot) -> None:
        for target, value in zip(
            (self.resident_k, self.resident_v, self.last_req_ids, self.slots, self.lru),
            (snapshot.resident_k, snapshot.resident_v, snapshot.last_req_ids, snapshot.slots, snapshot.lru),
            strict=True,
        ):
            target.copy_(value)

    def close(self) -> None:
        self.torch.npu.synchronize()
        self.offload.free(self.source_k.data_ptr(), 0)
        self.offload.free(self.source_v.data_ptr(), 0)

    def qli(self, budget: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.torch.ops._C_ascend.npu_lightning_indexer_quant_prefetch_resident(
            self.query, self.key, self.weights, self.query_scale, self.key_scale,
            self.query_ends, self.key_lengths, self.block_table, self.threshold,
            self.req_ids, self.token_to_req, self.stable, self.host_table, self.bases,
            self.resident_k, self.resident_v, self.last_req_ids, self.slots, self.lru,
            self.active, HOST_BLOCK, self.source_k.shape[0], budget, TOPK, 0.5,
        )

    def chain(self, budget: int, workspace: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sparse, candidates, stats, next_threshold = self.qli(budget)
        self.torch.ops._C_ascend.npu_sparse_kv_plan_transfer(
            self.req_ids, sparse[:, 0].contiguous(), self.stable, self.visible,
            self.token_to_req, self.host_table, self.active, self.last_req_ids,
            self.slots, self.lru, self.current_slots, self.miss_count, self.miss_tokens,
            self.miss_slots, workspace, self.bases, self.resident_k, self.resident_v,
            TOPK, CAPACITY, self.host_table.shape[1] * HOST_BLOCK, HOST_BLOCK,
            self.source_k.shape[0], KEY_BYTES, VALUE_BYTES,
        )
        return sparse, candidates, stats, next_threshold, self.miss_count


def specs(profile: str, dtype: DType) -> list[BenchmarkSpec]:
    if profile == "smoke":
        shapes = (((1,), 8192), ((2,) * 4, 32768))
    else:
        shapes = (
            ((1,), 8192), ((1,) * 4, 32768), ((2,) * 4, 32768),
            ((4,) * 4, 32768), ((4,) * 8, 131072), ((1, 2, 4, 1), 131072),
        )
    return [BenchmarkSpec(query_counts, context, dtype) for query_counts, context in shapes]


def make_case(torch: ModuleType, offload: ModuleType, spec: BenchmarkSpec) -> BenchmarkCase:
    torch.manual_seed(20261020 + spec.batch * 17 + spec.rows)
    rows = spec.rows
    index_pages = (spec.context + INDEX_BLOCK - 1) // INDEX_BLOCK
    host_pages = (spec.context + HOST_BLOCK - 1) // HOST_BLOCK
    index_table = spec.table(INDEX_BLOCK)
    host_table = spec.table(HOST_BLOCK)
    device = lambda value, dtype: torch.tensor(value, dtype=dtype, device="npu")
    query = torch.randint(-32, 32, (rows, HEADS, DIM), dtype=torch.int8, device="npu")
    key = torch.randint(-32, 32, (spec.batch * index_pages, INDEX_BLOCK, 1, DIM), dtype=torch.int8, device="npu")
    match spec.dtype:
        case "int8":
            auxiliary_dtype = torch.float16
        case "fp8":
            query = (query.float() / 16).to(torch.float8_e4m3fn)
            key = (key.float() / 16).to(torch.float8_e4m3fn)
            auxiliary_dtype = torch.float32
        case unreachable:
            assert_never(unreachable)
    query_scale = torch.full((rows, HEADS), 0.25, dtype=auxiliary_dtype, device="npu")
    key_scale = torch.full((spec.batch * index_pages, INDEX_BLOCK, 1), 0.125,
                           dtype=auxiliary_dtype, device="npu")
    weights = torch.ones((rows, HEADS), dtype=auxiliary_dtype, device="npu")
    source_k = offload.empty([spec.batch * host_pages, HOST_BLOCK, KEY_BYTES], dtype=torch.uint8)
    source_v = offload.empty([spec.batch * host_pages, HOST_BLOCK, VALUE_BYTES], dtype=torch.uint8)
    assert source_k.device.type == source_v.device.type == "cpu", "expected mapped DDR source"
    source_k.random_(0, 256)
    source_v.random_(0, 256)
    bases = device([int(offload.get_dva(source_k.data_ptr())), int(offload.get_dva(source_v.data_ptr()))], torch.int64)
    assert not bool(torch.any(bases == 0)), "MemFabric returned a zero DVA"
    resident_k = torch.full((rows, CAPACITY, KEY_BYTES), 173, dtype=torch.uint8, device="npu")
    resident_v = torch.full((rows, CAPACITY, VALUE_BYTES), 211, dtype=torch.uint8, device="npu")
    resident_k = resident_k.view(torch.bfloat16).reshape(rows, CAPACITY, 1, KEY_BYTES // 2)
    resident_v = resident_v.view(torch.bfloat16).reshape(rows, CAPACITY, 1, VALUE_BYTES // 2)
    visible = spec.visible
    return BenchmarkCase(
        torch, offload, spec, query, key, weights, query_scale, key_scale,
        device(spec.query_ends, torch.int32), device([spec.context] * spec.batch, torch.int32),
        device(index_table, torch.int32), torch.full((rows,), -float("inf"), dtype=torch.float32, device="npu"),
        device(np.arange(spec.batch) + 700, torch.int64), device(spec.requests, torch.int32),
        device(spec.stable, torch.int32), device(visible, torch.int32), device(host_table, torch.int32), bases,
        resident_k, resident_v, torch.full((rows,), -1, dtype=torch.int64, device="npu"),
        torch.full((rows, CAPACITY), -1, dtype=torch.int32, device="npu"),
        torch.arange(CAPACITY, dtype=torch.int32, device="npu").repeat(rows, 1),
        torch.tensor([rows], dtype=torch.int32, device="npu"), source_k, source_v,
        torch.full((rows, TOPK), -1, dtype=torch.int32, device="npu"),
        torch.zeros(rows, dtype=torch.int32, device="npu"),
        torch.full((rows, TOPK), -1, dtype=torch.int32, device="npu"),
        torch.full((rows, TOPK), -1, dtype=torch.int32, device="npu"),
    )
