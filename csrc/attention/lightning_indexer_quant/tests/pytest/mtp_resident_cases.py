from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mtp_resident_oracle import DType, Geometry, INDEX_BLOCK, KEY_BYTES, ScoreInputs, VALUE_BYTES

if TYPE_CHECKING:
    from torch import Tensor


def runtime() -> ModuleType:
    torch = pytest.importorskip("torch", reason="QLI NPU tests require PyTorch")
    pytest.importorskip("torch_npu", reason="QLI NPU tests require Ascend torch_npu")
    if not torch.npu.is_available():
        pytest.skip("QLI NPU tests require an available Ascend NPU")
    utils = importlib.import_module("vllm_ascend.utils")
    utils.bootstrap_custom_op_env(include_vendor_lib=True)
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    for name in ("npu_lightning_indexer_quant_prefetch", "npu_lightning_indexer_quant_prefetch_resident"):
        assert hasattr(torch.ops._C_ascend, name), f"installed NPU extension is missing {name}"
    return torch


@dataclass(frozen=True, slots=True)
class ResidentCase:
    torch: ModuleType
    geometry: Geometry
    query: Tensor
    key: Tensor
    weights: Tensor
    query_scale: Tensor
    key_scale: Tensor
    query_ends: Tensor
    key_lengths: Tensor
    table: Tensor
    threshold: Tensor
    req_ids: Tensor
    token_to_req: Tensor
    stable: Tensor
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
    golden: ScoreInputs

    def run(self, budget: int = 4, count: int = 7) -> tuple[Tensor, ...]:
        g = self.geometry
        return self.torch.ops._C_ascend.npu_lightning_indexer_quant_prefetch_resident(
            self.query, self.key, self.weights, self.query_scale, self.key_scale,
            self.query_ends, self.key_lengths, self.table, self.threshold, self.req_ids,
            self.token_to_req, self.stable, self.host_table, self.bases,
            self.resident_k, self.resident_v, self.last_req_ids, self.slots,
            self.lru, self.active, g.host_block, self.source_k.shape[0], budget, count, 0.5,
        )

    def cache_bytes(self, tensor: Tensor) -> Tensor:
        return tensor.view(self.torch.uint8).reshape(self.geometry.rows, self.geometry.capacity, -1)

    def expected_bytes(self, row: int, token: int) -> tuple[np.ndarray, np.ndarray]:
        request = int(self.geometry.requests[row])
        page = int(self.host_table[request, token // self.geometry.host_block].cpu())
        offset = token % self.geometry.host_block
        return (self.source_k[page, offset].cpu().numpy(), self.source_v[page, offset].cpu().numpy())

    def assert_payload(self, row: int, token: int) -> None:
        slots = self.slots[row].cpu().numpy()
        matches = np.flatnonzero(slots == token)
        assert matches.size == 1
        slot = int(matches[0])
        expected_k, expected_v = self.expected_bytes(row, token)
        np.testing.assert_array_equal(self.cache_bytes(self.resident_k)[row, slot].cpu().numpy(), expected_k)
        np.testing.assert_array_equal(self.cache_bytes(self.resident_v)[row, slot].cpu().numpy(), expected_v)


def make_case(torch: ModuleType, geometry: Geometry = Geometry(), *, dtype: DType = "int8",
              heads: int = 32, typed_cache: bool = False, offload: ModuleType | None = None) -> ResidentCase:
    torch.manual_seed(811)
    g = geometry
    table_cpu = g.table(INDEX_BLOCK)
    pages = table_cpu.size
    shape_q = (g.rows, heads, 128)
    shape_k = (pages, INDEX_BLOCK, 1, 128)
    if dtype == "int8":
        query = torch.randint(-32, 32, shape_q, dtype=torch.int8, device="npu")
        key = torch.randint(-32, 32, shape_k, dtype=torch.int8, device="npu")
        query_scale = torch.full(shape_q[:2], 0.25, dtype=torch.float16, device="npu")
        key_scale = torch.full(shape_k[:3], 0.125, dtype=torch.float16, device="npu")
    else:
        query_fp = torch.randn(shape_q, dtype=torch.float32, device="npu")
        key_fp = torch.randn(shape_k, dtype=torch.float32, device="npu")
        query_scale = query_fp.abs().amax(dim=-1).clamp_min(1e-6) / 448
        key_scale = key_fp.abs().amax(dim=-1).clamp_min(1e-6) / 448
        query = (query_fp / query_scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        key = (key_fp / key_scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    weights = torch.rand((g.rows, heads), dtype=query_scale.dtype, device="npu")
    host_table_cpu = g.table(g.host_block)
    sources = []
    for width in (KEY_BYTES, VALUE_BYTES):
        payload = torch.randint(0, 256, (host_table_cpu.size, g.host_block, width), dtype=torch.uint8)
        if offload is None:
            source = payload.to("npu")
        else:
            source = offload.empty(list(payload.shape), dtype=torch.uint8)
            source.copy_(payload)
        sources.append(source)
    addresses = [
        source.data_ptr() if offload is None else int(offload.get_dva(source.data_ptr()))
        for source in sources
    ]
    assert all(addresses), "MemFabric returned a zero DVA"
    k = torch.full((g.rows, g.capacity, KEY_BYTES), 173, dtype=torch.uint8, device="npu")
    v = torch.full((g.rows, g.capacity, VALUE_BYTES), 211, dtype=torch.uint8, device="npu")
    if typed_cache:
        k = k.view(torch.bfloat16).reshape(g.rows, g.capacity, 1, KEY_BYTES // 2)
        v = v.view(torch.bfloat16).reshape(g.rows, g.capacity, 1, VALUE_BYTES // 2)
    def device(values: np.ndarray, dtype: torch.dtype) -> Tensor:
        return torch.tensor(values, dtype=dtype, device="npu")
    golden = ScoreInputs(query.float().cpu().numpy(), key.float().cpu().numpy(), weights.cpu().numpy(),
                         query_scale.cpu().numpy(), key_scale.cpu().numpy(), table_cpu, g, dtype)
    return ResidentCase(
        torch, g, query, key, weights, query_scale, key_scale,
        device(g.query_ends, torch.int32), device(np.asarray(g.key_lengths), torch.int32),
        device(table_cpu, torch.int32), torch.full((g.rows,), -np.inf, dtype=torch.float32, device="npu"),
        device(np.arange(len(g.query_counts)) + 41, torch.int64), device(g.requests, torch.int32),
        device(g.visible, torch.int32), device(host_table_cpu, torch.int32),
        device(np.asarray(addresses), torch.int64), k, v,
        torch.full((g.rows,), -1, dtype=torch.int64, device="npu"),
        torch.full((g.rows, g.capacity), -1, dtype=torch.int32, device="npu"),
        torch.arange(g.capacity, dtype=torch.int32, device="npu").repeat(g.rows, 1),
        device(np.asarray([g.rows]), torch.int32), sources[0], sources[1], golden,
    )
