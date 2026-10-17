from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch

    Tensor = torch.Tensor
else:
    Tensor = object

from lightning_indexer_quant_prefetch_golden import PrefetchInputs

DType = Literal["int8", "fp8"]


class Case(NamedTuple):
    query: Tensor
    key: Tensor
    weights: Tensor
    query_scale: Tensor
    key_scale: Tensor
    payload: Tensor
    query_lengths: Tensor
    key_lengths: Tensor
    block_table: Tensor
    addresses: Tensor
    golden: PrefetchInputs
    host_rows: tuple[Tensor, ...]


def _runtime() -> ModuleType:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    utils = importlib.import_module("vllm_ascend.utils")
    utils.bootstrap_custom_op_env(include_vendor_lib=True)
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, "npu_lightning_indexer_quant_prefetch"):
        raise AssertionError("prefetch operator is not registered in _C_ascend")
    return torch


def _make_case(
    torch: ModuleType,
    *,
    threshold: float = -np.inf,
    budget: int = 4,
    offload: ModuleType | None = None,
    lengths: tuple[int, ...] = (13, 5, 0),
    physical: int = 12,
    block_size: int = 16,
    sparse_count: int = 7,
    negative_weights: bool = False,
    tied_scores: bool = False,
    row_bytes: int = 128,
    heads: int = 32,
    seed: int = 811,
    quantized_inputs: bool = False,
    dtype: DType = "int8",
) -> Case:
    if dtype not in {"int8", "fp8"}:
        raise ValueError(f"unsupported test dtype: {dtype}")
    torch.manual_seed(seed)
    batch, width = len(lengths), 128
    if dtype == "fp8":
        query_fp = torch.randn((batch, heads, width), dtype=torch.float32, device="npu")
        key_fp = torch.randn((physical, block_size, 1, width), dtype=torch.float32, device="npu")
        query_scale = torch.amax(torch.abs(query_fp), dim=-1).clamp_min(1.0e-6) / 448.0
        key_scale = torch.amax(torch.abs(key_fp), dim=-1).clamp_min(1.0e-6) / 448.0
        query = (query_fp / query_scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        key = (key_fp / key_scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    elif quantized_inputs:
        query_fp = torch.randn((batch, heads, width), dtype=torch.float32, device="npu")
        query_scale = torch.amax(torch.abs(query_fp), dim=-1).clamp_min(1.0e-6) / 127.0
        query = torch.round(query_fp / query_scale[..., None]).clamp(-127, 127).to(torch.int8)
        key_fp = torch.randn((physical, block_size, 1, width), dtype=torch.float32, device="npu")
        key_scale = torch.amax(torch.abs(key_fp), dim=-1, keepdim=True).clamp_min(1.0e-6).squeeze(-1) / 127.0
        key = torch.round(key_fp / key_scale[..., None]).clamp(-127, 127).to(torch.int8)
        query_scale = query_scale.to(torch.float16)
        key_scale = key_scale.to(torch.float16)
    else:
        query = torch.randint(-32, 32, (batch, heads, width), dtype=torch.int8, device="npu")
        key = torch.randint(-32, 32, (physical, block_size, 1, width), dtype=torch.int8, device="npu")
        query_scale = torch.full((batch, heads), 0.25, dtype=torch.float16, device="npu")
        key_scale = torch.full((physical, block_size, 1), 0.125, dtype=torch.float16, device="npu")
    payload = torch.randint(0, 256, (physical, block_size, 1, row_bytes), dtype=torch.uint8, device="npu")
    if tied_scores:
        key.zero_()
    weight_dtype = torch.float32 if dtype == "fp8" else torch.float16
    weights = torch.rand((batch, heads), dtype=weight_dtype, device="npu")
    if negative_weights:
        weights = torch.linspace(-1.0, 1.0, heads, dtype=weight_dtype, device="npu").repeat(batch, 1)
    key_lengths = torch.tensor(lengths, dtype=torch.int32, device="npu")
    query_lengths = torch.arange(1, batch + 1, dtype=torch.int32, device="npu")
    max_blocks = max(1, max((length + block_size - 1) // block_size for length in lengths))
    logical_blocks = torch.arange(max_blocks, dtype=torch.int32, device="npu")[None, :]
    request_offsets = torch.arange(batch, dtype=torch.int32, device="npu")[:, None]
    block_table = (logical_blocks * 5 + request_offsets * 3 + 1).remainder(physical)
    addresses_cpu = np.zeros((batch, block_table.shape[1] * block_size), dtype=np.int64)
    host_rows: list[Tensor] = []
    for row, length in enumerate(lengths):
        for token in range(length):
            page = int(block_table[row, token // block_size].cpu())
            offset = token % block_size
            if offload is None:
                addresses_cpu[row, token] = int(payload[page, offset, 0].data_ptr())
            else:
                host_row = offload.empty([row_bytes], dtype=torch.uint8)
                host_row.copy_(payload[page, offset, 0].cpu())
                host_rows.append(host_row)
                dva = int(offload.get_dva(host_row.data_ptr()))
                if dva == 0:
                    raise RuntimeError("MemFabric get_dva returned zero for an offload row")
                addresses_cpu[row, token] = dva
    addresses = torch.tensor(addresses_cpu, dtype=torch.int64, device="npu")
    if dtype == "fp8":
        query_cpu = query.float().cpu().numpy()
        key_cpu = key.float().cpu().numpy()
    else:
        query_cpu = query.cpu().numpy()
        key_cpu = key.cpu().numpy()
    golden = PrefetchInputs(
        query=query_cpu,
        key=key_cpu,
        weights=weights.cpu().numpy(),
        query_dequant_scale=query_scale.cpu().numpy(),
        key_dequant_scale=key_scale.cpu().numpy(),
        actual_seq_lengths_key=np.asarray(lengths, dtype=np.int32),
        block_table=block_table.cpu().numpy(),
        prefetch_threshold=np.full(batch, threshold, dtype=np.float32),
        kv_row_bytes=row_bytes,
        prefetch_budget=budget,
        sparse_count=sparse_count,
        kv_payload=payload.cpu().numpy(),
        dtype=dtype,
    )
    return Case(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        payload,
        query_lengths,
        key_lengths,
        block_table,
        addresses,
        golden,
        tuple(host_rows),
    )


def _run(
    torch: ModuleType,
    case: Case,
    *,
    threshold: float | Tensor,
    budget: int,
    query_lengths: Tensor | None = None,
    key_lengths: Tensor | None = None,
    block_table: Tensor | None = None,
) -> tuple[Tensor, ...]:
    threshold_tensor: float | Tensor = threshold
    if not hasattr(threshold, "device"):
        threshold_tensor = torch.full((case.query.size(0),), threshold, dtype=torch.float32, device="npu")
    return torch.ops._C_ascend.npu_lightning_indexer_quant_prefetch(
        case.query,
        case.key,
        case.weights,
        case.query_scale,
        case.key_scale,
        case.query_lengths if query_lengths is None else query_lengths,
        case.key_lengths if key_lengths is None else key_lengths,
        case.block_table if block_table is None else block_table,
        case.addresses,
        threshold_tensor,
        case.golden.kv_row_bytes,
        budget,
        case.golden.sparse_count,
        case.golden.ema_decay,
    )
