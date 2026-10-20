from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

DType = Literal["int8", "fp8"]
INDEX_BLOCK: Final = 16
KEY_BYTES: Final = 64
VALUE_BYTES: Final = 96


@dataclass(frozen=True, slots=True)
class Geometry:
    query_counts: tuple[int, ...] = (2, 1)
    key_lengths: tuple[int, ...] = (37, 21)
    host_block: int = 24
    capacity: int = 8

    @property
    def rows(self) -> int:
        return sum(self.query_counts)

    @property
    def query_ends(self) -> np.ndarray:
        return np.cumsum(self.query_counts, dtype=np.int32)

    @property
    def requests(self) -> np.ndarray:
        return np.repeat(np.arange(len(self.query_counts), dtype=np.int32), self.query_counts)

    @property
    def visible(self) -> np.ndarray:
        return np.asarray([
            length - count + offset + 1
            for count, length in zip(self.query_counts, self.key_lengths, strict=True)
            for offset in range(count)
        ], dtype=np.int32)

    def table(self, block_size: int) -> np.ndarray:
        columns = max(1, (max(self.key_lengths) + block_size - 1) // block_size)
        return np.arange(len(self.query_counts) * columns, dtype=np.int32).reshape(-1, columns)[:, ::-1].copy()


@dataclass(frozen=True, slots=True)
class ScoreInputs:
    query: np.ndarray
    key: np.ndarray
    weights: np.ndarray
    query_scale: np.ndarray
    key_scale: np.ndarray
    block_table: np.ndarray
    geometry: Geometry
    dtype: DType


def causal_scores(inputs: ScoreInputs) -> tuple[np.ndarray, ...]:
    """Compute causal scores numerically without calling a production operator."""
    qfactor = inputs.weights * inputs.query_scale
    if inputs.dtype == "int8":
        qfactor = qfactor.astype(np.float16)
    qfactor = qfactor.astype(np.float32)
    scores: list[np.ndarray] = []
    for row, visible in enumerate(inputs.geometry.visible):
        request = inputs.geometry.requests[row]
        row_scores = np.empty(int(visible), dtype=np.float32)
        for token in range(int(visible)):
            page = inputs.block_table[request, token // INDEX_BLOCK]
            offset = token % INDEX_BLOCK
            if inputs.dtype == "int8":
                dot = np.int32(inputs.query[row]) @ np.int32(inputs.key[page, offset, 0])
                dot = np.asarray(dot / 1024.0, dtype=np.float16).astype(np.float32)
            else:
                dot = inputs.query[row].astype(np.float32) @ inputs.key[page, offset, 0].astype(np.float32)
            products = np.maximum(dot, 0.0) * qfactor[row]
            total = np.float32(0)
            for product in products:
                total = np.float32(total + np.float32(product))
            row_scores[token] = np.float32(total * inputs.key_scale[page, offset, 0])
        scores.append(row_scores)
    return tuple(scores)


def topk(scores: tuple[np.ndarray, ...], count: int) -> np.ndarray:
    result = np.full((len(scores), 1, count), -1, dtype=np.int32)
    for row, values in enumerate(scores):
        order = np.lexsort((np.arange(values.size), -values))[:count]
        result[row, 0, :order.size] = order
    return result


def plan_workspace_elements(topk_count: int, capacity: int) -> int:
    hash_capacity = max(32, 1 << (2 * topk_count - 1).bit_length())
    return 3 * hash_capacity + 2 * capacity + topk_count
