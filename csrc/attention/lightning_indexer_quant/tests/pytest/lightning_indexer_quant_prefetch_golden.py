from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PrefetchInputs(NamedTuple):
    query: np.ndarray
    key: np.ndarray
    weights: np.ndarray
    query_dequant_scale: np.ndarray
    key_dequant_scale: np.ndarray
    actual_seq_lengths_key: np.ndarray
    block_table: np.ndarray
    prefetch_threshold: np.ndarray
    kv_row_bytes: int
    prefetch_budget: int
    sparse_count: int = 2048
    ema_decay: float = 0.5
    kv_payload: np.ndarray | None = None


class PrefetchExpected(NamedTuple):
    indices: np.ndarray
    gathered_kv: np.ndarray
    prefetch_indices: np.ndarray
    stats: np.ndarray
    next_threshold: np.ndarray
    scores: tuple[np.ndarray, ...]


def _stable_topk(scores: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int32)
    order = np.lexsort((np.arange(scores.size), -scores))
    return order[:count].astype(np.int32, copy=False)


def _row_bytes(key: np.ndarray, block: int, offset: int, row_bytes: int) -> np.ndarray:
    row = np.asarray(key[block, offset, 0]).view(np.uint8).reshape(-1)
    if row.size < row_bytes:
        return np.pad(row, (0, row_bytes - row.size), constant_values=0)
    return row[:row_bytes].copy()


def compute_expected(inputs: PrefetchInputs) -> PrefetchExpected:
    query = np.asarray(inputs.query, dtype=np.int8)
    key = np.asarray(inputs.key, dtype=np.int8)
    batch, heads, _ = query.shape
    block_size = key.shape[1]
    valid_k = np.asarray(inputs.actual_seq_lengths_key, dtype=np.int32)
    topk = inputs.sparse_count
    candidate_width = max(1, inputs.prefetch_budget)
    indices = np.full((batch, 1, topk), -1, dtype=np.int32)
    gathered = np.zeros((batch, topk, inputs.kv_row_bytes), dtype=np.uint8)
    candidates = np.full((batch, candidate_width), -1, dtype=np.int32)
    stats = np.zeros((batch, 2), dtype=np.int32)
    next_threshold = np.empty(batch, dtype=np.float32)
    score_rows: list[np.ndarray] = []
    qfactor = (
        (np.asarray(inputs.weights, dtype=np.float16) * np.asarray(inputs.query_dequant_scale, dtype=np.float16))
        .astype(np.float16)
        .astype(np.float32)
    )

    for row in range(batch):
        length = max(0, int(valid_k[row]))
        scores = np.full(length, -np.inf, dtype=np.float32)
        for token in range(length):
            page = int(inputs.block_table[row, token // block_size])
            offset = token % block_size
            qk = np.int32(query[row]) @ np.int32(key[page, offset, 0])
            qk_fp16 = np.asarray(qk / 1024.0, dtype=np.float16).astype(np.float32)
            relu_qk = np.maximum(qk_fp16, 0.0)
            scale = float(np.asarray(inputs.key_dequant_scale[page, offset, 0], dtype=np.float32))
            score = np.float32(0.0)
            for head in range(heads):
                score = np.float32(score + np.float32(relu_qk[head] * qfactor[row, head]))
            scores[token] = np.float32(score * scale)
        score_rows.append(scores)
        selected = _stable_topk(scores, min(topk, length))
        indices[row, 0, : selected.size] = selected
        threshold = float(np.asarray(inputs.prefetch_threshold[row], dtype=np.float32))
        speculative = [token for token, score in enumerate(scores) if score > threshold]
        speculative = speculative[: inputs.prefetch_budget]
        if speculative:
            candidates[row, : len(speculative)] = np.asarray(speculative, dtype=np.int32)
        selected_set = set(int(token) for token in selected)
        candidate_set = set(speculative)
        stats[row] = (len(speculative), sum(token not in candidate_set for token in selected_set))
        for out, token in enumerate(selected):
            page = int(inputs.block_table[row, token // block_size])
            payload = key if inputs.kv_payload is None else inputs.kv_payload
            gathered[row, out] = _row_bytes(payload, page, token % block_size, inputs.kv_row_bytes)
        if selected.size >= topk and topk > 0:
            actual = float(scores[int(selected[topk - 1])])
            old = threshold
            if np.isfinite(old):
                next_threshold[row] = np.float32(inputs.ema_decay * old + (1.0 - inputs.ema_decay) * actual)
            else:
                next_threshold[row] = np.float32(actual)
        else:
            next_threshold[row] = np.float32(threshold)
    return PrefetchExpected(indices, gathered, candidates, stats, next_threshold, tuple(score_rows))


def assert_contract(expected: PrefetchExpected, *, topk: int, row_bytes: int, budget: int) -> None:
    assert expected.indices.ndim == 3 and expected.indices.shape[1:] == (1, topk)
    assert expected.gathered_kv.shape == (expected.indices.shape[0], topk, row_bytes)
    assert expected.prefetch_indices.shape[1] == max(1, budget)
    for row in range(expected.indices.shape[0]):
        valid = expected.indices[row, 0]
        ids = valid[valid >= 0]
        assert np.unique(ids).size == ids.size
        candidates = expected.prefetch_indices[row]
        assert np.all((candidates == -1) | (candidates >= 0))
        if budget == 0:
            assert np.all(candidates == -1)
