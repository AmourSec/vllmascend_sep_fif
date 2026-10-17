from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lightning_indexer_quant_prefetch_golden import PrefetchExpected

if TYPE_CHECKING:
    import torch
    from prefetch_test_cases import Case

    Tensor = torch.Tensor
else:
    Tensor = object


def _assert_id_sets_equal(left: np.ndarray, right: np.ndarray) -> None:
    for lhs, rhs in zip(left.reshape(left.shape[0], -1), right.reshape(right.shape[0], -1), strict=True):
        assert sorted(int(value) for value in lhs if value >= 0) == sorted(int(value) for value in rhs if value >= 0)


def _assert_selected_and_rows(actual: tuple[Tensor, ...], case: Case, expected: PrefetchExpected) -> None:
    actual_ids = actual[0].cpu().numpy().reshape(case.query.size(0), -1)
    actual_rows = actual[1].cpu().numpy()
    actual_candidates = actual[2].cpu().numpy()
    np.testing.assert_array_equal(actual_candidates, expected.prefetch_indices)
    np.testing.assert_allclose(actual[4].cpu().numpy(), expected.next_threshold, rtol=2e-3, atol=2e-3, equal_nan=True)
    for row, ids in enumerate(actual_ids):
        valid_ids = [int(token) for token in ids if token >= 0]
        scores = expected.scores[row]
        count = min(expected.indices.shape[-1], scores.size)
        if count == 0:
            allowed = set()
            required = set()
        else:
            cutoff = np.partition(scores, scores.size - count)[scores.size - count]
            allowed = set(np.flatnonzero(scores >= cutoff).tolist())
            required = set(np.flatnonzero(scores > cutoff).tolist())
        assert len(valid_ids) == len(set(valid_ids))
        assert len(valid_ids) == count
        assert set(valid_ids).issubset(allowed)
        assert required.issubset(set(valid_ids))
        assert all(0 <= token < scores.size for token in valid_ids)
        assert np.all(actual_ids[row, count:] == -1)
        assert np.all(actual_rows[row, count:] == 0)
        for slot, token in enumerate(valid_ids):
            page = int(case.golden.block_table[row, token // case.golden.key.shape[1]])
            offset = token % case.golden.key.shape[1]
            payload = case.golden.kv_payload
            assert payload is not None
            expected_bytes = payload[page, offset, 0]
            np.testing.assert_array_equal(actual_rows[row, slot], expected_bytes)
    _assert_stats(actual, expected)


def _assert_stats(actual: tuple[Tensor, ...], expected: PrefetchExpected) -> None:
    stats = actual[3].cpu().numpy()
    candidates = actual[2].cpu().numpy()
    ids = actual[0].cpu().numpy().reshape(actual[0].shape[0], -1)
    for row, row_ids in enumerate(ids):
        selected = {int(value) for value in row_ids if value >= 0}
        prefetched = {int(value) for value in candidates[row] if value >= 0}
        assert tuple(stats[row]) == (len(prefetched), len(selected - prefetched))
