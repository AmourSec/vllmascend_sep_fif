from __future__ import annotations

import numpy as np
import pytest

from mtp_resident_oracle import Geometry, INDEX_BLOCK, ScoreInputs, causal_scores, plan_workspace_elements, topk


@pytest.mark.parametrize("geometry", [
    Geometry(), Geometry((1, 3, 2), (5, 35, 17)), Geometry((0, 2, 1), (0, 37, 21)),
    Geometry((2, 1), (19, 7)),
    Geometry((2, 1), (137, 129), capacity=129), Geometry(host_block=32),
])
def test_mtp_fixture_geometry_is_consistent_without_torch(geometry: Geometry) -> None:
    # Given each geometry used by the device tests.
    index = geometry.table(INDEX_BLOCK)
    host = geometry.table(geometry.host_block)
    # When source and persistent allocations are derived from that geometry.
    requests = geometry.requests
    # Then all visible tokens fit both independent block layouts and row state is T-sized.
    assert geometry.query_ends[-1] == geometry.rows
    assert len(requests) == len(geometry.visible) == geometry.rows
    assert np.all(geometry.visible > 0)
    for row, visible in enumerate(geometry.visible):
        request = requests[row]
        assert visible <= geometry.key_lengths[request]
        assert index.shape[1] * INDEX_BLOCK >= visible
        assert host.shape[1] * geometry.host_block >= visible
        assert np.max(index[request]) < index.size
        assert np.max(host[request]) < host.size
    assert plan_workspace_elements(7, geometry.capacity) == 96 + 2 * geometry.capacity + 7


def test_numeric_oracle_excludes_future_mtp_tokens() -> None:
    # Given a monotonically increasing score per token and two causal query rows.
    geometry = Geometry((2,), (5,))
    query = np.ones((2, 1, 128), dtype=np.int8)
    key = np.zeros((1, 16, 1, 128), dtype=np.int8)
    key[0, :5, 0] = np.arange(1, 6)[:, None]
    inputs = ScoreInputs(query, key, np.ones((2, 1)), np.ones((2, 1)),
                         np.ones((1, 16, 1)), np.zeros((1, 1), dtype=np.int32), geometry, "int8")
    # When the independent oracle computes per-query visibility and TopK.
    scores = causal_scores(inputs)
    # Then the first row cannot see token four, and the final row can.
    np.testing.assert_array_equal(scores[0], [0.125, 0.25, 0.375, 0.5])
    np.testing.assert_array_equal(scores[1], [0.125, 0.25, 0.375, 0.5, 0.625])
    np.testing.assert_array_equal(topk(scores, 3), [[[3, 2, 1]], [[4, 3, 2]]])
