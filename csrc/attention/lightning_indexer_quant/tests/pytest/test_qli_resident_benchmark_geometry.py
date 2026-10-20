from __future__ import annotations

import numpy as np

from resident_benchmark import CAPACITY, HOST_BLOCK, INDEX_BLOCK, TOPK, BenchmarkSpec, specs


def test_mixed_mtp_geometry_has_distinct_causal_and_stable_lengths() -> None:
    spec = BenchmarkSpec((1, 2, 4, 1), 131072, "int8")
    np.testing.assert_array_equal(spec.query_ends, [1, 3, 7, 8])
    np.testing.assert_array_equal(spec.requests, [0, 1, 1, 2, 2, 2, 2, 3])
    np.testing.assert_array_equal(
        spec.visible,
        [131072, 131071, 131072, 131069, 131070, 131071, 131072, 131072],
    )
    np.testing.assert_array_equal(
        spec.stable,
        [131071, 131070, 131070, 131068, 131068, 131068, 131068, 131071],
    )
    assert np.all(spec.visible - spec.stable == np.asarray([1, 1, 2, 1, 2, 3, 4, 1]))


def test_geometry_tables_cover_index_and_ddr_pages_without_aliasing() -> None:
    spec = BenchmarkSpec((1, 2, 4, 1), 131072, "fp8")
    index = spec.table(INDEX_BLOCK)
    host = spec.table(HOST_BLOCK)
    assert index.shape == (4, 1024)
    assert host.shape == (4, 1024)
    assert index.dtype == host.dtype == np.int32
    for request in range(spec.batch):
        np.testing.assert_array_equal(index[request], np.arange(request * 1024, (request + 1) * 1024))
        np.testing.assert_array_equal(host[request], np.arange(request * 1024, (request + 1) * 1024))
    assert not np.shares_memory(index, host)


def test_full_profile_covers_requested_batch_mtp_and_long_context_matrix() -> None:
    matrix = specs("full", "int8")
    identities = {(item.batch, item.query_counts, item.context) for item in matrix}
    expected = {
        (1, (1,), 8192),
        (4, (1, 1, 1, 1), 32768),
        (4, (2, 2, 2, 2), 32768),
        (4, (4, 4, 4, 4), 32768),
        (8, (4, 4, 4, 4, 4, 4, 4, 4), 131072),
        (4, (1, 2, 4, 1), 131072),
    }
    assert identities == expected
    assert {item.batch for item in matrix} == {1, 4, 8}
    assert {max(item.query_counts) for item in matrix} == {1, 2, 4}
    assert any(len(set(item.query_counts)) > 1 for item in matrix)
    assert any(item.context == 131072 for item in matrix)
    assert TOPK == 2048 and CAPACITY == 4096


def test_spec_rows_and_lengths_are_consistent_for_every_profile_case() -> None:
    for profile in ("smoke", "full"):
        for dtype in ("int8", "fp8"):
            for spec in specs(profile, dtype):
                assert spec.rows == int(spec.query_ends[-1])
                assert spec.requests.shape == (spec.rows,)
                assert spec.visible.shape == spec.stable.shape == (spec.rows,)
                assert np.all(spec.visible > spec.stable)
                assert np.all(spec.stable >= 0)
                assert spec.table(INDEX_BLOCK).shape[1] == (spec.context + INDEX_BLOCK - 1) // INDEX_BLOCK
                assert spec.table(HOST_BLOCK).shape[1] == (spec.context + HOST_BLOCK - 1) // HOST_BLOCK
