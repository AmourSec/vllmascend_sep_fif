import pytest
import torch

from tests.e2e.nightly.single_node.ops.singlecard_ops.quant_lightning_indexer_sparse_kv.selection_cases import (
    TOPK,
    fp32_to_bf16_ties_away,
    has_strict_margin,
    make_integer_tie_case,
    make_multitrunk_strict_margin_case,
    make_strict_margin_case,
    validate_exact_expected,
    validate_topk_selection,
)


def _selected_ids(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask, as_tuple=False).flatten()


def test_ties_away_rounding_and_strict_margin_boundary() -> None:
    rounded = fp32_to_bf16_ties_away(torch.tensor([2043.0, 2044.0, 2055.0, 2056.0]))
    torch.testing.assert_close(rounded, torch.tensor([2040.0, 2048.0, 2048.0, 2064.0]), rtol=0, atol=0)
    assert torch.tensor(2056.0).to(torch.bfloat16).to(torch.float32) == 2048

    tie_case = make_integer_tie_case()
    strict_case = make_strict_margin_case()
    assert not has_strict_margin(tie_case)
    assert has_strict_margin(strict_case)
    assert tie_case.scores[2040] < tie_case.scores[2044]
    assert bool((tie_case.scores[2044:2056] == 2048).all())
    assert strict_case.scores[:2048].max() == 2048
    assert strict_case.scores[2048:].min() == 4096


def test_tie_aware_validation_accepts_legal_reported_swap() -> None:
    case = make_integer_tie_case()
    expected = _selected_ids(case.expected_mask)
    reported = torch.cat((torch.arange(2044, 2048), expected[:4], expected[8:]))

    validate_topk_selection(case.scores, expected)
    validate_topk_selection(case.scores, reported)
    assert set(reported.tolist()) - set(expected.tolist()) == set(range(2044, 2048))
    assert set(expected.tolist()) - set(reported.tolist()) == set(range(2052, 2056))


def test_tie_aware_validation_rejects_worse_duplicate_and_invalid_selections() -> None:
    case = make_integer_tie_case()
    expected = _selected_ids(case.expected_mask)
    worse = expected.clone()
    worse[0] = 0
    duplicate = expected.clone()
    duplicate[0] = duplicate[1]
    invalid = expected.clone()
    invalid[0] = case.scores.numel()

    with pytest.raises(AssertionError, match="expected 2048 selected IDs"):
        validate_topk_selection(case.scores, expected[:-1])
    with pytest.raises(AssertionError, match="below unselected maximum"):
        validate_topk_selection(case.scores, worse)
    with pytest.raises(AssertionError, match="unique"):
        validate_topk_selection(case.scores, duplicate)
    with pytest.raises(AssertionError, match="out of range"):
        validate_topk_selection(case.scores, invalid)


def test_strict_single_and_multitrunk_cases_select_exact_expected_ids() -> None:
    for case in (make_strict_margin_case(), make_multitrunk_strict_margin_case()):
        expected = _selected_ids(case.expected_mask)
        assert expected.numel() == TOPK
        validate_topk_selection(case.scores, expected)
        validate_exact_expected(case, expected.to(torch.int32))


def test_declared_scores_match_generated_fp8_and_bf16_encoding() -> None:
    for case in (make_integer_tie_case(), make_strict_margin_case(), make_multitrunk_strict_margin_case()):
        query = case.query[0].to(torch.float32)
        keys = case.key.reshape(case.seq_len, -1).to(torch.float32)
        per_head = keys @ query.transpose(0, 1)
        encoded_scores = fp32_to_bf16_ties_away((per_head * case.weights[0].to(torch.float32)).sum(dim=1))
        torch.testing.assert_close(encoded_scores, case.scores, rtol=0, atol=0)

    multitrunk = make_multitrunk_strict_margin_case()
    expected = _selected_ids(multitrunk.expected_mask)
    assert multitrunk.seq_len == 32768
    assert torch.equal(expected, torch.arange(0, 32768, 16))
