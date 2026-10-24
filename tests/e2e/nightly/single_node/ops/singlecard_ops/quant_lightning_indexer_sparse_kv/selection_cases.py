from dataclasses import dataclass
from typing import Final

import torch

TOPK: Final = 2048
BLOCK_SIZE: Final = 128
QUERY_HEADS: Final = 32
HEAD_DIM: Final = 128


@dataclass(frozen=True)
class SelectionCase:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor
    expected_mask: torch.Tensor

    @property
    def seq_len(self) -> int:
        return self.key.shape[0] * self.key.shape[1]


def fp32_to_bf16_ties_away(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.float32 or bool((values < 0).any()) or not bool(torch.isfinite(values).all()):
        raise ValueError("rounding reference requires finite nonnegative float32 values")
    bits = values.contiguous().view(torch.int32).to(torch.int64) & 0xFFFF_FFFF
    # CANN CAST_ROUND rounds positive halfway cases upward, unlike PyTorch's ties-to-even BF16 conversion.
    rounded_bits = (((bits + 0x8000) >> 16) << 16).to(torch.int32)
    return rounded_bits.view(torch.float32)


def validate_topk_selection(scores: torch.Tensor, selected_ids: torch.Tensor, topk: int = TOPK) -> None:
    ids = selected_ids.to(dtype=torch.int64, device="cpu").flatten()
    score_values = scores.to(dtype=torch.float32, device="cpu").flatten()
    assert ids.numel() == topk, f"expected {topk} selected IDs, got {ids.numel()}"
    assert torch.unique(ids).numel() == topk, "selected IDs must be unique"
    assert bool(((ids >= 0) & (ids < score_values.numel())).all()), "selected ID is out of range"

    selected_mask = torch.zeros(score_values.numel(), dtype=torch.bool)
    selected_mask[ids] = True
    chosen = score_values[selected_mask]
    unselected = score_values[~selected_mask]
    chosen_min = chosen.min()
    unselected_max = unselected.max()
    assert bool(chosen_min >= unselected_max), (
        f"chosen minimum {chosen_min.item()} is below unselected maximum {unselected_max.item()}"
    )
    assert bool(selected_mask[score_values > chosen_min].all()), "a score strictly above the cutoff was omitted"


def has_strict_margin(case: SelectionCase) -> bool:
    return bool(case.scores[case.expected_mask].min() > case.scores[~case.expected_mask].max())


def validate_exact_expected(case: SelectionCase, indices: torch.Tensor) -> None:
    actual = torch.sort(indices.to(device="cpu").flatten()).values
    expected = torch.nonzero(case.expected_mask, as_tuple=False).flatten().to(dtype=actual.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def make_integer_tie_case() -> SelectionCase:
    token_ids = torch.arange(4096, dtype=torch.int64)
    lower_values = token_ids & 2047
    selected = token_ids >= 2048
    return _make_case(lower_values, selected, selection_weight=2048)


def make_strict_margin_case() -> SelectionCase:
    token_ids = torch.arange(4096, dtype=torch.int64)
    lower_values = token_ids & 2047
    selected = token_ids >= 2048
    return _make_case(lower_values, selected, selection_weight=4096)


def make_multitrunk_strict_margin_case() -> SelectionCase:
    token_ids = torch.arange(32768, dtype=torch.int64)
    lower_values = token_ids % 2048
    selected = token_ids % 16 == 0
    return _make_case(lower_values, selected, selection_weight=4096)


def _make_case(lower_values: torch.Tensor, selected: torch.Tensor, *, selection_weight: int) -> SelectionCase:
    seq_len = lower_values.numel()
    key_bits = torch.zeros((seq_len, 1, HEAD_DIM), dtype=torch.int8)
    for bit in range(11):
        key_bits[:, 0, bit] = ((lower_values >> bit) & 1).to(torch.int8)
    key_bits[:, 0, 11] = selected.to(torch.int8)

    query_bits = torch.zeros((1, QUERY_HEADS, HEAD_DIM), dtype=torch.int8)
    weights = torch.zeros((1, QUERY_HEADS), dtype=torch.bfloat16)
    for bit in range(11):
        query_bits[0, bit, bit] = 1
        weights[0, bit] = 1 << bit
    query_bits[0, 11, 11] = 1
    weights[0, 11] = selection_weight

    raw_scores = lower_values.to(torch.float32) + selected.to(torch.float32) * selection_weight
    return SelectionCase(
        query=query_bits.to(torch.float8_e4m3fn),
        key=key_bits.reshape(seq_len // BLOCK_SIZE, BLOCK_SIZE, 1, HEAD_DIM).to(torch.float8_e4m3fn),
        weights=weights,
        scores=fp32_to_bf16_ties_away(raw_scores),
        expected_mask=selected,
    )
