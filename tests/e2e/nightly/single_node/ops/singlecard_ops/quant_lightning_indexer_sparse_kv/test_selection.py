from dataclasses import replace

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.utils import bootstrap_custom_op_env  # noqa: E402

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

from .assertions import assert_state_matches, oracle_after  # noqa: E402
from .helpers import make_case, make_state, require_ops, run_fused, run_indexer, run_plan  # noqa: E402
from .selection_cases import (  # noqa: E402
    SelectionCase,
    has_strict_margin,
    make_integer_tie_case,
    make_multitrunk_strict_margin_case,
    make_strict_margin_case,
    validate_exact_expected,
    validate_topk_selection,
)


@pytest.fixture(scope="module", autouse=True)
def _ops_available() -> None:
    require_ops()


def _run_selection_case(selection: SelectionCase) -> tuple[torch.Tensor, torch.Tensor]:
    case = make_case([selection.seq_len], [1])
    case = replace(
        case,
        query=selection.query.npu(),
        key=selection.key.npu(),
        weights=selection.weights.npu(),
        query_scale=torch.ones_like(case.query_scale),
        key_scale=torch.ones_like(case.key_scale),
    )
    before = make_state(case)
    fused_state = before.clone()
    baseline_state = before.clone()

    fused_indices = run_fused(case, fused_state)
    baseline_indices = run_indexer(case)
    run_plan(case, baseline_state, baseline_indices)
    torch_npu.npu.synchronize()

    fused_cpu = fused_indices.cpu()
    baseline_cpu = baseline_indices.cpu()
    validate_topk_selection(selection.scores, fused_cpu)
    validate_topk_selection(selection.scores, baseline_cpu)
    assert_state_matches(fused_state, oracle_after(case, before, fused_indices))
    assert_state_matches(baseline_state, oracle_after(case, before, baseline_indices))
    return fused_cpu, baseline_cpu


@pytest.mark.npu_smoke
@pytest.mark.npu_inline
def test_integer_encoding_ties_are_validated_without_forcing_equal_indices() -> None:
    _run_selection_case(make_integer_tie_case())


@pytest.mark.npu_smoke
@pytest.mark.npu_inline
def test_strict_margin_single_trunk_selects_exact_ids() -> None:
    selection = make_strict_margin_case()
    assert has_strict_margin(selection)
    fused, baseline = _run_selection_case(selection)
    validate_exact_expected(selection, fused)
    validate_exact_expected(selection, baseline)


@pytest.mark.npu_smoke
@pytest.mark.npu_inline
def test_strict_margin_multitrunk_selects_distributed_exact_ids() -> None:
    selection = make_multitrunk_strict_margin_case()
    assert has_strict_margin(selection)
    fused, baseline = _run_selection_case(selection)
    validate_exact_expected(selection, fused)
    validate_exact_expected(selection, baseline)
