from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


@pytest.mark.parametrize("mode", ["keys", "selection", "preprocess"])
def test_prefetch_tiling_template(tmp_path: Path, request: FixtureRequest, mode: str) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A host C++ compiler is required for the tiling template check")
    configured = request.config.getoption("--cann-template-include")
    candidates = [Path(configured)] if configured else []
    if not configured:
        for variable in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
            if os.environ.get(variable):
                candidates.append(Path(os.environ[variable]) / "include")
        candidates.append(Path("/usr/local/Ascend/ascend-toolkit/latest/include"))
    relative_header = Path("ascendc/host_api/tiling/template_argument.h")
    include = next((root for root in candidates if (root / relative_header).is_file()), None)
    if include is None:
        if configured:
            pytest.fail(f"Missing {relative_header} under {configured}")
        pytest.skip("CANN template header not found; use --cann-template-include to select its include root")
    operator = Path(__file__).resolve().parents[2]
    executable = tmp_path / "tiling_key_probe"
    command = [compiler, "-std=c++17", "-I", str(include), "-I", str(operator / "op_kernel")]
    if mode == "preprocess":
        command.extend(("-E", "-P", "-x", "c++", "-DASCENDC_TPL_PRE", "-DASCENDC_TPL_KERNEL"))
        command.append(str(operator / "op_kernel/lightning_indexer_quant_template_tiling_key.h"))
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        selections = [line for line in result.stdout.splitlines() if "@@ASCENDC_TPL_LISTS" in line]
        assert len(selections) == 1, "The kernel compiler reads only the first template selection list"
        for dtype in (2, 36):
            for parameter in ("DT_Q", "DT_K"):
                assert f"@@ASCENDC_TPL_DTYPE_SEL_{parameter}@@ = {{{dtype}}}" in selections[0]
        return
    if mode == "selection":
        command.append("-DASCENDC_DEBUG")
    command.extend((str(operator / "tests/host/tiling_key_probe.cpp"), "-o", str(executable)))
    compiled = subprocess.run(command, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    if mode == "keys":
        assert result.stdout.split() == ["50528770", "50537508"]
    else:
        assert result.stdout.strip() == "7 supported combinations"
