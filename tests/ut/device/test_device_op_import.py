from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).parents[3] / "vllm_ascend/device/device_op.py"
TARGET = "_device_op_import_cycle_test"
OPS = {
    "vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt": "chunk_scaled_dot_kkt_fwd_kernel",
    "vllm_ascend.ops.triton.fla.solve_tril": "solve_tril_16x16_kernel",
    "vllm_ascend.ops.triton.fused_gdn_gating": "fused_gdn_gating_patch",
    "vllm_ascend.ops.triton.rms_norm": "triton_q_rms",
}


class _Kernel:
    def __init__(self, calls: list, name: str):
        self.calls = calls
        self.name = name

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((self.name, grid, kwargs))

        return launch


class _OpsLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, calls: list):
        self.fullname = fullname
        self.calls = calls

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        namespace = {}
        exec(f"from {TARGET} import DeviceOperator", namespace)
        name = OPS[self.fullname]
        self.calls.append(("import", name))
        if name.endswith("kernel"):
            setattr(module, name, _Kernel(self.calls, name))
        else:
            setattr(module, name, self._callable(name))

    def _callable(self, name):
        def call(*args):
            result = object()
            self.calls.append((name, args, result))
            return result

        return call


class _OpsFinder(importlib.abc.MetaPathFinder):
    def __init__(self, calls: list):
        self.calls = calls

    def find_spec(self, fullname, path=None, target=None):
        if fullname in OPS:
            return importlib.util.spec_from_loader(fullname, _OpsLoader(fullname, self.calls))
        return None


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_device_op(monkeypatch: pytest.MonkeyPatch, has_triton: bool):
    calls: list = []
    for name in OPS:
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)
    for name in ("vllm", "vllm_ascend", "vllm_ascend.device", "vllm_ascend.quantization"):
        monkeypatch.setitem(sys.modules, name, _package(name))
    for name in ("vllm_ascend.ops", "vllm_ascend.ops.triton", "vllm_ascend.ops.triton.fla"):
        monkeypatch.setitem(sys.modules, name, _package(name))

    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.HAS_TRITON = has_triton
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", triton_utils)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))

    utils = types.ModuleType("vllm_ascend.device.utils")
    utils.FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE = -1
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    hardware = types.ModuleType("vllm_ascend.device.hardware_profile")
    hardware.DeviceAdaptorFamily = SimpleNamespace(STANDARD=0, FP8_OPTIMIZED=1, COMPATIBILITY=2)
    hardware.get_current_hardware_profile = lambda: SimpleNamespace(device_adaptor_family=0)
    monkeypatch.setitem(sys.modules, hardware.__name__, hardware)

    quant_type = types.ModuleType("vllm_ascend.quantization.quant_type")
    quant_type.QuantType = type("QuantType", (), {})
    monkeypatch.setitem(sys.modules, quant_type.__name__, quant_type)
    quant_utils = types.ModuleType("vllm_ascend.quantization.utils")
    quant_utils.QUANT_DTYPES = ()
    quant_utils.get_dynamic_mx_quant_scale_alg = lambda: None
    monkeypatch.setitem(sys.modules, quant_utils.__name__, quant_utils)

    finder = _OpsFinder(calls)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    module = types.ModuleType(TARGET)
    module.__file__ = str(SOURCE)
    monkeypatch.setitem(sys.modules, TARGET, module)
    # The repository supports evaluated PEP 604 annotations; Mac Python 3.9 needs them deferred in this source harness.
    source = "from __future__ import annotations\n" + SOURCE.read_text()
    exec(compile(source, str(SOURCE), "exec"), module.__dict__)
    return module, calls


@pytest.mark.parametrize("has_triton", [False, True])
def test_module_initialization_does_not_import_ops(monkeypatch, has_triton):
    module, calls = _load_device_op(monkeypatch, has_triton)

    assert module.DeviceOperator is module.BaseDeviceAdaptor
    assert calls == []


def test_loader_restores_present_and_absent_ops_modules():
    present_name, absent_name = tuple(OPS)[:2]
    missing = object()
    saved_present = sys.modules.get(present_name, missing)
    saved_absent = sys.modules.get(absent_name, missing)
    original = types.ModuleType(present_name)
    sys.modules[present_name] = original
    sys.modules.pop(absent_name, None)

    try:
        with pytest.MonkeyPatch.context() as patch:
            _load_device_op(patch, True)
            importlib.import_module(present_name)
            importlib.import_module(absent_name)
        assert sys.modules[present_name] is original
        assert absent_name not in sys.modules
    finally:
        for name, saved in ((present_name, saved_present), (absent_name, saved_absent)):
            if saved is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


@pytest.mark.parametrize(
    ("adaptor_name", "solve_stride", "a5_flag"),
    [("BaseDeviceAdaptor", 2, False), ("A5DeviceAdaptor", 1, True)],
)
def test_kernels_are_imported_at_dispatch(monkeypatch, adaptor_name, solve_stride, a5_flag):
    module, calls = _load_device_op(monkeypatch, True)
    adaptor = getattr(module, adaptor_name)
    values = [object() for _ in range(10)]

    assert adaptor.chunk_scaled_dot_kkt_fwd(4, 5, 6, *values[:6], 7, 8, 9, 10, 11, 12, 13) is values[3]
    assert adaptor.solve_tril_16x16(*values[:4], 7, 8, 12, 64, 9, 10) is values[1]
    gating = adaptor.fused_gdn_gating(*values[:4])

    chunk = next(call for call in calls if call[0] == "chunk_scaled_dot_kkt_fwd_kernel" and len(call) == 3)
    assert chunk[1] == (4,)
    assert chunk[2] == {
        "k": values[0],
        "beta": values[1],
        "g_cumsum": values[2],
        "A": values[3],
        "cu_seqlens": values[4],
        "chunk_indices": values[5],
        "T": 7,
        "B": 8,
        "H": 9,
        "Hg": 10,
        "K": 11,
        "BT": 12,
        "BK": 13,
        "bh_step": 5,
        "task_num": 6,
        "num_core": 4,
        "num_warps": 8,
        "num_stages": 3,
        "multibuffer": True,
        **({"disable_tightly_coupled_buffer_reuse": True} if a5_flag else {}),
    }
    assert chunk[2].get("disable_tightly_coupled_buffer_reuse", False) is a5_flag
    solve = next(call for call in calls if call[0] == "solve_tril_16x16_kernel" and len(call) == 3)
    assert solve[1] == (9, 80)
    assert solve[2] == {
        "A": values[0],
        "Ad": values[1],
        "cu_seqlens": values[2],
        "chunk_indices": values[3],
        "T": 7,
        "H": 8,
        "BT": 12,
        "LARGE_BLOCK_T": 64,
        "EXTRACT_SLICE_STRIDE_1": solve_stride,
        "num_warps": 1,
        "num_stages": 4,
    }
    fused = next(call for call in calls if call[0] == "fused_gdn_gating_patch" and len(call) == 3)
    assert fused[1] == tuple(values[:4]) and gating is fused[2]


@pytest.mark.parametrize("adaptor_name", ["BaseDeviceAdaptor", "A5DeviceAdaptor"])
def test_rms_custom_triton_and_fallback_paths(monkeypatch, adaptor_name):
    module, calls = _load_device_op(monkeypatch, True)
    adaptor = getattr(module, adaptor_name)
    q = torch.tensor([[3.0, 4.0]])
    custom_result = object()

    assert adaptor.apply_dsa_q_rms(q, 0.25, lambda value: custom_result) is custom_result
    assert calls == []
    triton_result = adaptor.apply_dsa_q_rms(q, 0.25)
    rms = next(call for call in calls if call[0] == "triton_q_rms")
    assert rms[1] == (q, 0.25) and triton_result is rms[2]

    fallback_module, fallback_calls = _load_device_op(monkeypatch, False)
    actual = getattr(fallback_module, adaptor_name).apply_dsa_q_rms(q, 0.25)
    expected = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 0.25)
    torch.testing.assert_close(actual, expected)
    assert fallback_calls == []
