# SPDX-License-Identifier: Apache-2.0
"""CPU checks of private-op contracts and real build-script generation, not CANN compilation."""

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
UTIL = ROOT / "csrc/cmake/scripts/util"
OPERATORS = (
    ("moe", "vllm_causal_conv1d", "VllmCausalConv1d", 8),
    ("attention", "vllm_kda_gate_cumsum", "VllmKdaGateCumsum", 4),
)
DTYPES = {"FLOAT": "float32", "FLOAT16": "float16", "BF16": "bfloat16", "INT32": "int32",
          "INT64": "int64", "BOOL": "bool"}


def load_build_script(name):
    spec = importlib.util.spec_from_file_location(name, UTIL / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    # These generation paths use ordinary regex syntax; no TBE/CANN modules are imported.
    with patch.dict(sys.modules, {"regex": re}), patch.object(sys, "path", [str(UTIL), *sys.path]):
        spec.loader.exec_module(module)
    return module


def op_info_fixture(directory, op_type):
    """Read names, types and attribute order from the actual OpDef; choose one dtype combination."""
    source = (directory / "op_host" / (directory.name + "_def.cpp")).read_text()
    variables = dict(re.findall(r"initializer_list<ge::DataType>\s+(\w+)\s*=\s*\{(.*?)\};", source, re.S))
    lines = [f"[{op_type}]", f"opFile.value={directory.name}", f"opInterface.value={directory.name}",
             "dynamicShapeSupport.flag=true", "dynamicFormat.flag=true"]
    for kind in ("Input", "Output"):
        sections = re.findall(r'this->' + kind + r'\("([^\"]+)"\)(.*?);', source, re.S)
        for index, (name, body) in enumerate(sections):
            param_type = re.search(r"ParamType\((REQUIRED|OPTIONAL)\)", body)[1].lower()
            dtype_expr = re.search(r"DataType(?:List)?\((.*?)\)", body, re.S)[1].strip()
            dtype = re.search(r"ge::DT_(\w+)", variables.get(dtype_expr, dtype_expr))[1]
            prefix = f"{kind.lower()}{index}"
            lines += [f"{prefix}.name={name}", f"{prefix}.paramType={param_type}",
                      f"{prefix}.dtype={DTYPES[dtype]}", f"{prefix}.format=ND"]
    attrs = re.findall(r'this->Attr\("([^\"]+)"\)\.AttrType\((\w+)\)\.(\w+)\((.*?)\);', source)
    lines.append("attr.list=" + ",".join(a[0] for a in attrs))
    for name, mode, dtype, default in attrs:
        lines += [f"attr_{name}.type={dict(Int='int', Bool='bool', Float='float', String='str')[dtype]}",
                  f"attr_{name}.paramType={mode.lower()}"]
        if default:
            lines.append(f"attr_{name}.defaultValue=" + default.strip('"'))
    return "\n".join(lines) + "\n", attrs


class PrivateOpBuildTest(unittest.TestCase):
    def test_generated_metadata_and_compile_entry_use_private_names(self):
        binary = load_build_script("ascendc_bin_param_build")
        adapter = load_build_script("ascendc_impl_build")
        for group, name, op_type, input_count in OPERATORS:
            with self.subTest(op=op_type), tempfile.TemporaryDirectory() as temporary:
                directory = ROOT / "csrc" / group / name
                self.assertIn(f"OP_ADD({op_type})", (directory / "op_host" / f"{name}_def.cpp").read_text())
                fixture, attrs = op_info_fixture(directory, op_type)
                info = Path(temporary) / "op.ini"
                info.write_text(fixture)
                out = Path(temporary) / "out"
                out.mkdir()
                desc = binary.opdesc_parser.get_op_desc(str(info), [], [], binary.BinParamBuilder, None)[0]
                desc.set_out_path(str(out))
                desc.set_soc_version("ascend950")
                desc.gen_input_json(temporary)
                params = list(out.glob("*_param.json"))
                self.assertEqual(len(params), 1)
                data = json.loads(params[0].read_text())
                self.assertEqual(data["op_type"], op_type)
                self.assertEqual(len(data["op_list"][0]["inputs"]), input_count)
                emitted = data["op_list"][0]["attrs"]
                self.assertEqual([a["name"] for a in emitted], [a[0] for a in attrs])
                self.assertTrue(all(a is not None and a["value"] is not None for a in emitted))
                scripts = list(out.glob("*.sh"))
                self.assertEqual(len(scripts), 1)
                self.assertIn(f"--main_func={name}", scripts[0].read_text())
                desc = adapter.opdesc_parser.get_op_desc(str(info), [], [], adapter.AdpBuilder, None)[0]
                desc.write_adapt(str(directory / "op_kernel"), str(out))
                generated = out / "dynamic" / f"{name}.py"
                self.assertTrue(generated.is_file())
                ast.parse(generated.read_text())
                self.assertIn(op_type, generated.read_text())

    def test_kda_attributes_and_handwritten_aclnn_are_consistent(self):
        directory = ROOT / "csrc/attention/vllm_kda_gate_cumsum"
        _, attrs = op_info_fixture(directory, "VllmKdaGateCumsum")
        self.assertEqual([(a[0], a[1], a[2]) for a in attrs], [
            ("chunk_size", "REQUIRED", "Int"), ("use_gate_in_kernel", "REQUIRED", "Bool"),
            ("safe_gate", "REQUIRED", "Bool"), ("lower_bound", "REQUIRED", "Float"),
            ("layout", "OPTIONAL", "String"),
        ])
        self.assertEqual(attrs[-1][3], '"BSND"')
        api = directory / "op_host/op_api"
        l0 = (api / "vllm_kda_gate_cumsum.cpp").read_text()
        self.assertIn("OP_TYPE_REGISTER(VllmKdaGateCumsum)", l0)
        self.assertRegex(l0, r"ADD_TO_LAUNCHER_LIST_AICORE\(\s*VllmKdaGateCumsum,")
        self.assertIn("OP_ATTR(chunkSize, useGateInKernel, safeGate, static_cast<float>(lowerBound), layout)", l0)
        for suffix in ("GetWorkspaceSize", ""):
            symbol = "aclnnVllmKdaGateCumsum" + suffix
            self.assertIn(symbol + "(", (api / "aclnn_vllm_kda_gate_cumsum.cpp").read_text())
            self.assertIn(symbol + "(", (api / "aclnn_vllm_kda_gate_cumsum.h").read_text())
        self.assertIn("aclnnVllmKdaGateCumsum,", (directory / "kda_gate_cumsum_torch_adpt.h").read_text())
        self.assertIn("aclnnVllmCausalConv1d,", (ROOT / "csrc/torch_binding.cpp").read_text())

    def test_kernel_and_tiling_registration_match_private_types(self):
        for group, name, op_type, input_count in OPERATORS:
            with self.subTest(op=op_type):
                directory = ROOT / "csrc" / group / name
                source = (directory / "op_kernel" / f"{name}.cpp").read_text()
                signature = re.search(r"void " + name + r"\((.*?)\)", source, re.S)[1]
                self.assertEqual(signature.count("GM_ADDR"), input_count + 3)
                host = "\n".join(p.read_text() for p in (directory / "op_host").glob("*.cpp"))
                self.assertIn(f"IMPL_OP_OPTILING({op_type})", host)
                cmake = (directory / "op_host/CMakeLists.txt").read_text()
                self.assertIn(f"OPTYPE {name}", cmake)
                self.assertIn(f"OP_NAME {op_type}", cmake)
        causal = ROOT / "csrc/moe/vllm_causal_conv1d"
        self.assertIn("ASCENDC_TPL_ARGS_DECL(VllmCausalConv1d,",
                      (causal / "op_kernel/causal_conv1d_tiling_key.h").read_text())
        self.assertIn("IMPL_OP_INFERSHAPE(VllmCausalConv1d)",
                      (causal / "op_host/causal_conv1d_infershape.cpp").read_text())
        kda = ROOT / "csrc/attention/vllm_kda_gate_cumsum/op_host/kda_gate_cumsum_tiling.h"
        self.assertIn("REGISTER_TILING_DATA_CLASS(VllmKdaGateCumsum,", kda.read_text())

    def test_selected_soc_lists_keep_both_ops_and_310p(self):
        source = (ROOT / "csrc/build_aclnn.sh").read_text()
        branches = re.findall(r'CUSTOM_OPS_ARRAY=\((.*?)\)', source, re.S)
        self.assertEqual(len(branches), 4)
        self.assertIn('"causal_conv1d_v310"', branches[0])
        for branch in branches[1:]:
            self.assertIn('"vllm_causal_conv1d"', branch)
            self.assertIn('"vllm_kda_gate_cumsum"', branch)
        chunk = ROOT / "csrc/attention/chunk_kda_fwd"
        self.assertIn("attention/vllm_kda_gate_cumsum;", (chunk / "op_host/CMakeLists.txt").read_text())
        header = chunk / "op_kernel/chunk_kda_fwd_common.h"
        include = re.search(r'#include "([^"]*kda_gate_cumsum_kernel.h)"', header.read_text())[1]
        self.assertTrue((header.parent / include).is_file())
        binding = (ROOT / "csrc/torch_binding.cpp").read_text()
        self.assertIn('ops.impl("kda_gate_cumsum",', binding)
        self.assertIn('ops.impl("npu_causal_conv1d_custom",', binding)

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_reconfigure_rebuilds_derived_operator_lists(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.16)\nproject(registry NONE)\n'
                'set(PKG_NAME probe)\n'
                f'set(OPS_TRANSFORMER_DIR "{ROOT}/csrc")\n'
                f'include("{ROOT}/csrc/cmake/variables.cmake")\n'
                f'include("{ROOT}/csrc/cmake/func.cmake")\n'
                'macro(add_modules_sources)\nendmacro()\n'
                'function(add_ops_compile_options)\nendfunction()\n'
                'set(BUILD_OPEN_PROJECT OFF)\nset(BUILD_OPS_RTY_KERNEL ON)\n'
                'if(NOT PROBE_ONLY_KDA)\n'
                f'add_subdirectory("{ROOT}/csrc/moe/vllm_causal_conv1d/op_host" causal)\n'
                'endif()\n'
                f'add_subdirectory("{ROOT}/csrc/attention/vllm_kda_gate_cumsum/op_host" kda)\n'
                'file(WRITE "${CMAKE_BINARY_DIR}/ops.txt" "${COMPILED_OPS}")\n'
                'file(WRITE "${CMAKE_BINARY_DIR}/dirs.txt" "${COMPILED_OP_DIRS}")\n'
            )
            build = project / "build"
            command = ["cmake", "-S", str(project), "-B", str(build)]
            result = subprocess.run(command + ["-DCOMPILED_OPS=causal_conv1d;kda_gate_cumsum",
                                               "-DCOMPILED_OP_DIRS=/obsolete"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((build / "ops.txt").read_text().split(";"),
                             ["vllm_causal_conv1d", "vllm_kda_gate_cumsum"])
            self.assertNotIn("/obsolete", (build / "dirs.txt").read_text())
            result = subprocess.run(command + ["-DPROBE_ONLY_KDA=ON"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((build / "ops.txt").read_text(), "vllm_kda_gate_cumsum")
            self.assertNotIn("vllm_causal_conv1d", (build / "dirs.txt").read_text())

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_incremental_kernel_staging_refreshes_changed_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            source = project / "operator/op_kernel"
            source.mkdir(parents=True)
            header = source / "config.h"
            header.write_text("constexpr int threads = 1024;\n")
            (source / "kernel.cpp").write_text('#include "config.h"\n')
            (project / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.16)\nproject(staging NONE)\n'
                f'include("{ROOT}/csrc/cmake/func.cmake")\n'
                'set(OPS_ADV_UTILS_KERNEL_INC "missing")\n'
                'add_ops_src_copy(TARGET_NAME stage SRC "${CMAKE_SOURCE_DIR}/operator" '
                'DST "${CMAKE_BINARY_DIR}/src/operator" COMPUTE_UNIT ascend950)\n'
            )
            build = project / "build"
            subprocess.run(["cmake", "-S", str(project), "-B", str(build)], check=True, capture_output=True)
            command = ["cmake", "--build", str(build), "--target", "stage", "--parallel", "2"]
            subprocess.run(command, check=True, capture_output=True)
            staged = build / "src/operator/op_kernel/config.h"
            self.assertEqual(staged.read_text(), header.read_text())
            header.write_text("constexpr int threads = 2048;\n")
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(staged.read_text(), header.read_text())


if __name__ == "__main__":
    unittest.main()
