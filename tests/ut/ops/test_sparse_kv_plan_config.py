# SPDX-License-Identifier: Apache-2.0
"""Compile the real Plan config with its public kernel symbol, without CANN/NPU."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / "csrc/attention/sparse_kv_plan/op_kernel/sparse_kv_plan_config.h"


class SparseKvPlanConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which("g++") or shutil.which("clang++")
        if cls.compiler is None:
            raise unittest.SkipTest("g++ or clang++ is required for this compilation regression")

    def compile_probe(self, macro):
        source = macro + f'''#include "{HEADER}"
extern "C" void sparse_kv_plan();
using namespace sparse_kv_plan_config;
static_assert(PLAN_THREADS == 1024U);
static_assert(PLAN_WARP_COUNT == 32U);
static_assert(PLAN_SCAN_BYTES == 288U);
static_assert(PLAN_ROW_UB_LIMIT_BYTES == 112U * 1024U);
static_assert(PLAN_MAX_DYNAMIC_UB_BYTES == 288U + 112U * 1024U);
static_assert(PLAN_SCAN_STORAGE_ELEMENTS == 72U);
int main() {{ return 0; }}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.cpp"
            path.write_text(source)
            result = subprocess.run([self.compiler, "-std=c++17", "-fsyntax-only", str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_kernel_symbol_and_config_coexist(self):
        self.compile_probe("")

    def test_cann_kernel_rename_does_not_rename_config_namespace(self):
        self.compile_probe("#define sparse_kv_plan sparse_kv_plan_0_tilingkey\n")


if __name__ == "__main__":
    unittest.main()
