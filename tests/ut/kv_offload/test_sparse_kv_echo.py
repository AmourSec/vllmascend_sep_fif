# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free checks: python tests/ut/kv_offload/test_sparse_kv_echo.py.

The host harness executes the actual row code with CPU atomics/barriers and a
reduced thread count. It checks the algorithm, not CANN compilation, the memory
model of A5, DMA, or graph replay. Those still require hardware validation.
"""

import ast
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "csrc/attention/sparse_kv_echo_plan/op_kernel"


class TestSparseKVEcho(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "requires g++")
    def test_production_ub_layout(self):
        # Use the real 2048-thread header here; the row harness scales down
        # threads to keep CPU execution affordable.
        source = r"""
#include <cassert>
#include <cstdint>
#include "sparse_kv_echo_plan_config.h"
using namespace sparse_kv_echo_plan;
static_assert(PLAN_THREADS == 2048 && PLAN_WARP_COUNT == 64);
static_assert(LocalMemoryBytes(4096, 8192) == 75840);
static_assert(LocalMemoryBytes(4096, 262144) == 43072);
static_assert(CacheTokenMapInUb(4096, 8192));
static_assert(!CacheTokenMapInUb(4096, 262144));
static_assert(!CachePrioritiesInUb(16384));
int main() {
  for (uint64_t c = 1; c <= 20000; ++c) {
    const uint64_t mapLimit = (PLAN_UB_BUDGET_BYTES - PLAN_FIXED_UB_BYTES - PriorityCacheBytes(c)) / 4;
    for (int delta = -1; delta <= 1; ++delta) {
      if (mapLimit == 0 && delta < 0) continue;
      const uint64_t m = mapLimit + delta;
      const uint32_t bytes = LocalMemoryBytes(c, m);
      assert(bytes <= PLAN_UB_BUDGET_BYTES && bytes % 32 == 0);
      assert(CacheTokenMapInUb(c, m) == (m <= mapLimit));
      assert(bytes >= PLAN_FIXED_UB_BYTES + PriorityCacheBytes(c));
      if (CacheTokenMapInUb(c, m)) assert(bytes >= PLAN_FIXED_UB_BYTES + PriorityCacheBytes(c) + m * 4);
    }
  }
}
"""
        with tempfile.TemporaryDirectory(prefix="echo-layout-test-") as directory:
            tmp = Path(directory)
            test = tmp / "layout.cpp"
            test.write_text(source)
            binary = tmp / "layout"
            subprocess.run(["g++", "-std=c++17", "-I", str(KERNEL), str(test), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_workspace_layout(self):
        source = ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_echo.py"
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
        namespace = {}
        exec(compile(tree, str(source), "exec"), namespace)
        workspace = namespace["echo_workspace_elements"]
        self.assertEqual(workspace(1, 1024, 256, 512), 1792)
        self.assertEqual(workspace(100, 8192, 2048, 4096), 64 * 14336)
        self.assertEqual(workspace(3, 131, 23, 37), 3 * 192)

    @unittest.skipUnless(shutil.which("g++"), "requires g++ with C++20 support")
    def test_actual_row_code(self):
        body = (KERNEL / "sparse_kv_echo_plan_apt.cpp").read_text().split('extern "C"')[0]
        config = (KERNEL / "sparse_kv_echo_plan_config.h").read_text()
        with tempfile.TemporaryDirectory(prefix="echo-host-test-") as directory:
            tmp = Path(directory)
            (tmp / "echo_kernel_body.h").write_text(body)
            for header in re.findall(r'#include "([^\"]+)"', body):
                path = tmp / header
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            for threads in (32, 64):
                with self.subTest(threads=threads):
                    (tmp / "sparse_kv_echo_plan_config.h").write_text(
                        config.replace("PLAN_THREADS = 2048U", f"PLAN_THREADS = {threads}U")
                    )
                    executable = tmp / "echo-host-test"
                    subprocess.run(
                        [
                            "g++",
                            "-std=c++20",
                            "-O1",
                            "-g",
                            "-pthread",
                            "-fsanitize=address,undefined",
                            "-fno-omit-frame-pointer",
                            "-no-pie",
                            "-I",
                            str(tmp),
                            str(Path(__file__).with_name("sparse_kv_echo_host.cpp")),
                            "-o",
                            str(executable),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run([str(executable)], check=True, timeout=180)


if __name__ == "__main__":
    unittest.main()
