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
