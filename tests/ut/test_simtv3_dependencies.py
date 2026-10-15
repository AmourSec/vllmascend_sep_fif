# SPDX-License-Identifier: Apache-2.0
"""Check dependency metadata without importing Torch, vLLM, or an NPU backend."""

import ast
import shlex
import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]


def runtime_requirements():
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    requirements = [Requirement(line) for raw in lines if (line := raw.split("#", 1)[0].strip())]
    return {canonicalize_name(requirement.name): requirement for requirement in requirements}


def build_requirements():
    # This project supports Python 3.10, which does not provide tomllib.
    build_section = (ROOT / "pyproject.toml").read_text(encoding="utf-8").split("build-backend", 1)[0]
    requirements = [Requirement(item) for item in ast.literal_eval(build_section.split("requires =", 1)[1].strip())]
    return {canonicalize_name(requirement.name): requirement for requirement in requirements}


class Simtv3DependencyTest(unittest.TestCase):
    def test_web_requirements_match_vllm_028_and_build_metadata(self):
        runtime = runtime_requirements()
        build = build_requirements()
        for name, version in (("fastapi", "0.136.3"), ("starlette", "1.8.0")):
            with self.subTest(name=name):
                self.assertIn(version, runtime[name].specifier)
                self.assertEqual(runtime[name].specifier, build[name].specifier)
        # These are real vLLM API compatibility boundaries, not arbitrary upgrades.
        for version in ("0.123.10", "0.132.0", "0.137.0"):
            self.assertNotIn(version, runtime["fastapi"].specifier)
        self.assertNotIn("1.0.0", runtime["starlette"].specifier)

    def test_memfabric_matches_clean_fork_version(self):
        requirement = runtime_requirements()["memfabric-hybrid"]
        self.assertIn("1.2.1", requirement.specifier)
        for version in ("1.2.0", "1.3.0"):
            self.assertNotIn(version, requirement.specifier)

    def test_proxy_install_examples_allow_selected_web_stack(self):
        examples = (
            "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py",
            "examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py",
            "examples/epd_disaggregated/epd_load_balance_proxy_layerwise_server_example.py",
            "examples/dynamic_bucket_load_balancer/README.md",
        )
        for path in examples:
            with self.subTest(path=path):
                line = next(line for line in (ROOT / path).read_text().splitlines() if "pip install" in line)
                items = shlex.split(line.split("pip install", 1)[1])
                requirements = {canonicalize_name(item.name): item for text in items if (item := Requirement(text))}
                self.assertIn("0.136.3", requirements["fastapi"].specifier)
                self.assertIn("1.8.0", requirements["starlette"].specifier)

    def test_memcache_contract_is_optional_and_preserved(self):
        self.assertNotIn("memcache-hybrid", runtime_requirements())
        self.assertNotIn("memcache-hybrid", build_requirements())
        tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
        setup_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setup"
        )
        extras = ast.literal_eval(next(item.value for item in setup_call.keywords if item.arg == "extras_require"))
        self.assertEqual(extras["memcache"], ["memcache_hybrid==1.2.0"])

    def test_device_pins_are_not_relaxed_without_runtime_evidence(self):
        runtime = runtime_requirements()
        build = build_requirements()
        expected = {"torch": "==2.10.0", "torch-npu": "==2.10.0.post4", "triton-ascend": "==3.2.2"}
        for name, specifier in expected.items():
            with self.subTest(name=name):
                self.assertEqual(str(runtime[name].specifier), specifier)
                self.assertEqual(runtime[name].specifier, build[name].specifier)


if __name__ == "__main__":
    unittest.main()
