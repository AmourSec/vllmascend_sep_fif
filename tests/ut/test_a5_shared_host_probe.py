# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the actual probe orchestration and shared backing."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/a5_shared_host_probe"


def load_backend():
    spec = importlib.util.spec_from_file_location("a5_probe_backend_test", TOOL / "backend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SharedHostProbeTest(unittest.TestCase):
    def invoke(self, *arguments):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            command = [
                sys.executable,
                str(TOOL / "probe.py"),
                "--output",
                str(output),
                "--shm-dir",
                directory,
                "--timeout",
                "5",
                *arguments,
            ]
            result = subprocess.run(command, text=True, capture_output=True, timeout=30)
            events = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
        return result, events

    def test_two_mappings_access_one_backing_with_independent_addresses(self):
        backend = load_backend()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool"
            owner = backend.SharedRegion(path, 8192, create=True)
            peer = backend.SharedRegion(path, 8192)
            try:
                self.assertNotEqual(owner.hva, peer.hva)
                self.assertEqual(owner.info(), peer.info())
                owner.write(b"abcd", 0)
                peer.write(b"efgh", 4)
                self.assertEqual(owner.read()[:8], b"abcdefgh")
                self.assertEqual(peer.read()[:8], b"abcdefgh")
                owner.check_guards()
                peer.check_guards()
                with self.assertRaises(ValueError):
                    owner.write(b"wrong", 8190)
            finally:
                peer.close()
                owner.close()

    def test_shared_guard_corruption_is_reported(self):
        backend = load_backend()
        with tempfile.TemporaryDirectory() as directory:
            region = backend.SharedRegion(Path(directory) / "pool", 4096, create=True)
            try:
                region.mapping[0] ^= 1
                with self.assertRaisesRegex(RuntimeError, "guard overwritten"):
                    region.check_guards()
            finally:
                region.close()

    def test_real_two_and_four_process_cpu_protocol(self):
        page = load_backend().PAGE
        for devices, size in (("0,1", str(2 * page)), ("0,1,2,3", str(4 * page)), ("0,0", str(2 * page))):
            with self.subTest(devices=devices):
                result, events = self.invoke("--backend", "cpu", "--devices", devices, "--bytes", size)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(events[-1]["status"], "PASS")
                self.assertEqual(events[-1]["scope"], "CPU_ONLY")
                ready = [event for event in events if event["phase"] == "ready"]
                self.assertEqual(len(ready), len(devices.split(",")))
                backing = {
                    (event["backing_device"], event["backing_inode"], event["backing_offset"]) for event in ready
                }
                self.assertEqual(len(backing), 1)
                phases = {event["phase"] for event in events}
                self.assertTrue(
                    {
                        "initial_read",
                        "disjoint_cpu_writes",
                        "suffix_rewrite_visibility",
                        "cpu_rewrite_visibility",
                    }.issubset(phases)
                )
                self.assertNotIn("disjoint_device_writes", phases)

    def test_invalid_shard_alignment_is_rejected_before_runtime(self):
        result, events = self.invoke("--backend", "cpu", "--devices", "0,1", "--bytes", "4096")
        self.assertEqual(result.returncode, 2)
        self.assertIn("divisible", result.stderr)
        self.assertEqual(events, [])

    def test_native_load_failure_never_reports_success(self):
        result, events = self.invoke(
            "--backend",
            "pinned",
            "--devices",
            "0",
            "--bytes",
            str(load_backend().PAGE),
            "--library",
            "/no/such/a5_probe_library.so",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(events[-1]["status"], "FAIL")
        self.assertFalse(any(event.get("status") == "PASS" for event in events))

    def test_urma_requires_separate_source_and_cpu_addressable_destination(self):
        for arguments in (
            ("--backend", "posix-mapped", "--devices", "0,1", "--urma-source-device", "0"),
            ("--backend", "vmm-host", "--urma-source-device", "2"),
            ("--backend", "cpu", "--urma-source-device", "2"),
        ):
            with self.subTest(arguments=arguments):
                result, events = self.invoke(*arguments)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
