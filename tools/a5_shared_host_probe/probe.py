# SPDX-License-Identifier: Apache-2.0
"""Small multi-process A5 Host sharing experiments; no model or vLLM import."""

import argparse
import array
import contextlib
import ctypes
import hashlib
import ipaddress
import json
import multiprocessing
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

from backend import PAGE, SharedRegion, WorkerPool


def pattern(size, epoch):
    words = array.array("I", (((index * 2654435761) ^ (epoch * 2246822519)) & 0xFFFFFFFF for index in range(size // 4)))
    return words.tobytes()


def check_bytes(actual, expected, label):
    if len(actual) != len(expected):
        raise RuntimeError(f"{label}: byte count {len(actual)} != {len(expected)}")
    if actual != expected:
        mismatch = next(index for index, (a, b) in enumerate(zip(actual, expected)) if a != b)
        raise RuntimeError(
            f"{label}: first mismatch at byte {mismatch}, actual={actual[mismatch]}, expected={expected[mismatch]}"
        )


def dispatch(pool, message):
    command = message["command"]
    if command == "read":
        return {"data": pool.read(message["path"])}
    if command == "write":
        pool.write(message["data"], message["offset"], message["path"])
    elif command == "peer":
        for device in set(message["devices"]):
            pool.native.call("a5_peer", device)
    elif command == "export":
        return pool.native.export(message["numa"], message["huge"], message["pids"])
    elif command == "import":
        pool.native.call("a5_vmm_import", message["handle"], message["physical_bytes"])
    elif command == "info":
        return pool.info()
    elif command == "bind-shared":
        return pool.bind_shared(message["owner_hva"])
    elif command == "urma-init":
        return pool.init_urma(message["role"])
    elif command == "upload":
        pool.native.call("a5_upload", ctypes.create_string_buffer(message["data"]))
    elif command == "pull":
        offset, size = message["offset"], message["bytes"]
        result = pool.engine.batch_transfer_sync_read(
            message["session"],
            [pool.native.address("hva") + offset],
            [message["source"] + offset],
            [size],
        )
        if result != 0:
            raise RuntimeError(f"URMA pull failed: ret={result}, offset={offset}, bytes={size}")
    elif command == "close":
        pool.close()
    else:
        raise ValueError(f"unknown command {command!r}")
    return {}


def worker(connection, config, device, rank):
    pool = None
    phase = "initialize"
    try:
        pool = WorkerPool(config, device, rank)
        connection.send({"ok": True, "rank": rank, "device": device, "pid": os.getpid(), **pool.info()})
        while True:
            message = connection.recv()
            phase = message["command"]
            result = dispatch(pool, message)
            connection.send({"ok": True, "rank": rank, **result})
            if phase == "close":
                break
    except BaseException as error:
        traceback.print_exc()
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            connection.send({"ok": False, "rank": rank, "phase": phase, "error": str(error)})
    finally:
        try:
            if pool:
                pool.close()
        except Exception:
            traceback.print_exc()
        connection.close()


class Workers:
    def __init__(self, config, devices, timeout):
        self.timeout = timeout
        self.items = []
        self.closed = set()
        context = multiprocessing.get_context("spawn")
        for rank, device in enumerate(devices):
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(child, config, device, rank))
            process.start()
            child.close()
            self.items.append((parent, process))

    def receive(self, rank):
        connection, process = self.items[rank]
        if not connection.poll(self.timeout):
            raise TimeoutError(f"rank {rank} pid={process.pid} exceeded {self.timeout}s")
        try:
            result = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"rank {rank} exited before sending a result; exitcode={process.exitcode}") from error
        if not result["ok"]:
            raise RuntimeError(f"rank {rank} failed in {result['phase']}: {result['error']}")
        return result

    def call(self, messages, ranks=None):
        ranks = list(range(len(self.items))) if ranks is None else list(ranks)
        if len(messages) != len(ranks):
            raise ValueError("one command per rank is required")
        for rank, message in zip(ranks, messages):
            self.items[rank][0].send(message)
        results = [self.receive(rank) for rank in ranks]
        for rank, message in zip(ranks, messages):
            if message["command"] == "close":
                self.closed.add(rank)
        return results

    def broadcast(self, message, ranks=None):
        ranks = list(range(len(self.items))) if ranks is None else list(ranks)
        return self.call([message for _ in ranks], ranks)

    def shutdown(self):
        errors = []
        # Only processes spawned by this invocation are stopped.
        for rank in reversed(range(len(self.items))):
            connection, process = self.items[rank]
            if process.is_alive() and rank not in self.closed:
                try:
                    self.broadcast({"command": "close"}, [rank])
                except (RuntimeError, TimeoutError, BrokenPipeError, EOFError, OSError) as error:
                    errors.append(str(error))
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                errors.append(f"rank {rank} required termination")
            if process.exitcode != 0:
                errors.append(f"rank {rank} exitcode={process.exitcode}")
            connection.close()
        return errors


def record(output, phase, **fields):
    result = {"phase": phase, **fields}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")


def verify(workers, expected, args, label, region=None):
    replies = workers.broadcast({"command": "read", "path": args.path_id})
    for rank, result in enumerate(replies):
        check_bytes(result["data"], expected, f"{label} rank={rank}")
    if region and args.backend in ("cpu", "posix-mapped", "posix-pinned-mapped"):
        region.check_guards()
        check_bytes(region.read(), expected, f"{label} shared CPU view")
    if args.backend != "cpu" and args.path_id != 0:
        for rank, result in enumerate(workers.broadcast({"command": "read", "path": 0})):
            check_bytes(result["data"], expected, f"{label} runtime view rank={rank}")
    record(
        args.output,
        label,
        status="PASS",
        checked_bytes=len(expected),
        ranks=len(replies),
        sha256=hashlib.sha256(expected).hexdigest(),
        path=args.path,
    )


def write_commands(workers, expected, args):
    count = len(workers.items)
    shard = len(expected) // count
    replicated = args.backend == "pinned"
    messages = []
    for rank in range(count):
        offset = 0 if replicated else rank * shard
        size = len(expected) if replicated else shard
        messages.append(
            {"command": "write", "data": expected[offset : offset + size], "offset": offset, "path": args.path_id}
        )
    workers.call(messages)


def prepare(workers, ready, initial, args):
    if args.backend == "mf-shared":
        workers.broadcast({"command": "bind-shared", "owner_hva": ready[0]["owner_hva"]})
    if args.backend == "vmm-host":
        workers.broadcast({"command": "peer", "devices": args.devices})
        exported = workers.broadcast(
            {
                "command": "export",
                "numa": args.numa_node,
                "huge": int(args.huge),
                "pids": [item["bare_pid"] for item in ready],
            },
            [0],
        )[0]
        workers.broadcast(
            {"command": "import", "handle": exported["handle"], "physical_bytes": exported["physical_bytes"]},
            range(1, len(ready)),
        )
        record(args.output, "vmm_import", physical_bytes=exported["physical_bytes"], pid_whitelist=True)
    if args.backend in ("pinned", "vmm-host", "mf-shared"):
        ranks = None if args.backend == "pinned" else [0]
        workers.broadcast({"command": "write", "data": initial, "offset": 0, "path": 0}, ranks)
    addresses = workers.broadcast({"command": "info"})
    for item in addresses:
        record(args.output, "mapping", **item)


def test_urma(destinations, args, config, region):
    source_config = dict(config, backend="source")
    source = Workers(source_config, [args.urma_source_device], args.timeout)
    try:
        source.receive(0)
        source_info = source.broadcast({"command": "urma-init", "role": "Prefill"})[0]
        destinations.broadcast({"command": "urma-init", "role": "Decode"})
        expected = pattern(args.bytes, 5)
        source.broadcast({"command": "upload", "data": expected})
        count = len(destinations.items)
        shard = args.bytes // count
        messages = []
        for rank in range(count):
            offset = 0 if args.backend == "pinned" else rank * shard
            size = args.bytes if args.backend == "pinned" else shard
            messages.append(
                {
                    "command": "pull",
                    "session": source_info["session"],
                    "source": source_info["address"],
                    "offset": offset,
                    "bytes": size,
                }
            )
        destinations.call(messages)
        verify(destinations, expected, args, "urma_pull_then_aiv_read", region)
    finally:
        # Even on a failed pull, stop D clients before releasing the P allocation.
        failures = destinations.shutdown() + source.shutdown()
        if failures:
            raise RuntimeError("URMA cleanup: " + "; ".join(failures))


def run(args):
    started = time.monotonic()
    config = {key: value for key, value in vars(args).items() if key not in ("output",)}
    config["library"] = str(args.library.resolve())
    workers = None
    errors = []
    with tempfile.TemporaryDirectory(prefix="a5-host-probe-", dir=args.shm_dir) as directory:
        config["backing"] = str(Path(directory) / "pool")
        region = SharedRegion(config["backing"], args.bytes, create=True)
        initial = pattern(args.bytes, 1)
        region.write(initial)
        try:
            workers = Workers(config, args.devices, args.timeout)
            ready = [workers.receive(rank) for rank in range(len(args.devices))]
            for item in ready:
                record(args.output, "ready", **item)
            prepare(workers, ready, initial, args)
            verify(workers, initial, args, "initial_read", region)
            expected = pattern(args.bytes, 2)
            write_commands(workers, expected, args)
            label = "disjoint_cpu_writes" if args.backend == "cpu" else "disjoint_device_writes"
            if args.backend != "cpu" and args.path_id == 0:
                label = "disjoint_host_runtime_writes"
            if args.backend == "pinned":
                label = "replicated_host_runtime_writes" if args.path_id == 0 else "replicated_device_writes"
            verify(workers, expected, args, label, region)
            replacement = pattern(PAGE, 3)
            offset = args.bytes - PAGE
            ranks = None if args.backend == "pinned" else [0]
            workers.broadcast({"command": "write", "data": replacement, "offset": offset, "path": args.path_id}, ranks)
            expected = expected[:offset] + replacement
            verify(workers, expected, args, "suffix_rewrite_visibility", region)
            if args.backend in ("cpu", "posix-mapped", "posix-pinned-mapped"):
                expected = pattern(args.bytes, 4)
                region.write(expected)
                verify(workers, expected, args, "cpu_rewrite_visibility", region)
            elif args.backend == "mf-shared":
                expected = pattern(args.bytes, 4)
                workers.broadcast({"command": "write", "data": expected, "offset": 0, "path": 0}, [0])
                verify(workers, expected, args, "cpu_rewrite_visibility", region)
            if args.urma_source_device is not None:
                test_urma(workers, args, config, region)
        finally:
            if workers:
                errors = workers.shutdown()
            try:
                region.check_guards()
            finally:
                region.close()
        if errors:
            raise RuntimeError("worker cleanup: " + "; ".join(errors))
    scope = "CPU_ONLY" if args.backend == "cpu" else ("RUNTIME_COPY_ONLY" if args.path_id == 0 else "DEVICE_PROBE")
    record(
        args.output,
        "complete",
        status="PASS",
        scope=scope,
        backend=args.backend,
        elapsed_seconds=round(time.monotonic() - started, 3),
        note="Correctness experiment; elapsed time is not kernel latency or model throughput.",
    )


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("cpu", "pinned", "posix-mapped", "posix-pinned-mapped", "vmm-host", "mf-shared"),
        default="posix-pinned-mapped",
    )
    parser.add_argument(
        "--devices", default="0,1", help="Logical device IDs; 0,0 is a same-device cross-process control"
    )
    parser.add_argument("--path", choices=("runtime", "simt", "simt-cg", "mte"), default="simt")
    parser.add_argument("--bytes", type=int, default=256 * 1024)
    parser.add_argument("--library", type=Path, default=Path(__file__).parent / "build/liba5_host_probe.so")
    parser.add_argument("--shm-dir", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--output", type=Path, default=Path("a5_shared_host_probe.jsonl"))
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--numa-node", type=int, default=0)
    parser.add_argument("--huge", action="store_true", help="VMM: request P2P huge pages instead of normal pages")
    parser.add_argument(
        "--urma-source-device", type=int, help="Optional additional P device for real URMA HBM-to-shared-Host pull"
    )
    parser.add_argument("--bind-ip", help="Numeric IPv4 address reachable by MemFabric endpoints")
    parser.add_argument("--store-url", help="An unused tcp://IP:PORT for this test's Prefill store server")
    parser.add_argument("--shared-store-url", help="mf-shared: a separate unused tcp://IP:PORT for the offload group")
    parser.add_argument(
        "--shared-address",
        choices=("identity", "get-dva"),
        default="identity",
        help="mf-shared: test the legacy identity-address contract or query each rank's get_dva",
    )
    args = parser.parse_args()
    try:
        args.devices = [int(value) for value in args.devices.split(",")]
    except ValueError:
        parser.error("--devices must be comma-separated integers")
    if not args.devices or min(args.devices) < 0:
        parser.error("at least one non-negative device ID is required")
    if args.bytes <= 0 or args.bytes > 1 << 30 or args.bytes % (PAGE * len(args.devices)) != 0:
        parser.error(f"--bytes must be in (0, 1 GiB] and divisible by {PAGE} times the number of workers")
    if args.timeout <= 0 or args.numa_node < 0:
        parser.error("timeout must be positive and NUMA node non-negative")
    if args.backend == "mf-shared" and not (args.shared_store_url or "").startswith("tcp://"):
        parser.error("mf-shared requires --shared-store-url tcp://IP:PORT to avoid the production pool store")
    if sys.byteorder != "little":
        parser.error("this uint32 pattern probe requires a little-endian Host")
    if args.backend.startswith("posix"):
        kernel = tuple(int(value) for value in platform.release().split("-")[0].split(".")[:2])
        if kernel <= (5, 10):
            parser.error("POSIX registration probe requires Linux newer than 5.10 per CANN documentation")
    if args.urma_source_device is not None:
        if args.backend in ("cpu", "vmm-host") or args.path == "runtime":
            parser.error("URMA test requires pinned/POSIX backend and an AIV path")
        if args.urma_source_device < 0 or args.urma_source_device in args.devices:
            parser.error("use an additional, distinct P device to avoid same-device URMA loopback")
        try:
            ipaddress.IPv4Address(args.bind_ip)
        except (ipaddress.AddressValueError, TypeError):
            parser.error("--bind-ip must be a numeric IPv4 address for URMA")
        if not args.store_url or not args.store_url.startswith("tcp://"):
            parser.error("--store-url tcp://IP:PORT is required for URMA")
        if args.store_url == args.shared_store_url:
            parser.error("offload pool and transfer engine need separate store ports")
    args.path_id = ("runtime", "simt", "simt-cg", "mte").index(args.path)
    if args.backend == "cpu":
        args.path = "cpu"
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = arguments()
    record(
        args.output,
        "environment",
        backend=args.backend,
        devices=args.devices,
        bytes=args.bytes,
        kernel=platform.release(),
        machine=platform.machine(),
        python=sys.version.split()[0],
        page_alignment=PAGE,
        ascend_home=os.environ.get("ASCEND_HOME_PATH"),
        path=args.path,
    )
    try:
        run(args)
    except Exception as error:
        record(args.output, "complete", status="FAIL", error=str(error), backend=args.backend)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
