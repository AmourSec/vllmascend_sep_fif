# SPDX-License-Identifier: Apache-2.0
"""Isolated process backends. The parent process never initializes CANN."""

import ctypes
import mmap
import os
from pathlib import Path

PAGE = max(4096, mmap.PAGESIZE)
GUARD = b"\xbd" * PAGE
ACL_HOST_REG_MAPPED = 0x2
ACL_HOST_REG_PINNED = 0x10000000


class SharedRegion:
    def __init__(self, path, size, create=False):
        self.size = size
        self.fd = os.open(path, os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0), 0o600)
        if create:
            os.ftruncate(self.fd, size + 2 * PAGE)
        self.mapping = mmap.mmap(self.fd, size + 2 * PAGE, flags=mmap.MAP_SHARED)
        self.hva = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping, PAGE))
        if create:
            self.mapping[:PAGE] = GUARD
            self.mapping[-PAGE:] = GUARD

    def write(self, data, offset=0):
        if offset < 0 or offset + len(data) > self.size:
            raise ValueError("shared-memory write outside payload")
        self.mapping[PAGE + offset : PAGE + offset + len(data)] = data

    def read(self):
        return self.mapping[PAGE : PAGE + self.size]

    def check_guards(self):
        if self.mapping[:PAGE] != GUARD or self.mapping[-PAGE:] != GUARD:
            raise RuntimeError("shared Host guard overwritten")

    def info(self):
        stat = os.fstat(self.fd)
        return {"backing_device": stat.st_dev, "backing_inode": stat.st_ino, "backing_offset": PAGE}

    def close(self):
        self.mapping.close()
        os.close(self.fd)


class Native:
    def __init__(self, library, device, size):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.size = size
        self.state = ctypes.c_void_p()
        self._declare()
        try:
            self.call("a5_open", device, size, ctypes.byref(self.state), with_state=False)
        except BaseException:
            self.close()
            raise

    def _declare(self):
        pointer = ctypes.c_void_p
        u64 = ctypes.c_uint64
        integer = ctypes.c_int
        signatures = {
            "a5_open": [integer, u64, ctypes.POINTER(pointer)],
            "a5_close": [pointer],
            "a5_attach": [pointer, u64, ctypes.c_uint32],
            "a5_bind_existing": [pointer, u64, u64],
            "a5_pinned": [pointer],
            "a5_bare_pid": [pointer, ctypes.POINTER(ctypes.c_int32)],
            "a5_peer": [pointer, integer],
            "a5_vmm_create": [
                pointer,
                integer,
                integer,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_size_t,
                ctypes.POINTER(u64),
                ctypes.POINTER(u64),
            ],
            "a5_vmm_import": [pointer, u64, u64],
            "a5_read": [pointer, pointer, integer],
            "a5_write": [pointer, pointer, u64, u64, integer],
            "a5_upload": [pointer, pointer],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = arguments
            function.restype = integer
        for name in ("a5_hva", "a5_dva", "a5_hbm"):
            function = getattr(self.lib, name)
            function.argtypes = [pointer]
            function.restype = u64

    def call(self, name, *arguments, with_state=True):
        if with_state:
            arguments = (self.state, *arguments)
        result = getattr(self.lib, name)(*arguments)
        if result != 0:
            raise RuntimeError(f"{name} returned {result}; see the CANN call and error code in stderr")

    def address(self, name):
        return int(getattr(self.lib, f"a5_{name}")(self.state))

    def read(self, path):
        output = ctypes.create_string_buffer(self.size)
        self.call("a5_read", output, path)
        return output.raw

    def write(self, data, offset, path):
        if offset < 0 or offset + len(data) > self.size:
            raise ValueError("native write outside payload")
        buffer = ctypes.create_string_buffer(data)
        self.call("a5_write", buffer, offset, len(data), path)

    def export(self, numa, huge, pids):
        whitelist = (ctypes.c_int32 * len(pids))(*pids)
        handle, size = ctypes.c_uint64(), ctypes.c_uint64()
        self.call("a5_vmm_create", numa, huge, whitelist, len(pids), ctypes.byref(handle), ctypes.byref(size))
        return {"handle": handle.value, "physical_bytes": size.value}

    def close(self):
        if self.state.value:
            state, self.state = self.state, ctypes.c_void_p()
            self.call("a5_close", state, with_state=False)


class WorkerPool:
    def __init__(self, config, device, rank):
        self.config = config
        self.device = device
        self.native = None
        self.region = None
        self.engine = None
        self.registered_hva = 0
        self.offload = None
        self.offload_initialized = False
        self.owner_hva = 0
        mode = config["backend"]
        if mode in ("cpu", "posix-mapped", "posix-pinned-mapped"):
            self.region = SharedRegion(config["backing"], config["bytes"])
        try:
            if mode != "cpu":
                self.native = Native(config["library"], device, config["bytes"])
                if self.region is not None:
                    flags = ACL_HOST_REG_MAPPED | (ACL_HOST_REG_PINNED if mode == "posix-pinned-mapped" else 0)
                    self.native.call("a5_attach", self.region.hva, flags)
                elif mode == "pinned":
                    self.native.call("a5_pinned")
                elif mode == "mf-shared":
                    self.init_shared(rank)
        except BaseException:
            self.close()
            raise

    def init_shared(self, rank):
        from memfabric_hybrid import offload

        self.offload = offload
        config = offload.OffloadConfig()
        config.device_id = self.device
        config.reserve_size = 1 << 30
        config.alloc_size = (1 << 30) if rank == 0 else 0
        config.world_size = len(self.config["devices"])
        config.rank_id = rank
        config.scene = offload.Scene.SHARED
        config.store_url = self.config["shared_store_url"]
        result = offload.initialize(config)
        if result != 0:
            raise RuntimeError(f"MemFabric SHARED initialize failed: ret={result}, rank={rank}, device={self.device}")
        self.offload_initialized = True
        if rank == 0:
            self.owner_hva = int(offload.malloc(self.config["bytes"]))
            if not self.owner_hva:
                raise RuntimeError("MemFabric SHARED malloc returned a null address")

    def bind_shared(self, owner_hva):
        dva = owner_hva
        if self.config["shared_address"] == "get-dva":
            dva = int(self.offload.get_dva(owner_hva))
        self.native.call("a5_bind_existing", owner_hva, dva)
        return self.info()

    def info(self):
        result = {"hva": hex(self.region.hva if self.region else 0), "dva": "0x0"}
        if self.native:
            result.update(hva=hex(self.native.address("hva")), dva=hex(self.native.address("dva")))
        if self.region:
            result.update(self.region.info())
        if self.owner_hva:
            result["owner_hva"] = self.owner_hva
        if self.config["backend"] == "vmm-host":
            bare_pid = ctypes.c_int32()
            self.native.call("a5_bare_pid", ctypes.byref(bare_pid))
            result["bare_pid"] = bare_pid.value
        return result

    def read(self, path):
        return self.native.read(path) if self.native else self.region.read()

    def write(self, data, offset, path):
        if self.native:
            self.native.write(data, offset, path)
        else:
            self.region.write(data, offset)

    def init_urma(self, role):
        from memfabric_hybrid import TransDataOpType, TransferEngine, set_conf_store_tls, set_log_level

        if not self.native or (role == "Decode" and not self.native.address("hva")):
            raise RuntimeError(
                "URMA destination requires a real CPU HVA, not a mapped DVA or an unverified VMM pointer"
            )
        set_log_level(2)
        set_conf_store_tls(False, "")
        self.engine = TransferEngine()
        result = self.engine.initialize(
            self.config["store_url"],
            self.config["bind_ip"],
            role,
            self.device,
            store_server_role="Prefill",
            data_op_type=TransDataOpType.DEVICE_URMA,
        )
        if result != 0:
            raise RuntimeError(f"URMA initialize failed: ret={result}, role={role}, device={self.device}")
        address = self.native.address("hbm" if role == "Prefill" else "hva")
        result = self.engine.register_memory(address, self.config["bytes"])
        if result != 0:
            raise RuntimeError(f"URMA register_memory failed: ret={result}, address={address:#x}")
        self.registered_hva = address
        return {"session": f"{self.config['bind_ip']}:{self.engine.get_rpc_port()}", "address": address}

    def close(self):
        errors = []
        if self.engine:
            actions = [("destroy", ()), ("unInitialize", ())]
            if self.registered_hva:
                actions.insert(0, ("unregister_memory", (self.registered_hva,)))
            for method, arguments in actions:
                try:
                    result = getattr(self.engine, method)(*arguments)
                    if result not in (None, 0):
                        errors.append(f"{method} returned {result}")
                except Exception as error:
                    errors.append(f"{method}: {error}")
            self.engine = None
        if self.offload_initialized:
            self.owner_hva = 0
            try:
                result = self.offload.uninitialize()
                if result not in (None, 0):
                    errors.append(f"offload.uninitialize returned {result}")
            except Exception as error:
                errors.append(str(error))
            self.offload_initialized = False
        if self.native:
            try:
                self.native.close()
            except Exception as error:
                errors.append(str(error))
            self.native = None
        if self.region:
            try:
                self.region.check_guards()
                self.region.close()
            except Exception as error:
                errors.append(str(error))
            self.region = None
        if errors:
            raise RuntimeError("cleanup failed: " + "; ".join(errors))
