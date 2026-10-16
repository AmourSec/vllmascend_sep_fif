// SPDX-License-Identifier: Apache-2.0
#include "native.h"

#include <acl/acl.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int a5_kernel_copy(void* source, void* destination, uint64_t words, int path, aclrtStream stream);

namespace {
constexpr size_t GUARD_WORDS = 32;
constexpr uint32_t GUARD_VALUE = 0xA5C39E71U;

struct State {
  int device = -1;
  size_t bytes = 0;
  bool initialized = false;
  bool deviceSet = false;
  bool registered = false;
  void* hva = nullptr;
  void* dva = nullptr;
  void* pinned = nullptr;
  void* hbm = nullptr;
  void* reserved = nullptr;
  bool mapped = false;
  aclrtDrvMemHandle physical = nullptr;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
};

void Check(aclError code, const char* call) {
  if (code != ACL_SUCCESS) {
    throw std::runtime_error(std::string(call) + " failed, ret=" + std::to_string(code));
  }
}

State& Use(void* opaque) {
  auto& state = *static_cast<State*>(opaque);
  Check(aclrtSetCurrentContext(state.context), "aclrtSetCurrentContext");
  return state;
}

template <typename Function>
int Guard(const char* operation, Function function) {
  try {
    function();
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "ERROR %s: %s\n", operation, error.what());
    return -1;
  }
}

uint32_t* Payload(State& state) { return static_cast<uint32_t*>(state.hbm) + GUARD_WORDS; }

void ResetScratch(State& state) {
  std::vector<uint32_t> initial(state.bytes / sizeof(uint32_t) + 2 * GUARD_WORDS, GUARD_VALUE);
  Check(aclrtMemcpy(state.hbm, initial.size() * sizeof(uint32_t), initial.data(), initial.size() * sizeof(uint32_t),
                    ACL_MEMCPY_HOST_TO_DEVICE),
        "initialize HBM guards");
}

void DownloadScratch(State& state, void* output) {
  std::vector<uint32_t> result(state.bytes / sizeof(uint32_t) + 2 * GUARD_WORDS);
  Check(aclrtMemcpy(result.data(), result.size() * sizeof(uint32_t), state.hbm, result.size() * sizeof(uint32_t),
                    ACL_MEMCPY_DEVICE_TO_HOST),
        "download HBM and guards");
  const auto front = result.begin() + GUARD_WORDS;
  const auto back = result.end() - GUARD_WORDS;
  const auto valid = [](uint32_t word) { return word == GUARD_VALUE; };
  if (!std::all_of(result.begin(), front, valid) || !std::all_of(back, result.end(), valid)) {
    throw std::runtime_error("HBM output guard overwritten");
  }
  std::memcpy(output, result.data() + GUARD_WORDS, state.bytes);
}

void MapPhysical(State& state, uint64_t bytes) {
  Check(aclrtReserveMemAddress(&state.reserved, bytes, 0, nullptr, 0), "aclrtReserveMemAddress");
  Check(aclrtMapMem(state.reserved, bytes, 0, state.physical, 0), "aclrtMapMem");
  state.mapped = true;
  state.dva = state.reserved;
  // A VMM address is not assumed to be CPU-dereferenceable. No HVA is exposed.
}

void LogCleanup(aclError code, const char* call, int& result) {
  if (code != ACL_SUCCESS) {
    std::fprintf(stderr, "ERROR cleanup %s failed, ret=%d\n", call, code);
    result = -1;
  }
}
}  // namespace

extern "C" int a5_open(int device, uint64_t bytes, void** output) {
  return Guard("a5_open", [&] {
    auto state = std::make_unique<State>();
    state->device = device;
    state->bytes = bytes;
    *output = state.release();  // Caller can clean up even if initialization fails.
    auto& current = *static_cast<State*>(*output);
    Check(aclInit(nullptr), "aclInit");
    current.initialized = true;
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    current.deviceSet = true;
    Check(aclrtGetCurrentContext(&current.context), "aclrtGetCurrentContext");
    Check(aclrtCreateStream(&current.stream), "aclrtCreateStream");
    Check(aclrtMalloc(&current.hbm, bytes + 2 * GUARD_WORDS * sizeof(uint32_t), ACL_MEM_MALLOC_NORMAL_ONLY),
          "aclrtMalloc HBM scratch");
    aclrtHostMemMapCapability capability = ACL_RT_HOST_MEM_MAP_NOT_SUPPORTED;
    const auto ret = aclrtHostMemMapCapabilities(device, ACL_RT_HAC_TYPE_AIV, &capability);
    std::fprintf(stderr, "INFO device=%d soc=%s AIV_host_map_query_ret=%d capability=%d bytes=%llu\n", device,
                 aclrtGetSocName(), ret, capability, static_cast<unsigned long long>(bytes));
    // A missing capability query is diagnostic; the actual kernel check remains decisive.
  });
}

extern "C" int a5_close(void* opaque) {
  if (opaque == nullptr) {
    return 0;
  }
  std::unique_ptr<State> state(static_cast<State*>(opaque));
  int result = 0;
  if (state->context != nullptr) {
    LogCleanup(aclrtSetCurrentContext(state->context), "set context", result);
  }
  if (state->stream != nullptr) {
    LogCleanup(aclrtSynchronizeStream(state->stream), "synchronize stream", result);
  }
  if (state->registered) {
    LogCleanup(aclrtHostUnregister(state->hva), "aclrtHostUnregister", result);
  }
  if (state->pinned != nullptr) {
    LogCleanup(aclrtFreeHost(state->pinned), "aclrtFreeHost", result);
  }
  if (state->mapped) {
    LogCleanup(aclrtUnmapMem(state->reserved), "aclrtUnmapMem", result);
  }
  if (state->reserved != nullptr) {
    LogCleanup(aclrtReleaseMemAddress(state->reserved), "aclrtReleaseMemAddress", result);
  }
  if (state->physical != nullptr) {
    LogCleanup(aclrtFreePhysical(state->physical), "aclrtFreePhysical", result);
  }
  if (state->hbm != nullptr) {
    LogCleanup(aclrtFree(state->hbm), "aclrtFree HBM", result);
  }
  if (state->stream != nullptr) {
    LogCleanup(aclrtDestroyStream(state->stream), "aclrtDestroyStream", result);
  }
  if (state->deviceSet) {
    LogCleanup(aclrtResetDevice(state->device), "aclrtResetDevice", result);
  }
  if (state->initialized) {
    LogCleanup(aclFinalize(), "aclFinalize", result);
  }
  return result;
}

extern "C" int a5_attach(void* opaque, uint64_t hva, uint32_t flags) {
  return Guard("a5_attach", [&] {
    auto& state = Use(opaque);
    state.hva = reinterpret_cast<void*>(hva);
    Check(aclrtHostRegisterV2(state.hva, state.bytes, flags), "aclrtHostRegisterV2 mmap");
    state.registered = true;
    Check(aclrtHostGetDevicePointer(state.hva, &state.dva, 0), "aclrtHostGetDevicePointer mmap");
  });
}

extern "C" int a5_pinned(void* opaque) {
  return Guard("a5_pinned", [&] {
    auto& state = Use(opaque);
    Check(aclrtMallocHost(&state.pinned, state.bytes), "aclrtMallocHost");
    state.hva = state.pinned;
    // Already pinned by the allocator: test MAPPED alone, without re-pinning it.
    Check(aclrtHostRegisterV2(state.hva, state.bytes, ACL_HOST_REG_MAPPED), "aclrtHostRegisterV2 pinned");
    state.registered = true;
    Check(aclrtHostGetDevicePointer(state.hva, &state.dva, 0), "aclrtHostGetDevicePointer pinned");
  });
}

extern "C" int a5_bind_existing(void* opaque, uint64_t hva, uint64_t dva) {
  return Guard("a5_bind_existing", [&] {
    auto& state = Use(opaque);
    if (hva == 0 || dva == 0) {
      throw std::runtime_error("existing shared mapping has a null HVA or DVA");
    }
    // Mapping and lifetime belong to MemFabric; do not register/free it again.
    state.hva = reinterpret_cast<void*>(hva);
    state.dva = reinterpret_cast<void*>(dva);
  });
}

extern "C" int a5_bare_pid(void* opaque, int32_t* pid) {
  return Guard("a5_bare_pid", [&] {
    Use(opaque);
    Check(aclrtDeviceGetBareTgid(pid), "aclrtDeviceGetBareTgid");
  });
}

extern "C" int a5_peer(void* opaque, int peer) {
  return Guard("a5_peer", [&] {
    auto& state = Use(opaque);
    if (peer == state.device) {
      return;
    }
    int32_t supported = 0;
    Check(aclrtDeviceCanAccessPeer(&supported, state.device, peer), "aclrtDeviceCanAccessPeer");
    if (supported == 0) {
      throw std::runtime_error("peer access unavailable for device=" + std::to_string(state.device) +
                               " peer=" + std::to_string(peer));
    }
    Check(aclrtDeviceEnablePeerAccess(peer, 0), "aclrtDeviceEnablePeerAccess");
  });
}

extern "C" int a5_vmm_create(void* opaque, int numa, int huge, const int32_t* pids, size_t count, uint64_t* shared,
                             uint64_t* physicalBytes) {
  return Guard("a5_vmm_create", [&] {
    auto& state = Use(opaque);
    aclrtPhysicalMemProp property{};
    property.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    property.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    property.memAttr = huge ? ACL_MEM_P2P_HUGE : ACL_MEM_P2P_NORMAL;
    property.location.type = ACL_MEM_LOCATION_TYPE_HOST_NUMA;
    property.location.id = numa;
    size_t granularity = 0;
    Check(aclrtMemGetAllocationGranularity(&property, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &granularity),
          "aclrtMemGetAllocationGranularity HOST_NUMA");
    if (granularity == 0) {
      throw std::runtime_error("zero VMM allocation granularity");
    }
    *physicalBytes = ((state.bytes + granularity - 1) / granularity) * granularity;
    Check(aclrtMallocPhysical(&state.physical, *physicalBytes, &property, 0), "aclrtMallocPhysical HOST_NUMA");
    aclrtPhysicalMemProp actual{};
    Check(aclrtMemGetAllocationPropertiesFromHandle(state.physical, &actual), "query physical allocation properties");
    if (actual.location.type != ACL_MEM_LOCATION_TYPE_HOST_NUMA && actual.location.type != ACL_MEM_LOCATION_TYPE_HOST) {
      throw std::runtime_error("VMM allocation is not Host memory; refusing an HBM sharing false positive");
    }
    std::fprintf(stderr, "INFO VMM location_type=%d numa=%u physical_bytes=%llu\n", actual.location.type,
                 actual.location.id, static_cast<unsigned long long>(*physicalBytes));
    Check(aclrtMemExportToShareableHandleV2(state.physical, ACL_RT_VMM_EXPORT_FLAG_DEFAULT,
                                            ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, shared),
          "VMM export V2");
    Check(
        aclrtMemSetPidToShareableHandleV2(shared, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, const_cast<int32_t*>(pids), count),
        "VMM PID whitelist V2");
    MapPhysical(state, *physicalBytes);
  });
}

extern "C" int a5_vmm_import(void* opaque, uint64_t shared, uint64_t physicalBytes) {
  return Guard("a5_vmm_import", [&] {
    auto& state = Use(opaque);
    Check(aclrtMemImportFromShareableHandleV2(&shared, ACL_MEM_SHARE_HANDLE_TYPE_DEFAULT, 0, &state.physical),
          "VMM import V2");
    MapPhysical(state, physicalBytes);
  });
}

extern "C" uint64_t a5_hva(void* opaque) { return reinterpret_cast<uint64_t>(static_cast<State*>(opaque)->hva); }
extern "C" uint64_t a5_dva(void* opaque) { return reinterpret_cast<uint64_t>(static_cast<State*>(opaque)->dva); }
extern "C" uint64_t a5_hbm(void* opaque) { return reinterpret_cast<uint64_t>(Payload(*static_cast<State*>(opaque))); }

extern "C" int a5_read(void* opaque, void* output, int path) {
  return Guard("a5_read", [&] {
    auto& state = Use(opaque);
    if (path == 0) {
      // Never pass a HostGetDevicePointer result to an ACL memory-copy API.
      const void* source = state.hva != nullptr ? state.hva : state.reserved;
      Check(aclrtMemcpy(output, state.bytes, source, state.bytes,
                        state.hva != nullptr ? ACL_MEMCPY_HOST_TO_HOST : ACL_MEMCPY_DEVICE_TO_HOST),
            "runtime copy using HVA or ordinary VMM address");
      return;
    }
    ResetScratch(state);
    Check(a5_kernel_copy(state.dva, Payload(state), state.bytes / sizeof(uint32_t), path, state.stream),
          "AIV Host-to-HBM kernel and stream synchronization");
    DownloadScratch(state, output);
  });
}

extern "C" int a5_write(void* opaque, const void* input, uint64_t offset, uint64_t bytes, int path) {
  return Guard("a5_write", [&] {
    auto& state = Use(opaque);
    if (offset > state.bytes || bytes > state.bytes - offset || bytes % 32 != 0 || offset % 32 != 0) {
      throw std::runtime_error("write range invalid: offset=" + std::to_string(offset) +
                               " bytes=" + std::to_string(bytes) + " capacity=" + std::to_string(state.bytes));
    }
    if (path == 0) {
      if (state.hva != nullptr) {
        std::memcpy(static_cast<char*>(state.hva) + offset, input, bytes);
      } else {
        Check(aclrtMemcpy(static_cast<char*>(state.reserved) + offset, bytes, input, bytes, ACL_MEMCPY_HOST_TO_DEVICE),
              "initialize ordinary VMM mapping");
      }
      return;
    }
    ResetScratch(state);
    Check(aclrtMemcpy(Payload(state), state.bytes, input, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "upload write pattern");
    Check(a5_kernel_copy(Payload(state), static_cast<char*>(state.dva) + offset, bytes / sizeof(uint32_t), path,
                         state.stream),
          "AIV HBM-to-Host kernel and stream synchronization");
    std::vector<uint32_t> unused(state.bytes / sizeof(uint32_t));
    DownloadScratch(state, unused.data());
  });
}

extern "C" int a5_upload(void* opaque, const void* input) {
  return Guard("a5_upload", [&] {
    auto& state = Use(opaque);
    ResetScratch(state);
    Check(aclrtMemcpy(Payload(state), state.bytes, input, state.bytes, ACL_MEMCPY_HOST_TO_DEVICE), "upload P HBM");
  });
}
