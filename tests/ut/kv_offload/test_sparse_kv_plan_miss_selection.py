import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KERNEL_DIR = ROOT / "csrc/attention/sparse_kv_plan/op_kernel"
SHARED = KERNEL_DIR / "sparse_kv_plan_shared.h"


def _write_ascend_stubs(include_dir: Path) -> None:
    (include_dir / "simt_api").mkdir(parents=True)
    (include_dir / "kernel_operator.h").write_text(
        """#include <cstdint>
#define __gm__
#define __ubuf__
#define __simt_callee__
#define __simt_vf__
#define __aicore__
#define LAUNCH_BOUND(x)
struct ThreadIdxStub { uint32_t x; };
inline ThreadIdxStub threadIdx{0};
"""
    )
    (include_dir / "simt_api/asc_simt.h").write_text(
        """#include <cstdint>
inline int32_t asc_shfl_up(int32_t value, uint32_t, uint32_t) { return value; }
"""
    )
    (include_dir / "simt_api/device_atomic_functions.h").write_text(
        """#include <cstdint>
template <typename T> inline T asc_atomic_cas(T* value, T expected, T desired) {
  const T old = *value; if (old == expected) *value = desired; return old;
}
template <typename T> inline T asc_atomic_min(T* value, T other) {
  const T old = *value; if (other < *value) *value = other; return old;
}
template <typename T> inline T asc_atomic_max(T* value, T other) {
  const T old = *value; if (other > *value) *value = other; return old;
}
"""
    )
    (include_dir / "simt_api/device_sync_functions.h").write_text("inline void asc_syncthreads() {}\n")
    (include_dir / "simt_api/device_warp_functions.h").write_text("")


def _phase6_candidate() -> str:
    # Execute the kernel predicate itself; CPU stubs do not model SIMT synchronization.
    source = SHARED.read_text()
    phase6 = source[source.index("int32_t localMissCount = 0;") :]
    body = phase6[phase6.index("    if (position < topk) {") :]
    return body[: body.index("    const int32_t rank = BlockExclusiveScan")]


def _cold_miss_candidate() -> str:
    source = SHARED.read_text()
    cold = source[source.index("if (control[CONTROL_RESET] != 0) {") :]
    body = cold[cold.index("      if (position < topk &&") :]
    return body[: body.index("      const int32_t rank = BlockExclusiveScan")]


def _compile_and_run(tmp_path: Path, candidate: str) -> str:
    cold_candidate = _cold_miss_candidate()
    include_dir = tmp_path / "ascend_stubs"
    _write_ascend_stubs(include_dir)
    source = tmp_path / "duplicate_miss.cpp"
    source.write_text(
        f"""#include <climits>
#include <cstdint>
#include <iostream>
#include <vector>
#include \"{SHARED}\"

int EvaluateCandidate(int32_t* currentSlots, int32_t* hashFirstPos,
                      int32_t* missPositions, int32_t* hashOwner,
                      int32_t* lruSlots, uint32_t capacity,
                      uint32_t position, uint32_t topk) {{
  const int64_t topkBase = 0;
  const int64_t capacityBase = 0;
  int32_t missFlag = 0;
{candidate}
  return missFlag;
}}

int EvaluateColdCandidate(int32_t* missPositions, int32_t* hashFirstPos, uint32_t position, uint32_t topk) {{
  int32_t flag = 0;
{cold_candidate}
  return flag;
}}

int main() {{
  constexpr uint32_t topk = 4096;
  constexpr uint32_t hashCapacity = 8192;
  std::vector<int32_t> keys(hashCapacity, -1);
  std::vector<int32_t> first(hashCapacity, INT32_MAX);
  std::vector<int32_t> indices(topk, -1);
  std::vector<int32_t> current(topk, -1);
  std::vector<int32_t> visible(1, 4096);
  std::vector<int32_t> requests(1, 0);
  std::vector<int32_t> blocks(4096, 0);
  std::vector<int32_t> cached(topk, -1);
  std::vector<int32_t> owners(hashCapacity, -1);
  indices[0] = 3; indices[1] = 3; indices[2] = 1; indices[3] = 2;
  // Repeated tokens straddle both SIMTv3 tile boundaries (1023/1024, 2047/2048).
  indices[1023] = 3; indices[1024] = 8; indices[2047] = 8; indices[2048] = 8; indices[2049] = 4;
  indices[2050] = 4; indices[2051] = 5; indices[2052] = 6;
  indices[3000] = 7;
  blocks[7] = -1;
  for (int32_t position = static_cast<int32_t>(topk) - 1; position >= 0; --position) {{
    if (IsSourceTokenValid(0, indices[position], visible.data(), requests.data(), blocks.data(),
                           4096, 1, 4096, 4096, 1)) {{
      cached[position] = static_cast<int32_t>(
          InsertTopkToken(keys.data(), first.data(), indices[position], position, hashCapacity));
    }}
  }}
  const std::vector<uint32_t> expectedCold{{0, 2, 3, 1024, 2049, 2051, 2052}};
  const std::vector<uint32_t> expectedResident{{2, 3, 1024, 2049, 2051, 2052}};
  std::vector<uint32_t> coldMisses;
  for (uint32_t position = 0; position < topk; ++position) {{
    if (EvaluateColdCandidate(cached.data(), first.data(), position, topk) != 0) coldMisses.push_back(position);
  }}
  if (coldMisses != expectedCold) return 1;
  for (uint32_t position : coldMisses) std::cout << position << " ";
  std::cout << "\\n";
  for (uint32_t capacity : {{2048U, 32768U, 32769U}}) {{
    std::vector<int32_t> lruSlots(capacity, -1);
    const uint32_t order = capacity - 1U;
    const uint32_t slot = capacity - 2U;
    lruSlots[order] = static_cast<int32_t>(slot);
    owners[cached[0]] = static_cast<int32_t>(capacity <= OWNER_PACK_CAPACITY
                                              ? (order << OWNER_SLOT_BITS) | slot : order);
    current.assign(topk, -1);
    std::vector<uint32_t> misses;
    for (uint32_t position = 0; position < topk; ++position) {{
      if (EvaluateCandidate(current.data(), first.data(), cached.data(), owners.data(),
                            lruSlots.data(), capacity, position, topk) != 0) {{
        misses.push_back(position);
      }}
    }}
    if (misses != expectedResident || current[0] != static_cast<int32_t>(slot)) return 2;
    for (uint32_t position = 1; position < topk; ++position) {{
      if (current[position] != -1) return 3;
    }}
    if (capacity == 2048U) {{
      for (uint32_t position : misses) std::cout << position << " ";
      std::cout << "\\n";
    }}
  }}
  return 0;
}}
"""
    )
    compiler = shutil.which("clang++") or shutil.which("g++")
    assert compiler is not None, "a C++17 compiler is required"
    binary = tmp_path / "duplicate_miss"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(include_dir),
            "-I",
            str(KERNEL_DIR),
            str(source),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], check=False, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, f"miss positions: {result.stdout}\n{result.stderr}"
    return result.stdout


def test_phase6_deduplicates_cold_misses_across_topk_tiles(tmp_path: Path) -> None:
    candidate = _phase6_candidate()
    # Both cold and warm predicates are compiled from the production kernel; no SIMT emulator is involved.
    assert _compile_and_run(tmp_path, candidate) == "0 2 3 1024 2049 2051 2052 \n2 3 1024 2049 2051 2052 \n"
