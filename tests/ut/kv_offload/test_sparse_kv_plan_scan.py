"""Execute the production second-level scan with a CPU warp shuffle model."""

import shutil
import subprocess
from pathlib import Path

from tests.ut.kv_offload.test_sparse_kv_plan_miss_selection import KERNEL_DIR, SHARED, _write_ascend_stubs


def test_parallel_warp_bases_match_serial_prefix_for_packed_counts(tmp_path: Path) -> None:
    # Given one SIMTv3 block's warp totals, including all-hit/all-evict and partial tiles.
    include_dir = tmp_path / "ascend_stubs"
    _write_ascend_stubs(include_dir)
    (include_dir / "simt_api/asc_simt.h").write_text(
        """#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>

class WarpBarrier {
 public:
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const uint32_t generation = generation_;
    if (++arrived_ == 32U) {
      arrived_ = 0U;
      ++generation_;
      condition_.notify_all();
    } else {
      condition_.wait(lock, [&] { return generation_ != generation; });
    }
  }
 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  uint32_t arrived_ = 0U;
  uint32_t generation_ = 0U;
};
inline WarpBarrier shuffleBarrier;
inline std::array<int32_t, 32> shuffleValues{};
inline thread_local uint32_t shuffleLane = 0U;
inline int32_t asc_shfl_up(int32_t value, uint32_t offset, uint32_t) {
  shuffleValues[shuffleLane] = value;
  shuffleBarrier.Wait();
  const int32_t prior = shuffleValues[shuffleLane >= offset ? shuffleLane - offset : shuffleLane];
  shuffleBarrier.Wait();
  return prior;
}
"""
    )
    source = tmp_path / "scan.cpp"
    source.write_text(
        f"""#include <array>
#include <iostream>
#include <thread>
#include "{SHARED}"

int main() {{
  static_assert(PLAN_THREADS == 1024U && PLAN_WARP_COUNT == 32U);
  for (uint32_t pattern = 0; pattern < 12U; ++pattern) {{
    std::array<int32_t, PLAN_SCAN_STORAGE_ELEMENTS> workspace{{}};
    std::array<int32_t, PLAN_WARP_COUNT> expected{{}};
    int32_t total = 0;
    for (uint32_t warp = 0; warp < PLAN_WARP_COUNT; ++warp) {{
      uint32_t evict = 0U;
      uint32_t hit = 0U;
      if (pattern == 1U) evict = 32U;
      if (pattern == 2U) hit = 32U;
      if (pattern > 2U && warp < 35U - pattern) {{
        evict = (warp * 17U + pattern) % 33U;
        hit = (warp * 13U + pattern) % (33U - evict);
      }}
      workspace[WARP_TOTALS_OFFSET + warp] = static_cast<int32_t>(evict + (hit << COUNT_SHIFT));
      expected[warp] = total;
      total += workspace[WARP_TOTALS_OFFSET + warp];
    }}
    std::array<std::thread, PLAN_WARP_COUNT> lanes;
    for (uint32_t lane = 0; lane < PLAN_WARP_COUNT; ++lane) {{
      lanes[lane] = std::thread([&, lane] {{
        shuffleLane = lane;
        PublishWarpBases(workspace.data(), lane);
      }});
    }}
    for (auto& lane : lanes) lane.join();
    for (uint32_t warp = 0; warp < PLAN_WARP_COUNT; ++warp) {{
      if (workspace[WARP_BASES_OFFSET + warp] != expected[warp]) {{
        std::cerr << "wrong prefix pattern=" << pattern << " warp=" << warp;
        return 1;
      }}
    }}
    if (workspace[CONTROL_OFFSET + CONTROL_TOTAL] != total) return 2;
    const auto completed = workspace;
    PublishWarpBases(workspace.data(), 32U);
    if (workspace != completed) return 3;
  }}
}}
"""
    )
    compiler = shutil.which("clang++") or shutil.which("g++")
    assert compiler is not None, "a C++17 compiler is required"
    binary = tmp_path / "scan"
    # When the actual kernel helper runs, only shuffles are replaced by synchronized CPU lanes.
    compiled = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
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
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False, timeout=20)
    # Then all stable prefix ranks and the packed block total equal the serial reference.
    assert result.returncode == 0, result.stdout + result.stderr
