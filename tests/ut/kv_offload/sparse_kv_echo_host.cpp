// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// CPU execution model for the actual SIMT row code. This is not an NPU simulator.
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <thread>
#include <vector>

#define __gm__
#define __ubuf__
#define __aicore__
#define __simt_callee__
#define __simt_vf__
#define LAUNCH_BOUND(n)
#include "sparse_kv_echo_plan_config.h"
constexpr int kThreads = sparse_kv_echo_plan::PLAN_THREADS;
thread_local struct {
  unsigned x;
} threadIdx;
std::barrier syncPoint(kThreads);
int32_t shuffleValues[kThreads];
int64_t downValues[kThreads];
uint64_t blockBarriers = 0;
std::atomic<uint64_t> mapAtomics{0}, histogramAtomics{0};
void asc_syncthreads() {
  if (threadIdx.x == 0) ++blockBarriers;
  syncPoint.arrive_and_wait();
}
int32_t asc_shfl_up(int32_t value, unsigned delta, unsigned width) {
  shuffleValues[threadIdx.x] = value;
  syncPoint.arrive_and_wait();
  const unsigned lane = threadIdx.x % width;
  const int32_t result = shuffleValues[threadIdx.x - (lane >= delta ? delta : 0)];
  syncPoint.arrive_and_wait();
  return result;
}
int64_t asc_shfl_down(int64_t value, unsigned delta, unsigned width) {
  downValues[threadIdx.x] = value;
  syncPoint.arrive_and_wait();
  const unsigned lane = threadIdx.x % width;
  const int64_t result = downValues[threadIdx.x + (lane + delta < width ? delta : 0)];
  syncPoint.arrive_and_wait();
  return result;
}
int32_t asc_atomic_add(int32_t* ptr, int32_t value) {
  ++histogramAtomics;
  return std::atomic_ref<int32_t>(*ptr).fetch_add(value);
}
template <typename T>
T asc_atomic_exch(T* ptr, T value) {
  ++mapAtomics;
  return std::atomic_ref<T>(*ptr).exchange(value);
}
template <typename T>
T asc_atomic_max(T* ptr, T value) {
  ++mapAtomics;
  std::atomic_ref<T> atomic(*ptr);
  T old = atomic.load();
  while (old < value && !atomic.compare_exchange_weak(old, value)) {
  }
  return old;
}
#include "echo_kernel_body.h"

void RunCase(int capacity, int topk, int maxToken, int rows, int rounds, bool mtp = false) {
  constexpr int blockSize = 16;
  constexpr int guard = 16;
  constexpr int32_t poison = 0x35465768;
  const int requests = 3;
  const int blocks = (maxToken + blockSize - 1) / blockSize;
  const int grid = std::min(rows, 2);
  const auto stride = sparse_kv_echo_plan::WorkspaceBlockElements(maxToken, topk, capacity);
  std::mt19937 rng(712 + capacity);
  std::vector<int64_t> reqIds{31, 32, 33}, last(rows, -1), priority(rows * (capacity + 1), -1);
  std::vector<int32_t> slots(rows * capacity, -1), indices(rows * topk), stable(rows), visible(rows), tokenReq(rows);
  std::vector<int32_t> table(requests * blocks), current(rows * topk, -99), misses(rows, -99);
  std::vector<int32_t> missTokens(rows * topk, -99), missSlots(rows * topk, -99);
  std::vector<int32_t> scratch(grid * stride + 2 * guard, poison);
  const int localElements = sparse_kv_echo_plan::LocalMemoryBytes(capacity, maxToken) / sizeof(int32_t);
  std::vector<int32_t> scan(localElements + 2 * guard, poison);
  // Two different layers replay the owner's descriptors into different payloads.
  std::vector<int64_t> host(requests * maxToken), resident(rows * capacity, -1), mirrored(rows * capacity, -1);
  for (int i = 0; i < requests * maxToken; ++i) host[i] = i + 1;
  for (int step = 0; step < rounds; ++step) {
    int active = step % 5 == 4 ? rows - 1 : rows;
    for (int r = 0; r < requests; ++r) {
      for (int b = 0; b < blocks; ++b) table[r * blocks + b] = b;
    }
    if (step % 3 == 2 && blocks > 1) table[blocks + 1] = -1;
    if (step == 9) table[2 * blocks] = blocks;
    if (step == 3) reqIds[0] += 100;
    for (int r = 0; r < rows; ++r) {
      tokenReq[r] = r % requests;
      if (step % 6 == 5 && r == 0) tokenReq[r] = -1;
      // Multiple MTP rows can share a request, with distinct visible lengths.
      if (step == 4) tokenReq[r] = 0;
      visible[r] = maxToken - (step == 4 ? rows - 1 - r : 0);
      stable[r] = step % 3 == 1 ? maxToken / 2 : maxToken;
      for (int t = 0; t < topk; ++t) {
        int token = step == 0 ? t % maxToken : rng() % (maxToken + 3);
        if (t % 7 == 1) token = indices[r * topk + t - 1];
        if (t % 19 == 3) token = -1;
        if (step != 2) indices[r * topk + t] = token;
      }
    }
    if (step == 7) std::fill(stable.begin(), stable.end(), 0);
    if (step == 8) {
      for (int r = 0; r < rows; ++r) tokenReq[r] = (r + 1) % requests;
    }
    if (step == 10) tokenReq[0] = requests;
    if (mtp) {
      // Draft [100,101,102] is verified, then only position 100 is accepted.
      // Positions >=101 are rewritten on the next iteration. All rows must
      // reload this suffix and mask positions beyond their own query.
      for (int r = 0; r < rows; ++r) {
        tokenReq[r] = 0;
        stable[r] = step == 0 ? 100 : 101;
        visible[r] = stable[r] + r + 1;
        for (int t = 0; t < topk; ++t) indices[r * topk + t] = 99 + t;
      }
      if (step == 1) {
        for (int t = 101; t < maxToken; ++t) host[t] += 9000000;
      }
    }
    if (step == 1 && !mtp) {
      for (int r = 0; r < requests; ++r)
        for (int t = maxToken / 2; t < maxToken; ++t) host[r * maxToken + t] += 1000000;
    }
    if (step == 6) {
      for (int r = 0; r < rows; ++r) priority[r * (capacity + 1) + capacity] = int64_t{1} << 33;
    }
    const auto oldSlots = slots;
    const auto oldPriority = priority;
    const auto oldLast = last;
    const auto oldCurrent = current;
    // Any scratch left by previous rows/layers is legal, including stale
    // negative TopK markers. The inverse check must reject false hits.
    std::fill(scratch.begin() + guard, scratch.end() - guard, -3);
    std::fill(scan.begin() + guard, scan.end() - guard, -3);
    mapAtomics = 0;
    histogramAtomics = 0;
    for (int b = 0; b < grid; ++b) {
      std::vector<std::thread> workers;
      for (int tid = 0; tid < kThreads; ++tid) {
        workers.emplace_back([&, tid, b] {
          threadIdx.x = tid;
          SparseKvEchoPlanVf(scan.data() + guard, reqIds.data(), last.data(), indices.data(), stable.data(),
                             visible.data(), tokenReq.data(), table.data(), slots.data(), priority.data(),
                             current.data(), misses.data(), missTokens.data(), missSlots.data(), scratch.data() + guard,
                             stride, b, grid, active, topk, capacity, maxToken, requests, blocks, blocks, blockSize);
        });
      }
      for (auto& worker : workers) worker.join();
    }
    for (int i = 0; i < guard; ++i) {
      assert(scratch[i] == poison && scratch[scratch.size() - 1 - i] == poison);
      assert(scan[i] == poison && scan[scan.size() - 1 - i] == poison);
    }
    uint64_t validInputs = 0;
    for (int r = 0; r < active; ++r) {
      const auto req = tokenReq[r];
      const auto id = req >= 0 && req < requests ? reqIds[req] : -1;
      const auto clock = oldLast[r] != id ? 0 : oldPriority[r * (capacity + 1) + capacity];
      std::vector<int32_t> before(oldSlots.begin() + r * capacity, oldSlots.begin() + (r + 1) * capacity);
      std::vector<int64_t> ages(oldPriority.begin() + r * (capacity + 1),
                                oldPriority.begin() + r * (capacity + 1) + capacity);
      for (int s = 0; s < capacity; ++s) {
        if (oldLast[r] != id || before[s] < 0 || before[s] >= stable[r]) {
          before[s] = -1;
          ages[s] = -1;
        }
      }
      std::set<int> selected;
      std::vector<int> expectedMiss;
      for (int t = 0; t < topk; ++t) {
        const int token = indices[r * topk + t];
        const bool valid = req >= 0 && req < requests && token >= 0 && token < maxToken && token < visible[r] &&
                           table[req * blocks + token / blockSize] >= 0 &&
                           table[req * blocks + token / blockSize] < blocks;
        if (!valid) {
          assert(current[r * topk + t] == -1);
          continue;
        }
        ++validInputs;
        if (!selected.insert(token).second) continue;
        auto found = std::find(before.begin(), before.end(), token);
        if (found == before.end())
          expectedMiss.push_back(token);
        else
          ages[found - before.begin()] = clock + 1;
      }
      assert(misses[r] == static_cast<int>(expectedMiss.size()));
      if (step == 2) assert(misses[r] == 0);
      std::vector<int> evictionOrder(capacity);
      std::iota(evictionOrder.begin(), evictionOrder.end(), 0);
      std::sort(evictionOrder.begin(), evictionOrder.end(),
                [&](int a, int b) { return ages[a] < ages[b] || (ages[a] == ages[b] && a < b); });
      std::set<int> expectedVictims(evictionOrder.begin(), evictionOrder.begin() + expectedMiss.size()), actualVictims;
      for (int m = 0; m < topk; ++m) {
        const int token = missTokens[r * topk + m], slot = missSlots[r * topk + m];
        if (m >= misses[r]) {
          assert(token == -1 && slot == -1);
          continue;
        }
        assert(token == expectedMiss[m] && slot >= 0 && slot < capacity);
        actualVictims.insert(slot);
        before[slot] = token;
        ages[slot] = clock + 2;
        resident[r * capacity + slot] = host[req * maxToken + token];
        mirrored[r * capacity + slot] = host[req * maxToken + token] * 7;
      }
      assert(actualVictims == expectedVictims);
      assert(std::equal(before.begin(), before.end(), slots.begin() + r * capacity));
      assert(std::equal(ages.begin(), ages.end(), priority.begin() + r * (capacity + 1)));
      assert(last[r] == id && priority[r * (capacity + 1) + capacity] == clock + 2);
      for (int t = 0; t < topk; ++t) {
        const int slot = current[r * topk + t];
        if (slot < 0) continue;
        const int token = indices[r * topk + t];
        assert(slots[r * capacity + slot] == token);
        assert(resident[r * capacity + slot] == host[req * maxToken + token]);
        assert(mirrored[r * capacity + slot] == host[req * maxToken + token] * 7);
      }
    }
    assert(mapAtomics == 2 * validInputs);
    if (step == 2) assert(histogramAtomics == 0);
    for (int r = active; r < rows; ++r) {
      assert(std::equal(oldSlots.begin() + r * capacity, oldSlots.begin() + (r + 1) * capacity,
                        slots.begin() + r * capacity));
      assert(
          std::equal(oldCurrent.begin() + r * topk, oldCurrent.begin() + (r + 1) * topk, current.begin() + r * topk));
      assert(last[r] == oldLast[r]);
    }
  }
}
void RunSelection(const std::vector<int64_t>& priority, int misses, int64_t hitPriority, bool report = false) {
  constexpr int guard = 16;
  constexpr int poison = 0x65656565;
  const int capacity = priority.size();
  const int bytes = sparse_kv_echo_plan::LocalMemoryBytes(capacity, 1);
  std::vector<int32_t> local(bytes / sizeof(int32_t) + 2 * guard, poison);
  std::vector<int32_t> victims(capacity + 2 * guard, poison);
  auto input = priority;
  EchoRow<int32_t*> row{nullptr,
                        input.data(),
                        nullptr,
                        nullptr,
                        victims.data() + guard,
                        nullptr,
                        nullptr,
                        nullptr,
                        local.data() + guard,
                        static_cast<uint32_t>(capacity),
                        0};
  histogramAtomics = 0;
  blockBarriers = 0;
  std::vector<std::thread> workers;
  for (int tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      threadIdx.x = tid;
      SelectVictims(row, misses, hitPriority, tid);
    });
  }
  for (auto& worker : workers) worker.join();
  std::vector<int> order(capacity);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return priority[a] < priority[b] || (priority[a] == priority[b] && a < b); });
  assert(misses > 0 && priority[order[misses - 1]] < hitPriority);
  const int64_t threshold = priority[order[misses - 1]];
  std::vector<int> expected;
  for (int slot = 0; slot < capacity; ++slot)
    if (priority[slot] < threshold) expected.push_back(slot);
  for (int slot = 0; slot < capacity && static_cast<int>(expected.size()) < misses; ++slot)
    if (priority[slot] == threshold) expected.push_back(slot);
  assert(std::equal(expected.begin(), expected.end(), victims.begin() + guard));
  assert(input == priority);
  assert(histogramAtomics <= 8U * capacity);  // At most eight byte passes, independent of clock magnitude.
  for (int i = 0; i < guard; ++i) {
    assert(local[i] == poison && local[local.size() - 1 - i] == poison);
    assert(victims[i] == poison && victims[victims.size() - 1 - i] == poison);
  }
  if (report) {
    std::cout << "Selection logical counts: capacity=" << capacity << " misses=" << misses
              << " barriers=" << blockBarriers << " UB_histogram_atomics=" << histogramAtomics
              << " (host model threads=" << kThreads << ", not NPU timings)\n";
  }
}

void RunSelectionCases() {
  std::mt19937_64 rng(7312);
  for (int pattern = 0; pattern < 7; ++pattern) {
    std::vector<int64_t> priority(67);
    const int64_t high = (int64_t{1} << 60);
    for (int s = 0; s < 67; ++s) {
      switch (pattern) {
        case 0:
          priority[s] = -1;
          break;
        case 1:
          priority[s] = 255;
          break;
        case 2:
          priority[s] = s % 4 == 0 ? -1 : 255 + s % 3;
          break;
        case 3:
          priority[s] = high + (rng() & 65535);
          break;
        case 4:
          priority[s] = (int64_t{1} << (8 * (s % 8))) + s;
          break;
        case 5:
          priority[s] = INT64_MAX - 10000 + s;
          break;
        default:
          priority[s] = rng() % high;
          break;
      }
    }
    for (int misses : {1, 19, 67}) RunSelection(priority, misses, INT64_MAX);
    // An exact boundary cannot select protected hits, including when topk==capacity.
    priority[0] = INT64_MAX;
    RunSelection(priority, 66, INT64_MAX);
  }
  const int boundary = sparse_kv_echo_plan::PLAN_PRIORITY_UB_LIMIT_BYTES / sizeof(int64_t);
  for (int capacity : {boundary, boundary + 1}) {
    std::vector<int64_t> priority(capacity);
    for (int s = 0; s < capacity; ++s) priority[s] = (capacity - s) % 701;
    RunSelection(priority, 37, 1000);
  }
  std::vector<int64_t> priority(4096, 20001);
  for (int s = 0; s < 2150; ++s) priority[s] = 20000 - 4096 + s;
  RunSelection(priority, 102, 20001, true);
}

int main() {
  RunSelectionCases();
  RunCase(8, 5, 128, 3, 2, true);
  RunCase(1, 1, 5, 1, 8);
  RunCase(37, 23, 131, 5, 12);
  RunCase(64, 32, 17, 2, 8);
  RunCase(512, 256, 1024, 3, 8);
  RunCase(4096, 2048, 8192, 1, 4);
  RunCase(8192, 4096, 16384, 1, 3);
  RunCase(16384, 256, 32768, 1, 3);  // Both the map and priority snapshot use GM.
  RunCase(16384, 32, 17, 1, 3);      // UB map, GM priority snapshot.
  RunCase(37, 23, 262144, 5, 12);    // GM map reused by multiple rows/layers.
  std::cout << "ECHO row code: all shapes, eviction sets, payloads and guards passed (" << kThreads
            << " CPU threads)\n";
}
