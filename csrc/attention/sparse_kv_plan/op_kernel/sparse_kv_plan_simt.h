/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */
#ifndef VLLM_ASCEND_SPARSE_KV_PLAN_SIMT_H
#define VLLM_ASCEND_SPARSE_KV_PLAN_SIMT_H

#include <climits>
#include <cstdint>

#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "sparse_kv_plan_config.h"

namespace sparse_kv_plan_simt {

constexpr int32_t HASH_EMPTY = -1;
constexpr int32_t HASH_POSITION_EMPTY = INT32_MAX;
constexpr uint32_t HASH_BUCKET_INVALID = UINT32_MAX;
constexpr uint32_t COUNT_SHIFT = 12U;
constexpr int32_t COUNT_MASK = (1U << COUNT_SHIFT) - 1U;

template <uint32_t Threads = sparse_kv_plan_cfg::PLAN_THREADS>
inline constexpr uint32_t kWarpCount = Threads / sparse_kv_plan_cfg::PLAN_WARP_SIZE;

template <uint32_t Threads = sparse_kv_plan_cfg::PLAN_THREADS>
inline constexpr uint32_t kControlOffset = 2U * kWarpCount<Threads>;

template <uint32_t Threads = sparse_kv_plan_cfg::PLAN_THREADS>
__simt_callee__ __aicore__ inline int32_t BlockExclusiveScan(__ubuf__ int32_t* workspace, int32_t value,
                                                               uint32_t thread) {
  static_assert(Threads > 0 && Threads % sparse_kv_plan_cfg::PLAN_WARP_SIZE == 0, "whole warps required");
  static_assert(Threads < (1U << COUNT_SHIFT), "packed scan would carry");
  constexpr uint32_t warpSize = sparse_kv_plan_cfg::PLAN_WARP_SIZE;
  constexpr uint32_t warpCount = kWarpCount<Threads>;
  __ubuf__ int32_t* warpTotals = workspace;
  __ubuf__ int32_t* warpBases = workspace + warpCount;
  __ubuf__ int32_t* control = workspace + kControlOffset<Threads>;
  const uint32_t lane = thread & (warpSize - 1U);
  const uint32_t warp = thread / warpSize;
  int32_t inclusive = value;
  for (uint32_t offset = 1U; offset < warpSize; offset <<= 1U) {
    const int32_t prior = asc_shfl_up(inclusive, offset, warpSize);
    if (lane >= offset) {
      inclusive += prior;
    }
  }
  if (lane == warpSize - 1U) {
    warpTotals[warp] = inclusive;
  }
  asc_syncthreads();

  if (thread == 0U) {
    int32_t blockTotal = 0;
    for (uint32_t warpIndex = 0U; warpIndex < warpCount; ++warpIndex) {
      warpBases[warpIndex] = blockTotal;
      blockTotal += warpTotals[warpIndex];
    }
    control[0U] = blockTotal;
  }
  asc_syncthreads();

  const int32_t exclusive = warpBases[warp] + inclusive - value;
  asc_syncthreads();
  return exclusive;
}

__simt_callee__ __aicore__ inline uint32_t HashToken(int32_t token, uint32_t hashCapacity) {
  uint32_t value = static_cast<uint32_t>(token);
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return value & (hashCapacity - 1U);
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline void InsertTopkToken(ScratchPtr keys, ScratchPtr firstPositions, int32_t token,
                                                        int32_t position, uint32_t hashCapacity) {
  uint32_t bucket = HashToken(token, hashCapacity);
  const uint32_t mask = hashCapacity - 1U;
  for (uint32_t probe = 0U; probe < hashCapacity; ++probe) {
    const int32_t old = asc_atomic_cas(keys + bucket, HASH_EMPTY, token);
    if (old == HASH_EMPTY || old == token) {
      asc_atomic_min(firstPositions + bucket, position);
      return;
    }
    bucket = (bucket + 1U) & mask;
  }
}

template <typename ScratchPtr>
__simt_callee__ __aicore__ inline uint32_t LookupTopkToken(ScratchPtr keys, int32_t token, uint32_t hashCapacity) {
  uint32_t bucket = HashToken(token, hashCapacity);
  const uint32_t mask = hashCapacity - 1U;
  for (uint32_t probe = 0U; probe < hashCapacity; ++probe) {
    const int32_t key = keys[bucket];
    if (key == token) {
      return bucket;
    }
    if (key == HASH_EMPTY) {
      return HASH_BUCKET_INVALID;
    }
    bucket = (bucket + 1U) & mask;
  }
  return HASH_BUCKET_INVALID;
}

template <uint32_t Threads = sparse_kv_plan_cfg::PLAN_THREADS, typename EvictPtr, typename HitPtr>
__simt_callee__ __aicore__ inline int32_t PartitionSlots(__ubuf__ int32_t* scanWorkspace, uint32_t thread, int32_t slot,
                                                       int32_t evictFlag, int32_t hitFlag, EvictPtr evictSlots,
                                                       HitPtr hitSlots, int32_t evictCount, int32_t hitCount) {
  const int32_t packed = evictFlag + (hitFlag << COUNT_SHIFT);
  const int32_t ranks = BlockExclusiveScan<Threads>(scanWorkspace, packed, thread);
  __ubuf__ int32_t* control = scanWorkspace + kControlOffset<Threads>;
  const int32_t totals = control[0U];
  const int32_t evictRank = ranks & COUNT_MASK;
  const int32_t hitRank = ranks >> COUNT_SHIFT;
  if (evictFlag != 0) {
    evictSlots[evictCount + evictRank] = slot;
  }
  if (hitFlag != 0) {
    hitSlots[hitCount + hitRank] = slot;
  }
  asc_syncthreads();
  return totals;
}

template <uint32_t Threads = sparse_kv_plan_cfg::PLAN_THREADS, typename EvictPtr, typename HitPtr, typename MissPtr,
          typename LruPtr>
__simt_callee__ __aicore__ inline void MergeLru(EvictPtr evictSlots, HitPtr hitSlots, MissPtr missSlots,
                                                LruPtr lruOutput, int32_t evictCount, int32_t hitCount,
                                                int32_t assignCount, uint32_t thread) {
  const int32_t unassignedCount = evictCount - assignCount;
  const int32_t validLruCount = evictCount + hitCount;
  for (uint32_t output = thread; output < static_cast<uint32_t>(validLruCount); output += Threads) {
    int32_t slot;
    if (output < static_cast<uint32_t>(unassignedCount)) {
      slot = evictSlots[assignCount + output];
    } else if (output < static_cast<uint32_t>(evictCount)) {
      slot = missSlots[output - unassignedCount];
    } else {
      slot = hitSlots[output - evictCount];
    }
    lruOutput[output] = slot;
  }
  asc_syncthreads();
}

}

#endif
