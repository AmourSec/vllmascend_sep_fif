/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * ECHO-style direct lookup and priority eviction, adapted to per-query rows.
 * The block scan and mixed SIMD/SIMT entry derive from SparseKvPlan.
 * Unlike ECHO's persistent per-layer reverse map, the reverse map here is
 * scratch: rebuild touched entries from slotToToken, then reuse across layers.
 * All capacity-dependent metadata stays in GM; UB contains only the scan.
 */
#include <climits>
#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "sparse_kv_echo_plan_config.h"

namespace {
using sparse_kv_echo_plan::PLAN_THREADS;
using sparse_kv_echo_plan::PLAN_WARP_COUNT;
using sparse_kv_echo_plan::PLAN_WARP_SIZE;
constexpr int32_t INVALID_TOKEN = -1;
constexpr int32_t MAP_EMPTY = INT32_MIN;
constexpr uint32_t CONTROL_OFFSET = 2U * PLAN_WARP_COUNT;
constexpr uint32_t COUNT_SHIFT = 12U;
constexpr int32_t COUNT_MASK = (1U << COUNT_SHIFT) - 1U;
static_assert(PLAN_THREADS < (1U << COUNT_SHIFT), "packed scan would carry");

struct SparseKvEchoPlanTilingData {
  uint64_t workspaceBlockElements;
  int64_t maxRows;
  int64_t topk;
  int64_t capacity;
  int64_t maxToken;
  int64_t maxRequests;
  int64_t maxNumBlocks;
  int64_t hostNumBlocks;
  int32_t blockSize;
  uint32_t localMemoryBytes;
};

__simt_callee__ __aicore__ inline int32_t BlockExclusiveScan(__ubuf__ int32_t* workspace, int32_t value,
                                                             uint32_t thread) {
  __ubuf__ int32_t* warpBases = workspace + PLAN_WARP_COUNT;
  const uint32_t lane = thread % PLAN_WARP_SIZE;
  const uint32_t warp = thread / PLAN_WARP_SIZE;
  int32_t inclusive = value;
  for (uint32_t offset = 1U; offset < PLAN_WARP_SIZE; offset <<= 1U) {
    const int32_t prior = asc_shfl_up(inclusive, offset, PLAN_WARP_SIZE);
    if (lane >= offset) {
      inclusive += prior;
    }
  }
  if (lane == PLAN_WARP_SIZE - 1U) {
    workspace[warp] = inclusive;
  }
  asc_syncthreads();
  if (thread == 0U) {
    int32_t total = 0;
    for (uint32_t i = 0U; i < PLAN_WARP_COUNT; ++i) {
      warpBases[i] = total;
      total += workspace[i];
    }
    workspace[CONTROL_OFFSET] = total;
  }
  asc_syncthreads();
  const int32_t exclusive = warpBases[warp] + inclusive - value;
  asc_syncthreads();
  return exclusive;
}

__simt_callee__ __aicore__ inline bool IsSourceTokenValid(int64_t row, int32_t token, __gm__ int32_t* visibleSeqLens,
                                                          __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable,
                                                          int32_t maxToken, int64_t maxRequests, int64_t maxNumBlocks,
                                                          int64_t hostNumBlocks, int32_t blockSize) {
  if (token < 0 || token >= maxToken || token >= visibleSeqLens[row]) {
    return false;
  }
  const int32_t request = tokenToReq[row];
  if (request < 0 || request >= maxRequests) {
    return false;
  }
  const int64_t blockId = token / blockSize;
  if (blockId >= maxNumBlocks) {
    return false;
  }
  const int32_t physicalBlock = blockTable[static_cast<int64_t>(request) * maxNumBlocks + blockId];
  return physicalBlock >= 0 && physicalBlock < hostNumBlocks;
}

// All pointers refer to one query row, except tokenMap which belongs to the AIV.
struct EchoRow {
  __gm__ int32_t* slotToToken;
  __gm__ int64_t* priority;
  __gm__ int32_t* tokens;
  __gm__ int32_t* tokenMap;
  __gm__ int32_t* victims;
  __gm__ int32_t* missPositions;
  __gm__ int32_t* missTokens;
  __gm__ int32_t* missSlots;
  __ubuf__ int32_t* scan;
  uint32_t capacity;
  uint32_t topk;
};

__simt_callee__ __aicore__ inline void ResetResident(const EchoRow& row, bool reset, int32_t stablePrefix,
                                                     int32_t maxToken, uint32_t thread) {
  for (uint32_t slot = thread; slot < row.capacity; slot += PLAN_THREADS) {
    int32_t token = row.slotToToken[slot];
    if (reset || token < 0 || token >= stablePrefix || token >= maxToken) {
      row.slotToToken[slot] = INVALID_TOKEN;
      row.priority[slot] = -1;
    } else {
      // Repeated TopK/resident tokens may touch the same map entry.
      asc_atomic_exch(row.tokenMap + token, MAP_EMPTY);
    }
  }
}

__simt_callee__ __aicore__ inline int32_t MarkAndCompactMisses(const EchoRow& row, int64_t hitPriority,
                                                               uint32_t thread) {
  for (uint32_t slot = thread; slot < row.capacity; slot += PLAN_THREADS) {
    const int32_t token = row.slotToToken[slot];
    if (token >= 0) {
      asc_atomic_max(row.tokenMap + token, static_cast<int32_t>(slot));
    }
  }
  asc_syncthreads();
  for (uint32_t pos = thread; pos < row.topk; pos += PLAN_THREADS) {
    const int32_t token = row.tokens[pos];
    if (token >= 0) {
      // Resident slots are nonnegative. Negative values elect the first TopK
      // occurrence of a missing token, without a hash table or duplicate copies.
      asc_atomic_max(row.tokenMap + token, -static_cast<int32_t>(pos) - 1);
    }
  }
  asc_syncthreads();
  int32_t misses = 0;
  for (uint32_t tile = 0U; tile < row.topk; tile += PLAN_THREADS) {
    const uint32_t pos = tile + thread;
    int32_t missing = 0;
    if (pos < row.topk && row.tokens[pos] >= 0) {
      const int32_t mapped = row.tokenMap[row.tokens[pos]];
      if (mapped >= 0) {
        asc_atomic_exch(row.priority + mapped, hitPriority);
      } else {
        missing = mapped == -static_cast<int32_t>(pos) - 1;
      }
    }
    const int32_t rank = BlockExclusiveScan(row.scan, missing, thread);
    const int32_t total = row.scan[CONTROL_OFFSET];
    if (missing != 0) {
      row.missPositions[misses + rank] = static_cast<int32_t>(pos);
    }
    asc_syncthreads();
    misses += total;
  }
  return misses;
}

__simt_callee__ __aicore__ inline int32_t CountAtMost(const EchoRow& row, int64_t bound, uint32_t thread) {
  int32_t count = 0;
  for (uint32_t slot = thread; slot < row.capacity; slot += PLAN_THREADS) {
    count += row.priority[slot] <= bound;
  }
  BlockExclusiveScan(row.scan, count, thread);
  const int32_t total = row.scan[CONTROL_OFFSET];
  asc_syncthreads();
  return total;
}

__simt_callee__ __aicore__ inline int64_t EvictionThreshold(const EchoRow& row, int32_t misses, int64_t hitPriority,
                                                            uint32_t thread) {
  if (CountAtMost(row, -1, thread) >= misses) {
    return -1;
  }
  // Select the kth-smallest priority without sorting the whole cache. Hits
  // have hitPriority, so capacity >= topk keeps every selected hit protected.
  int64_t low = 0;
  int64_t high = hitPriority - 1;
  while (low < high) {
    const int64_t mid = low + (high - low) / 2;
    if (CountAtMost(row, mid, thread) >= misses) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return low;
}

__simt_callee__ __aicore__ inline void CollectVictims(const EchoRow& row, int64_t threshold, int32_t misses,
                                                      uint32_t thread) {
  const int32_t smallerTotal = CountAtMost(row, threshold - 1, thread);
  int32_t smallerBase = 0;
  int32_t equalBase = 0;
  for (uint32_t tile = 0U; tile < row.capacity; tile += PLAN_THREADS) {
    const uint32_t slot = tile + thread;
    const int32_t smaller = slot < row.capacity && row.priority[slot] < threshold;
    const int32_t equal = slot < row.capacity && row.priority[slot] == threshold;
    const int32_t rank = BlockExclusiveScan(row.scan, smaller + (equal << COUNT_SHIFT), thread);
    const int32_t totals = row.scan[CONTROL_OFFSET];
    if (smaller != 0) {
      row.victims[smallerBase + (rank & COUNT_MASK)] = static_cast<int32_t>(slot);
    }
    const int32_t equalRank = smallerTotal + equalBase + (rank >> COUNT_SHIFT);
    if (equal != 0 && equalRank < misses) {
      row.victims[equalRank] = static_cast<int32_t>(slot);
    }
    asc_syncthreads();
    smallerBase += totals & COUNT_MASK;
    equalBase += totals >> COUNT_SHIFT;
  }
}

__simt_callee__ __aicore__ inline void AssignMisses(const EchoRow& row, int32_t misses, int64_t newPriority,
                                                    uint32_t thread) {
  for (uint32_t miss = thread; miss < static_cast<uint32_t>(misses); miss += PLAN_THREADS) {
    const int32_t slot = row.victims[miss];
    const int32_t token = row.tokens[row.missPositions[miss]];
    row.slotToToken[slot] = token;
    row.priority[slot] = newPriority;
    row.missTokens[miss] = token;
    row.missSlots[miss] = slot;
    row.tokenMap[token] = slot;
  }
  asc_syncthreads();
  for (uint32_t pos = thread; pos < row.topk; pos += PLAN_THREADS) {
    const int32_t token = row.tokens[pos];
    row.tokens[pos] = token >= 0 ? row.tokenMap[token] : INVALID_TOKEN;
  }
  asc_syncthreads();
}

__simt_callee__ __aicore__ inline void ProcessRuntimeRow(
    __ubuf__ int32_t* scanWorkspace, int64_t rowIndex, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds,
    __gm__ int32_t* topkIndices, __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens,
    __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken, __gm__ int64_t* slotPriorities,
    __gm__ int32_t* currentSlots, __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,
    __gm__ int32_t* blockWorkspace, uint32_t topk, uint32_t capacity, int32_t maxToken, int64_t maxRequests,
    int64_t maxNumBlocks, int64_t hostNumBlocks, int32_t blockSize) {
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  const int64_t topkBase = rowIndex * static_cast<int64_t>(topk);
  EchoRow row{slotToToken + rowIndex * capacity,
              slotPriorities + rowIndex * (static_cast<int64_t>(capacity) + 1),
              currentSlots + topkBase,
              blockWorkspace,
              blockWorkspace + maxToken,
              blockWorkspace + maxToken + capacity,
              missTokens + topkBase,
              missSlots + topkBase,
              scanWorkspace,
              capacity,
              topk};
  const int32_t request = tokenToReq[rowIndex];
  const int64_t reqId = request >= 0 && request < maxRequests ? reqIds[request] : -1;
  const bool reset = lastReqIds[rowIndex] != reqId;
  const int64_t clock = reset ? 0 : row.priority[capacity];
  ResetResident(row, reset, stablePrefixLens[rowIndex], maxToken, thread);
  for (uint32_t pos = thread; pos < topk; pos += PLAN_THREADS) {
    const int32_t token = topkIndices[topkBase + pos];
    const bool valid = IsSourceTokenValid(rowIndex, token, visibleSeqLens, tokenToReq, blockTable, maxToken,
                                          maxRequests, maxNumBlocks, hostNumBlocks, blockSize);
    // currentSlots temporarily holds validated tokens until the final gather.
    row.tokens[pos] = valid ? token : INVALID_TOKEN;
    row.missTokens[pos] = INVALID_TOKEN;
    row.missSlots[pos] = INVALID_TOKEN;
    if (valid) {
      asc_atomic_exch(row.tokenMap + token, MAP_EMPTY);
    }
  }
  asc_syncthreads();
  const int32_t misses = MarkAndCompactMisses(row, clock + 1, thread);
  if (misses > 0) {
    const int64_t threshold = EvictionThreshold(row, misses, clock + 1, thread);
    CollectVictims(row, threshold, misses, thread);
  }
  AssignMisses(row, misses, clock + 2, thread);
  if (thread == 0U) {
    missCount[rowIndex] = misses;
    lastReqIds[rowIndex] = reqId;
    row.priority[capacity] = clock + 2;
  }
  asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(PLAN_THREADS) inline void SparseKvEchoPlanVf(
    __ubuf__ int32_t* scanWorkspace, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds, __gm__ int32_t* topkIndices,
    __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,
    __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken, __gm__ int64_t* slotPriorities,
    __gm__ int32_t* currentSlots, __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,
    __gm__ int32_t* compactWorkspace, uint64_t workspaceBlockElements, int64_t rowStart, int64_t rowStride,
    int64_t numRows, int64_t topk, int64_t capacity, int64_t maxToken, int64_t maxRequests, int64_t maxNumBlocks,
    int64_t hostNumBlocks, int32_t blockSize) {
  __gm__ int32_t* blockWorkspace = compactWorkspace + static_cast<uint64_t>(rowStart) * workspaceBlockElements;
  for (int64_t row = rowStart; row < numRows; row += rowStride) {
    ProcessRuntimeRow(scanWorkspace, row, reqIds, lastReqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq,
                      blockTable, slotToToken, slotPriorities, currentSlots, missCount, missTokens, missSlots,
                      blockWorkspace, static_cast<uint32_t>(topk), static_cast<uint32_t>(capacity),
                      static_cast<int32_t>(maxToken), maxRequests, maxNumBlocks, hostNumBlocks, blockSize);
  }
}
}  // namespace

extern "C" __global__ __aicore__ void sparse_kv_echo_plan(GM_ADDR reqIds, GM_ADDR topkIndices, GM_ADDR stablePrefixLens,
                                                          GM_ADDR visibleSeqLens, GM_ADDR tokenToReq,
                                                          GM_ADDR blockTable, GM_ADDR activeRows, GM_ADDR lastReqIds,
                                                          GM_ADDR slotToToken, GM_ADDR slotPriorities,
                                                          GM_ADDR currentSlots, GM_ADDR missCount, GM_ADDR missTokens,
                                                          GM_ADDR missSlots, GM_ADDR compactWorkspace,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::InitSocState();
  REGISTER_TILING_DEFAULT(SparseKvEchoPlanTilingData);
  GET_TILING_DATA_WITH_STRUCT(SparseKvEchoPlanTilingData, tilingData, tiling);
  const int32_t requestedRows = *reinterpret_cast<__gm__ int32_t*>(activeRows);
  const int64_t numRows = requestedRows > 0 && requestedRows < tilingData.maxRows
                              ? requestedRows
                              : (requestedRows > 0 ? tilingData.maxRows : 0);
  const int64_t rowStart = static_cast<int64_t>(AscendC::GetBlockIdx());
  const int64_t rowStride = static_cast<int64_t>(AscendC::GetBlockNum());
  if (rowStart >= numRows) {
    return;
  }
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> localWorkspace;
  pipe.InitBuffer(localWorkspace, tilingData.localMemoryBytes);
  __ubuf__ int32_t* scanWorkspace = reinterpret_cast<__ubuf__ int32_t*>(localWorkspace.Get<int32_t>().GetPhyAddr());
  asc_vf_call<SparseKvEchoPlanVf>(
      dim3(PLAN_THREADS), scanWorkspace, reinterpret_cast<__gm__ int64_t*>(reqIds),
      reinterpret_cast<__gm__ int64_t*>(lastReqIds), reinterpret_cast<__gm__ int32_t*>(topkIndices),
      reinterpret_cast<__gm__ int32_t*>(stablePrefixLens), reinterpret_cast<__gm__ int32_t*>(visibleSeqLens),
      reinterpret_cast<__gm__ int32_t*>(tokenToReq), reinterpret_cast<__gm__ int32_t*>(blockTable),
      reinterpret_cast<__gm__ int32_t*>(slotToToken), reinterpret_cast<__gm__ int64_t*>(slotPriorities),
      reinterpret_cast<__gm__ int32_t*>(currentSlots), reinterpret_cast<__gm__ int32_t*>(missCount),
      reinterpret_cast<__gm__ int32_t*>(missTokens), reinterpret_cast<__gm__ int32_t*>(missSlots),
      reinterpret_cast<__gm__ int32_t*>(compactWorkspace), tilingData.workspaceBlockElements, rowStart, rowStride,
      numRows, tilingData.topk, tilingData.capacity, tilingData.maxToken, tilingData.maxRequests,
      tilingData.maxNumBlocks, tilingData.hostNumBlocks, tilingData.blockSize);
}
