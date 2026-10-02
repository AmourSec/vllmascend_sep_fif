/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * ECHO-style direct lookup and priority eviction, adapted to per-query rows.
 * The block scan and mixed SIMD/SIMT entry derive from SparseKvPlan.
 * Unlike ECHO's persistent per-layer reverse map, the reverse map here is
 * scratch: rebuild touched entries from slotToToken, then reuse across layers.
 * Priority selection uses byte-wise radix histograms, with a bounded UB
 * snapshot for common capacities and the same GM algorithm for larger rows.
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
using sparse_kv_echo_plan::PLAN_RADIX_BINS;
using sparse_kv_echo_plan::PLAN_RADIX_BITS;
using sparse_kv_echo_plan::PLAN_RADIX_SHARDS;
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
  // Callers consume the total, then synchronize before reusing this storage.
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
template <typename MapPtr>
struct EchoRow {
  __gm__ int32_t* slotToToken;
  __gm__ int64_t* priority;
  __gm__ int32_t* tokens;
  MapPtr tokenMap;
  __gm__ int32_t* victims;
  __gm__ int32_t* missPositions;
  __gm__ int32_t* missTokens;
  __gm__ int32_t* missSlots;
  __ubuf__ int32_t* scan;
  uint32_t capacity;
  uint32_t topk;
};

template <typename MapPtr>
__simt_callee__ __aicore__ inline void ResetResident(const EchoRow<MapPtr>& row, bool reset, int32_t stablePrefix,
                                                     int32_t maxToken, uint32_t thread) {
  for (uint32_t slot = thread; slot < row.capacity; slot += PLAN_THREADS) {
    int32_t token = row.slotToToken[slot];
    if (reset || token < 0 || token >= stablePrefix || token >= maxToken) {
      row.slotToToken[slot] = INVALID_TOKEN;
      row.priority[slot] = -1;
    }
  }
}

template <typename MapPtr>
__simt_callee__ __aicore__ inline void PublishResidentHits(const EchoRow<MapPtr>& row, int64_t hitPriority,
                                                           uint32_t thread) {
  for (uint32_t slot = thread; slot < row.capacity; slot += PLAN_THREADS) {
    const int32_t token = row.slotToToken[slot];
    if (token < 0) {
      continue;
    }
    const int32_t mapped = row.tokenMap[token];
    // Scratch outside this TopK is stale. A reverse check disambiguates it
    // without clearing maxToken entries or resetting resident map entries.
    if (mapped < 0 && mapped != MAP_EMPTY) {
      const uint32_t pos = static_cast<uint32_t>(-(mapped + 1));
      if (pos < row.topk && row.tokens[pos] == token) {
        // Resident tokens are unique: their owning threads write disjoint
        // map entries and priorities, even when TopK contains duplicates.
        row.tokenMap[token] = static_cast<int32_t>(slot);
        row.priority[slot] = hitPriority;
      }
    }
  }
  asc_syncthreads();
}

template <typename MapPtr>
__simt_callee__ __aicore__ inline int32_t MarkAndCompactMisses(const EchoRow<MapPtr>& row, int64_t hitPriority,
                                                               bool empty, uint32_t thread) {
  for (uint32_t pos = thread; pos < row.topk; pos += PLAN_THREADS) {
    const int32_t token = row.tokens[pos];
    if (token >= 0) {
      // Elect the first TopK occurrence before publishing resident hits.
      asc_atomic_max(row.tokenMap + token, -static_cast<int32_t>(pos) - 1);
    }
  }
  asc_syncthreads();
  if (!empty) {
    PublishResidentHits(row, hitPriority, thread);
  }
  int32_t misses = 0;
  for (uint32_t tile = 0U; tile < row.topk; tile += PLAN_THREADS) {
    const uint32_t pos = tile + thread;
    int32_t missing = 0;
    if (pos < row.topk && row.tokens[pos] >= 0) {
      const int32_t mapped = row.tokenMap[row.tokens[pos]];
      missing = mapped == -static_cast<int32_t>(pos) - 1;
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

struct PriorityStats {
  int32_t freeCount;
  int64_t minimum;
  int64_t maximum;
};

__simt_callee__ __aicore__ inline PriorityStats ReducePriorityStats(__ubuf__ int64_t* stats, int64_t freeCount,
                                                                    int64_t minimum, int64_t maximum, uint32_t thread) {
  for (uint32_t offset = PLAN_WARP_SIZE / 2U; offset > 0U; offset >>= 1U) {
    freeCount += asc_shfl_down(freeCount, offset, PLAN_WARP_SIZE);
    const int64_t otherMin = asc_shfl_down(minimum, offset, PLAN_WARP_SIZE);
    const int64_t otherMax = asc_shfl_down(maximum, offset, PLAN_WARP_SIZE);
    minimum = otherMin < minimum ? otherMin : minimum;
    maximum = otherMax > maximum ? otherMax : maximum;
  }
  if (thread % PLAN_WARP_SIZE == 0U) {
    const uint32_t warp = thread / PLAN_WARP_SIZE;
    stats[warp] = freeCount;
    stats[PLAN_WARP_COUNT + warp] = minimum;
    stats[2U * PLAN_WARP_COUNT + warp] = maximum;
  }
  asc_syncthreads();
  if (thread == 0U) {
    freeCount = 0;
    minimum = INT64_MAX;
    maximum = -1;
    for (uint32_t warp = 0U; warp < PLAN_WARP_COUNT; ++warp) {
      freeCount += stats[warp];
      minimum = stats[PLAN_WARP_COUNT + warp] < minimum ? stats[PLAN_WARP_COUNT + warp] : minimum;
      maximum = stats[2U * PLAN_WARP_COUNT + warp] > maximum ? stats[2U * PLAN_WARP_COUNT + warp] : maximum;
    }
    stats[3U * PLAN_WARP_COUNT] = freeCount;
    stats[3U * PLAN_WARP_COUNT + 1U] = minimum;
    stats[3U * PLAN_WARP_COUNT + 2U] = maximum;
  }
  asc_syncthreads();
  return {static_cast<int32_t>(stats[3U * PLAN_WARP_COUNT]), stats[3U * PLAN_WARP_COUNT + 1U],
          stats[3U * PLAN_WARP_COUNT + 2U]};
}

template <typename MapPtr>
__simt_callee__ __aicore__ inline PriorityStats GatherPriorityStats(const EchoRow<MapPtr>& row, int64_t hitPriority,
                                                                    uint32_t thread) {
  __ubuf__ int64_t* stats = reinterpret_cast<__ubuf__ int64_t*>(
      row.scan + (sparse_kv_echo_plan::PLAN_SCAN_BYTES + sparse_kv_echo_plan::PLAN_HISTOGRAM_BYTES) / sizeof(int32_t));
  __ubuf__ int64_t* snapshot =
      reinterpret_cast<__ubuf__ int64_t*>(row.scan + sparse_kv_echo_plan::PLAN_FIXED_UB_BYTES / sizeof(int32_t));
  int64_t freeCount = 0;
  int64_t minimum = INT64_MAX;
  int64_t maximum = -1;
  for (uint32_t slot = thread; slot < row.capacity; slot += PLAN_THREADS) {
    const int64_t priority = row.priority[slot];
    if (sparse_kv_echo_plan::CachePrioritiesInUb(row.capacity)) {
      snapshot[slot] = priority;
    }
    freeCount += priority < 0;
    if (priority >= 0 && priority < hitPriority) {
      minimum = priority < minimum ? priority : minimum;
      maximum = priority > maximum ? priority : maximum;
    }
  }
  return ReducePriorityStats(stats, freeCount, minimum, maximum, thread);
}

struct RadixBucket {
  uint32_t digit;
  int32_t smaller;
};

// Return the digit containing the requested order statistic, not a full sort.
template <typename MapPtr>
__simt_callee__ __aicore__ inline RadixBucket SelectHistogramBucket(const EchoRow<MapPtr>& row,
                                                                    __ubuf__ int32_t* histogram, int32_t remaining,
                                                                    uint32_t thread) {
  int32_t base = 0;
  for (uint32_t tile = 0U; tile < PLAN_RADIX_BINS; tile += PLAN_THREADS) {
    const uint32_t bin = tile + thread;
    int32_t count = 0;
    if (bin < PLAN_RADIX_BINS) {
      for (uint32_t shard = 0U; shard < PLAN_RADIX_SHARDS; ++shard) {
        count += histogram[shard * PLAN_RADIX_BINS + bin];
      }
    }
    const int32_t rank = BlockExclusiveScan(row.scan, count, thread);
    const int32_t total = row.scan[CONTROL_OFFSET];
    if (remaining > base + rank && remaining <= base + rank + count) {
      row.scan[CONTROL_OFFSET + 1U] = static_cast<int32_t>(bin);
      row.scan[CONTROL_OFFSET + 2U] = base + rank;
    }
    asc_syncthreads();
    base += total;
  }
  return {static_cast<uint32_t>(row.scan[CONTROL_OFFSET + 1U]), row.scan[CONTROL_OFFSET + 2U]};
}

template <typename MapPtr, typename PriorityPtr>
__simt_callee__ __aicore__ inline RadixBucket RadixPass(const EchoRow<MapPtr>& row, PriorityPtr priorities,
                                                        uint64_t prefix, uint32_t shift, int32_t remaining,
                                                        int64_t hitPriority, uint32_t thread) {
  __ubuf__ int32_t* histogram = row.scan + sparse_kv_echo_plan::PLAN_SCAN_BYTES / sizeof(int32_t);
  for (uint32_t i = thread; i < sparse_kv_echo_plan::PLAN_HISTOGRAM_ELEMENTS; i += PLAN_THREADS) {
    histogram[i] = 0;
  }
  asc_syncthreads();
  const uint32_t shard = (thread / PLAN_WARP_SIZE) % PLAN_RADIX_SHARDS;
  for (uint32_t slot = thread; slot < row.capacity; slot += PLAN_THREADS) {
    const int64_t priority = priorities[slot];
    const uint64_t digits = static_cast<uint64_t>(priority) >> shift;
    if (priority >= 0 && priority < hitPriority &&
        (digits >> PLAN_RADIX_BITS) == ((prefix >> shift) >> PLAN_RADIX_BITS)) {
      asc_atomic_add(histogram + shard * PLAN_RADIX_BINS + (digits & (PLAN_RADIX_BINS - 1U)), 1);
    }
  }
  asc_syncthreads();
  return SelectHistogramBucket(row, histogram, remaining, thread);
}

struct EvictionCut {
  int64_t threshold;
  int32_t smaller;
};

template <typename MapPtr, typename PriorityPtr>
__simt_callee__ __aicore__ inline EvictionCut EvictionThreshold(const EchoRow<MapPtr>& row, PriorityPtr priorities,
                                                                PriorityStats stats, int32_t misses,
                                                                int64_t hitPriority, uint32_t thread) {
  if (stats.freeCount >= misses) {
    return {-1, 0};
  }
  if (stats.minimum == stats.maximum) {
    return {stats.minimum, stats.freeCount};
  }
  uint32_t shift = 0U;
  uint64_t differing = static_cast<uint64_t>(stats.minimum ^ stats.maximum);
  while (differing >= PLAN_RADIX_BINS) {
    differing >>= PLAN_RADIX_BITS;
    shift += PLAN_RADIX_BITS;
  }
  // Skip common high bytes, even when the int64 clock has run for a long time.
  uint64_t prefix = ((static_cast<uint64_t>(stats.minimum) >> shift) & ~uint64_t{PLAN_RADIX_BINS - 1U}) << shift;
  int32_t remaining = misses - stats.freeCount;
  int32_t smaller = stats.freeCount;
  for (;;) {
    const RadixBucket bucket = RadixPass(row, priorities, prefix, shift, remaining, hitPriority, thread);
    prefix |= static_cast<uint64_t>(bucket.digit) << shift;
    remaining -= bucket.smaller;
    smaller += bucket.smaller;
    if (shift == 0U) {
      return {static_cast<int64_t>(prefix), smaller};
    }
    shift -= PLAN_RADIX_BITS;
  }
}

template <typename MapPtr, typename PriorityPtr>
__simt_callee__ __aicore__ inline void CollectVictims(const EchoRow<MapPtr>& row, PriorityPtr priorities,
                                                      EvictionCut cut, int32_t misses, uint32_t thread) {
  int32_t smallerBase = 0;
  int32_t equalBase = 0;
  for (uint32_t tile = 0U; tile < row.capacity; tile += PLAN_THREADS) {
    const uint32_t slot = tile + thread;
    const int64_t priority = slot < row.capacity ? priorities[slot] : INT64_MAX;
    const int32_t smaller = priority < cut.threshold;
    const int32_t equal = priority == cut.threshold;
    const int32_t rank = BlockExclusiveScan(row.scan, smaller + (equal << COUNT_SHIFT), thread);
    const int32_t totals = row.scan[CONTROL_OFFSET];
    if (smaller != 0) {
      row.victims[smallerBase + (rank & COUNT_MASK)] = static_cast<int32_t>(slot);
    }
    const int32_t equalRank = cut.smaller + equalBase + (rank >> COUNT_SHIFT);
    if (equal != 0 && equalRank < misses) {
      row.victims[equalRank] = static_cast<int32_t>(slot);
    }
    asc_syncthreads();
    smallerBase += totals & COUNT_MASK;
    equalBase += totals >> COUNT_SHIFT;
  }
}

template <typename MapPtr>
__simt_callee__ __aicore__ inline void SelectVictims(const EchoRow<MapPtr>& row, int32_t misses, int64_t hitPriority,
                                                     uint32_t thread) {
  const PriorityStats stats = GatherPriorityStats(row, hitPriority, thread);
  if (sparse_kv_echo_plan::CachePrioritiesInUb(row.capacity)) {
    __ubuf__ int64_t* priorities =
        reinterpret_cast<__ubuf__ int64_t*>(row.scan + sparse_kv_echo_plan::PLAN_FIXED_UB_BYTES / sizeof(int32_t));
    const EvictionCut cut = EvictionThreshold(row, priorities, stats, misses, hitPriority, thread);
    CollectVictims(row, priorities, cut, misses, thread);
  } else {
    const EvictionCut cut = EvictionThreshold(row, row.priority, stats, misses, hitPriority, thread);
    CollectVictims(row, row.priority, cut, misses, thread);
  }
}

template <typename MapPtr>
__simt_callee__ __aicore__ inline void AssignMisses(const EchoRow<MapPtr>& row, int32_t misses, int64_t newPriority,
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

template <typename MapPtr>
__simt_callee__ __aicore__ inline void ProcessRuntimeRow(
    __ubuf__ int32_t* scanWorkspace, MapPtr tokenMap, int64_t rowIndex, __gm__ int64_t* reqIds,
    __gm__ int64_t* lastReqIds, __gm__ int32_t* topkIndices, __gm__ int32_t* stablePrefixLens,
    __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken,
    __gm__ int64_t* slotPriorities, __gm__ int32_t* currentSlots, __gm__ int32_t* missCount, __gm__ int32_t* missTokens,
    __gm__ int32_t* missSlots, __gm__ int32_t* blockWorkspace, uint32_t topk, uint32_t capacity, int32_t maxToken,
    int64_t maxRequests, int64_t maxNumBlocks, int64_t hostNumBlocks, int32_t blockSize) {
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  const int64_t topkBase = rowIndex * static_cast<int64_t>(topk);
  EchoRow<MapPtr> row{slotToToken + rowIndex * capacity,
                      slotPriorities + rowIndex * (static_cast<int64_t>(capacity) + 1),
                      currentSlots + topkBase,
                      tokenMap,
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
  const bool empty = reset || stablePrefixLens[rowIndex] <= 0;
  const int32_t misses = MarkAndCompactMisses(row, clock + 1, empty, thread);
  if (misses > 0) {
    if (empty) {
      for (uint32_t miss = thread; miss < static_cast<uint32_t>(misses); miss += PLAN_THREADS) {
        row.victims[miss] = static_cast<int32_t>(miss);
      }
      asc_syncthreads();
    } else {
      SelectVictims(row, misses, clock + 1, thread);
    }
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
    if (sparse_kv_echo_plan::CacheTokenMapInUb(capacity, maxToken)) {
      __ubuf__ int32_t* tokenMap = scanWorkspace + (sparse_kv_echo_plan::PLAN_FIXED_UB_BYTES +
                                                    sparse_kv_echo_plan::PriorityCacheBytes(capacity)) /
                                                       sizeof(int32_t);
      ProcessRuntimeRow(scanWorkspace, tokenMap, row, reqIds, lastReqIds, topkIndices, stablePrefixLens, visibleSeqLens,
                        tokenToReq, blockTable, slotToToken, slotPriorities, currentSlots, missCount, missTokens,
                        missSlots, blockWorkspace, static_cast<uint32_t>(topk), static_cast<uint32_t>(capacity),
                        static_cast<int32_t>(maxToken), maxRequests, maxNumBlocks, hostNumBlocks, blockSize);
    } else {
      ProcessRuntimeRow(scanWorkspace, blockWorkspace, row, reqIds, lastReqIds, topkIndices, stablePrefixLens,
                        visibleSeqLens, tokenToReq, blockTable, slotToToken, slotPriorities, currentSlots, missCount,
                        missTokens, missSlots, blockWorkspace, static_cast<uint32_t>(topk),
                        static_cast<uint32_t>(capacity), static_cast<int32_t>(maxToken), maxRequests, maxNumBlocks,
                        hostNumBlocks, blockSize);
    }
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
