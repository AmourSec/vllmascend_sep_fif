/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 */

#ifndef VLLM_ASCEND_SPARSE_KV_PLAN_SHARED_H
#define VLLM_ASCEND_SPARSE_KV_PLAN_SHARED_H

#include <climits>
#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "sparse_kv_plan_config.h"

namespace {

using sparse_kv_plan_cfg::PLAN_ROW_UB_LIMIT_ELEMENTS;
using sparse_kv_plan_cfg::PLAN_SCAN_STORAGE_ELEMENTS;
using sparse_kv_plan_cfg::PLAN_THREADS;
using sparse_kv_plan_cfg::PLAN_WARP_COUNT;
using sparse_kv_plan_cfg::PLAN_WARP_SIZE;
constexpr int32_t INVALID_TOKEN = -1;
constexpr int32_t HASH_EMPTY = -1;
constexpr int32_t HASH_POSITION_EMPTY = INT32_MAX;
constexpr uint32_t HASH_BUCKET_INVALID = UINT32_MAX;

// scan 的动态 UB 布局，单位为 int32。warp 区域均由线程数推导，
// control 之后的 padding 保证单行 workspace 从 32 字节边界开始。
constexpr uint32_t WARP_TOTALS_OFFSET = 0U;
constexpr uint32_t WARP_BASES_OFFSET = PLAN_WARP_COUNT;
constexpr uint32_t CONTROL_OFFSET = 2U * PLAN_WARP_COUNT;
constexpr uint32_t CONTROL_TOTAL = 0U;
constexpr uint32_t CONTROL_RESET = 1U;
constexpr uint32_t OWNER_SLOT_BITS = 16U;
constexpr uint32_t OWNER_SLOT_MASK = (1U << OWNER_SLOT_BITS) - 1U;
constexpr uint32_t OWNER_PACK_CAPACITY = 1U << (31U - OWNER_SLOT_BITS);

// 这是主动选择的快路径上限，不是硬件 UB 容量上限。
// 动态 UB 最多为 112 KiB + scan workspace，为 SIMT DCache 和系统预留区域留下空间。
// 同一 block 顺序处理多行时复用这块内存，因此占用不随 batch 增长。
// 低 12 位表示 evict 数，高位表示 hit 数。
// 每个 tile 最多 PLAN_THREADS 项，而 2^12=4096，因此低位不会进位到 hit 字段。
constexpr uint32_t COUNT_SHIFT = 12U;
constexpr int32_t COUNT_MASK = (1U << COUNT_SHIFT) - 1U;

static_assert(PLAN_THREADS < (1U << COUNT_SHIFT), "packed scan would carry");

struct SparseKvPlanRuntimeContext {
  __gm__ int64_t* reqIds;
  __gm__ int64_t* lastReqIds;
  __gm__ int32_t* stablePrefixLens;
  __gm__ int32_t* visibleSeqLens;
  __gm__ int32_t* tokenToReq;
  __gm__ int32_t* blockTable;
  __gm__ int32_t* slotToToken;
  __gm__ int32_t* lruSlots;
  __gm__ int32_t* currentSlots;
  __gm__ int32_t* missCount;
  __gm__ int32_t* missTokens;
  __gm__ int32_t* missSlots;
  __gm__ int32_t* compactWorkspace;
  int64_t numRows;
  uint64_t hashCapacity;
  uint64_t workspaceRowElements;
  int64_t topk;
  int64_t capacity;
  int64_t maxToken;
  int64_t maxRequests;
  int64_t maxNumBlocks;
  int64_t hostNumBlocks;
  int32_t blockSize;
};


/**
 * 对完整 SIMT block 做独占前缀和，返回当前线程之前的输入总和。
 * 输入可以是 0/1，也可以是打包后的 evict + (hit << 12)。
 *
 * 第一级：每个 warp 的 32 个线程用 shuffle 计算 inclusive scan。
 * 第二级：warp 0 并行扫描 warp total，生成每个 warp 的起点。
 * warp 数完全由 PLAN_THREADS 推导，后续调整线程数无需重写 scan 布局。
 *
 * 所有线程必须参加所有 barrier。尾 tile 的无效线程输入 0，不能提前返回。
 * 调用者必须在 scatter/读取 warpBases 后执行 barrier，再复用 workspace。
 */
__simt_callee__ __aicore__ inline void PublishWarpBases(__ubuf__ int32_t* workspace, uint32_t thread) {
  // Warp 0 only; PLAN_THREADS is 1024, so every lane executes the same shuffles.
  if (thread >= PLAN_WARP_SIZE) {
    return;
  }
  __ubuf__ int32_t* warpTotals = workspace + WARP_TOTALS_OFFSET;
  __ubuf__ int32_t* warpBases = workspace + WARP_BASES_OFFSET;
  __ubuf__ int32_t* control = workspace + CONTROL_OFFSET;
  const uint32_t lane = thread;
  const int32_t value = lane < PLAN_WARP_COUNT ? warpTotals[lane] : 0;
  int32_t inclusive = value;
  for (uint32_t offset = 1U; offset < PLAN_WARP_SIZE; offset <<= 1U) {
    const int32_t prior = asc_shfl_up(inclusive, offset, PLAN_WARP_SIZE);
    if (lane >= offset) {
      inclusive += prior;
    }
  }
  if (lane < PLAN_WARP_COUNT) {
    warpBases[lane] = inclusive - value;
  }
  if (lane == PLAN_WARP_SIZE - 1U) {
    control[CONTROL_TOTAL] = inclusive;
  }
}

__simt_callee__ __aicore__ inline int32_t BlockExclusiveScan(__ubuf__ int32_t* workspace, int32_t value,
                                                             uint32_t thread) {
  __ubuf__ int32_t* warpTotals = workspace + WARP_TOTALS_OFFSET;
  __ubuf__ int32_t* warpBases = workspace + WARP_BASES_OFFSET;
  const uint32_t lane = thread & (PLAN_WARP_SIZE - 1U);
  const uint32_t warp = thread / PLAN_WARP_SIZE;
  int32_t inclusive = value;
  for (uint32_t offset = 1U; offset < PLAN_WARP_SIZE; offset <<= 1U) {
    const int32_t prior = asc_shfl_up(inclusive, offset, PLAN_WARP_SIZE);
    if (lane >= offset) {
      inclusive += prior;
    }
  }
  if (lane == PLAN_WARP_SIZE - 1U) {
    warpTotals[warp] = inclusive;
  }
  asc_syncthreads();

  PublishWarpBases(workspace, thread);
  asc_syncthreads();
  const int32_t exclusive = warpBases[warp] + inclusive - value;
  return exclusive;
}

// H 是 2 的幂，线性探测可以用 & (H-1) 环绕。bucket 使用无符号类型，
// 不把合法高位索引误判为 -1。沿用原工程的 hash 容量与 token 混合函数。
__simt_callee__ __aicore__ inline uint32_t HashToken(int32_t token, uint32_t hashCapacity) {
  uint32_t value = static_cast<uint32_t>(token);
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return value & (hashCapacity - 1U);
}

/**
 * ScratchPtr 由实参推导为 __ubuf__ int32_t* 或 __gm__ int32_t*。
 * 编译器分别生成两个地址空间的实现，不做 UB/GM 之间的指针强转。
 * 昇腾 950 的 int32 CAS/min/max 支持 UB 与 GM。
 * CAS 决定哪个 token 占桶，atomic_min 确保重复 token 保留最早 TopK 位置。
 */
template <typename ScratchPtr>
__simt_callee__ __aicore__ inline uint32_t InsertTopkToken(ScratchPtr keys, ScratchPtr firstPositions, int32_t token,
                                                       int32_t position, uint32_t hashCapacity) {
  uint32_t bucket = HashToken(token, hashCapacity);
  const uint32_t mask = hashCapacity - 1U;
  for (uint32_t probe = 0U; probe < hashCapacity; ++probe) {
    const int32_t old = asc_atomic_cas(keys + bucket, HASH_EMPTY, token);
    if (old == HASH_EMPTY || old == token) {
      asc_atomic_min(firstPositions + bucket, position);
      return bucket;
    }
    bucket = (bucket + 1U) & mask;
  }
  // 合法 helper 产生至少 2*topk 个桶，因此不会填满；保留有界循环防止死循环。
  return HASH_BUCKET_INVALID;
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

__simt_callee__ __aicore__ inline bool IsSourceTokenValid(int64_t row, int32_t token, __gm__ int32_t* visibleSeqLens,
                                                          __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable,
                                                          int64_t maxToken, int64_t maxRequests, int64_t maxNumBlocks,
                                                          int64_t hostNumBlocks, int32_t blockSize) {
  if (token < 0 || token >= maxToken || token >= visibleSeqLens[row]) {
    return false;
  }
  const int32_t request = tokenToReq[row];
  if (request < 0 || request >= maxRequests || blockSize <= 0) {
    return false;
  }
  const int64_t blockId = token / blockSize;
  if (blockId < 0 || blockId >= maxNumBlocks) {
    return false;
  }
  const int64_t tableOffset = static_cast<int64_t>(request) * maxNumBlocks + blockId;
  const int32_t physicalBlock = blockTable[tableOffset];
  return physicalBlock >= 0 && physicalBlock < hostNumBlocks;
}

__simt_callee__ __aicore__ inline void ClearMissTail(int64_t base, uint32_t assigned, uint32_t topk,
                                                      uint32_t thread, __gm__ int32_t* missTokens,
                                                      __gm__ int32_t* missSlots) {
  for (uint32_t pos = assigned + thread; pos < topk; pos += PLAN_THREADS) {
    missTokens[base + pos] = INVALID_TOKEN;
    missSlots[base + pos] = INVALID_TOKEN;
  }
}

/**
 * 一次调用只处理一条 query 行。只有临时数组允许放 UB；reqId、resident、LRU 等
 * 持久状态始终保存在 GM，并由 row 定位，不能依赖上一轮由哪个物理核处理。
 *
 * rowWorkspace 的五段布局（单位 int32）：
 * keys[H]、firstPos[H]、owner[H]、partitionSlots[C]、missPositions[T]。
 * H=hashCapacity，C=capacity，T=topk，总大小为 (3H+C+T)*4 字节。
 */
template <typename ScratchPtr, typename TopkPtr>
__simt_callee__ __aicore__ inline void ProcessRuntimeRow(
    __ubuf__ int32_t* scanWorkspace, int64_t row, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds,
    TopkPtr topkIndices, __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens,
    __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots,
    __gm__ int32_t* currentSlots, __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots,
    ScratchPtr rowWorkspace, uint32_t hashCapacity, uint32_t topk, uint32_t capacity, int32_t maxToken,
    int64_t maxRequests, int64_t maxNumBlocks, int64_t hostNumBlocks, int32_t blockSize) {
  const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
  __ubuf__ int32_t* control = scanWorkspace + CONTROL_OFFSET;
  const int64_t topkBase = row * static_cast<int64_t>(topk);
  const int64_t capacityBase = row * static_cast<int64_t>(capacity);
  ScratchPtr hashKeys = rowWorkspace;
  ScratchPtr hashFirstPos = hashKeys + hashCapacity;
  ScratchPtr hashOwner = hashFirstPos + hashCapacity;
  ScratchPtr partitionSlots = hashOwner + hashCapacity;
  ScratchPtr missPositions = partitionSlots + capacity;

  // Stage 1: initialize outputs and hash metadata before publishing cached buckets.
  for (uint32_t pos = thread; pos < topk; pos += PLAN_THREADS) {
    currentSlots[topkBase + pos] = INVALID_TOKEN;
  }
  for (uint32_t i = thread; i < hashCapacity; i += PLAN_THREADS) {
    hashKeys[i] = HASH_EMPTY;
    hashFirstPos[i] = HASH_POSITION_EMPTY;
  }
  if (thread == 0U) {
    missCount[row] = 0;
    const int32_t request = tokenToReq[row];
    const int64_t currentReqId = request >= 0 && request < maxRequests ? reqIds[request] : -1;
    control[CONTROL_RESET] = lastReqIds[row] != currentReqId ? 1 : 0;
  }
  asc_syncthreads();

  // Prepare keys/firstPos and cached bucket for all rows before any lookup.
  for (uint32_t pos = thread; pos < topk; pos += PLAN_THREADS) {
    const int32_t token = topkIndices[pos];
    if (IsSourceTokenValid(row, token, visibleSeqLens, tokenToReq, blockTable, maxToken, maxRequests, maxNumBlocks,
                           hostNumBlocks, blockSize)) {
      missPositions[pos] = static_cast<int32_t>(InsertTopkToken(hashKeys, hashFirstPos, token,
                                                                  static_cast<int32_t>(pos), hashCapacity));
    } else {
      missPositions[pos] = static_cast<int32_t>(HASH_BUCKET_INVALID);
    }
  }
  asc_syncthreads();

  // Request changes reset the row; cold rows skip resident classification.
  if (control[CONTROL_RESET] != 0) {
    for (uint32_t slot = thread; slot < capacity; slot += PLAN_THREADS) {
      slotToToken[capacityBase + slot] = INVALID_TOKEN;
    }
    // Cold path still uses the hash to retain only the first duplicate occurrence.
    int32_t validCount = 0;
    for (uint32_t tile = 0U; tile < topk; tile += PLAN_THREADS) {
      const uint32_t position = tile + thread;
      int32_t flag = 0;
      if (position < topk && static_cast<uint32_t>(missPositions[position]) != HASH_BUCKET_INVALID) {
        const uint32_t bucket = static_cast<uint32_t>(missPositions[position]);
        flag = hashFirstPos[bucket] == static_cast<int32_t>(position);
      }
      // All bucket reads precede the scan barrier. Compaction writes at or
      // before position, so future tiles' cached buckets remain untouched.
      const int32_t rank = BlockExclusiveScan(scanWorkspace, flag, thread);
      const int32_t total = control[CONTROL_TOTAL];
      if (flag != 0) {
        missPositions[validCount + rank] = static_cast<int32_t>(position);
      }
      asc_syncthreads();  // Includes slot invalidation and consumption of scan control.
      validCount += total;
    }
    const int32_t assigned = validCount < static_cast<int32_t>(capacity) ? validCount : static_cast<int32_t>(capacity);
    for (uint32_t miss = thread; miss < static_cast<uint32_t>(assigned); miss += PLAN_THREADS) {
      const int32_t position = missPositions[miss];
      const int32_t token = topkIndices[position];
      slotToToken[capacityBase + miss] = token;
      currentSlots[topkBase + position] = static_cast<int32_t>(miss);
      missTokens[topkBase + miss] = token;
      missSlots[topkBase + miss] = static_cast<int32_t>(miss);
    }
    ClearMissTail(topkBase, static_cast<uint32_t>(assigned), topk, thread, missTokens, missSlots);
    const uint32_t unused = capacity - static_cast<uint32_t>(assigned);
    for (uint32_t order = thread; order < capacity; order += PLAN_THREADS) {
      lruSlots[capacityBase + order] = static_cast<int32_t>(order < unused ? assigned + order : order - unused);
    }
    if (thread == 0U) {
      const int32_t request = tokenToReq[row];
      lastReqIds[row] = request >= 0 && request < maxRequests ? reqIds[request] : -1;
      missCount[row] = assigned;
    }
    asc_syncthreads();  // Uniform return; persistent writes finish before scratch reuse.
    return;
  }

  // Only warm rows need owner state; cold rows returned before this initialization.
  for (uint32_t i = thread; i < hashCapacity; i += PLAN_THREADS) {
    hashOwner[i] = INVALID_TOKEN;
  }
  asc_syncthreads();

  int32_t stablePrefix = stablePrefixLens[row];
  if (stablePrefix < 0) {
    stablePrefix = 0;
  } else if (stablePrefix > maxToken) {
    stablePrefix = maxToken;
  }

  // Stage 4: stable LRU classification. One partition array grows from both ends.
  // 同时产生 evictRank 和 hitRank；两者仍分别保持原来的相对顺序。
  int32_t evictableCount = 0;
  int32_t hitCount = 0;
  for (uint32_t tile = 0U; tile < capacity; tile += PLAN_THREADS) {
    const uint32_t order = tile + thread;
    int32_t slot = INVALID_TOKEN;
    int32_t evictFlag = 0;
    int32_t hitFlag = 0;
    if (order < capacity) {
      slot = lruSlots[capacityBase + order];
      if (slot >= 0 && static_cast<uint32_t>(slot) < capacity) {
        int32_t token = slotToToken[capacityBase + slot];
        // speculative suffix 不再可靠，先失效再判断 hit。
        if (token >= 0 && token < maxToken && token >= stablePrefix) {
          slotToToken[capacityBase + slot] = INVALID_TOKEN;
          token = INVALID_TOKEN;
        }
        uint32_t bucket = HASH_BUCKET_INVALID;
        if (token >= 0 && token < maxToken) {
          bucket = LookupTopkToken(hashKeys, token, hashCapacity);
        }
        if (bucket != HASH_BUCKET_INVALID) {
          hitFlag = 1;
          const int32_t owner = capacity <= OWNER_PACK_CAPACITY
                                    ? static_cast<int32_t>((order << OWNER_SLOT_BITS) |
                                                           static_cast<uint32_t>(slot))
                                    : static_cast<int32_t>(order);
          asc_atomic_max(hashOwner + bucket, owner);
          // 同一 token 占多个 slot 时，旧 LRU 中最后一个 owner 提供命中结果。
        } else {
          evictFlag = 1;
        }
      }
    }
    // 越界 LRU 项和尾部线程两种 flag 都为 0；不能把它们误当 evict。
    const int32_t packed = evictFlag + (hitFlag << COUNT_SHIFT);
    const int32_t ranks = BlockExclusiveScan(scanWorkspace, packed, thread);
    const int32_t totals = control[CONTROL_TOTAL];
    const int32_t evictRank = ranks & COUNT_MASK;
    const int32_t hitRank = ranks >> COUNT_SHIFT;
    if (evictFlag != 0) {
      partitionSlots[evictableCount + evictRank] = slot;
    }
    if (hitFlag != 0) {
      partitionSlots[capacity - 1U - static_cast<uint32_t>(hitCount + hitRank)] = slot;
    }
    asc_syncthreads();
    evictableCount += totals & COUNT_MASK;
    hitCount += totals >> COUNT_SHIFT;
  }

  // Stage 6: consume cached buckets before in-place miss compaction.
  // The scan barrier orders every bucket read before scatter; each destination
  // is <= its source position. The tile-end barrier protects the next scan.
  int32_t localMissCount = 0;
  for (uint32_t tile = 0U; tile < topk; tile += PLAN_THREADS) {
    const uint32_t position = tile + thread;
    int32_t missFlag = 0;
    if (position < topk) {
      const uint32_t bucket = static_cast<uint32_t>(missPositions[position]);
      if (bucket != HASH_BUCKET_INVALID && hashFirstPos[bucket] == static_cast<int32_t>(position)) {
        const int32_t owner = hashOwner[bucket];
        if (owner >= 0) {
          currentSlots[topkBase + position] = capacity <= OWNER_PACK_CAPACITY
                                                  ? (owner & OWNER_SLOT_MASK)
                                                  : lruSlots[capacityBase + owner];
        } else {
          missFlag = 1;
        }
      }
    }
    const int32_t rank = BlockExclusiveScan(scanWorkspace, missFlag, thread);
    const int32_t tileMissCount = control[CONTROL_TOTAL];
    if (missFlag != 0) {
      missPositions[localMissCount + rank] = static_cast<int32_t>(position);
    }
    asc_syncthreads();
    localMissCount += tileMissCount;
  }

  // 阶段 7：miss 多于可淘汰 slot 时，仅分配能够容纳的前缀。
  const int32_t assignCount = localMissCount < evictableCount ? localMissCount : evictableCount;

  for (uint32_t miss = thread; miss < static_cast<uint32_t>(assignCount); miss += PLAN_THREADS) {
    const int32_t slot = partitionSlots[miss];
    const int32_t position = missPositions[miss];
    const int32_t token = topkIndices[position];
    slotToToken[capacityBase + slot] = token;
    currentSlots[topkBase + position] = slot;
    missTokens[topkBase + miss] = token;
    missSlots[topkBase + miss] = slot;
  }
  ClearMissTail(topkBase, static_cast<uint32_t>(assignCount), topk, thread, missTokens, missSlots);

  // 阶段 8：LRU = 未使用 evictable + 新 miss + 原 hit，顺序与原算子一致。
  const int32_t unassignedCount = evictableCount - assignCount;
  const int32_t validLruCount = evictableCount + hitCount;
  for (uint32_t output = thread; output < static_cast<uint32_t>(validLruCount); output += PLAN_THREADS) {
    int32_t slot;
    if (output < static_cast<uint32_t>(unassignedCount)) {
      slot = partitionSlots[assignCount + output];
    } else if (output < static_cast<uint32_t>(evictableCount)) {
      slot = partitionSlots[output - unassignedCount];
    } else {
      slot = partitionSlots[capacity - 1U - (output - static_cast<uint32_t>(evictableCount))];
    }
    lruSlots[capacityBase + output] = slot;
  }
  if (thread == 0U) {
    missCount[row] = assignCount;
  }
  // 下一行可能立刻复用 UB，必须等当前行全部线程写完。
  asc_syncthreads();
}

template <typename TopkPtr, typename ScratchPtr>
__simt_vf__ __aicore__ LAUNCH_BOUND(PLAN_THREADS) inline void SparseKvPlanRuntimeRowVf(
    __ubuf__ int32_t* scanWorkspace, int64_t row, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds,
    TopkPtr topkIndices, __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens,
    __gm__ int32_t* tokenToReq, __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken,
    __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots, __gm__ int32_t* missCount,
    __gm__ int32_t* missTokens, __gm__ int32_t* missSlots, ScratchPtr rowWorkspace,
    uint64_t hashCapacity, int64_t topk, int64_t capacity, int64_t maxToken,
    int64_t maxRequests, int64_t maxNumBlocks, int64_t hostNumBlocks, int32_t blockSize) {
  ProcessRuntimeRow(scanWorkspace, row, reqIds, lastReqIds, topkIndices, stablePrefixLens, visibleSeqLens,
                    tokenToReq, blockTable, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots,
                    rowWorkspace, static_cast<uint32_t>(hashCapacity), static_cast<uint32_t>(topk),
                    static_cast<uint32_t>(capacity), static_cast<int32_t>(maxToken), maxRequests, maxNumBlocks,
                    hostNumBlocks, blockSize);
}

/**
 * SIMT 调度层。是否使用 UB 对整个 block 一致，不会在 barrier 两侧发生线程分歧。
 * UB 路径复用当前 AIV 的单行 scratch；GM 路径按 row 切分用户的完整 workspace。
 */
__simt_vf__ __aicore__ LAUNCH_BOUND(PLAN_THREADS) inline void SparseKvPlanRuntimeVf(
    __ubuf__ int32_t* scanWorkspace, __gm__ int64_t* reqIds, __gm__ int64_t* lastReqIds, __gm__ int32_t* topkIndices,
    __gm__ int32_t* stablePrefixLens, __gm__ int32_t* visibleSeqLens, __gm__ int32_t* tokenToReq,
    __gm__ int32_t* blockTable, __gm__ int32_t* slotToToken, __gm__ int32_t* lruSlots, __gm__ int32_t* currentSlots,
    __gm__ int32_t* missCount, __gm__ int32_t* missTokens, __gm__ int32_t* missSlots, __gm__ int32_t* compactWorkspace,
    uint64_t hashCapacity, uint64_t workspaceRowElements, int64_t rowStart, int64_t rowStride, int64_t numReqs,
    int64_t topk, int64_t capacity, int64_t maxToken, int64_t maxRequests, int64_t maxNumBlocks, int64_t hostNumBlocks,
    int32_t blockSize) {
  for (int64_t row = rowStart; row < numReqs; row += rowStride) {
    if (workspaceRowElements <= PLAN_ROW_UB_LIMIT_ELEMENTS) {
      __ubuf__ int32_t* rowWorkspace = scanWorkspace + PLAN_SCAN_STORAGE_ELEMENTS;
      ProcessRuntimeRow(scanWorkspace, row, reqIds, lastReqIds,
                        topkIndices + row * static_cast<int64_t>(topk), stablePrefixLens, visibleSeqLens,
                        tokenToReq, blockTable, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots,
                        rowWorkspace, static_cast<uint32_t>(hashCapacity), static_cast<uint32_t>(topk),
                        static_cast<uint32_t>(capacity), static_cast<int32_t>(maxToken), maxRequests, maxNumBlocks,
                        hostNumBlocks, blockSize);
    } else {
      __gm__ int32_t* rowWorkspace = compactWorkspace + static_cast<uint64_t>(row) * workspaceRowElements;
      ProcessRuntimeRow(scanWorkspace, row, reqIds, lastReqIds,
                        topkIndices + row * static_cast<int64_t>(topk), stablePrefixLens, visibleSeqLens,
                        tokenToReq, blockTable, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots,
                        rowWorkspace, static_cast<uint32_t>(hashCapacity), static_cast<uint32_t>(topk),
                        static_cast<uint32_t>(capacity), static_cast<int32_t>(maxToken), maxRequests, maxNumBlocks,
                        hostNumBlocks, blockSize);
    }
  }
}

}  // namespace

#endif  // VLLM_ASCEND_SPARSE_KV_PLAN_SHARED_H
