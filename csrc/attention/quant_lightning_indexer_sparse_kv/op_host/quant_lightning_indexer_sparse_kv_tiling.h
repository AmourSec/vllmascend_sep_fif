/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QUANT_LIGHTNING_INDEXER_SPARSE_KV_TILING_H_
#define QUANT_LIGHTNING_INDEXER_SPARSE_KV_TILING_H_

#include <cstdint>

namespace optiling::qli_sparse_kv_contract {

constexpr uint32_t TOPK = 2048U;
constexpr uint64_t HASH_CAPACITY = 4096U;
constexpr uint32_t SCAN_BYTES = 288U;
constexpr uint32_t ROW_UB_LIMIT_BYTES = 112U * 1024U;
constexpr uint32_t QLI_TOPK_SLAB_BYTES = 89344U;
constexpr uint32_t QLI_UB_SMALL_T_BYTES = 150528U;
constexpr uint32_t QLI_UB_LARGE_T_BYTES = 186880U;
constexpr uint32_t TOTAL_UB_BYTES = 256U * 1024U;
constexpr uint32_t SYSTEM_UB_RESERVE_BYTES = 8U * 1024U;
constexpr uint32_t MIN_DATA_CACHE_BYTES = 32U * 1024U;

static_assert(QLI_UB_LARGE_T_BYTES + SCAN_BYTES <= TOTAL_UB_BYTES - SYSTEM_UB_RESERVE_BYTES - MIN_DATA_CACHE_BYTES);
static_assert(QLI_UB_LARGE_T_BYTES + SCAN_BYTES + ROW_UB_LIMIT_BYTES - QLI_TOPK_SLAB_BYTES <=
              TOTAL_UB_BYTES - SYSTEM_UB_RESERVE_BYTES - MIN_DATA_CACHE_BYTES);

struct PlannerSizing {
  uint64_t hashCapacity;
  uint64_t workspaceRowElements;
  uint32_t planLocalMemoryBytes;
  uint32_t dynamicUbBytes;
  bool usesUbWorkspace;
};

constexpr uint64_t AlignUp32(uint64_t value) { return (value + 31U) & ~static_cast<uint64_t>(31U); }

constexpr PlannerSizing ComputePlannerSizing(int64_t capacity, uint32_t tokenCount) {
  const uint64_t rowElements = 3U * HASH_CAPACITY + static_cast<uint64_t>(capacity) + TOPK;
  const uint64_t alignedRowBytes = AlignUp32(rowElements * sizeof(int32_t));
  const bool usesUb = alignedRowBytes <= ROW_UB_LIMIT_BYTES;
  const uint32_t rowUbBytes = usesUb ? static_cast<uint32_t>(alignedRowBytes) : 0U;
  const uint32_t localBytes = SCAN_BYTES + rowUbBytes;
  const uint32_t extraPlannerBytes = rowUbBytes > QLI_TOPK_SLAB_BYTES ? rowUbBytes - QLI_TOPK_SLAB_BYTES : 0U;
  const uint32_t qliBytes = tokenCount > 64U ? QLI_UB_LARGE_T_BYTES : QLI_UB_SMALL_T_BYTES;
  return {HASH_CAPACITY, rowElements, localBytes, qliBytes + SCAN_BYTES + extraPlannerBytes, usesUb};
}

constexpr bool CompactWorkspaceShapeIsValid(int64_t rows, int64_t width, const PlannerSizing& sizing,
                                            uint32_t tokenCount) {
  if (sizing.usesUbWorkspace) {
    const bool minimalPlaceholder = rows == 1 && width == 1;
    const bool fullWorkspace = rows >= tokenCount && width >= static_cast<int64_t>(sizing.workspaceRowElements);
    return minimalPlaceholder || fullWorkspace;
  }
  return rows >= tokenCount && width == static_cast<int64_t>(sizing.workspaceRowElements);
}

constexpr uint32_t ResolveStride(int64_t requested, uint32_t defaultStride) {
  return requested == -1 ? defaultStride : static_cast<uint32_t>(requested);
}

constexpr bool PlannerBlockTableShapeIsValid(int64_t rows, int64_t width, int64_t maxRequests) {
  return rows == maxRequests && width > 0;
}

constexpr uint64_t AdditionalWorkspaceBytes(uint64_t sequenceLength, uint32_t aicCount) {
  constexpr uint64_t sequenceTile = 128U;
  constexpr uint64_t scoreRows = 4U;
  constexpr uint64_t scoreElementBytes = sizeof(uint16_t);
  return scoreRows * ((sequenceLength + sequenceTile - 1U) / sequenceTile) * sequenceTile * scoreElementBytes *
         aicCount;
}

}  // namespace optiling::qli_sparse_kv_contract

#ifndef QLI_SPARSE_KV_FIXED_CONTRACT_ONLY

  #include "exe_graph/runtime/tiling_context.h"
  #include "platform/platform_info.h"
  #include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(QuantLightningIndexerSparseKvTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, tSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, keyDequantScaleStride0)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint64_t, hashCapacity)
TILING_DATA_FIELD_DEF(uint64_t, workspaceRowElements)
TILING_DATA_FIELD_DEF(int64_t, maxRows)
TILING_DATA_FIELD_DEF(int64_t, topk)
TILING_DATA_FIELD_DEF(int64_t, capacity)
TILING_DATA_FIELD_DEF(int64_t, maxToken)
TILING_DATA_FIELD_DEF(int64_t, maxRequests)
TILING_DATA_FIELD_DEF(int64_t, maxNumBlocks)
TILING_DATA_FIELD_DEF(int64_t, hostNumBlocks)
TILING_DATA_FIELD_DEF(int32_t, planBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, planLocalMemoryBytes)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantLightningIndexerSparseKv, QuantLightningIndexerSparseKvTilingData)

struct QuantLightningIndexerSparseKvCompileInfo {};

struct QuantLightningIndexerSparseKvInfo {
  fe::PlatFormInfos* platformInfo = nullptr;
  uint32_t bSize = 0;
  uint32_t tSize = 0;
  uint32_t s2Size = 0;
  uint32_t maxBlockNumPerBatch = 0;
  uint32_t keyStride0 = 0;
  uint32_t keyDequantScaleStride0 = 0;
  int64_t capacity = 0;
  int64_t maxToken = 0;
  int64_t maxRequests = 0;
  int64_t maxNumBlocks = 0;
  int64_t hostNumBlocks = 0;
  qli_sparse_kv_contract::PlannerSizing planner{};
};

ge::graphStatus TilingForQuantLightningIndexerSparseKv(gert::TilingContext* context);

}  // namespace optiling

#endif
#endif
