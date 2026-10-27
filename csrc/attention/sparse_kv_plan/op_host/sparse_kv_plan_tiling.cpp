/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sparse_kv_plan_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sparse_kv_plan_config.h"

namespace optiling {
namespace {
constexpr uint64_t MIN_HASH_CAPACITY = 32U;
constexpr uint32_t PLAN_BLOCKS = 64U;
constexpr uint32_t INPUT_COUNT = 15U;
constexpr uint32_t LAST_REQ_IDS_INDEX = 7U;
constexpr uint32_t SLOT_TO_TOKEN_INDEX = 8U;
constexpr uint32_t LRU_SLOTS_INDEX = 9U;
constexpr uint32_t CURRENT_SLOTS_INDEX = 10U;
constexpr uint32_t MISS_COUNT_INDEX = 11U;
constexpr uint32_t MISS_TOKENS_INDEX = 12U;
constexpr uint32_t MISS_SLOTS_INDEX = 13U;
constexpr uint32_t COMPACT_WORKSPACE_INDEX = 14U;

uint64_t HashCapacity(int64_t topk) {
  uint64_t target = static_cast<uint64_t>(topk) * 2U;
  uint64_t capacity = MIN_HASH_CAPACITY;
  while (capacity < target) {
    capacity <<= 1U;
  }
  return capacity;
}

ge::graphStatus Fail(gert::TilingContext* context, const char* message) {
  OP_LOGE(context->GetNodeName(), "%s", message);
  return ge::GRAPH_FAILED;
}

bool InputIsContiguous(gert::TilingContext* context, uint32_t index, const gert::Shape& shape) {
  const gert::Stride* stride = context->GetRequiredInputStride(index);
  if (stride == nullptr) {
    stride = context->GetInputStride(index);
  }
  if (stride == nullptr || stride->GetDimNum() != shape.GetDimNum()) {
    return true;
  }
  uint64_t expected = 1U;
  for (int64_t dim = static_cast<int64_t>(shape.GetDimNum()) - 1; dim >= 0; --dim) {
    const int64_t size = shape.GetDim(static_cast<size_t>(dim));
    const int64_t actual = stride->GetStride(static_cast<size_t>(dim));
    if (size > 1 && (actual < 0 || static_cast<uint64_t>(actual) != expected)) {
      return false;
    }
    expected *= static_cast<uint64_t>(size);
  }
  return true;
}

bool InputHasDataType(gert::TilingContext* context, uint32_t index, ge::DataType dataType) {
  const gert::CompileTimeTensorDesc* desc = context->GetInputDesc(index);
  return desc != nullptr && desc->GetDataType() == dataType;
}
}  // namespace

static ge::graphStatus SparseKvPlanTiling(gert::TilingContext* context) {
  const gert::StorageShape* shapes[INPUT_COUNT];
  for (uint32_t index = 0; index < INPUT_COUNT; ++index) {
    shapes[index] = context->GetInputShape(index);
    if (shapes[index] == nullptr) {
      return Fail(context, "SparseKvPlan requires every input shape");
    }
  }
  const auto* attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return Fail(context, "SparseKvPlan requires shapes and attributes");
  }
  const ge::DataType dataTypes[INPUT_COUNT] = {
      ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
      ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
  };
  for (uint32_t index = 0; index < INPUT_COUNT; ++index) {
    if (!InputHasDataType(context, index, dataTypes[index])) {
      return Fail(context, "SparseKvPlan input dtype mismatch");
    }
  }

  const gert::Shape& reqOrigin = shapes[0]->GetOriginShape();
  const gert::Shape& topkOrigin = shapes[1]->GetOriginShape();
  const gert::Shape& stablePrefixOrigin = shapes[2]->GetOriginShape();
  const gert::Shape& visibleSeqOrigin = shapes[3]->GetOriginShape();
  const gert::Shape& tokenToReqOrigin = shapes[4]->GetOriginShape();
  const gert::Shape& blockTableOrigin = shapes[5]->GetOriginShape();
  const gert::Shape& activeRowsOrigin = shapes[6]->GetOriginShape();
  const gert::Shape& lastReqIdsOrigin = shapes[7]->GetOriginShape();
  const gert::Shape& slotToTokenOrigin = shapes[8]->GetOriginShape();
  const gert::Shape& lruSlotsOrigin = shapes[9]->GetOriginShape();
  const gert::Shape& currentSlotsOrigin = shapes[10]->GetOriginShape();
  const gert::Shape& missCountOrigin = shapes[11]->GetOriginShape();
  const gert::Shape& missTokensOrigin = shapes[12]->GetOriginShape();
  const gert::Shape& missSlotsOrigin = shapes[13]->GetOriginShape();
  const gert::Shape& compactWorkspaceOrigin = shapes[14]->GetOriginShape();
  if (reqOrigin.GetDimNum() != 1 || topkOrigin.GetDimNum() != 2 || stablePrefixOrigin.GetDimNum() != 1 ||
      visibleSeqOrigin.GetDimNum() != 1 || tokenToReqOrigin.GetDimNum() != 1 || blockTableOrigin.GetDimNum() != 2 ||
      activeRowsOrigin.GetDimNum() > 1 || lastReqIdsOrigin.GetDimNum() != 1 || slotToTokenOrigin.GetDimNum() != 2 ||
      lruSlotsOrigin.GetDimNum() != 2 || currentSlotsOrigin.GetDimNum() != 2 || missCountOrigin.GetDimNum() != 1 ||
      missTokensOrigin.GetDimNum() != 2 || missSlotsOrigin.GetDimNum() != 2 ||
      (compactWorkspaceOrigin.GetDimNum() != 1 && compactWorkspaceOrigin.GetDimNum() != 2)) {
    return Fail(context, "SparseKvPlan input rank mismatch");
  }

  const int64_t maxRows = topkOrigin.GetDim(0);
  const int64_t maxRequests = reqOrigin.GetDim(0);
  const int64_t maxNumBlocks = blockTableOrigin.GetDim(1);
  const int64_t* topkAttr = attrs->GetInt(0);
  const int64_t* capacityAttr = attrs->GetInt(1);
  const int64_t* maxTokenAttr = attrs->GetInt(2);
  const int64_t* blockSizeAttr = attrs->GetInt(3);
  const int64_t* hostNumBlocksAttr = attrs->GetInt(4);
  if (topkAttr == nullptr || capacityAttr == nullptr || maxTokenAttr == nullptr || blockSizeAttr == nullptr ||
      hostNumBlocksAttr == nullptr) {
    return Fail(context, "SparseKvPlan requires every attribute");
  }
  const int64_t topk = *topkAttr;
  const int64_t capacity = *capacityAttr;
  const int64_t maxToken = *maxTokenAttr;
  const int64_t blockSize = *blockSizeAttr;
  const int64_t hostNumBlocks = *hostNumBlocksAttr;
  if (maxRows <= 0 || maxRequests <= 0 || topk <= 0 || capacity <= 0 || maxToken <= 0 || maxNumBlocks <= 0 ||
      blockSize <= 0 || hostNumBlocks <= 0) {
    return Fail(context, "SparseKvPlan received a non-positive shape or attribute");
  }
  if (blockTableOrigin.GetDim(0) != maxRequests || topkOrigin.GetDim(1) != topk) {
    return Fail(context, "SparseKvPlan input shapes do not match request/topk attributes");
  }
  if (topk > std::numeric_limits<int32_t>::max() / 2 || capacity > std::numeric_limits<int32_t>::max() ||
      maxToken > std::numeric_limits<int32_t>::max() || blockSize > std::numeric_limits<int32_t>::max()) {
    return Fail(context, "SparseKvPlan shape exceeds int32 kernel limits");
  }

  const uint64_t hashCapacity = HashCapacity(topk);
  const uint64_t rowElements = 3U * hashCapacity + static_cast<uint64_t>(capacity) + static_cast<uint64_t>(topk);
  if (static_cast<uint64_t>(maxRows) > std::numeric_limits<uint64_t>::max() / rowElements) {
    return Fail(context, "SparseKvPlan workspace size overflow");
  }
  const bool activeRowsHasValue = activeRowsOrigin.GetDimNum() == 0 || activeRowsOrigin.GetDim(0) >= 1;
  if (stablePrefixOrigin.GetDim(0) < maxRows || visibleSeqOrigin.GetDim(0) < maxRows ||
      tokenToReqOrigin.GetDim(0) < maxRows || !activeRowsHasValue || lastReqIdsOrigin.GetDim(0) < maxRows ||
      slotToTokenOrigin.GetDim(0) < maxRows || lruSlotsOrigin.GetDim(0) < maxRows ||
      currentSlotsOrigin.GetDim(0) < maxRows || missCountOrigin.GetDim(0) < maxRows ||
      missTokensOrigin.GetDim(0) < maxRows || missSlotsOrigin.GetDim(0) < maxRows ||
      slotToTokenOrigin.GetDim(1) != capacity || lruSlotsOrigin.GetDim(1) != capacity ||
      currentSlotsOrigin.GetDim(1) != topk || missTokensOrigin.GetDim(1) != topk || missSlotsOrigin.GetDim(1) != topk) {
    return Fail(context, "SparseKvPlan input shape does not provide the required row storage");
  }

  const bool usesUb = rowElements <= sparse_kv_plan_cfg::PLAN_ROW_UB_LIMIT_ELEMENTS;
  bool workspaceIsValid = false;
  if (compactWorkspaceOrigin.GetDimNum() == 1) {
    const uint64_t workspaceElements = static_cast<uint64_t>(compactWorkspaceOrigin.GetDim(0));
    workspaceIsValid =
        (usesUb && workspaceElements == 1U) || workspaceElements >= static_cast<uint64_t>(maxRows) * rowElements;
  } else {
    const bool isPlaceholder = compactWorkspaceOrigin.GetDim(0) == 1 && compactWorkspaceOrigin.GetDim(1) == 1;
    const bool hasFullRows = compactWorkspaceOrigin.GetDim(0) >= maxRows;
    workspaceIsValid =
        usesUb
            ? isPlaceholder || (hasFullRows && static_cast<uint64_t>(compactWorkspaceOrigin.GetDim(1)) >= rowElements)
            : hasFullRows && static_cast<uint64_t>(compactWorkspaceOrigin.GetDim(1)) == rowElements;
  }
  if (!workspaceIsValid) {
    return Fail(context, "SparseKvPlan compact workspace does not satisfy the UB/GM storage contract");
  }

  const uint32_t mutableIndices[] = {LAST_REQ_IDS_INDEX,  SLOT_TO_TOKEN_INDEX,    LRU_SLOTS_INDEX,
                                     CURRENT_SLOTS_INDEX, MISS_COUNT_INDEX,       MISS_TOKENS_INDEX,
                                     MISS_SLOTS_INDEX,    COMPACT_WORKSPACE_INDEX};
  for (uint32_t index : mutableIndices) {
    if (!InputIsContiguous(context, index, shapes[index]->GetOriginShape())) {
      return Fail(context, "SparseKvPlan mutable input must be contiguous");
    }
  }

  const uint64_t rowBytes = rowElements * sizeof(int32_t);
  const uint64_t alignedRowBytes = (rowBytes + 31U) & ~static_cast<uint64_t>(31U);
  const uint64_t localBytes = sparse_kv_plan_cfg::PLAN_SCAN_BYTES +
                              (alignedRowBytes <= sparse_kv_plan_cfg::PLAN_ROW_UB_LIMIT_BYTES ? alignedRowBytes : 0U);
  if (localBytes > std::numeric_limits<uint32_t>::max()) {
    return Fail(context, "SparseKvPlan local memory size overflow");
  }
  const uint32_t localMemoryBytes = static_cast<uint32_t>(localBytes);
  if (context->SetDynUBufSize(localMemoryBytes) != ge::GRAPH_SUCCESS) {
    return Fail(context, "SparseKvPlan failed to set dynamic UB size");
  }

  SparseKvPlanTilingData tiling;
  tiling.set_hashCapacity(hashCapacity);
  tiling.set_workspaceRowElements(rowElements);
  tiling.set_maxRows(maxRows);
  tiling.set_topk(topk);
  tiling.set_capacity(capacity);
  tiling.set_maxToken(maxToken);
  tiling.set_maxRequests(maxRequests);
  tiling.set_maxNumBlocks(maxNumBlocks);
  tiling.set_hostNumBlocks(hostNumBlocks);
  tiling.set_blockSize(static_cast<int32_t>(blockSize));
  tiling.set_localMemoryBytes(localMemoryBytes);

  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return Fail(context, "SparseKvPlan cannot query platform information");
  }
  const platform_ascendc::PlatformAscendC platform(platformInfo);
  const uint32_t blockDim = std::min<uint32_t>(PLAN_BLOCKS, platform.GetCoreNumAiv());
  if (blockDim == 0) {
    return Fail(context, "SparseKvPlan found zero AIV cores");
  }
  context->SetBlockDim(blockDim);
  size_t* workspaceSize = context->GetWorkspaceSizes(1);
  workspaceSize[0] = platform.GetLibApiWorkSpaceSize();
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseSparseKvPlan(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(SparseKvPlan).Tiling(SparseKvPlanTiling).TilingParse<SparseKvPlanCompileInfo>(ParseSparseKvPlan);
}  // namespace optiling
