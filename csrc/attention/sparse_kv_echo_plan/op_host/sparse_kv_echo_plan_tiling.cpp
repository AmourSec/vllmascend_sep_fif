/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sparse_kv_echo_plan_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sparse_kv_echo_plan_config.h"

namespace optiling {
namespace {
ge::graphStatus Fail(gert::TilingContext* context, const char* message) {
  OP_LOGE(context->GetNodeName(), "%s", message);
  return ge::GRAPH_FAILED;
}
}  // namespace

static ge::graphStatus SparseKvEchoPlanTiling(gert::TilingContext* context) {
  const gert::StorageShape* reqShape = context->GetInputShape(0);
  const gert::StorageShape* topkShape = context->GetInputShape(1);
  const gert::StorageShape* blockTableShape = context->GetInputShape(5);
  const gert::StorageShape* priorityShape = context->GetInputShape(9);
  const gert::StorageShape* workspaceShape = context->GetInputShape(14);
  const auto* attrs = context->GetAttrs();
  if (reqShape == nullptr || topkShape == nullptr || blockTableShape == nullptr || priorityShape == nullptr ||
      workspaceShape == nullptr || attrs == nullptr) {
    return Fail(context, "SparseKvEchoPlan requires shapes and attributes");
  }
  const gert::Shape& reqOrigin = reqShape->GetOriginShape();
  const gert::Shape& topkOrigin = topkShape->GetOriginShape();
  const gert::Shape& blockTableOrigin = blockTableShape->GetOriginShape();
  if (reqOrigin.GetDimNum() != 1 || topkOrigin.GetDimNum() != 2 || blockTableOrigin.GetDimNum() != 2) {
    return Fail(context, "SparseKvEchoPlan expects req_ids rank 1 and topk/block_table rank 2");
  }

  const int64_t maxRows = topkOrigin.GetDim(0);
  const int64_t maxRequests = reqOrigin.GetDim(0);
  const int64_t maxNumBlocks = blockTableOrigin.GetDim(1);
  const int64_t topk = *attrs->GetInt(0);
  const int64_t capacity = *attrs->GetInt(1);
  const int64_t maxToken = *attrs->GetInt(2);
  const int64_t blockSize = *attrs->GetInt(3);
  const int64_t hostNumBlocks = *attrs->GetInt(4);
  if (maxRows <= 0 || maxRequests <= 0 || topk <= 0 || capacity <= 0 || maxToken <= 0 || maxNumBlocks <= 0 ||
      blockSize <= 0 || hostNumBlocks <= 0) {
    return Fail(context, "SparseKvEchoPlan received a non-positive shape or attribute");
  }
  if (blockTableOrigin.GetDim(0) != maxRequests || topkOrigin.GetDim(1) != topk) {
    return Fail(context, "SparseKvEchoPlan input shapes do not match request/topk attributes");
  }
  if (topk > capacity || capacity > std::numeric_limits<int32_t>::max() ||
      maxToken > std::numeric_limits<int32_t>::max() || blockSize > std::numeric_limits<int32_t>::max()) {
    return Fail(context, "SparseKvEchoPlan shape exceeds int32 kernel limits");
  }
  const auto& priorityOrigin = priorityShape->GetOriginShape();
  if (priorityOrigin.GetDimNum() != 2 || priorityOrigin.GetDim(0) < maxRows ||
      priorityOrigin.GetDim(1) != capacity + 1) {
    return Fail(context, "SparseKvEchoPlan priorities must have shape [at least rows, capacity + 1]");
  }
  const uint64_t blockElements = sparse_kv_echo_plan::WorkspaceBlockElements(maxToken, topk, capacity);
  const uint32_t maxBlocks = std::min<int64_t>(maxRows, sparse_kv_echo_plan::PLAN_MAX_BLOCKS);
  if (workspaceShape->GetOriginShape().GetShapeSize() < static_cast<int64_t>(maxBlocks * blockElements)) {
    return Fail(context, "SparseKvEchoPlan workspace is smaller than the per-AIV direct map and allocation scratch");
  }
  const uint32_t localMemoryBytes = sparse_kv_echo_plan::LocalMemoryBytes(capacity, maxToken);
  if (context->SetDynUBufSize(localMemoryBytes) != ge::GRAPH_SUCCESS) {
    return Fail(context, "SparseKvEchoPlan failed to set dynamic UB size");
  }

  SparseKvEchoPlanTilingData tiling;
  tiling.set_workspaceBlockElements(blockElements);
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
    return Fail(context, "SparseKvEchoPlan cannot query platform information");
  }
  const platform_ascendc::PlatformAscendC platform(platformInfo);
  const uint32_t blockDim = std::min<uint32_t>(maxBlocks, platform.GetCoreNumAiv());
  if (blockDim == 0) {
    return Fail(context, "SparseKvEchoPlan found zero AIV cores");
  }
  context->SetBlockDim(blockDim);
  size_t* workspaceSize = context->GetWorkspaceSizes(1);
  workspaceSize[0] = platform.GetLibApiWorkSpaceSize();
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ParseSparseKvEchoPlan(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(SparseKvEchoPlan)
    .Tiling(SparseKvEchoPlanTiling)
    .TilingParse<SparseKvEchoPlanCompileInfo>(ParseSparseKvEchoPlan);
}  // namespace optiling
