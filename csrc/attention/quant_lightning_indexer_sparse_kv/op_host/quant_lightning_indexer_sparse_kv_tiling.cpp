/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "quant_lightning_indexer_sparse_kv_tiling.h"

#include <cstring>
#include <limits>

#include "err/ops_err.h"
#include "platform/soc_spec.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "../../sparse_kv_plan/op_kernel/sparse_kv_plan_config.h"
#include "../op_kernel/quant_lightning_indexer_sparse_kv_template_tiling_key.h"

namespace optiling {
namespace {
constexpr uint32_t QUERY_INDEX = 0U;
constexpr uint32_t KEY_INDEX = 1U;
constexpr uint32_t WEIGHTS_INDEX = 2U;
constexpr uint32_t QUERY_SCALE_INDEX = 3U;
constexpr uint32_t KEY_SCALE_INDEX = 4U;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5U;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 6U;
constexpr uint32_t INDEXER_BLOCK_TABLE_INDEX = 7U;
constexpr uint32_t REQ_IDS_INDEX = 8U;
constexpr uint32_t STABLE_PREFIX_LENS_INDEX = 9U;
constexpr uint32_t VISIBLE_SEQ_LENS_INDEX = 10U;
constexpr uint32_t TOKEN_TO_REQ_INDEX = 11U;
constexpr uint32_t KV_BLOCK_TABLE_INDEX = 12U;
constexpr uint32_t ACTIVE_ROWS_INDEX = 13U;
constexpr uint32_t LAST_REQ_IDS_INDEX = 14U;
constexpr uint32_t SLOT_TO_TOKEN_INDEX = 15U;
constexpr uint32_t LRU_SLOTS_INDEX = 16U;
constexpr uint32_t CURRENT_SLOTS_INDEX = 17U;
constexpr uint32_t MISS_COUNT_INDEX = 18U;
constexpr uint32_t MISS_TOKENS_INDEX = 19U;
constexpr uint32_t MISS_SLOTS_INDEX = 20U;
constexpr uint32_t COMPACT_WORKSPACE_INDEX = 21U;
constexpr uint32_t OUTPUT_INDEX = 0U;

constexpr uint32_t ATTR_QUERY_QUANT_MODE_INDEX = 0U;
constexpr uint32_t ATTR_KEY_QUANT_MODE_INDEX = 1U;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2U;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 3U;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 4U;
constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 5U;
constexpr uint32_t ATTR_PRE_TOKENS_INDEX = 6U;
constexpr uint32_t ATTR_NEXT_TOKENS_INDEX = 7U;
constexpr uint32_t ATTR_KEY_STRIDE0_INDEX = 8U;
constexpr uint32_t ATTR_KEY_SCALE_STRIDE0_INDEX = 9U;
constexpr uint32_t ATTR_CAPACITY_INDEX = 10U;
constexpr uint32_t ATTR_MAX_TOKEN_INDEX = 11U;
constexpr uint32_t ATTR_PLAN_BLOCK_SIZE_INDEX = 12U;
constexpr uint32_t ATTR_HOST_NUM_BLOCKS_INDEX = 13U;

constexpr uint32_t HEADS = 32U;
constexpr uint32_t HEAD_DIM = 128U;
constexpr uint32_t BLOCK_SIZE = 128U;
constexpr int32_t SPARSE_MODE = 3;
constexpr int64_t UNBOUNDED_TOKEN = std::numeric_limits<int64_t>::max();

static_assert(qli_sparse_kv_contract::SCAN_BYTES == sparse_kv_plan_cfg::PLAN_SCAN_BYTES);
static_assert(qli_sparse_kv_contract::ROW_UB_LIMIT_BYTES == sparse_kv_plan_cfg::PLAN_ROW_UB_LIMIT_BYTES);

ge::graphStatus Fail(gert::TilingContext* context, const char* message) {
  OP_LOGE(context->GetNodeName(), "%s", message);
  return ge::GRAPH_FAILED;
}

bool HasDataType(gert::TilingContext* context, uint32_t index, ge::DataType expected) {
  const gert::CompileTimeTensorDesc* desc = context->GetInputDesc(index);
  return desc != nullptr && desc->GetDataType() == expected;
}

bool ShapeEquals(const gert::Shape& shape, const int64_t* dimensions, size_t count) {
  if (shape.GetDimNum() != count) {
    return false;
  }
  for (size_t index = 0; index < count; ++index) {
    if (shape.GetDim(index) != dimensions[index]) {
      return false;
    }
  }
  return true;
}

bool IsContiguous(gert::TilingContext* context, uint32_t index, const gert::Shape& shape) {
  const gert::Stride* stride = context->GetRequiredInputStride(index);
  if (stride == nullptr) {
    stride = context->GetInputStride(index);
  }
  if (stride == nullptr || stride->GetDimNum() != shape.GetDimNum()) {
    return true;
  }
  uint64_t expected = 1U;
  for (int64_t dimension = static_cast<int64_t>(shape.GetDimNum()) - 1; dimension >= 0; --dimension) {
    const int64_t size = shape.GetDim(static_cast<size_t>(dimension));
    const int64_t actual = stride->GetStride(static_cast<size_t>(dimension));
    if (size > 1 && (actual < 0 || static_cast<uint64_t>(actual) != expected)) {
      return false;
    }
    expected *= static_cast<uint64_t>(size);
  }
  return true;
}

bool InnerDimensionsAreContiguous(gert::TilingContext* context, uint32_t index, const gert::Shape& shape) {
  const gert::Stride* stride = context->GetDynamicInputStride(index, 0);
  if (stride == nullptr || stride->GetDimNum() == 0U) {
    return true;
  }
  if (stride->GetDimNum() != shape.GetDimNum()) {
    return false;
  }
  uint64_t expected = 1U;
  for (int64_t dimension = static_cast<int64_t>(shape.GetDimNum()) - 1; dimension >= 1; --dimension) {
    const int64_t size = shape.GetDim(static_cast<size_t>(dimension));
    const int64_t actual = stride->GetStride(static_cast<size_t>(dimension));
    if (size > 1 && (actual < 0 || static_cast<uint64_t>(actual) != expected)) {
      return false;
    }
    expected *= static_cast<uint64_t>(size);
  }
  return true;
}

ge::graphStatus ValidateFixedIndexer(gert::TilingContext* context, QuantLightningIndexerSparseKvInfo& info) {
  const gert::StorageShape* query = context->GetInputShape(QUERY_INDEX);
  const gert::StorageShape* key = context->GetInputShape(KEY_INDEX);
  const gert::StorageShape* weights = context->GetInputShape(WEIGHTS_INDEX);
  const gert::StorageShape* queryScale = context->GetInputShape(QUERY_SCALE_INDEX);
  const gert::StorageShape* keyScale = context->GetInputShape(KEY_SCALE_INDEX);
  const gert::CompileTimeTensorDesc* outputDesc = context->GetOutputDesc(OUTPUT_INDEX);
  const gert::StorageShape* output = context->GetOutputShape(OUTPUT_INDEX);
  if (query == nullptr || key == nullptr || weights == nullptr || queryScale == nullptr || keyScale == nullptr ||
      outputDesc == nullptr || output == nullptr) {
    return Fail(context, "fixed QLI requires all query/key/scale/weight/output descriptions");
  }
  if (!HasDataType(context, QUERY_INDEX, ge::DT_FLOAT8_E4M3FN) ||
      !HasDataType(context, KEY_INDEX, ge::DT_FLOAT8_E4M3FN) || !HasDataType(context, WEIGHTS_INDEX, ge::DT_BF16) ||
      !HasDataType(context, QUERY_SCALE_INDEX, ge::DT_FLOAT) || !HasDataType(context, KEY_SCALE_INDEX, ge::DT_FLOAT) ||
      outputDesc->GetDataType() != ge::DT_INT32) {
    return Fail(context, "fixed QLI requires FP8 query/key, BF16 weights, FP32 scales, and INT32 output");
  }

  const gert::Shape& queryShape = query->GetStorageShape();
  const gert::Shape& keyShape = key->GetStorageShape();
  const gert::Shape& weightsShape = weights->GetStorageShape();
  const gert::Shape& queryScaleShape = queryScale->GetStorageShape();
  const gert::Shape& keyScaleShape = keyScale->GetStorageShape();
  const gert::Shape& outputShape = output->GetStorageShape();
  if (queryShape.GetDimNum() != 3U || queryShape.GetDim(0) <= 0 || queryShape.GetDim(1) != HEADS ||
      queryShape.GetDim(2) != HEAD_DIM || queryShape.GetDim(0) > std::numeric_limits<uint32_t>::max()) {
    return Fail(context, "query must have fixed positive shape [T,32,128]");
  }
  const int64_t tokenCount = queryShape.GetDim(0);
  const int64_t keyDimensions[] = {keyShape.GetDimNum() == 4U ? keyShape.GetDim(0) : 0, BLOCK_SIZE, 1, HEAD_DIM};
  if (keyDimensions[0] <= 0 || !ShapeEquals(keyShape, keyDimensions, 4U)) {
    return Fail(context, "key must have fixed positive shape [blocks,128,1,128]");
  }
  const int64_t tokenHeadDimensions[] = {tokenCount, HEADS};
  const int64_t keyScaleDimensions[] = {keyDimensions[0], BLOCK_SIZE, 1};
  const int64_t outputDimensions[] = {tokenCount, 1, qli_sparse_kv_contract::TOPK};
  if (!ShapeEquals(weightsShape, tokenHeadDimensions, 2U) || !ShapeEquals(queryScaleShape, tokenHeadDimensions, 2U) ||
      !ShapeEquals(keyScaleShape, keyScaleDimensions, 3U) || !ShapeEquals(outputShape, outputDimensions, 3U)) {
    return Fail(context, "weights, scales, or output do not match the fixed QLI shape contract");
  }
  if (!InnerDimensionsAreContiguous(context, KEY_INDEX, keyShape) ||
      !InnerDimensionsAreContiguous(context, KEY_SCALE_INDEX, keyScaleShape)) {
    return Fail(context, "key and key scale may be strided only on axis zero");
  }

  const gert::Tensor* actualQ = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
  const gert::Tensor* actualK = context->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
  const gert::Tensor* indexerBlockTable = context->GetOptionalInputTensor(INDEXER_BLOCK_TABLE_INDEX);
  const gert::CompileTimeTensorDesc* actualQDesc = context->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
  const gert::CompileTimeTensorDesc* actualKDesc = context->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
  const gert::CompileTimeTensorDesc* indexerBlockTableDesc = context->GetOptionalInputDesc(INDEXER_BLOCK_TABLE_INDEX);
  if (actualQ == nullptr || actualK == nullptr || indexerBlockTable == nullptr || actualQDesc == nullptr ||
      actualKDesc == nullptr || indexerBlockTableDesc == nullptr || actualQDesc->GetDataType() != ge::DT_INT32 ||
      actualKDesc->GetDataType() != ge::DT_INT32 || indexerBlockTableDesc->GetDataType() != ge::DT_INT32) {
    return Fail(context, "TND/PA_BSND requires INT32 sequence lengths and indexer block table");
  }
  const int64_t batchSize = actualQ->GetShapeSize();
  const gert::Shape& indexerBlockShape = indexerBlockTable->GetStorageShape();
  if (batchSize <= 0 || batchSize > std::numeric_limits<uint32_t>::max() || actualK->GetShapeSize() != batchSize ||
      indexerBlockShape.GetDimNum() != 2U || indexerBlockShape.GetDim(0) != batchSize ||
      indexerBlockShape.GetDim(1) <= 0 || indexerBlockShape.GetDim(1) > std::numeric_limits<uint32_t>::max()) {
    return Fail(context, "sequence lengths and indexer block table must describe the same positive batch");
  }

  const auto* attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return Fail(context, "fixed QLI attributes are missing");
  }
  const int32_t* queryQuantMode = attrs->GetAttrPointer<int32_t>(ATTR_QUERY_QUANT_MODE_INDEX);
  const int32_t* keyQuantMode = attrs->GetAttrPointer<int32_t>(ATTR_KEY_QUANT_MODE_INDEX);
  const char* queryLayout = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
  const char* keyLayout = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
  const int32_t* sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX);
  const int32_t* sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
  const int64_t* preTokens = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX);
  const int64_t* nextTokens = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX);
  const int64_t* keyStride = attrs->GetAttrPointer<int64_t>(ATTR_KEY_STRIDE0_INDEX);
  const int64_t* keyScaleStride = attrs->GetAttrPointer<int64_t>(ATTR_KEY_SCALE_STRIDE0_INDEX);
  const int64_t* capacity = attrs->GetInt(ATTR_CAPACITY_INDEX);
  const int64_t* maxToken = attrs->GetInt(ATTR_MAX_TOKEN_INDEX);
  const int64_t* planBlockSize = attrs->GetInt(ATTR_PLAN_BLOCK_SIZE_INDEX);
  const int64_t* hostNumBlocks = attrs->GetInt(ATTR_HOST_NUM_BLOCKS_INDEX);
  if (queryQuantMode == nullptr || keyQuantMode == nullptr || queryLayout == nullptr || keyLayout == nullptr ||
      sparseCount == nullptr || sparseMode == nullptr || preTokens == nullptr || nextTokens == nullptr ||
      keyStride == nullptr || keyScaleStride == nullptr || capacity == nullptr || maxToken == nullptr ||
      planBlockSize == nullptr || hostNumBlocks == nullptr) {
    return Fail(context, "fixed QLI requires all fourteen attributes");
  }
  if (*queryQuantMode != 0 || *keyQuantMode != 0 || std::strcmp(queryLayout, "TND") != 0 ||
      std::strcmp(keyLayout, "PA_BSND") != 0 || *sparseCount != static_cast<int32_t>(qli_sparse_kv_contract::TOPK) ||
      *sparseMode != SPARSE_MODE || *preTokens != UNBOUNDED_TOKEN || *nextTokens != UNBOUNDED_TOKEN ||
      (*keyStride < 0 && *keyStride != -1) || (*keyScaleStride < 0 && *keyScaleStride != -1) ||
      *keyStride > std::numeric_limits<uint32_t>::max() || *keyScaleStride > std::numeric_limits<uint32_t>::max() ||
      *capacity < qli_sparse_kv_contract::TOPK || *capacity > std::numeric_limits<int32_t>::max() || *maxToken <= 0 ||
      *maxToken > std::numeric_limits<int32_t>::max() || *planBlockSize != BLOCK_SIZE || *hostNumBlocks <= 0) {
    return Fail(context, "attribute values are outside the fixed QLI contract");
  }

  info.bSize = static_cast<uint32_t>(batchSize);
  info.tSize = static_cast<uint32_t>(tokenCount);
  info.maxBlockNumPerBatch = static_cast<uint32_t>(indexerBlockShape.GetDim(1));
  const uint64_t sequenceLength = static_cast<uint64_t>(info.maxBlockNumPerBatch) * BLOCK_SIZE;
  if (sequenceLength > std::numeric_limits<uint32_t>::max()) {
    return Fail(context, "page-attention sequence length exceeds uint32 tiling limits");
  }
  info.s2Size = static_cast<uint32_t>(sequenceLength);
  info.keyStride0 = qli_sparse_kv_contract::ResolveStride(*keyStride, BLOCK_SIZE * HEAD_DIM);
  info.keyDequantScaleStride0 = qli_sparse_kv_contract::ResolveStride(*keyScaleStride, BLOCK_SIZE);
  info.capacity = *capacity;
  info.maxToken = *maxToken;
  info.maxRequests = batchSize;
  info.hostNumBlocks = *hostNumBlocks;
  info.planner = qli_sparse_kv_contract::ComputePlannerSizing(info.capacity, info.tSize);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidatePlanner(gert::TilingContext* context, QuantLightningIndexerSparseKvInfo& info) {
  constexpr uint32_t indices[] = {
      REQ_IDS_INDEX,     STABLE_PREFIX_LENS_INDEX, VISIBLE_SEQ_LENS_INDEX, TOKEN_TO_REQ_INDEX,     KV_BLOCK_TABLE_INDEX,
      ACTIVE_ROWS_INDEX, LAST_REQ_IDS_INDEX,       SLOT_TO_TOKEN_INDEX,    LRU_SLOTS_INDEX,        CURRENT_SLOTS_INDEX,
      MISS_COUNT_INDEX,  MISS_TOKENS_INDEX,        MISS_SLOTS_INDEX,       COMPACT_WORKSPACE_INDEX};
  constexpr ge::DataType types[] = {ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                    ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  const gert::StorageShape* shapes[sizeof(indices) / sizeof(indices[0])]{};
  for (size_t index = 0; index < sizeof(indices) / sizeof(indices[0]); ++index) {
    shapes[index] = context->GetInputShape(indices[index]);
    if (shapes[index] == nullptr || !HasDataType(context, indices[index], types[index])) {
      return Fail(context, "planner inputs require their fixed shapes and integer dtypes");
    }
  }
  const gert::Shape& req = shapes[0]->GetOriginShape();
  const gert::Shape& stable = shapes[1]->GetOriginShape();
  const gert::Shape& visible = shapes[2]->GetOriginShape();
  const gert::Shape& tokenToReq = shapes[3]->GetOriginShape();
  const gert::Shape& blockTable = shapes[4]->GetOriginShape();
  const gert::Shape& activeRows = shapes[5]->GetOriginShape();
  const gert::Shape& lastReq = shapes[6]->GetOriginShape();
  const gert::Shape& slotToToken = shapes[7]->GetOriginShape();
  const gert::Shape& lruSlots = shapes[8]->GetOriginShape();
  const gert::Shape& currentSlots = shapes[9]->GetOriginShape();
  const gert::Shape& missCount = shapes[10]->GetOriginShape();
  const gert::Shape& missTokens = shapes[11]->GetOriginShape();
  const gert::Shape& missSlots = shapes[12]->GetOriginShape();
  const gert::Shape& workspace = shapes[13]->GetOriginShape();
  const bool ranksValid = req.GetDimNum() == 1U && stable.GetDimNum() == 1U && visible.GetDimNum() == 1U &&
                          tokenToReq.GetDimNum() == 1U && blockTable.GetDimNum() == 2U &&
                          activeRows.GetDimNum() <= 1U && lastReq.GetDimNum() == 1U && slotToToken.GetDimNum() == 2U &&
                          lruSlots.GetDimNum() == 2U && currentSlots.GetDimNum() == 2U && missCount.GetDimNum() == 1U &&
                          missTokens.GetDimNum() == 2U && missSlots.GetDimNum() == 2U && workspace.GetDimNum() == 2U;
  if (!ranksValid) {
    return Fail(context, "planner input rank mismatch");
  }
  const bool activeRowsHasValue = activeRows.GetDimNum() == 0U || activeRows.GetDim(0) >= 1;
  if (req.GetDim(0) != info.maxRequests ||
      !qli_sparse_kv_contract::PlannerBlockTableShapeIsValid(blockTable.GetDim(0), blockTable.GetDim(1),
                                                             info.maxRequests) ||
      stable.GetDim(0) < info.tSize || visible.GetDim(0) < info.tSize || tokenToReq.GetDim(0) < info.tSize ||
      !activeRowsHasValue || lastReq.GetDim(0) < info.tSize || slotToToken.GetDim(0) < info.tSize ||
      slotToToken.GetDim(1) != info.capacity || lruSlots.GetDim(0) < info.tSize ||
      lruSlots.GetDim(1) != info.capacity || currentSlots.GetDim(0) < info.tSize ||
      currentSlots.GetDim(1) != qli_sparse_kv_contract::TOPK || missCount.GetDim(0) < info.tSize ||
      missTokens.GetDim(0) < info.tSize || missTokens.GetDim(1) != qli_sparse_kv_contract::TOPK ||
      missSlots.GetDim(0) < info.tSize || missSlots.GetDim(1) != qli_sparse_kv_contract::TOPK ||
      !qli_sparse_kv_contract::CompactWorkspaceShapeIsValid(workspace.GetDim(0), workspace.GetDim(1), info.planner,
                                                            info.tSize)) {
    return Fail(context, "planner input shape does not satisfy the fixed row-storage contract");
  }
  constexpr size_t mutableShapeOffsets[] = {6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U};
  for (size_t offset : mutableShapeOffsets) {
    if (!IsContiguous(context, indices[offset], shapes[offset]->GetOriginShape())) {
      return Fail(context, "planner mutable inputs must be contiguous");
    }
  }
  info.maxNumBlocks = blockTable.GetDim(1);
  return ge::GRAPH_SUCCESS;
}

uint64_t WorkspaceSize(const platform_ascendc::PlatformAscendC& platform, const QuantLightningIndexerSparseKvInfo& info,
                       uint32_t aicCount) {
  return platform.GetLibApiWorkSpaceSize() + qli_sparse_kv_contract::AdditionalWorkspaceBytes(info.s2Size, aicCount);
}

ge::graphStatus WriteTiling(gert::TilingContext* context, const QuantLightningIndexerSparseKvInfo& info) {
  const platform_ascendc::PlatformAscendC platform(info.platformInfo);
  const uint32_t aivCount = platform.GetCoreNumAiv();
  const uint32_t aicCount = platform.GetCoreNumAic();
  const uint32_t blockDim = platform.CalcTschBlockDim(aivCount, aicCount, aivCount);
  context->SetBlockDim(blockDim);
  context->GetWorkspaceSizes(1)[0] = WorkspaceSize(platform, info, aicCount);

  if (context->SetDynUBufSize(info.planner.dynamicUbBytes) != ge::GRAPH_SUCCESS) {
    return Fail(context, "failed to set fixed fused dynamic UB size");
  }

  QuantLightningIndexerSparseKvTilingData tiling;
  tiling.set_bSize(info.bSize);
  tiling.set_tSize(info.tSize);
  tiling.set_n2Size(1U);
  tiling.set_gSize(HEADS);
  tiling.set_s1Size(0U);
  tiling.set_s2Size(info.s2Size);
  tiling.set_sparseCount(qli_sparse_kv_contract::TOPK);
  tiling.set_keyStride0(info.keyStride0);
  tiling.set_keyDequantScaleStride0(info.keyDequantScaleStride0);
  tiling.set_usedCoreNum(blockDim);
  tiling.set_blockSize(BLOCK_SIZE);
  tiling.set_maxBlockNumPerBatch(info.maxBlockNumPerBatch);
  tiling.set_sparseMode(SPARSE_MODE);
  tiling.set_hashCapacity(info.planner.hashCapacity);
  tiling.set_workspaceRowElements(info.planner.workspaceRowElements);
  tiling.set_maxRows(info.tSize);
  tiling.set_topk(qli_sparse_kv_contract::TOPK);
  tiling.set_capacity(info.capacity);
  tiling.set_maxToken(info.maxToken);
  tiling.set_maxRequests(info.maxRequests);
  tiling.set_maxNumBlocks(info.maxNumBlocks);
  tiling.set_hostNumBlocks(info.hostNumBlocks);
  tiling.set_planBlockSize(BLOCK_SIZE);
  tiling.set_planLocalMemoryBytes(info.planner.planLocalMemoryBytes);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  const uint32_t tilingKey =
      GET_TPL_TILING_KEY(static_cast<uint32_t>(ge::DT_FLOAT8_E4M3FN), static_cast<uint32_t>(ge::DT_FLOAT8_E4M3FN),
                         static_cast<uint32_t>(ge::DT_INT32), 1U, 1U, 2U);
  context->SetTilingKey(tilingKey);
  context->SetScheduleMode(1);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus Prepare(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
}  // namespace

ge::graphStatus TilingForQuantLightningIndexerSparseKv(gert::TilingContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context->GetNodeName() == nullptr) {
    OP_LOGE("QuantLightningIndexerSparseKv", "operator name is null");
    return ge::GRAPH_FAILED;
  }
  QuantLightningIndexerSparseKvInfo info;
  info.platformInfo = context->GetPlatformInfo();
  if (info.platformInfo == nullptr || context->GetWorkspaceSizes(1) == nullptr ||
      context->GetRawTilingData() == nullptr) {
    return Fail(context, "fixed QLI requires platform, workspace, and raw tiling buffers");
  }
  const platform_ascendc::PlatformAscendC platform(info.platformInfo);
  if (platform.GetCurNpuArch() != NpuArch::DAV_3510 || platform.GetCoreNumAic() == 0U ||
      platform.GetCoreNumAiv() == 0U) {
    return Fail(context, "fixed QLI supports only DAV3510 with nonzero AIC/AIV cores");
  }
  if (ValidateFixedIndexer(context, info) != ge::GRAPH_SUCCESS || ValidatePlanner(context, info) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return WriteTiling(context, info);
}

IMPL_OP_OPTILING(QuantLightningIndexerSparseKv)
    .Tiling(TilingForQuantLightningIndexerSparseKv)
    .TilingParse<QuantLightningIndexerSparseKvCompileInfo>(Prepare);

}  // namespace optiling
