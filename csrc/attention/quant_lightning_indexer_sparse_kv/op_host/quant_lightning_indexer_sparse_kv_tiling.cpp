/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_lightning_indexer_tiling.cpp
 * \brief
 */

#include "quant_lightning_indexer_sparse_kv_tiling.h"

#include <algorithm>
#include <limits>

#include "../op_kernel/quant_lightning_indexer_sparse_kv_template_tiling_key.h"
#include "../../sparse_kv_plan/op_kernel/sparse_kv_plan_config.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {

namespace {
constexpr uint32_t REQ_IDS_INDEX = 8;
constexpr uint32_t STABLE_PREFIX_LENS_INDEX = 9;
constexpr uint32_t VISIBLE_SEQ_LENS_INDEX = 10;
constexpr uint32_t TOKEN_TO_REQ_INDEX = 11;
constexpr uint32_t KV_BLOCK_TABLE_INDEX = 12;
constexpr uint32_t ACTIVE_ROWS_INDEX = 13;
constexpr uint32_t LAST_REQ_IDS_INDEX = 14;
constexpr uint32_t SLOT_TO_TOKEN_INDEX = 15;
constexpr uint32_t LRU_SLOTS_INDEX = 16;
constexpr uint32_t CURRENT_SLOTS_INDEX = 17;
constexpr uint32_t MISS_COUNT_INDEX = 18;
constexpr uint32_t MISS_TOKENS_INDEX = 19;
constexpr uint32_t MISS_SLOTS_INDEX = 20;
constexpr uint32_t COMPACT_WORKSPACE_INDEX = 21;
constexpr uint32_t ATTR_CAPACITY_INDEX = 10;
constexpr uint32_t ATTR_MAX_TOKEN_INDEX = 11;
constexpr uint32_t ATTR_PLAN_BLOCK_SIZE_INDEX = 12;
constexpr uint32_t ATTR_HOST_NUM_BLOCKS_INDEX = 13;
constexpr uint64_t MIN_HASH_CAPACITY = 32U;
constexpr uint32_t QLI_UB_LARGE_T_BYTES = 186880U;
constexpr uint32_t QLI_UB_SMALL_T_BYTES = 150528U;
constexpr uint32_t PLAN_SCAN_UB_BYTES = 544U;
constexpr uint32_t QLI_TOPK_SLAB_BYTES = 89344U;
constexpr uint32_t TOTAL_UB_BYTES = 256U * 1024U;
constexpr uint32_t SYSTEM_UB_RESERVE_BYTES = 8U * 1024U;
constexpr uint32_t MIN_DATA_CACHE_BYTES = 32U * 1024U;
static_assert(QLI_UB_LARGE_T_BYTES + PLAN_SCAN_UB_BYTES <=
              TOTAL_UB_BYTES - SYSTEM_UB_RESERVE_BYTES - MIN_DATA_CACHE_BYTES);
static_assert(QLI_UB_LARGE_T_BYTES + PLAN_SCAN_UB_BYTES +
                  sparse_kv_plan_cfg::PLAN_ROW_UB_LIMIT_BYTES - QLI_TOPK_SLAB_BYTES <=
              TOTAL_UB_BYTES - SYSTEM_UB_RESERVE_BYTES - MIN_DATA_CACHE_BYTES);

uint64_t PlannerHashCapacity(int64_t topk)
{
    uint64_t target = static_cast<uint64_t>(topk) * 2U;
    uint64_t capacity = MIN_HASH_CAPACITY;
    while (capacity < target) {
        capacity <<= 1U;
    }
    return capacity;
}

bool PlannerInputIsContiguous(gert::TilingContext* context, uint32_t index, const gert::Shape& shape)
{
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

bool PlannerInputHasDataType(gert::TilingContext* context, uint32_t index, ge::DataType dataType)
{
    const gert::CompileTimeTensorDesc* desc = context->GetInputDesc(index);
    return desc != nullptr && desc->GetDataType() == dataType;
}

ge::graphStatus ParsePlannerInfo(gert::TilingContext* context, QuantLightningIndexerSparseKvInfo& info)
{
    const gert::StorageShape* reqShape = context->GetInputShape(REQ_IDS_INDEX);
    const gert::StorageShape* stablePrefixShape = context->GetInputShape(STABLE_PREFIX_LENS_INDEX);
    const gert::StorageShape* visibleSeqShape = context->GetInputShape(VISIBLE_SEQ_LENS_INDEX);
    const gert::StorageShape* tokenToReqShape = context->GetInputShape(TOKEN_TO_REQ_INDEX);
    const gert::StorageShape* kvBlockTableShape = context->GetInputShape(KV_BLOCK_TABLE_INDEX);
    const gert::StorageShape* activeRowsShape = context->GetInputShape(ACTIVE_ROWS_INDEX);
    const gert::StorageShape* lastReqIdsShape = context->GetInputShape(LAST_REQ_IDS_INDEX);
    const gert::StorageShape* slotToTokenShape = context->GetInputShape(SLOT_TO_TOKEN_INDEX);
    const gert::StorageShape* lruSlotsShape = context->GetInputShape(LRU_SLOTS_INDEX);
    const gert::StorageShape* currentSlotsShape = context->GetInputShape(CURRENT_SLOTS_INDEX);
    const gert::StorageShape* missCountShape = context->GetInputShape(MISS_COUNT_INDEX);
    const gert::StorageShape* missTokensShape = context->GetInputShape(MISS_TOKENS_INDEX);
    const gert::StorageShape* missSlotsShape = context->GetInputShape(MISS_SLOTS_INDEX);
    const gert::StorageShape* compactWorkspaceShape = context->GetInputShape(COMPACT_WORKSPACE_INDEX);
    const auto* attrs = context->GetAttrs();
    if (reqShape == nullptr || stablePrefixShape == nullptr || visibleSeqShape == nullptr ||
        tokenToReqShape == nullptr || kvBlockTableShape == nullptr || activeRowsShape == nullptr ||
        lastReqIdsShape == nullptr || slotToTokenShape == nullptr || lruSlotsShape == nullptr ||
        currentSlotsShape == nullptr || missCountShape == nullptr || missTokensShape == nullptr ||
        missSlotsShape == nullptr || compactWorkspaceShape == nullptr || attrs == nullptr) {
        OP_LOGE(context->GetNodeName(), "planner shapes or attributes are null");
        return ge::GRAPH_FAILED;
    }
    const uint32_t plannerIndices[] = {REQ_IDS_INDEX, STABLE_PREFIX_LENS_INDEX, VISIBLE_SEQ_LENS_INDEX,
                                       TOKEN_TO_REQ_INDEX, KV_BLOCK_TABLE_INDEX, ACTIVE_ROWS_INDEX,
                                       LAST_REQ_IDS_INDEX, SLOT_TO_TOKEN_INDEX, LRU_SLOTS_INDEX,
                                       CURRENT_SLOTS_INDEX, MISS_COUNT_INDEX, MISS_TOKENS_INDEX,
                                       MISS_SLOTS_INDEX, COMPACT_WORKSPACE_INDEX};
    const ge::DataType plannerDataTypes[] = {ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                             ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
                                             ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                             ge::DT_INT32, ge::DT_INT32};
    for (size_t index = 0; index < sizeof(plannerIndices) / sizeof(plannerIndices[0]); ++index) {
        if (!PlannerInputHasDataType(context, plannerIndices[index], plannerDataTypes[index])) {
            OP_LOGE(context->GetNodeName(), "planner input dtype mismatch");
            return ge::GRAPH_FAILED;
        }
    }
    const gert::Shape& reqOrigin = reqShape->GetOriginShape();
    const gert::Shape& stablePrefixOrigin = stablePrefixShape->GetOriginShape();
    const gert::Shape& visibleSeqOrigin = visibleSeqShape->GetOriginShape();
    const gert::Shape& tokenToReqOrigin = tokenToReqShape->GetOriginShape();
    const gert::Shape& kvBlockTableOrigin = kvBlockTableShape->GetOriginShape();
    const gert::Shape& activeRowsOrigin = activeRowsShape->GetOriginShape();
    const gert::Shape& lastReqIdsOrigin = lastReqIdsShape->GetOriginShape();
    const gert::Shape& slotToTokenOrigin = slotToTokenShape->GetOriginShape();
    const gert::Shape& lruSlotsOrigin = lruSlotsShape->GetOriginShape();
    const gert::Shape& currentSlotsOrigin = currentSlotsShape->GetOriginShape();
    const gert::Shape& missCountOrigin = missCountShape->GetOriginShape();
    const gert::Shape& missTokensOrigin = missTokensShape->GetOriginShape();
    const gert::Shape& missSlotsOrigin = missSlotsShape->GetOriginShape();
    const gert::Shape& compactWorkspaceOrigin = compactWorkspaceShape->GetOriginShape();
    if (reqOrigin.GetDimNum() != 1 || stablePrefixOrigin.GetDimNum() != 1 ||
        visibleSeqOrigin.GetDimNum() != 1 || tokenToReqOrigin.GetDimNum() != 1 ||
        kvBlockTableOrigin.GetDimNum() != 2 || activeRowsOrigin.GetDimNum() > 1 ||
        lastReqIdsOrigin.GetDimNum() != 1 || slotToTokenOrigin.GetDimNum() != 2 ||
        lruSlotsOrigin.GetDimNum() != 2 || currentSlotsOrigin.GetDimNum() != 2 ||
        missCountOrigin.GetDimNum() != 1 || missTokensOrigin.GetDimNum() != 2 ||
        missSlotsOrigin.GetDimNum() != 2 || compactWorkspaceOrigin.GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "planner input rank mismatch");
        return ge::GRAPH_FAILED;
    }
    const int64_t* capacity = attrs->GetInt(ATTR_CAPACITY_INDEX);
    const int64_t* maxToken = attrs->GetInt(ATTR_MAX_TOKEN_INDEX);
    const int64_t* planBlockSize = attrs->GetInt(ATTR_PLAN_BLOCK_SIZE_INDEX);
    const int64_t* hostNumBlocks = attrs->GetInt(ATTR_HOST_NUM_BLOCKS_INDEX);
    if (capacity == nullptr || maxToken == nullptr || planBlockSize == nullptr || hostNumBlocks == nullptr) {
        OP_LOGE(context->GetNodeName(), "planner attributes are null");
        return ge::GRAPH_FAILED;
    }
    info.maxRows = info.tSize;
    info.topk = info.sparseCount;
    info.capacity = *capacity;
    info.maxToken = *maxToken;
    info.maxRequests = reqOrigin.GetDim(0);
    info.maxNumBlocks = kvBlockTableOrigin.GetDim(1);
    info.hostNumBlocks = *hostNumBlocks;
    if (*planBlockSize <= 0 || *planBlockSize > std::numeric_limits<int32_t>::max()) {
        OP_LOGE(context->GetNodeName(), "planner block_size is outside int32 positive range");
        return ge::GRAPH_FAILED;
    }
    info.planBlockSize = static_cast<int32_t>(*planBlockSize);
    if (info.maxRows <= 0 || info.topk <= 0 || info.capacity < info.topk || info.maxToken <= 0 ||
        info.maxRequests <= 0 || info.maxNumBlocks <= 0 || info.hostNumBlocks <= 0 || info.planBlockSize <= 0 ||
        kvBlockTableOrigin.GetDim(0) != info.maxRequests || info.maxRequests != info.bSize ||
        info.planBlockSize != info.blockSize) {
        OP_LOGE(context->GetNodeName(), "planner received an invalid shape or attribute");
        return ge::GRAPH_FAILED;
    }
    if (info.topk > std::numeric_limits<int32_t>::max() / 2 ||
        info.capacity > std::numeric_limits<int32_t>::max() ||
        info.maxToken > std::numeric_limits<int32_t>::max()) {
        OP_LOGE(context->GetNodeName(), "planner dimensions exceed int32 kernel limits");
        return ge::GRAPH_FAILED;
    }
    info.hashCapacity = PlannerHashCapacity(info.topk);
    info.workspaceRowElements =
        3U * info.hashCapacity + 2U * static_cast<uint64_t>(info.capacity) + static_cast<uint64_t>(info.topk);
    const bool activeRowsHasValue = activeRowsOrigin.GetDimNum() == 0 || activeRowsOrigin.GetDim(0) >= 1;
    if (stablePrefixOrigin.GetDim(0) < info.maxRows || visibleSeqOrigin.GetDim(0) < info.maxRows ||
        tokenToReqOrigin.GetDim(0) < info.maxRows || !activeRowsHasValue ||
        lastReqIdsOrigin.GetDim(0) < info.maxRows || slotToTokenOrigin.GetDim(0) < info.maxRows ||
        lruSlotsOrigin.GetDim(0) < info.maxRows || currentSlotsOrigin.GetDim(0) < info.maxRows ||
        missCountOrigin.GetDim(0) < info.maxRows || missTokensOrigin.GetDim(0) < info.maxRows ||
        missSlotsOrigin.GetDim(0) < info.maxRows || compactWorkspaceOrigin.GetDim(0) < info.maxRows ||
        slotToTokenOrigin.GetDim(1) != info.capacity || lruSlotsOrigin.GetDim(1) != info.capacity ||
        currentSlotsOrigin.GetDim(1) != info.topk || missTokensOrigin.GetDim(1) != info.topk ||
        missSlotsOrigin.GetDim(1) != info.topk ||
        compactWorkspaceOrigin.GetDim(1) < static_cast<int64_t>(info.workspaceRowElements)) {
        OP_LOGE(context->GetNodeName(), "planner input shape does not provide the required row storage");
        return ge::GRAPH_FAILED;
    }
    const uint32_t mutableIndices[] = {LAST_REQ_IDS_INDEX, SLOT_TO_TOKEN_INDEX, LRU_SLOTS_INDEX,
                                       CURRENT_SLOTS_INDEX, MISS_COUNT_INDEX, MISS_TOKENS_INDEX,
                                       MISS_SLOTS_INDEX, COMPACT_WORKSPACE_INDEX};
    const gert::Shape* mutableShapes[] = {&lastReqIdsOrigin, &slotToTokenOrigin, &lruSlotsOrigin,
                                         &currentSlotsOrigin, &missCountOrigin, &missTokensOrigin,
                                         &missSlotsOrigin, &compactWorkspaceOrigin};
    for (size_t index = 0; index < sizeof(mutableIndices) / sizeof(mutableIndices[0]); ++index) {
        if (!PlannerInputIsContiguous(context, mutableIndices[index], *mutableShapes[index])) {
            OP_LOGE(context->GetNodeName(), "planner mutable input must be contiguous");
            return ge::GRAPH_FAILED;
        }
    }
    const uint64_t rowBytes = info.workspaceRowElements * sizeof(int32_t);
    const uint64_t alignedRowBytes = (rowBytes + 31U) & ~static_cast<uint64_t>(31U);
    const uint64_t localBytes = sparse_kv_plan_cfg::PLAN_SCAN_BYTES +
        (alignedRowBytes <= sparse_kv_plan_cfg::PLAN_ROW_UB_LIMIT_BYTES ? alignedRowBytes : 0U);
    if (localBytes > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(context->GetNodeName(), "planner local memory size overflow");
        return ge::GRAPH_FAILED;
    }
    info.planLocalMemoryBytes = static_cast<uint32_t>(localBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateFusedTarget(gert::TilingContext* context,
                                    const QuantLightningIndexerSparseKvInfo& info)
{
    const auto& params = info.opParamInfo;
    const gert::Shape& queryShape = params.query.shape->GetStorageShape();
    if (info.npuArch != NpuArch::DAV_3510 || info.gSize != 32U || info.n2Size != 1U ||
        info.blockSize != 128U ||
        params.query.desc->GetDataType() != ge::DT_FLOAT8_E4M3FN ||
        params.key.desc->GetDataType() != ge::DT_FLOAT8_E4M3FN ||
        params.weights.desc->GetDataType() != ge::DT_BF16 ||
        params.query_dequant_scale.desc->GetDataType() != ge::DT_FLOAT ||
        params.key_dequant_scale.desc->GetDataType() != ge::DT_FLOAT ||
        params.attenOut.desc->GetDataType() != ge::DT_INT32 ||
        queryShape.GetDimNum() != 3U ||
        queryShape.GetDim(1) != 32 || queryShape.GetDim(2) != 128 ||
        info.sparseCount != 2048U || info.sparseMode != 3 ||
        info.inputQLayout != FusedQliDataLayout::TND ||
        info.inputKLayout != FusedQliDataLayout::PA_BSND ||
        params.queryQuantMode == nullptr || *params.queryQuantMode != 0 ||
        params.keyQuantMode == nullptr || *params.keyQuantMode != 0) {
        OP_LOGE(context->GetNodeName(), "unsupported fused QuantLightningIndexer target configuration");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}

const std::map<ge::DataType, std::string> FUSED_QLI_DATATYPE_TO_STRING_MAP = {
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},   // fp8_e4m3 type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},             // hifloat8 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                    // dt_variant type
};

static std::string QLIDataTypeToSerialString(ge::DataType type)
{
    const auto qliIt = FUSED_QLI_DATATYPE_TO_STRING_MAP.find(type);
    if (qliIt != FUSED_QLI_DATATYPE_TO_STRING_MAP.end()) {
        return qliIt->second;
    } else {
        OP_LOGE("QuantLightningIndexerSparseKv", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

static std::string ToStringRaw(const gert::Shape& shape)
{
    std::ostringstream oss;
    const size_t qliDimensionCount = shape.GetDimNum();
    for (size_t qliDimension = 0; qliDimension < qliDimensionCount; ++qliDimension) {
        oss << (qliDimension == 0U ? "" : ", ") << shape.GetDim(qliDimension);
    }
    return oss.str();
}

ge::graphStatus QuantLightningIndexerSparseKvParser::CheckTensorShapes() const
{
    OP_CHECK_IF(qliParams_.query.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query", "The shape of query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qliParams_.key.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key", "The shape of key is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qliParams_.weights.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "weights", "The shape of weights is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qliParams_.query_dequant_scale.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query_dequant_scale",
                                                         "The shape of query_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qliParams_.key_dequant_scale.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key_dequant_scale",
                                                         "The shape of key_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        qliParams_.attenOut.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_indices", "The shape of sparse_indices is nullptr"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::CheckTensorDescriptions() const
{
    OP_CHECK_IF(qliParams_.query.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query", "The desc of query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qliParams_.key.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key", "The desc of key is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qliParams_.weights.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "weights", "The desc of weights is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qliParams_.query_dequant_scale.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query_dequant_scale",
                                                         "The desc of query_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qliParams_.key_dequant_scale.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key_dequant_scale",
                                                         "The desc of key_dequant_scale is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        qliParams_.attenOut.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_indices", "The desc of sparse_indices is nullptr"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// --------------------------QuantLightningIndexerSparseKvParser类成员函数定义-------------------------------------
ge::graphStatus QuantLightningIndexerSparseKvParser::CheckRequiredInOutExistence() const
{
    ge::graphStatus status = CheckTensorShapes();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    status = CheckTensorDescriptions();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(qliParams_.layOutQuery == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "layout_query", "Layout_query is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qliParams_.layOutKey == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "layout_key", "Layout_key is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qliParams_.sparseCount == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_count", "Sparse_count is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qliParams_.sparseMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_mode", "Sparse_mode is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qliParams_.queryQuantMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query_quant_mode", "Layout_query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qliParams_.keyQuantMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "key_quant_mode", "Key_quant_mode is nullptr"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::CheckQliRequiredParameters() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("QuantLightningIndexerSparseKv", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_2201 && npuArch_ != NpuArch::DAV_3510) {
        OP_LOGE(opName_, "Npu Arch Version[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr, OP_LOGE(opName_, "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                OP_LOGE(opName_, "RawTilingData got from GE context is nullptr"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QuantLightningIndexerSparseKvParser::GetOptionalInputParaInfo()
{
    qliParams_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    qliParams_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_Q_INDEX);
    qliParams_.actualSeqLengthsK.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    qliParams_.actualSeqLengthsK.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
    qliParams_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    qliParams_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void QuantLightningIndexerSparseKvParser::GetInputParaInfo()
{
    qliParams_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    qliParams_.query.shape = context_->GetInputShape(QUERY_INDEX);
    qliParams_.key.desc = context_->GetInputDesc(KEY_INDEX);
    qliParams_.key.shape = context_->GetInputShape(KEY_INDEX);
    qliParams_.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
    qliParams_.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    qliParams_.query_dequant_scale.desc = context_->GetInputDesc(QUERY_DEQUANT_SCALE_INDEX);
    qliParams_.query_dequant_scale.shape = context_->GetInputShape(QUERY_DEQUANT_SCALE_INDEX);
    qliParams_.key_dequant_scale.desc = context_->GetInputDesc(KEY_DEQUANT_SCALE_INDEX);
    qliParams_.key_dequant_scale.shape = context_->GetInputShape(KEY_DEQUANT_SCALE_INDEX);
    GetOptionalInputParaInfo();
}

void QuantLightningIndexerSparseKvParser::GetOutputParaInfo()
{
    qliParams_.attenOut.desc = context_->GetOutputDesc(quant_lightning_indexer);
    qliParams_.attenOut.shape = context_->GetOutputShape(quant_lightning_indexer);
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(opName_, "attrs got from GE is nullptr"), return ge::GRAPH_FAILED);

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");
    qliParams_.layOutQuery = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    qliParams_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);

    qliParams_.queryQuantMode = attrs->GetAttrPointer<int32_t>(ATTR_QUERY_QUANT_MODE_INDEX);
    qliParams_.keyQuantMode = attrs->GetAttrPointer<int32_t>(ATTR_KEY_QUANT_MODE_INDEX);
    qliParams_.layOutQuery = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    qliParams_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    qliParams_.sparseCount = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_COUNT_INDEX);
    qliParams_.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE_INDEX);
    qliParams_.preTokens = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX);
    qliParams_.nextTokens = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX);
    qliParams_.keyStride0 = *(attrs->GetAttrPointer<int64_t>(ATTR_KEY_STRIDE0_INDEX));
    qliParams_.keyDequantScaleStride0 = *(attrs->GetAttrPointer<int64_t>(ATTR_KEY_DEQUANT_SCALE_STRIDE0_INDEX));

    if (qliParams_.layOutQuery != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout_query is:%s", qliParams_.layOutQuery);
    }
    if (qliParams_.layOutKey != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout_key is:%s", qliParams_.layOutKey);
    }
    if (qliParams_.sparseCount != nullptr) {
        OP_LOGI(context_->GetNodeName(), "selscted count is:%d", *qliParams_.sparseCount);
    }
    if (qliParams_.sparseMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "sparse mode is:%d", *qliParams_.sparseMode);
    }
    if (qliParams_.preTokens != nullptr) {
        OP_LOGI(context_->GetNodeName(), "preTokens is:%d", *qliParams_.preTokens);
    }
    if (qliParams_.nextTokens != nullptr) {
        OP_LOGI(context_->GetNodeName(), "nextTokens is:%d", *qliParams_.nextTokens);
    }
    if (qliParams_.queryQuantMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "query_quant_mode mode is:%d", *qliParams_.queryQuantMode);
    }
    if (qliParams_.keyQuantMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "key_quant_mode mode is:%d", *qliParams_.keyQuantMode);
    }
    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo end");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::CheckAttrParaInfo()
{
    OP_CHECK_IF(((std::string(qliParams_.layOutKey) == "BNSD") || (std::string(qliParams_.layOutKey) == "PA_BBND")),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_key", std::string(qliParams_.layOutKey).c_str(),
                                                      "Layout_key only supported PA_BSND, PA_BBND, BSND or TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ((std::string(qliParams_.layOutQuery) != "BSND") && (std::string(qliParams_.layOutQuery) != "TND")),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_query", std::string(qliParams_.layOutQuery).c_str(),
                                              "Layout_query only supported BSND or TND"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((std::string(qliParams_.layOutKey) != "PA_BSND") &&
                 (std::string(qliParams_.layOutQuery)) != (std::string(qliParams_.layOutKey))),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "layout_query and layout_key",
                    std::string(qliParams_.layOutQuery) + " and " + std::string(qliParams_.layOutKey),
                    "When layout_key is not PA_BSND, layout_query and layout_key must be same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!((*qliParams_.sparseCount > 0) && (*qliParams_.sparseCount <= SPARSE_LIMIT)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sparse_count", std::to_string(*qliParams_.sparseCount),
                                                      "sparse_count must > 0 and <= 2048"),
                return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_3510) {
        auto queryType = qliParams_.query.desc->GetDataType();
        OP_CHECK_IF((queryType == ge::DT_HIFLOAT8) && (*qliParams_.sparseCount != SPARSE_LIMIT),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sparse_count",
                                                          std::to_string(*qliParams_.sparseCount).c_str(),
                                                          "When query type is hifloat8, sparse_count must be 2048"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        !((*qliParams_.sparseMode == 0) || (*qliParams_.sparseMode == SPARSE_MODE_LOWER)),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "sparse_mode", std::to_string(*qliParams_.sparseMode).c_str(),
                                              "Sparse_mode must be 0 or 3"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *qliParams_.preTokens != 9223372036854775807,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "pre_tokens", std::to_string(*qliParams_.preTokens).c_str(),
                                              "Pre_tokens must be 9223372036854775807"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        *qliParams_.nextTokens != 9223372036854775807,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "next_tokens", std::to_string(*qliParams_.nextTokens).c_str(),
                                              "Next_tokens must be 9223372036854775807"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((qliParams_.keyStride0 < 0) && (qliParams_.keyStride0 != -1)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName_, "key_stride0", std::to_string(qliParams_.keyStride0).c_str(), "key_stride0 must > 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((qliParams_.keyDequantScaleStride0 < 0) && (qliParams_.keyDequantScaleStride0 != -1)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "key_dequant_scale_stride0",
                                                      std::to_string(qliParams_.keyDequantScaleStride0).c_str(),
                                                      "Key_dequant_scale_stride0 must >= 0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(*qliParams_.queryQuantMode != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "query_quant_mode",
                                                      std::to_string(*qliParams_.queryQuantMode).c_str(),
                                                      "Query_quant_mode must be 0"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        *qliParams_.keyQuantMode != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "key_quant_mode", std::to_string(*qliParams_.keyQuantMode).c_str(), "Key_quant_mode must be 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetAndCheckInOutDataType()
{
    inputQType_ = qliParams_.query.desc->GetDataType();
    inputKType_ = qliParams_.key.desc->GetDataType();
    weightsType_ = qliParams_.weights.desc->GetDataType();
    inputQueryScaleType_ = qliParams_.query_dequant_scale.desc->GetDataType();
    inputKeyScaleType_ = qliParams_.key_dequant_scale.desc->GetDataType();
    outputType_ = qliParams_.attenOut.desc->GetDataType();

    OP_CHECK_IF(!(inputQType_ == inputKType_),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "query and key",
                    QLIDataTypeToSerialString(inputQType_) + " and " + QLIDataTypeToSerialString(inputKType_),
                    "The dtype of query and key must be same"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !(inputQueryScaleType_ == inputKeyScaleType_),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName_, "query_dequant_scale and key_dequant_scale",
            QLIDataTypeToSerialString(inputQueryScaleType_) + " and " + QLIDataTypeToSerialString(inputKeyScaleType_),
            "The dtype of query_dequant_scale and key_dequant_scale must be same"),
        return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(inputQType_ != ge::DT_FLOAT8_E4M3FN && inputQType_ != ge::DT_HIFLOAT8 && inputQType_ != ge::DT_INT8,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "query and key",
                        QLIDataTypeToSerialString(inputQType_) + " and " + QLIDataTypeToSerialString(inputKType_),
                        "The dtype of query and key must be float8_e4m3, hifloat8 or int8"),
                    return ge::GRAPH_FAILED);
        // hifloat8只支持bf16的weights和fp32的scale
        if (inputQType_ == ge::DT_FLOAT8_E4M3FN) {
            OP_CHECK_IF(
                (weightsType_ != ge::DT_BF16 || inputQueryScaleType_ != ge::DT_FLOAT) &&
                    (weightsType_ != ge::DT_FLOAT16 || inputQueryScaleType_ != ge::DT_FLOAT16),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "weights, query_dequant_scale and key_dequant_scale",
                    QLIDataTypeToSerialString(weightsType_) + ", " + QLIDataTypeToSerialString(inputQueryScaleType_) +
                        " and " + QLIDataTypeToSerialString(inputKeyScaleType_),
                    "When input query and key are float8_e4m3, "
                    "the dtype of weights, query_dequant_scale and key_dequant_scale must be "
                    "(bfloat16, float32, float32) or (float16, float16, float16)"),
                return ge::GRAPH_FAILED);
        } else if (inputKType_ == ge::DT_HIFLOAT8) {
            OP_CHECK_IF(
                weightsType_ != ge::DT_BF16 || inputQueryScaleType_ != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "weights, query_dequant_scale and key_dequant_scale",
                                                       QLIDataTypeToSerialString(weightsType_) + ", " +
                                                           QLIDataTypeToSerialString(inputQueryScaleType_) + " and " +
                                                           QLIDataTypeToSerialString(inputKeyScaleType_),
                                                       "When input query and key are hifloat8, the dtype of weights, "
                                                       "queryScale and keyScale must be (bfloat16, float32, float32)"),
                return ge::GRAPH_FAILED);
        } else if (inputQType_ == ge::DT_INT8) {
            OP_CHECK_IF(
                weightsType_ != ge::DT_FLOAT16 || inputQueryScaleType_ != ge::DT_FLOAT16,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName_, "weights, query_dequant_scale and key_dequant_scale",
                                                       QLIDataTypeToSerialString(weightsType_) + ", " +
                                                           QLIDataTypeToSerialString(inputQueryScaleType_) + " and " +
                                                           QLIDataTypeToSerialString(inputKeyScaleType_),
                                                       "When input query and key are int8, the dtype of weights, "
                                                       "queryScale and keyScale must be float16"),
                return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(inputQType_ != ge::DT_INT8,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "query and key",
                        QLIDataTypeToSerialString(inputQType_) + " and " + QLIDataTypeToSerialString(inputKType_),
                        "The dtype of query and key must be int8"),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            weightsType_ != ge::DT_FLOAT16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "weights", QLIDataTypeToSerialString(weightsType_).c_str(),
                                                  "The dtype of weights must be float16"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(inputQueryScaleType_ != ge::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        opName_, "query_dequant_scale and key_dequant_scale",
                        QLIDataTypeToSerialString(inputQueryScaleType_) + " and " +
                            QLIDataTypeToSerialString(inputKeyScaleType_),
                        "The dtype of query_dequant_scale and key_dequant_scale must be float16"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        outputType_ != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "sparse_indices", QLIDataTypeToSerialString(outputType_).c_str(),
                                              "The dtype of sparse_indices must be int32"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
    const map<string, FusedQliDataLayout> layoutQueryMap = {{"BSND", FusedQliDataLayout::BSND}, {"TND", FusedQliDataLayout::TND}};

    std::string layout_query(qliParams_.layOutQuery);
    auto QLayout_ = layoutQueryMap.find(layout_query);
    if (QLayout_ != layoutQueryMap.end()) {
        qLayout_ = QLayout_->second;
    }

    const map<string, FusedQliDataLayout> layoutKeyMap = {{"BSND", FusedQliDataLayout::BSND},
                                                  {"TND", FusedQliDataLayout::TND},
                                                  {"PA_BSND", FusedQliDataLayout::PA_BSND},
                                                  {"PA_BBND", FusedQliDataLayout::PA_BSND}};
    std::string layout_key(qliParams_.layOutKey);
    auto KLayout = layoutKeyMap.find(layout_key);
    if (KLayout != layoutKeyMap.end()) {
        kLayout_ = KLayout->second;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == FusedQliDataLayout::PA_BSND) {
        OP_CHECK_IF(qliParams_.blockTable.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "block_table", "Layout_key only supports PA_BSND, block_table must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(qliParams_.actualSeqLengthsK.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "actual_seq_lengths_key",
                        "Layout_key only supports PA_BSND, actual_seq_lengths_key must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            qliParams_.blockTable.desc->GetDataType() != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName_, "block_table", QLIDataTypeToSerialString(qliParams_.blockTable.desc->GetDataType()).c_str(),
                "The dtype of block_table must be int32"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(qliParams_.blockTable.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "block_table", "When Layout_key is not PA_BSND, block_table must be null"),
                    return ge::GRAPH_FAILED);
    }

    if (kLayout_ == FusedQliDataLayout::TND) {
        OP_CHECK_IF(
            qliParams_.actualSeqLengthsK.tensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "actual_seq_lengths_key",
                                                     "When layout_key is TND, actual_seq_lengths_key must not be null"),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(qliParams_.actualSeqLengthsK.tensor != nullptr &&
                    qliParams_.actualSeqLengthsK.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "actual_seq_lengths_key",
                    QLIDataTypeToSerialString(qliParams_.actualSeqLengthsK.desc->GetDataType()).c_str(),
                    "The dtype of actual_seq_lengths_key must be int32"),
                return ge::GRAPH_FAILED);
    if (qLayout_ == FusedQliDataLayout::TND) {
        OP_CHECK_IF(qliParams_.actualSeqLengthsQ.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "actual_seq_lengths_query",
                        "When layout_query is TND, actual_seq_lengths_query must not be null"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(qliParams_.actualSeqLengthsQ.tensor != nullptr &&
                    qliParams_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "actual_seq_lengths_query",
                    QLIDataTypeToSerialString(qliParams_.actualSeqLengthsQ.desc->GetDataType()).c_str(),
                    "The dtype of actual_seq_lengths_query must be int32"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::CheckShapeDim()
{
    OP_CHECK_IF(
        (qliParams_.blockTable.tensor != nullptr) &&
            (qliParams_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            opName_, "block_table", std::to_string(qliParams_.blockTable.tensor->GetStorageShape().GetDimNum()).c_str(),
            "The shape dim of block_table must be 2"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((kLayout_ == FusedQliDataLayout::PA_BSND) || (kLayout_ == FusedQliDataLayout::BSND)) &&
                    (qliParams_.key.shape->GetShape().GetDimNum() != DIM_NUM_FOUR),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "key", std::to_string(qliParams_.key.shape->GetShape().GetDimNum()).c_str(),
                    "The shape dim of key must be 4"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((kLayout_ == FusedQliDataLayout::TND) && (qliParams_.key.shape->GetShape().GetDimNum() != DIM_NUM_THREE),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "key", std::to_string(qliParams_.key.shape->GetShape().GetDimNum()).c_str(),
                    "The shape dim of key must be 3"),
                return ge::GRAPH_FAILED);

    uint32_t qShapeDim = qliParams_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = qliParams_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = qliParams_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t expectShapeDim = DIM_NUM_FOUR;
    if (qLayout_ == FusedQliDataLayout::TND) {
        expectShapeDim = DIM_NUM_THREE;
    }
    OP_CHECK_IF(
        qShapeDim != expectShapeDim,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "query", std::to_string(qShapeDim).c_str(),
                                                 "The shape dim of query must be " + std::to_string(expectShapeDim)),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShapeDim != expectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "sparse_indices", std::to_string(outShapeDim).c_str(),
                    "The shape dim of sparse_indices must be " + std::to_string(expectShapeDim)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightsShapeDim == expectShapeDim - 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "weights", std::to_string(weightsShapeDim).c_str(),
                    "The shape dim of weights must be " + std::to_string(expectShapeDim - 1)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetN1Size()
{
    if (qLayout_ == FusedQliDataLayout::BSND) {
        n1Size_ = qliParams_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO);
    } else {
        // TND
        n1Size_ = qliParams_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE);
    }
    OP_LOGI(context_->GetNodeName(), "n1Size is %d", n1Size_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus QliGetActualSeqLenSize(int64_t& qliSize, const gert::Tensor* qliTensor,
                                              const std::string& qliActualSeqLenName, const char* qliOpName)
{
    qliSize = qliTensor->GetShapeSize();
    if (qliSize == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            qliOpName, qliActualSeqLenName.c_str(), std::to_string(qliSize).c_str(),
            "The shape size of " + qliActualSeqLenName + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetActualSeqLenSize(int64_t& size, const gert::Tensor* tensor,
                                                   const std::string& actualSeqLenName) const
{
    return QliGetActualSeqLenSize(size, tensor, actualSeqLenName, opName_);
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetAndCheckN2Size()
{
    // PA_BSND
    if (kLayout_ == FusedQliDataLayout::TND) {
        n2Size_ = qliParams_.key.shape->GetShape().GetDim(DIM_IDX_ONE);
    } else {
        n2Size_ = qliParams_.key.shape->GetShape().GetDim(DIM_IDX_TWO);
    }
    OP_LOGI(context_->GetNodeName(), "N2 is %d", n2Size_);
    OP_CHECK_IF(n2Size_ != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "key",
                                                      ToStringRaw(qliParams_.query.shape->GetStorageShape()).c_str(),
                                                      "The head num of key must be 1"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "query",
                                              ToStringRaw(qliParams_.query.shape->GetStorageShape()).c_str(),
                                              "The head num of query can not be a multiple of the head num of key");
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;

    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(gSize_ != G_SIZE_LIMIT_950 && gSize_ != G_SIZE_LIMIT && gSize_ != G_SIZE_LIMIT_32_950 &&
                        gSize_ != G_SIZE_LIMIT_16_950 && gSize_ != G_SIZE_LIMIT_8_950,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "query", ToStringRaw(qliParams_.query.shape->GetStorageShape()).c_str(),
                        "The head num of query divided by the head num of key must equal 64, 32, 24, 16 or 8"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(gSize_ > G_SIZE_LIMIT,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "query", ToStringRaw(qliParams_.query.shape->GetStorageShape()).c_str(),
                        "The head num of query divided by the head num of key must <= " + std::to_string(G_SIZE_LIMIT)),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == FusedQliDataLayout::TND) {
        return GetActualSeqLenSize(bSize_, qliParams_.actualSeqLengthsQ.tensor, "actual_seq_lengths_query");
    } else { // BSND
        bSize_ = qliParams_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
        OP_LOGI(context_->GetNodeName(), "b: %d, s: %d, n: %d,d :%d",
                qliParams_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO),
                qliParams_.query.shape->GetStorageShape().GetDim(DIM_IDX_ONE),
                qliParams_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO),
                qliParams_.query.shape->GetStorageShape().GetDim(DIM_IDX_THREE));
        return ge::GRAPH_SUCCESS;
    }
}

static ge::graphStatus QliGetHeadDim(const FusedQliRequiredParaInfo& qliQuery, FusedQliDataLayout qliQLayout,
                                     const char* qliOpName, int64_t& qliHeadDim)
{
    uint32_t qliDIndex = DIM_IDX_TWO;
    switch (qliQLayout) {
        case FusedQliDataLayout::TND:
            qliDIndex = DIM_IDX_TWO;
            break;
        case FusedQliDataLayout::BSND:
            qliDIndex = DIM_IDX_THREE;
            break;
        default:
            OP_LOGE(qliOpName, "unsupported layout for getting head dim.");
            return ge::GRAPH_FAILED;
    }
    qliHeadDim = qliQuery.shape->GetStorageShape().GetDim(qliDIndex);
    OP_CHECK_IF(qliHeadDim != HEAD_DIM_LIMIT,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(qliOpName, "query",
                                                      ToStringRaw(qliQuery.shape->GetStorageShape()).c_str(),
                                                      "The last dim of query only support 128"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetHeadDim()
{
    return QliGetHeadDim(qliParams_.query, qLayout_, opName_, headDim_);
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetS1Size()
{
    if (qLayout_ == FusedQliDataLayout::BSND) {
        s1Size_ = qliParams_.query.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetAndCheckBlockSize()
{
    blockSize_ = qliParams_.key.shape->GetShape().GetDim(1);
    OP_LOGI(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OP_CHECK_IF(((blockSize_ % BLOCK_SIZE_FACTOR != 0) || (blockSize_ == 0) || (blockSize_ > BLOCK_SIZE_LIMIT)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "key", ToStringRaw(qliParams_.key.shape->GetShape()).c_str(),
                    "The block_size of key must be a multiple of 16 and be within the range (0, 1024]"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int32_t blockCount_ = static_cast<uint32_t>(qliParams_.key.shape->GetShape().GetDim(0));
    OP_CHECK_IF(
        (blockCount_ == 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "key", ToStringRaw(qliParams_.key.shape->GetShape()).c_str(),
                                              "The block_count of key cannot be 0"),
        return ge::GRAPH_FAILED);

    maxBlockNumPerBatch_ = qliParams_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    OP_LOGI(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
            maxBlockNumPerBatch_, blockSize_, s2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetS2SizeForBatchContinuous()
{
    std::string layout_key(qliParams_.layOutKey);
    if (kLayout_ == FusedQliDataLayout::BSND) {
        s2Size_ = qliParams_.key.shape->GetShape().GetDim(DIM_IDX_ONE);
    } else if (kLayout_ == FusedQliDataLayout::TND) {
        s2Size_ = qliParams_.key.shape->GetShape().GetDim(DIM_IDX_ZERO);
    }
    OP_CHECK_IF((kLayout_ != FusedQliDataLayout::BSND) && (kLayout_ != FusedQliDataLayout::TND),
                OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_key", layout_key.c_str(), "BSND or TND"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kLayout_ == FusedQliDataLayout::PA_BSND) {
        return GetS2SizeForPageAttention();
    }
    return GetS2SizeForBatchContinuous();
}

ge::graphStatus QuantLightningIndexerSparseKvParser::ValidateInputShapesMatch()
{
    /*
    TND:
    query [T,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [T,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize],
    out [T,N2,topk]
    QuantLightningIndexerSparseKv 输入输出shape约束:
    ----------------------
    BSND:
    query [BatchSize,S1,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [BatchSize,S1,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize] 可选
    out [BatchSize,S1,N2,topk]
    */
    uint32_t qliQueryWeightsN1Dim = 1;
    uint32_t qliOutN2Dim = 1;

    if (qLayout_ == FusedQliDataLayout::TND) {
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于act_seq_q
        OP_CHECK_IF((kLayout_ == FusedQliDataLayout::PA_BSND) &&
                        ((qliParams_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_) ||
                         (qliParams_.blockTable.tensor != nullptr &&
                          qliParams_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query, actual_seq_lengths_key and block_table",
                        Ops::Base::ToString(qliParams_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(qliParams_.actualSeqLengthsK.tensor->GetStorageShape()) + " and " +
                            Ops::Base::ToString(qliParams_.blockTable.tensor->GetStorageShape()),
                        "TND case actual_seq_lengths_query, actual_seq_lengths_key, block_table dim 0 are " +
                            std::to_string(bSize_) + ", " +
                            std::to_string(qliParams_.actualSeqLengthsK.tensor->GetShapeSize()) + ", " +
                            std::to_string(qliParams_.blockTable.tensor->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            ((kLayout_ != FusedQliDataLayout::PA_BSND) && qliParams_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "query and actual_seq_lengths_key",
                Ops::Base::ToString(qliParams_.query.shape->GetStorageShape()) + " and " +
                    Ops::Base::ToString(qliParams_.actualSeqLengthsK.tensor->GetStorageShape()),
                "TND case actual_seq_lengths_query, actual_seq_lengths_key are " + std::to_string(bSize_) + ", " +
                    std::to_string(qliParams_.actualSeqLengthsK.tensor->GetShapeSize()) +
                    " respectively, they must be same"),
            return ge::GRAPH_FAILED);
        // -----------------------check T-------------------
        uint32_t qTsize = qliParams_.query.shape->GetStorageShape().GetDim(0);
        OP_CHECK_IF((qliParams_.weights.shape->GetStorageShape().GetDim(0) != qTsize) ||
                        (qliParams_.attenOut.shape->GetStorageShape().GetDim(0) != qTsize),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query, weights and sparse_indices",
                        Ops::Base::ToString(qliParams_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(qliParams_.weights.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(qliParams_.attenOut.shape->GetStorageShape()),
                        "TND case query, weights, sparse_indices dim 0 are " + std::to_string(qTsize) + ", " +
                            std::to_string(qliParams_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                            std::to_string(qliParams_.attenOut.shape->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        tSize_ = qTsize;
    } else {
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于query
        OP_CHECK_IF((kLayout_ == FusedQliDataLayout::PA_BSND) &&
                        ((qliParams_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                         (qliParams_.blockTable.tensor != nullptr &&
                          qliParams_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_) ||
                         (qliParams_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_) ||
                         (qliParams_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query, weights, actual_seq_lengths_key, block_table and sparse_indices",
                        Ops::Base::ToString(qliParams_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(qliParams_.weights.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(qliParams_.actualSeqLengthsK.tensor->GetStorageShape()) + ", " +
                            Ops::Base::ToString(qliParams_.blockTable.tensor->GetStorageShape()) + " and " +
                            Ops::Base::ToString(qliParams_.attenOut.shape->GetStorageShape()),
                        "BSND case query, weights, actual_seq_lengths_key, block_table, sparse_indices dim 0 are " +
                            std::to_string(bSize_) + ", " +
                            std::to_string(qliParams_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                            std::to_string(qliParams_.actualSeqLengthsK.tensor->GetShapeSize()) + ", " +
                            std::to_string(qliParams_.blockTable.tensor->GetStorageShape().GetDim(0)) + ", " +
                            std::to_string(qliParams_.attenOut.shape->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (kLayout_ != FusedQliDataLayout::PA_BSND) && ((qliParams_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                                                  (qliParams_.actualSeqLengthsK.tensor != nullptr &&
                                                   qliParams_.actualSeqLengthsK.tensor->GetShapeSize() != bSize_) ||
                                                  (qliParams_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_)),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "query, weights, actual_seq_lengths_key and sparse_indices",
                Ops::Base::ToString(qliParams_.query.shape->GetStorageShape()) + ", " +
                    Ops::Base::ToString(qliParams_.weights.shape->GetStorageShape()) + ", " +
                    Ops::Base::ToString(qliParams_.actualSeqLengthsK.tensor->GetStorageShape()) + " and " +
                    Ops::Base::ToString(qliParams_.attenOut.shape->GetStorageShape()),
                "BSND case query, weights, actual_seq_lengths_key, sparse_indices dim 0 are " + std::to_string(bSize_) +
                    ", " + std::to_string(qliParams_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                    std::to_string(qliParams_.actualSeqLengthsK.tensor->GetShapeSize()) + ", " +
                    std::to_string(qliParams_.attenOut.shape->GetStorageShape().GetDim(0)) +
                    " respectively, they must be same"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((qliParams_.actualSeqLengthsQ.tensor != nullptr) &&
                        (qliParams_.actualSeqLengthsQ.tensor->GetShapeSize() != bSize_),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query and actual_seq_lengths_query",
                        Ops::Base::ToString(qliParams_.query.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(qliParams_.actualSeqLengthsQ.tensor->GetStorageShape()),
                        "BSND case query, actual_seq_lengths_query dim 0 are " + std::to_string(bSize_) + ", " +
                            std::to_string(qliParams_.actualSeqLengthsQ.tensor->GetShapeSize()) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        // -----------------------check S1-------------------
        OP_CHECK_IF((qliParams_.weights.shape->GetStorageShape().GetDim(1) != s1Size_) ||
                        (qliParams_.attenOut.shape->GetStorageShape().GetDim(1) != s1Size_),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "query, weights and sparse_indices",
                        Ops::Base::ToString(qliParams_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(qliParams_.weights.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(qliParams_.attenOut.shape->GetStorageShape()),
                        "BSND case query, weights, sparse_indices dim 1 are " + std::to_string(s1Size_) + ", " +
                            std::to_string(qliParams_.weights.shape->GetStorageShape().GetDim(1)) + ", " +
                            std::to_string(qliParams_.attenOut.shape->GetStorageShape().GetDim(1)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
        qliQueryWeightsN1Dim = DIM_IDX_TWO;
        qliOutN2Dim = DIM_IDX_TWO;
        tSize_ = bSize_ * s1Size_;
    }
    // -----------------------check N1-------------------
    OP_CHECK_IF((qliParams_.weights.shape->GetStorageShape().GetDim(qliQueryWeightsN1Dim) != n1Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "query and weights",
                    Ops::Base::ToString(qliParams_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(qliParams_.weights.shape->GetStorageShape()),
                    "The head num of query, weights are " + std::to_string(n1Size_) + ", " +
                        std::to_string(qliParams_.weights.shape->GetStorageShape().GetDim(1)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    OP_CHECK_IF(((kLayout_ != FusedQliDataLayout::TND && qliParams_.key.shape->GetShape().GetDim(DIM_IDX_THREE) != headDim_) ||
                 (kLayout_ == FusedQliDataLayout::TND && qliParams_.key.shape->GetShape().GetDim(DIM_IDX_TWO) != headDim_)),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query and key",
                                                       Ops::Base::ToString(qliParams_.query.shape->GetStorageShape()) +
                                                           " and " +
                                                           Ops::Base::ToString(qliParams_.key.shape->GetStorageShape()),
                                                       "Query, key shape last dim must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check N2-------------------
    OP_CHECK_IF(
        (qliParams_.attenOut.shape->GetStorageShape().GetDim(qliOutN2Dim) != n2Size_),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "key and sparse_indices",
                                               Ops::Base::ToString(qliParams_.key.shape->GetStorageShape()) + " and " +
                                                   Ops::Base::ToString(qliParams_.attenOut.shape->GetStorageShape()),
                                               "The head num of key and output sparse_indices must be same"),
        return ge::GRAPH_FAILED);
    // -----------------------check N2-------------------
    // -----------------------check sparse_count-------------------
    OP_CHECK_IF(
        (qliParams_.attenOut.shape->GetStorageShape().GetDim(qliOutN2Dim + 1) != *qliParams_.sparseCount),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "sparse_indices and sparse_count",
                                               Ops::Base::ToString(qliParams_.attenOut.shape->GetStorageShape()) +
                                                   " and " + std::to_string(*qliParams_.sparseCount),
                                               "The last dim of sparse_indices and sparse_count must be same"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::CheckScaleShape()
{
    uint32_t qShapeDim = qliParams_.query.shape->GetStorageShape().GetDimNum();
    uint32_t kShapeDim = qliParams_.key.shape->GetShape().GetDimNum();
    uint32_t qDequantScaleShapeDim = qliParams_.query_dequant_scale.shape->GetStorageShape().GetDimNum();
    uint32_t kDequantScaleShapeDim = qliParams_.key_dequant_scale.shape->GetShape().GetDimNum();
    OP_CHECK_IF(qDequantScaleShapeDim != (qShapeDim - 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "query_dequant_scale", std::to_string(qDequantScaleShapeDim),
                    "The shape dim of query_dequant_scale must be " + std::to_string(qShapeDim - 1)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDequantScaleShapeDim != (kShapeDim - 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, "key_dequant_scale", std::to_string(kDequantScaleShapeDim),
                    "The shape dim of key_dequant_scale must be " + std::to_string(kShapeDim - 1)),
                return ge::GRAPH_FAILED);
    // check q scale
    for (uint32_t i = 0; i < (qShapeDim - 1); i++) {
        uint32_t dimValueQueryScale = qliParams_.query_dequant_scale.shape->GetStorageShape().GetDim(i);
        uint32_t dimValueQuery = qliParams_.query.shape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(
            dimValueQueryScale != dimValueQuery,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "query and query_dequant_scale",
                Ops::Base::ToString(qliParams_.query.shape->GetStorageShape()) + " and " +
                    Ops::Base::ToString(qliParams_.query_dequant_scale.shape->GetStorageShape()),
                "Query_dequant_scale's shape[" + std::to_string(i) + "] " + std::to_string(dimValueQueryScale) +
                    " and query's shape[" + std::to_string(i) + "] " + std::to_string(dimValueQuery) + " is not same"),
            return ge::GRAPH_FAILED);
    }
    // check k scale
    for (uint32_t i = 0; i < (kShapeDim - 1); i++) {
        uint32_t dimValueKeyScale = qliParams_.key_dequant_scale.shape->GetShape().GetDim(i);
        uint32_t dimValueKey = qliParams_.key.shape->GetShape().GetDim(i);
        OP_CHECK_IF(
            dimValueKeyScale != dimValueKey,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "key and key_dequant_scale",
                Ops::Base::ToString(qliParams_.key.shape->GetStorageShape()) + " and " +
                    Ops::Base::ToString(qliParams_.key_dequant_scale.shape->GetStorageShape()),
                "Key_dequant_scale's shape[" + std::to_string(i) + "] " + std::to_string(dimValueKeyScale) +
                    " and key's shape[" + std::to_string(i) + "] " + std::to_string(dimValueKey) + " is not same"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// key非连续校验：通过shape计算expected stride进行校验
// PA_BSND时，只允许0轴非连续，其余轴必须连续
// 非PA_BSND时，所有轴都必须连续
ge::graphStatus QuantLightningIndexerSparseKvParser::CheckContiguous()
{
    if (npuArch_ != NpuArch::DAV_3510) {
        return ge::GRAPH_SUCCESS;
    }

    std::vector<uint32_t> keyStridesVec_;
    std::vector<uint32_t> keyDequantScaleStridesVec_;

    auto qliKeyStrides = context_->GetDynamicInputStride(KEY_INDEX, 0);
    auto keyDequantScaleStrides = context_->GetDynamicInputStride(KEY_DEQUANT_SCALE_INDEX, 0);
    if (qliKeyStrides != nullptr && qliKeyStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < qliKeyStrides->GetDimNum(); i++) {
            keyStridesVec_.push_back(qliKeyStrides->GetStride(i));
        }
    }
    if (keyDequantScaleStrides != nullptr && keyDequantScaleStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < keyDequantScaleStrides->GetDimNum(); i++) {
            keyDequantScaleStridesVec_.push_back(keyDequantScaleStrides->GetStride(i));
        }
    }

    bool keyNonContiguous = false;
    bool scaleNonContiguous = false;
    // PA_BSND: 0轴允许非连续，从1轴开始检查；非PA_BSND: 从0轴开始检查
    // PA_BSND: axis 0 allows non-contiguous, check starts from axis 1
    // Non-PA_BSND: check starts from axis 0
    size_t checkStartIdx = (kLayout_ == FusedQliDataLayout::PA_BSND) ? 1 : 0;
    if (!keyStridesVec_.empty() && qliParams_.key.shape != nullptr) {
        auto& shape = qliParams_.key.shape->GetStorageShape();
        std::vector<uint32_t> qliExpectedStrides;
        if (kLayout_ == FusedQliDataLayout::BSND || kLayout_ == FusedQliDataLayout::PA_BSND) {
            qliExpectedStrides = {shape.GetDim(1) * shape.GetDim(2) * shape.GetDim(3),
                                  shape.GetDim(2) * shape.GetDim(3), shape.GetDim(3), 1};
        } else if (kLayout_ == FusedQliDataLayout::TND) {
            qliExpectedStrides = {shape.GetDim(1) * shape.GetDim(2), shape.GetDim(2), 1};
        }
        for (size_t i = checkStartIdx; i < qliExpectedStrides.size(); ++i) {
            if (i < keyStridesVec_.size() && keyStridesVec_[i] != qliExpectedStrides[i]) {
                keyNonContiguous = true;
                break;
            }
        }
    }
    if (!keyDequantScaleStridesVec_.empty() && qliParams_.key_dequant_scale.shape != nullptr) {
        auto& shape = qliParams_.key_dequant_scale.shape->GetStorageShape();
        std::vector<uint32_t> expectedStrides;
        if (kLayout_ == FusedQliDataLayout::BSND || kLayout_ == FusedQliDataLayout::PA_BSND) {
            expectedStrides = {shape.GetDim(1) * shape.GetDim(2), shape.GetDim(2), 1};
        } else if (kLayout_ == FusedQliDataLayout::TND) {
            expectedStrides = {shape.GetDim(1), 1};
        }
        for (size_t i = checkStartIdx; i < expectedStrides.size(); ++i) {
            if (i < keyDequantScaleStridesVec_.size() && keyDequantScaleStridesVec_[i] != expectedStrides[i]) {
                scaleNonContiguous = true;
                break;
            }
        }
    }
    OP_CHECK_IF(keyNonContiguous || scaleNonContiguous,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "key and key_dequant_scale",
                    "Key and key_dequant_scale only supports non-contiguous tensor on the 0-axis in PA scenarios"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QuantLightningIndexerSparseKvParser::GenerateInfo(QuantLightningIndexerSparseKvInfo& QLIInfo)
{
    QLIInfo.opName = opName_;
    QLIInfo.platformInfo = platformInfo_;
    QLIInfo.opParamInfo = qliParams_;
    QLIInfo.npuArch = npuArch_;

    QLIInfo.bSize = bSize_;
    QLIInfo.tSize = tSize_;
    QLIInfo.n1Size = n1Size_;
    QLIInfo.n2Size = n2Size_;
    QLIInfo.s1Size = s1Size_;
    QLIInfo.s2Size = s2Size_;
    QLIInfo.gSize = gSize_;

    QLIInfo.inputQType = inputQType_;
    QLIInfo.inputKType = inputKType_;
    QLIInfo.outputType = outputType_;

    QLIInfo.blockSize = blockSize_;
    QLIInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    QLIInfo.pageAttentionFlag = (kLayout_ == FusedQliDataLayout::PA_BSND);
    QLIInfo.sparseMode = *qliParams_.sparseMode;
    QLIInfo.sparseCount = *qliParams_.sparseCount;
    QLIInfo.preTokens = *qliParams_.preTokens;
    QLIInfo.nextTokens = *qliParams_.nextTokens;

    if (qliParams_.keyStride0 != -1) {
        QLIInfo.keyStride0 = qliParams_.keyStride0;
    } else {
        QLIInfo.keyStride0 = blockSize_ * n2Size_ * headDim_;
    }
    if (qliParams_.keyDequantScaleStride0 != -1) {
        QLIInfo.keyDequantScaleStride0 = qliParams_.keyDequantScaleStride0;
    } else {
        QLIInfo.keyDequantScaleStride0 = blockSize_;
    }

    QLIInfo.inputQLayout = qLayout_;
    QLIInfo.inputKLayout = kLayout_;
}

ge::graphStatus QuantLightningIndexerSparseKvParser::ParseAndCheck(QuantLightningIndexerSparseKvInfo& QLIInfo)
{
    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckQliRequiredParameters()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetAndCheckInOutDataType() || ge::GRAPH_SUCCESS != GetQueryKeyAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetAndCheckOptionalInput()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckShapeDim() || ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetAndCheckN2Size() || ge::GRAPH_SUCCESS != GetGSize()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch() || ge::GRAPH_SUCCESS != CheckScaleShape()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckContiguous()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(QLIInfo);

    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForQuantLightningIndexerSparseKv(gert::TilingParseContext* /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------QuantLightningIndexerSparseKvTiling类成员函数定义-----------------------
static uint64_t QliCalcWorkspaceSize(const platform_ascendc::PlatformAscendC& qliPlatform, int64_t qliS2Size,
                                     uint32_t qliAicNum, ge::DataType inputQType)
{
    constexpr uint32_t qliMm1ResElemSize = 4;
    constexpr uint32_t qliDoubleBuffer = 2;
    constexpr uint32_t qliMBaseSize = 512;
    constexpr uint32_t qliS2BaseSize = 512;
    constexpr uint32_t qliV1ResElemSize = 4;
    constexpr uint32_t qliV1ResElemType = 2;
    constexpr uint32_t qliV1DecodeParamElemSize = 8;
    constexpr uint32_t qliV1DecodeParamNum = 16;
    constexpr uint32_t qliV1DecodeDataNum = 2;
    constexpr uint32_t qliS1BaseSize = 8;
    constexpr uint32_t qliTopkMaxSize = 2048;

    uint64_t qliWorkspaceSize = qliPlatform.GetLibApiWorkSpaceSize();
    if (qliPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        constexpr uint32_t qli3510S1Base = 4;
        constexpr uint32_t qli3510S2Base = 128;
        uint32_t scoreElemSize = (inputQType == ge::DT_INT8) ? sizeof(uint32_t) : sizeof(uint16_t);
        qliWorkspaceSize += qli3510S1Base * ((qliS2Size + qli3510S2Base - 1) / qli3510S2Base) * qli3510S2Base *
                            scoreElemSize * qliAicNum;
    } else {
        constexpr uint32_t qliMm1ResSize = qliMBaseSize * qliS2BaseSize;
        qliWorkspaceSize += qliMm1ResSize * qliMm1ResElemSize * qliDoubleBuffer * qliAicNum;
        qliWorkspaceSize +=
            qliV1DecodeDataNum * qliS1BaseSize * qliV1ResElemType * qliTopkMaxSize * qliV1ResElemSize * qliAicNum;
        qliWorkspaceSize +=
            qliV1DecodeDataNum * qliS1BaseSize * qliV1DecodeParamNum * qliV1DecodeParamElemSize * qliAicNum;
    }
    return qliWorkspaceSize;
}

ge::graphStatus QuantLightningIndexerSparseKvTiling::DoTiling(QuantLightningIndexerSparseKvInfo* tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    uint64_t workspaceSize = QliCalcWorkspaceSize(ascendcPlatform, tilingInfo->s2Size, aicNum, tilingInfo->inputQType);
    size_t* workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    const uint32_t qliLocalMemoryBytes =
        tilingInfo->tSize > 64U ? QLI_UB_LARGE_T_BYTES : QLI_UB_SMALL_T_BYTES;
    const uint32_t plannerRowUbBytes = tilingInfo->planLocalMemoryBytes > PLAN_SCAN_UB_BYTES
        ? tilingInfo->planLocalMemoryBytes - PLAN_SCAN_UB_BYTES
        : 0U;
    const uint32_t extraPlannerUbBytes = plannerRowUbBytes > QLI_TOPK_SLAB_BYTES
        ? plannerRowUbBytes - QLI_TOPK_SLAB_BYTES
        : 0U;
    const uint32_t dynamicUbBytes = qliLocalMemoryBytes + PLAN_SCAN_UB_BYTES + extraPlannerUbBytes;
    if (context_->SetDynUBufSize(dynamicUbBytes) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "failed to set fused dynamic UB size");
        return ge::GRAPH_FAILED;
    }

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_tSize(tilingInfo->tSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_s1Size(tilingInfo->s1Size);
    tilingData_.set_sparseCount(tilingInfo->sparseCount);
    tilingData_.set_keyStride0(tilingInfo->keyStride0);
    tilingData_.set_keyDequantScaleStride0(tilingInfo->keyDequantScaleStride0);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_sparseMode(tilingInfo->sparseMode);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.set_hashCapacity(tilingInfo->hashCapacity);
    tilingData_.set_workspaceRowElements(tilingInfo->workspaceRowElements);
    tilingData_.set_maxRows(tilingInfo->maxRows);
    tilingData_.set_topk(tilingInfo->topk);
    tilingData_.set_capacity(tilingInfo->capacity);
    tilingData_.set_maxToken(tilingInfo->maxToken);
    tilingData_.set_maxRequests(tilingInfo->maxRequests);
    tilingData_.set_maxNumBlocks(tilingInfo->maxNumBlocks);
    tilingData_.set_hostNumBlocks(tilingInfo->hostNumBlocks);
    tilingData_.set_planBlockSize(tilingInfo->planBlockSize);
    tilingData_.set_planLocalMemoryBytes(tilingInfo->planLocalMemoryBytes);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo->inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint32_t tilingKey =
        GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout, inputKLayout);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1); // 1: batchmode模式

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForQuantLightningIndexerSparseKv(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("QuantLightningIndexerSparseKv", "Tilingcontext is null"), return ge::GRAPH_FAILED);
    QuantLightningIndexerSparseKvInfo QLIInfo;
    QuantLightningIndexerSparseKvParser QuantLightningIndexerSparseKvParser(context);
    if (QuantLightningIndexerSparseKvParser.ParseAndCheck(QLIInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateFusedTarget(context, QLIInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ParsePlannerInfo(context, QLIInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QuantLightningIndexerSparseKvTiling QLITiling(context);
    return QLITiling.DoTiling(&QLIInfo);
}

// --------------------------Tiling及函数TilingPrepare函数注册--------
IMPL_OP_OPTILING(QuantLightningIndexerSparseKv)
    .Tiling(TilingForQuantLightningIndexerSparseKv)
    .TilingParse<QuantLightningIndexerSparseKvCompileInfo>(TilingPrepareForQuantLightningIndexerSparseKv);

} // namespace optiling
