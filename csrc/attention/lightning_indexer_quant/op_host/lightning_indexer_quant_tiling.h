/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_quant_tiling.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_QUANT_TILING_H_
#define LIGHTNING_INDEXER_QUANT_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
// ------------------公共定义--------------------------
struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class DataLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGTHS_INDEX = 2;
constexpr uint32_t QUERY_DEQUANT_SCALE_INDEX = 3;
constexpr uint32_t KEY_DEQUANT_SCALE_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 6;
constexpr uint32_t BLOCK_TABLE_INDEX = 7;
constexpr uint32_t KV_ADDRESSES_INDEX = 8;
constexpr uint32_t PREFETCH_THRESHOLD_INDEX = 9;
constexpr uint32_t REQ_IDS_INDEX = 10;
constexpr uint32_t TOKEN_TO_REQ_INDEX = 11;
constexpr uint32_t STABLE_PREFIX_LENS_INDEX = 12;
constexpr uint32_t HOST_BLOCK_TABLE_INDEX = 13;
constexpr uint32_t HOST_CACHE_BASES_INDEX = 14;
constexpr uint32_t RESIDENT_K_INDEX = 15;
constexpr uint32_t RESIDENT_V_INDEX = 16;
constexpr uint32_t LAST_REQ_IDS_INDEX = 17;
constexpr uint32_t SLOT_TO_TOKEN_INDEX = 18;
constexpr uint32_t LRU_SLOTS_INDEX = 19;
constexpr uint32_t ACTIVE_ROWS_INDEX = 20;
constexpr uint32_t LIGHTNING_INDEXER_QUANT = 0;
constexpr uint32_t GATHERED_KV_INDEX = 1;
constexpr uint32_t PREFETCH_INDICES_INDEX = 2;
constexpr uint32_t PREFETCH_STATS_INDEX = 3;
constexpr uint32_t NEXT_THRESHOLD_INDEX = 4;
// Attributes Index
constexpr uint32_t ATTR_QUERY_QUANT_MODE_INDEX = 0;
constexpr uint32_t ATTR_KEY_QUANT_MODE_INDEX = 1;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 4;
constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 5;
constexpr uint32_t ATTR_KV_ROW_BYTES_INDEX = 6;
constexpr uint32_t ATTR_PREFETCH_BUDGET_INDEX = 7;
constexpr uint32_t ATTR_EMA_DECAY_INDEX = 8;
constexpr uint32_t ATTR_HOST_BLOCK_SIZE_INDEX = 9;
constexpr uint32_t ATTR_HOST_NUM_BLOCKS_INDEX = 10;
// Dim Index
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
// Dim Num
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_LIMIT = 2048;
constexpr uint32_t G_SIZE_LIMIT = 64;
constexpr uint32_t BLOCK_SIZE_LIMIT = 1024;
constexpr uint32_t BLOCK_SIZE_FACTOR = 16;
constexpr uint32_t SPARSE_MODE_LOWER = 3;

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(LIQTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint32_t, kvRowBytes)
TILING_DATA_FIELD_DEF(uint32_t, prefetchBudget)
TILING_DATA_FIELD_DEF(float, emaDecay)
TILING_DATA_FIELD_DEF(uint32_t, isArch35)
TILING_DATA_FIELD_DEF(uint32_t, keyBlockCount)
TILING_DATA_FIELD_DEF(uint32_t, queryRows)
TILING_DATA_FIELD_DEF(uint32_t, residentMode)
TILING_DATA_FIELD_DEF(uint32_t, residentCapacity)
TILING_DATA_FIELD_DEF(uint32_t, residentKBytes)
TILING_DATA_FIELD_DEF(uint32_t, residentVBytes)
TILING_DATA_FIELD_DEF(uint32_t, hostBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, hostBlockCount)
TILING_DATA_FIELD_DEF(uint32_t, hostMaxBlockNum)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(LightningIndexerQuant, LIQTilingData)

// -----------算子CompileInfo定义-------------------
struct LIQCompileInfo {};

// -----------算子Tiling入参结构体定义---------------
struct LIQParaInfo {
    TilingRequiredParaInfo query = {nullptr, nullptr};
    TilingRequiredParaInfo key = {nullptr, nullptr};
    TilingRequiredParaInfo weights = {nullptr, nullptr};
    TilingRequiredParaInfo query_dequant_scale = {nullptr, nullptr};
    TilingRequiredParaInfo key_dequant_scale = {nullptr, nullptr};
    TilingOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    TilingOptionalParaInfo actualSeqLengthsK = {nullptr, nullptr};
    TilingOptionalParaInfo blockTable = {nullptr, nullptr};
    TilingOptionalParaInfo kvAddresses = {nullptr, nullptr};
    TilingOptionalParaInfo prefetchThreshold = {nullptr, nullptr};
    TilingOptionalParaInfo reqIds = {nullptr, nullptr};
    TilingOptionalParaInfo tokenToReq = {nullptr, nullptr};
    TilingOptionalParaInfo stablePrefixLens = {nullptr, nullptr};
    TilingOptionalParaInfo hostBlockTable = {nullptr, nullptr};
    TilingOptionalParaInfo hostCacheBases = {nullptr, nullptr};
    TilingOptionalParaInfo residentK = {nullptr, nullptr};
    TilingOptionalParaInfo residentV = {nullptr, nullptr};
    TilingOptionalParaInfo lastReqIds = {nullptr, nullptr};
    TilingOptionalParaInfo slotToToken = {nullptr, nullptr};
    TilingOptionalParaInfo lruSlots = {nullptr, nullptr};
    TilingOptionalParaInfo activeRows = {nullptr, nullptr};
    TilingRequiredParaInfo attenOut = {nullptr, nullptr};

    const int64_t *queryQuantMode = nullptr;
    const int64_t *keyQuantMode = nullptr;
    const char *layOutQuery = nullptr;
    const char *layOutKey = nullptr;
    const int64_t *blockSize = nullptr;
    const int64_t *sparseMode = nullptr;
    const int64_t *sparseCount = nullptr;
    const int64_t *kvRowBytes = nullptr;
    const int64_t *prefetchBudget = nullptr;
    const float *emaDecay = nullptr;
    const int64_t *hostBlockSize = nullptr;
    const int64_t *hostNumBlocks = nullptr;
};

// -----------算子Tiling入参信息类---------------
class LIQTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    LIQParaInfo opParamInfo;
    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t gSize = 0;
    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t keyBlockCount = 0;
    uint32_t queryRows = 0;
    bool residentMode = false;
    uint32_t residentCapacity = 0;
    uint32_t residentKBytes = 0;
    uint32_t residentVBytes = 0;
    uint32_t hostBlockSize = 0;
    uint32_t hostBlockCount = 0;
    uint32_t hostMaxBlockNum = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    uint32_t sparseCount = 0;
    uint32_t kvRowBytes = 0;
    uint32_t prefetchBudget = 0;
    float emaDecay = 0.5F;
    bool isArch35 = false;
    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout
    DataLayout inputQLayout = DataLayout::BSND;
    DataLayout inputKLayout = DataLayout::PA_BSND;
};

// -----------算子Tiling入参信息解析及Check类---------------
class LIQInfoParser {
public:
    explicit LIQInfoParser(gert::TilingContext *context) : context_(context) {}
    ~LIQInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                        const std::string &actualSeqLenName);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus CheckAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus CheckScaleShape();
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetAndCheckBlockSize();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetQueryKeyAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetAttenMaskInfo();
    ge::graphStatus GetActualSeqInfo();
    void GenerateInfo(LIQTilingInfo &liqInfo);
    ge::graphStatus ParseAndCheck(LIQTilingInfo &liqInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    LIQParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    // Layout
    DataLayout qLayout_ = DataLayout::BSND;
    DataLayout kLayout_ = DataLayout::PA_BSND;
    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKType_ = ge::DT_FLOAT16;
    ge::DataType weightsType_ = ge::DT_FLOAT16;
    ge::DataType inputQueryScaleType_ = ge::DT_FLOAT16;
    ge::DataType inputKeyScaleType_ = ge::DT_FLOAT16;
    ge::DataType blockTableType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
};

// ---------------算子Tiling类---------------
class LightningIndexerQuantTiling {
public:
    explicit LightningIndexerQuantTiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus DoTiling(LIQTilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    LIQTilingData tilingData_;
};

}  // namespace optiling
#endif  // LIGHTNING_INDEXER_QUANT_TILING_H_
