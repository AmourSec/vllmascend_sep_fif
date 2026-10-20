/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository.
 */
#ifndef LIGHTNING_INDEXER_QUANT_KERNEL_ARCH35_H
#define LIGHTNING_INDEXER_QUANT_KERNEL_ARCH35_H

#include "kernel_operator.h"
#if __has_include("../../quant_lightning_indexer_v2/arch35/quant_lightning_indexer_v2_service_cube_arch35.h")
#include "../../quant_lightning_indexer_v2/arch35/quant_lightning_indexer_v2_service_cube_arch35.h"
#else
#include "../../../quant_lightning_indexer_v2/op_kernel/arch35/quant_lightning_indexer_v2_service_cube_arch35.h"
#endif
#include "lightning_indexer_quant_score_arch35.h"
#include "lightning_indexer_quant_topk_arch35.h"
#include "lightning_indexer_quant_prefetch_arch35.h"
#include "lightning_indexer_quant_query_rows.h"
#include "lightning_indexer_quant_resident_arch35.h"

namespace LIQ950 {
using namespace AscendC;
template <typename QueryT, typename QkT, typename ScaleT>
class DecodePrefetch {
public:
    using CubeType = QLIV2Common::QLIV2Type<QueryT, QueryT, QkT, uint16_t, int32_t, true,
        QLIV2Common::LI_LAYOUT::TND, QLIV2Common::LI_LAYOUT::PA_BBND, ScaleT, ScaleT>;

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR weights, GM_ADDR qScale, GM_ADDR kScale,
        GM_ADDR qLengths, GM_ADDR kLengths, GM_ADDR blocks, GM_ADDR addresses, GM_ADDR threshold,
        GM_ADDR reqIds, GM_ADDR tokenToReq, GM_ADDR stablePrefix, GM_ADDR hostBlocks, GM_ADDR hostBases,
        GM_ADDR residentK, GM_ADDR residentV, GM_ADDR lastReqIds, GM_ADDR slotToToken,
        GM_ADDR lruSlots, GM_ADDR activeRows,
        GM_ADDR indices, GM_ADDR gathered, GM_ADDR candidates, GM_ADDR stats, GM_ADDR nextThreshold,
        GM_ADDR workspace, const LIQTilingData *tiling, TPipe *pipe)
    {
        t_ = tiling;
        query_.SetGlobalBuffer((__gm__ QueryT *)query);
        key_.SetGlobalBuffer((__gm__ QueryT *)key);
        weights_.SetGlobalBuffer((__gm__ ScaleT *)weights);
        qScale_.SetGlobalBuffer((__gm__ ScaleT *)qScale);
        kScale_.SetGlobalBuffer((__gm__ ScaleT *)kScale);
        qLengths_.SetGlobalBuffer((__gm__ int32_t *)qLengths);
        kLengths_.SetGlobalBuffer((__gm__ int32_t *)kLengths);
        blocks_.SetGlobalBuffer((__gm__ int32_t *)blocks);
        if (t_->residentMode != 0) {
            tokenToReq_.SetGlobalBuffer((__gm__ int32_t *)tokenToReq);
            activeRows_.SetGlobalBuffer((__gm__ int32_t *)activeRows);
            hostBlocks_.SetGlobalBuffer((__gm__ int32_t *)hostBlocks);
        }
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(blocks_);
        if ASCEND_IS_AIC {
            QLIV2Common::ConstInfo info{};
            info.mBaseSize = 2 * t_->gSize;
            info.mBaseSizeMax = 128;
            info.s1BaseSize = 2;
            info.s2BaseSize = 128;
            info.batchSize = t_->bSize;
            info.gSize = t_->gSize;
            info.qHeadNum = t_->gSize;
            info.kHeadNum = 1;
            info.headDim = 128;
            info.sparseCount = t_->sparseCount;
            info.kSeqSize = t_->s2Size;
            info.qSeqSize = 1;
            info.kCacheBlockSize = t_->blockSize;
            info.maxBlockNumPerBatch = t_->maxBlockNumPerBatch;
            info.keyStride0 = t_->blockSize * 128;
            info.keyDequantScaleStride0 = t_->blockSize;
            info.maxSeqlenQ = -1;
            info.quantMode = std::is_same_v<QueryT, int8_t> ? 2 : 1;
            cube_.InitParams(info);
            cube_.InitBuffers(pipe);
            cube_.InitMm1GlobalTensor(blocks_, key_, query_);
        }
        if ASCEND_IS_AIV {
            // FIX dual destinations require the identical first UB allocation on AIC and AIV.
            pipe->InitBuffer(qkBuf_, 65536);
            qk_ = qkBuf_.Get<QkT>();
            pipe->InitBuffer(scaleBuf_, 256 * sizeof(ScaleT));
            scales_ = scaleBuf_.Get<ScaleT>();
            pipe->InitBuffer(scoreBuf_, 512);
            scores_ = scoreBuf_.Get<float>();
            selected_.Init(pipe, t_->sparseCount);
            if (t_->residentMode != 0) {
                resident_.Init(pipe, t_, reqIds, stablePrefix, hostBlocks, hostBases, residentK,
                    residentV, lastReqIds, slotToToken, lruSlots, threshold, candidates, stats,
                    nextThreshold, indices, gathered, workspace);
            } else {
                rows_.Init(pipe, t_->kvRowBytes, t_->prefetchBudget, t_->sparseCount, t_->s2Size,
                    addresses, threshold, workspace, gathered, candidates, stats, nextThreshold, indices);
            }
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t core = GetBlockIdx();
        if ASCEND_IS_AIV {
            core /= GetTaskRation();
        }
        if (core >= t_->usedCoreNum) {
            return;
        }
        if ASCEND_IS_AIC {
            cube_.AllocEventID();
        }
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<4, PIPE_V>(0);
            CrossCoreSetFlag<4, PIPE_V>(1);
        }
        uint32_t loop = 0;
        for (uint32_t row = core; row < t_->queryRows; row += t_->usedCoreNum) {
            QueryRow queryRow{};
            bool inactive = false;
            bool valid = ReadRow(row, queryRow, inactive);
            uint32_t length = valid ? queryRow.visible : 0;
            uint32_t request = queryRow.request;
            if ASCEND_IS_AIV {
                if (GetSubBlockIdx() == 0) {
                    selected_.Reset();
                    if (t_->residentMode != 0) {
                        resident_.Reset(row, request, length, valid);
                    } else {
                        rows_.Reset(row, request);
                    }
                    if (length > 0) {
                        uint64_t queryOffset = static_cast<uint64_t>(row) * t_->gSize;
                        DataCopy(scales_, weights_[queryOffset], t_->gSize);
                        DataCopy(scales_[64], qScale_[queryOffset], t_->gSize);
                    }
                }
            }
            for (uint32_t start = 0; start < length; start += 128, ++loop) {
                uint32_t size = length - start < 128 ? length - start : 128;
                if ASCEND_IS_AIC {
                    QLIV2Common::RunInfo run{};
                    run.loop = loop;
                    run.bIdx = request;
                    run.s2Idx = start / 128;
                    run.actMBaseSize = t_->gSize;
                    run.actS1Size = 1;
                    run.actS2Size = length;
                    run.actualSingleProcessSInnerSize = size;
                    run.tensorQueryOffset = static_cast<uint64_t>(row) * t_->gSize * 128;
                    run.isFirstS2InnerLoop = start == 0;
                    run.isLastS2InnerLoop = start + size == length;
                    cube_.ComputeMm1(run);
                }
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag<4, PIPE_V>(2 + loop % 2);
                    if (GetSubBlockIdx() == 0) {
                        LoadKeyScales(request, start, size);
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        ComputeScores(scores_, qk_[(loop % 2) * 64], scales_, scales_[64], scales_[128], t_->gSize);
                        SetFlag<HardEvent::V_S>(EVENT_ID0);
                        WaitFlag<HardEvent::V_S>(EVENT_ID0);
                    }
                    // Release the QK bank before fetching KV, allowing the next AIC tile to run.
                    CrossCoreSetFlag<4, PIPE_V>(loop % 2);
                    if (GetSubBlockIdx() == 0) {
                        for (uint32_t token = 0; token < size; ++token) {
                            float score = scores_.GetValue(token);
                            selected_.Append(score, start + token);
                            if (t_->residentMode != 0) {
                                resident_.Consider(start + token, score);
                            } else {
                                rows_.Consider(start + token, score);
                            }
                        }
                        if (t_->residentMode != 0) {
                            resident_.FlushTile();
                        }
                    }
                }
            }
            if ASCEND_IS_AIV {
                if (GetSubBlockIdx() == 0) {
                    selected_.Finish();
                    if (t_->residentMode != 0) {
                        resident_.Finish(selected_, t_->emaDecay, valid || inactive);
                    } else {
                        rows_.Finish(selected_, t_->emaDecay, valid);
                    }
                }
            }
        }
        if ASCEND_IS_AIC {
            cube_.FreeEventID();
            CrossCoreWaitFlag<4, PIPE_FIX>(0);
            CrossCoreWaitFlag<4, PIPE_FIX>(1);
            CrossCoreWaitFlag<4, PIPE_FIX>(16);
            CrossCoreWaitFlag<4, PIPE_FIX>(17);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline bool ReadRow(uint32_t row, QueryRow &queryRow, bool &inactive)
    {
        uint32_t active = t_->queryRows;
        if (t_->residentMode != 0) {
            int32_t activeValue = activeRows_.GetValue(0);
            if (activeValue < 0 || static_cast<uint32_t>(activeValue) > t_->queryRows) {
                return false;
            }
            active = static_cast<uint32_t>(activeValue);
            inactive = row >= active;
        }
        if (!ResolveQueryRow(row, t_->bSize, t_->queryRows, active, t_->s2Size,
                             qLengths_, kLengths_, queryRow)) {
            return false;
        }
        uint32_t request = queryRow.request;
        uint32_t pages = (queryRow.visible + t_->blockSize - 1) / t_->blockSize;
        for (uint32_t page = 0; page < pages; ++page) {
            int32_t physical = blocks_.GetValue(static_cast<uint64_t>(request) * t_->maxBlockNumPerBatch + page);
            if (physical < 0 || physical >= t_->keyBlockCount) {
                return false;
            }
        }
        if (t_->residentMode != 0) {
            if (tokenToReq_.GetValue(row) != static_cast<int32_t>(request)) {
                return false;
            }
            uint32_t hostPages = (queryRow.visible + t_->hostBlockSize - 1) / t_->hostBlockSize;
            if (hostPages > t_->hostMaxBlockNum) {
                return false;
            }
            for (uint32_t page = 0; page < hostPages; ++page) {
                int32_t physical = hostBlocks_.GetValue(static_cast<uint64_t>(request) * t_->hostMaxBlockNum + page);
                if (physical < 0 || physical >= t_->hostBlockCount) {
                    return false;
                }
            }
        }
        return true;
    }

    __aicore__ inline void LoadKeyScales(uint32_t batch, uint32_t start, uint32_t size)
    {
        uint32_t done = 0;
        while (done < size) {
            uint32_t logical = start + done;
            uint32_t pageOffset = logical % t_->blockSize;
            uint32_t part = t_->blockSize - pageOffset;
            part = part < size - done ? part : size - done;
            int32_t physical = blocks_.GetValue(
                static_cast<uint64_t>(batch) * t_->maxBlockNumPerBatch + logical / t_->blockSize);
            DataCopyPad(scales_[128 + done], kScale_[static_cast<uint64_t>(physical) * t_->blockSize + pageOffset],
                        {1, static_cast<uint32_t>(part * sizeof(ScaleT)), 0, 0, 0}, {false, 0, 0, 0});
            done += part;
        }
    }

    const LIQTilingData *t_;
    QLIV2Kernel::QLIV2Matmul<CubeType> cube_;
    StreamingTopK selected_;
    PrefetchRows rows_;
    ResidentRows resident_;
    GlobalTensor<QueryT> query_, key_;
    GlobalTensor<ScaleT> weights_, qScale_, kScale_;
    GlobalTensor<int32_t> qLengths_, kLengths_, blocks_;
    GlobalTensor<int32_t> tokenToReq_, activeRows_, hostBlocks_;
    TBuf<TPosition::VECCALC> qkBuf_, scaleBuf_, scoreBuf_;
    LocalTensor<QkT> qk_;
    LocalTensor<ScaleT> scales_;
    LocalTensor<float> scores_;
};
}
#endif
