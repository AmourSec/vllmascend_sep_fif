/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#ifndef LIGHTNING_INDEXER_QUANT_RESIDENT_ARCH35_H_
#define LIGHTNING_INDEXER_QUANT_RESIDENT_ARCH35_H_

#include "kernel_operator.h"
#include "lightning_indexer_quant_resident_memory.h"
#include "lightning_indexer_quant_resident_transfer.h"
#include "../lightning_indexer_quant_resident_source.h"

namespace LIQ950 {
class ResidentRows {
public:
    __aicore__ inline void Init(TPipe *pipe, const LIQTilingData *tiling,
        GM_ADDR reqIds, GM_ADDR stablePrefix, GM_ADDR hostBlocks, GM_ADDR hostBases,
        GM_ADDR residentK, GM_ADDR residentV, GM_ADDR lastReqIds, GM_ADDR slotToToken,
        GM_ADDR lruSlots, GM_ADDR threshold, GM_ADDR candidates, GM_ADDR stats,
        GM_ADDR nextThreshold, GM_ADDR indices, GM_ADDR gatheredDummy, GM_ADDR workspace)
    {
        t_ = tiling;
        width_ = t_->prefetchBudget > 0 ? t_->prefetchBudget : 1;
        pipe->InitBuffer(candidateBuf_, ((width_ * 4 + 31) / 32) * 32);
        pipe->InitBuffer(indexBuf_, ((t_->sparseCount * 4 + 31) / 32) * 32);
        pipe->InitBuffer(stateBuf_, 32);
        candidate_ = candidateBuf_.Get<int32_t>();
        index_ = indexBuf_.Get<int32_t>();
        state_ = stateBuf_.Get<int32_t>();
        memory_.Init(state_);
        storage_.Init(slotToToken, lruSlots, lastReqIds, workspace, candidate_, &memory_, t_->residentCapacity, pipe);
        transfers_.Init(pipe);
        reqIds_.SetGlobalBuffer((__gm__ int64_t *)reqIds);
        stablePrefix_.SetGlobalBuffer((__gm__ int32_t *)stablePrefix);
        hostBlocks_.SetGlobalBuffer((__gm__ int32_t *)hostBlocks);
        hostBases_.SetGlobalBuffer((__gm__ int64_t *)hostBases);
        residentK_.SetGlobalBuffer(residentK);
        residentV_.SetGlobalBuffer(residentV);
        threshold_.SetGlobalBuffer((__gm__ float *)threshold);
        candidates_.SetGlobalBuffer((__gm__ int32_t *)candidates);
        stats_.SetGlobalBuffer((__gm__ int32_t *)stats);
        nextThreshold_.SetGlobalBuffer((__gm__ float *)nextThreshold);
        indices_.SetGlobalBuffer((__gm__ int32_t *)indices);
        gatheredDummy_.SetGlobalBuffer(gatheredDummy);
    }

    __aicore__ inline void Reset(uint32_t row, uint32_t request, uint32_t visible, bool valid)
    {
        row_ = row;
        request_ = request;
        visible_ = visible;
        thresholdValue_ = memory_.Read(threshold_, row);
        metadataOk_ = !valid || ValidateSources();
        valid_ = valid && metadataOk_;
        for (uint32_t i = 0; i < t_->sparseCount; ++i) index_.SetValue(i, -1);
        storage_.Row(row);
        policy_.Init(storage_, t_->residentCapacity, t_->prefetchBudget, thresholdValue_);
        const int64_t requestId = valid_ ? memory_.Read(reqIds_, request) : 0;
        const int32_t stable = valid_ ? memory_.Read(stablePrefix_, row) : 0;
        touched_ = valid_;
        if (!policy_.Reset(requestId, stable, visible, valid_) && valid_) {
            valid_ = false;
            metadataOk_ = false;
        }
    }

    __aicore__ inline void Consider(uint32_t token, float score) { policy_.Consider(token, score); }

    __aicore__ inline void FlushTile()
    {
        // The preceding tile can remain in flight while the next scores/TopK are
        // computed. Publish only after both streams finish; consume reuse events
        // before the next tile recycles descriptor storage or either UB bank.
        transfers_.Drain(policy_);
        if (!valid_) return;
        policy_.FlushTile(transfers_);
        for (uint32_t i = 0; i < transfers_.Count(); ++i) {
            uint64_t sourceK = 0, sourceV = 0;
            if (!Source(static_cast<uint32_t>(transfers_.Token(i)), sourceK, sourceV)) {
                valid_ = metadataOk_ = false;
                break;
            }
            const uint64_t slot = static_cast<uint64_t>(row_) * t_->residentCapacity + transfers_.Slot(i);
            transfers_.Submit(sourceK, residentK_[slot * t_->residentKBytes], t_->residentKBytes);
            transfers_.Submit(sourceV, residentV_[slot * t_->residentVBytes], t_->residentVBytes);
            transfers_.Submitted();
        }
    }

    template <typename TopK>
    __aicore__ inline void Finish(TopK &selected, float decay, bool validMetadata)
    {
        FlushTile();
        transfers_.Drain(policy_);
        const bool metadata = validMetadata && metadataOk_;
        const auto result = policy_.Finish(metadata);
        if (touched_) storage_.FinishRow();
        float next = thresholdValue_;
        if (valid_ && metadata) {
            for (uint32_t i = 0; i < selected.Count() && i < t_->sparseCount; ++i)
                index_.SetValue(i, static_cast<int32_t>(selected.Index(i)));
            if (selected.Count() == t_->sparseCount) {
                union { float value; uint32_t bits; } previous;
                previous.value = thresholdValue_;
                const float kth = selected.MinScore();
                next = (previous.bits & 0x7f800000U) == 0x7f800000U ? kth :
                       decay * thresholdValue_ + (1.0F - decay) * kth;
            }
        }
        if (!metadata) {
            for (uint32_t i = 0; i < width_; ++i) candidate_.SetValue(i, -1);
        }
        auto zero = transfers_.ZeroBuffer();
        for (uint32_t i = 0; i < t_->sparseCount; ++i) zero.SetValue(i, 0);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(indices_[static_cast<uint64_t>(row_) * t_->sparseCount], index_,
                    {1, t_->sparseCount * 4, 0, 0, 0});
        DataCopyPad(candidates_[static_cast<uint64_t>(row_) * width_], candidate_, {1, width_ * 4, 0, 0, 0});
        DataCopyPad(gatheredDummy_[static_cast<uint64_t>(row_) * t_->sparseCount], zero,
                    {1, t_->sparseCount, 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        state_.SetValue(0, result.admitted);
        state_.SetValue(1, result.copiedMisses);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(stats_[static_cast<uint64_t>(row_) * 2], state_, {1, 8, 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        memory_.Write(nextThreshold_, row_, next);
    }

private:
    __aicore__ inline bool ValidateSources()
    {
        if (request_ >= t_->bSize || t_->hostBlockSize == 0 || t_->hostBlockCount == 0 ||
            t_->residentKBytes == 0 || t_->residentKBytes > 4096 ||
            t_->residentVBytes == 0 || t_->residentVBytes > 4096) return false;
        baseK_ = static_cast<uint64_t>(memory_.Read(hostBases_, 0));
        baseV_ = static_cast<uint64_t>(memory_.Read(hostBases_, 1));
        if (baseK_ == 0 || baseV_ == 0) return false;
        const uint64_t pages = (static_cast<uint64_t>(visible_) + t_->hostBlockSize - 1) / t_->hostBlockSize;
        if (pages > t_->hostMaxBlockNum) return false;
        for (uint64_t page = 0; page < pages; ++page) {
            const uint64_t end = (page + 1) * t_->hostBlockSize;
            const uint32_t last = static_cast<uint32_t>((end < visible_ ? end : visible_) - 1);
            uint64_t sourceK = 0, sourceV = 0;
            if (!Source(last, sourceK, sourceV)) return false;
        }
        return true;
    }
    __aicore__ inline bool Source(uint32_t token, uint64_t &sourceK, uint64_t &sourceV)
    {
        if (token >= visible_ || t_->hostBlockSize == 0) return false;
        const uint32_t page = token / t_->hostBlockSize;
        if (page >= t_->hostMaxBlockNum) return false;
        const int32_t physical = memory_.Read(hostBlocks_, static_cast<uint64_t>(request_) * t_->hostMaxBlockNum + page);
        if (physical < 0 || static_cast<uint32_t>(physical) >= t_->hostBlockCount) return false;
        const uint32_t offset = token % t_->hostBlockSize;
        return ResidentSourceAddress(baseK_, physical, offset, t_->hostBlockSize, t_->residentKBytes, sourceK) &&
               ResidentSourceAddress(baseV_, physical, offset, t_->hostBlockSize, t_->residentVBytes, sourceV);
    }

    const LIQTilingData *t_;
    ResidentMetadataMemory memory_;
    ResidentStorage storage_;
    ResidentPolicy<ResidentStorage> policy_;
    ResidentTransfers transfers_;
    TBuf<TPosition::VECCALC> candidateBuf_, indexBuf_, stateBuf_;
    LocalTensor<int32_t> candidate_, index_, state_;
    GlobalTensor<int64_t> reqIds_, hostBases_;
    GlobalTensor<int32_t> stablePrefix_, hostBlocks_, candidates_, stats_, indices_;
    GlobalTensor<float> threshold_, nextThreshold_;
    GlobalTensor<uint8_t> residentK_, residentV_, gatheredDummy_;
    uint32_t row_ = 0, request_ = 0, visible_ = 0, width_ = 0;
    uint64_t baseK_ = 0, baseV_ = 0;
    float thresholdValue_ = 0.0F;
    bool valid_ = false, metadataOk_ = false, touched_ = false;
};
} // namespace LIQ950
#endif
