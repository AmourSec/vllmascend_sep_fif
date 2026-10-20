/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository.
 */
#ifndef LIGHTNING_INDEXER_QUANT_PREFETCH_ARCH35_H
#define LIGHTNING_INDEXER_QUANT_PREFETCH_ARCH35_H

#include "kernel_operator.h"

namespace LIQ950 {
using namespace AscendC;

__aicore__ inline uint64_t AlignBytes(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

// A call owns its staging rows. Candidate DMA is speculative: candidate_ records
// admitted slots immediately, while Drain() makes every row ready before Find().
class PrefetchRows {
public:
    __aicore__ inline void Init(TPipe *pipe, uint32_t rowBytes, uint32_t budget, uint32_t topK,
                               uint32_t addressStride, GM_ADDR addresses, GM_ADDR threshold,
                               GM_ADDR staging, GM_ADDR gathered, GM_ADDR candidates,
                               GM_ADDR stats, GM_ADDR nextThreshold, GM_ADDR indices)
    {
        rowBytes_ = rowBytes;
        budget_ = budget;
        topK_ = topK;
        addressStride_ = addressStride;
        rowStride_ = AlignBytes(rowBytes, 32);
        batchStride_ = AlignBytes((budget > 0 ? budget : 1) * rowStride_, 512);
        pipe->InitBuffer(copyBuf_, 4096);
        pipe->InitBuffer(gatherBuf_, AlignBytes(rowBytes_ > 0 ? rowBytes_ : 1, 32));
        pipe->InitBuffer(candidateBuf_, AlignBytes((budget > 0 ? budget : 1) * 4, 32));
        pipe->InitBuffer(indexBuf_, AlignBytes(topK * 4, 32));
        pipe->InitBuffer(stateBuf_, 32);
        copy_ = copyBuf_.Get<uint8_t>();
        gather_[0] = copy_;
        gather_[1] = gatherBuf_.Get<uint8_t>();
        banks_[0] = gather_[0];
        banks_[1] = gather_[1];
        candidate_ = candidateBuf_.Get<int32_t>();
        index_ = indexBuf_.Get<int32_t>();
        state_ = stateBuf_.Get<int32_t>();
        addresses_.SetGlobalBuffer((__gm__ int64_t *)addresses);
        threshold_.SetGlobalBuffer((__gm__ float *)threshold);
        staging_.SetGlobalBuffer(staging);
        gathered_.SetGlobalBuffer(gathered);
        candidates_.SetGlobalBuffer((__gm__ int32_t *)candidates);
        stats_.SetGlobalBuffer((__gm__ int32_t *)stats);
        nextThreshold_.SetGlobalBuffer((__gm__ float *)nextThreshold);
        indices_.SetGlobalBuffer((__gm__ int32_t *)indices);
    }

    __aicore__ inline void Reset(uint32_t batch)
    {
        batch_ = batch;
        count_ = 0;
        fallback_ = 0;
        thresholdValue_ = rowBytes_ > 0 ? threshold_.GetValue(batch) : 0.0F;
        for (uint32_t i = 0; i < (budget_ > 0 ? budget_ : 1); ++i) {
            candidate_.SetValue(i, -1);
        }
        for (uint32_t i = 0; i < topK_; ++i) {
            index_.SetValue(i, -1);
        }
        bankPending_[0] = false;
        bankPending_[1] = false;
    }

    __aicore__ inline void Consider(uint32_t token, float score)
    {
        if (rowBytes_ == 0 || count_ >= budget_ || !(score > thresholdValue_)) {
            return;
        }
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer((__gm__ uint8_t *)(uint64_t)addresses_.GetValue(
            static_cast<uint64_t>(batch_) * addressStride_ + token));
        const uint32_t slot = count_;
        const uint32_t bank = slot & 1U;
        const uint32_t eventId = bank == 0 ? EVENT_ID1 : EVENT_ID2;
        if (bankPending_[bank]) {
            // This consumes the previous bank's reuse event before overwriting UB.
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        }
        SetFlag<HardEvent::S_MTE2>(eventId);
        WaitFlag<HardEvent::S_MTE2>(eventId);
        DataCopyPad(banks_[bank], source, {1, rowBytes_, 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        DataCopyPad(staging_[static_cast<uint64_t>(batch_) * batchStride_ + slot * rowStride_],
                    banks_[bank], {1, rowBytes_, 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        bankPending_[bank] = true;
        // The ID is admitted now; Drain(), not Consider(), waits for its DMA.
        candidate_.SetValue(slot, token);
        ++count_;
    }

    template <typename TopK>
    __aicore__ inline void Finish(TopK &selected, float decay, bool validMetadata)
    {
        Drain();
        gatherPending_[0] = false;
        gatherPending_[1] = false;
        if (rowBytes_ > 0 && selected.Count() > 0) {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        }
        for (uint32_t i = 0; i < selected.Count(); ++i) {
            int32_t token = selected.Index(i);
            index_.SetValue(i, token);
            if (rowBytes_ == 0) {
                continue;
            }
            int32_t slot = Find(token);
            GlobalTensor<uint8_t> source;
            if (slot >= 0) {
                source = staging_[static_cast<uint64_t>(batch_) * batchStride_ + slot * rowStride_];
            } else {
                source.SetGlobalBuffer((__gm__ uint8_t *)(uint64_t)addresses_.GetValue(
                    static_cast<uint64_t>(batch_) * addressStride_ + token));
                ++fallback_;
            }
            uint32_t bank = i & 1U;
            uint32_t eventId = bank == 0 ? EVENT_ID1 : EVENT_ID2;
            if (gatherPending_[bank]) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            }
            DataCopyPad(gather_[bank], source, {1, rowBytes_, 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopyPad(gathered_[(static_cast<uint64_t>(batch_) * topK_ + i) * rowBytes_],
                        gather_[bank], {1, rowBytes_, 0, 0, 0});
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
            gatherPending_[bank] = true;
        }
        for (uint32_t bank = 0; bank < 2; ++bank) {
            if (gatherPending_[bank]) {
                uint32_t eventId = bank == 0 ? EVENT_ID1 : EVENT_ID2;
                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            }
        }
        if (rowBytes_ > 0 && selected.Count() > 0) {
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
        if (rowBytes_ > 0) {
            for (uint32_t byte = 0; byte < rowBytes_; ++byte) {
                copy_.SetValue(byte, 0);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            for (uint32_t i = selected.Count(); i < topK_; ++i) {
                DataCopyPad(gathered_[(static_cast<uint64_t>(batch_) * topK_ + i) * rowBytes_],
                            copy_, {1, rowBytes_, 0, 0, 0});
            }
            PipeBarrier<PIPE_ALL>();
        }
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(indices_[static_cast<uint64_t>(batch_) * topK_], index_, {1, topK_ * 4, 0, 0, 0});
        uint32_t candidateWidth = budget_ > 0 ? budget_ : 1;
        DataCopyPad(candidates_[static_cast<uint64_t>(batch_) * candidateWidth], candidate_,
                    {1, candidateWidth * 4, 0, 0, 0});
        state_.SetValue(0, validMetadata ? count_ : -1);
        state_.SetValue(1, validMetadata ? fallback_ : -1);
        float next = thresholdValue_;
        if (rowBytes_ > 0 && selected.Count() == topK_) {
            union { float value; uint32_t bits; } previous;
            previous.value = thresholdValue_;
            float kth = selected.MinScore();
            next = (previous.bits & 0x7f800000U) == 0x7f800000U ? kth :
                   decay * thresholdValue_ + (1.0F - decay) * kth;
        }
        auto floatState = state_.ReinterpretCast<float>();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(stats_[batch_ * 2], state_, {1, 8, 0, 0, 0});
        // DataCopyPad's UB source must be 32-byte aligned.
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        floatState.SetValue(0, next);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(nextThreshold_[batch_], floatState, {1, 4, 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

private:
    __aicore__ inline void Drain()
    {
        for (uint32_t bank = 0; bank < 2; ++bank) {
            if (bankPending_[bank]) {
                const uint32_t eventId = bank == 0 ? EVENT_ID1 : EVENT_ID2;
                // Drain candidate DMA before final gather reuses both UB banks and events 1/2.
                WaitFlag<HardEvent::MTE3_MTE2>(eventId);
                bankPending_[bank] = false;
            }
        }
        // Always consume the gate, including empty/zero-byte/budget-zero calls.
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline int32_t Find(int32_t token)
    {
        uint32_t left = 0;
        uint32_t right = count_;
        while (left < right) {
            uint32_t mid = (left + right) / 2;
            if (candidate_.GetValue(mid) < token) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left < count_ && candidate_.GetValue(left) == token ? static_cast<int32_t>(left) : -1;
    }

    uint32_t rowBytes_, budget_, topK_, addressStride_, batch_, count_, fallback_;
    uint64_t rowStride_, batchStride_;
    float thresholdValue_;
    TBuf<TPosition::VECCALC> copyBuf_, candidateBuf_, indexBuf_, stateBuf_;
    TBuf<TPosition::VECCALC> gatherBuf_;
    LocalTensor<uint8_t> copy_;
    LocalTensor<uint8_t> banks_[2];
    LocalTensor<uint8_t> gather_[2];
    LocalTensor<int32_t> candidate_, index_, state_;
    GlobalTensor<int64_t> addresses_;
    GlobalTensor<float> threshold_, nextThreshold_;
    GlobalTensor<uint8_t> staging_, gathered_;
    GlobalTensor<int32_t> candidates_, stats_, indices_;
    bool bankPending_[2];
    bool gatherPending_[2];
};
} // namespace LIQ950
#endif
