/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#ifndef LIGHTNING_INDEXER_QUANT_RESIDENT_TRANSFER_H_
#define LIGHTNING_INDEXER_QUANT_RESIDENT_TRANSFER_H_

#include "kernel_operator.h"

namespace LIQ950 {
using namespace AscendC;

class ResidentTransfers {
public:
    __aicore__ inline void Init(TPipe *pipe)
    {
        pipe->InitBuffer(firstBankBuf_, 4096);
        pipe->InitBuffer(secondBankBuf_, 4096);
        pipe->InitBuffer(pendingBuf_, 128 * 3 * sizeof(int32_t));
        banks_[0] = firstBankBuf_.Get<uint8_t>();
        banks_[1] = secondBankBuf_.Get<uint8_t>();
        pending_ = pendingBuf_.Get<int32_t>();
    }
    __aicore__ inline void BeginTile(uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i) pending_.SetValue(i, 0);
    }
    __aicore__ inline void Protect(uint32_t index) { pending_.SetValue(index, 1); }
    __aicore__ inline bool HasHit(uint32_t index) const { return pending_.GetValue(index) != 0; }
    // Policy invokes Enqueue only while classifying one <=128-score tile.
    __aicore__ inline void Enqueue(int32_t token, int32_t slot)
    {
        pending_.SetValue(128 + count_, token);
        pending_.SetValue(256 + count_, slot);
        ++count_;
    }
    __aicore__ inline void Submitted() { ++submitted_; }
    __aicore__ inline uint32_t Count() const { return count_; }
    __aicore__ inline int32_t Token(uint32_t index) const { return pending_.GetValue(128 + index); }
    __aicore__ inline int32_t Slot(uint32_t index) const { return pending_.GetValue(256 + index); }
    __aicore__ inline LocalTensor<uint8_t> ZeroBuffer() const { return banks_[0]; }

    __aicore__ inline void Submit(uint64_t source, GlobalTensor<uint8_t> destination, uint32_t bytes)
    {
        const uint32_t bank = nextBank_;
        nextBank_ ^= 1U;
        const uint32_t event = bank == 0 ? EVENT_ID1 : EVENT_ID2;
        if (busy_[bank]) WaitFlag<HardEvent::MTE3_MTE2>(event);
        GlobalTensor<uint8_t> input;
        input.SetGlobalBuffer((__gm__ uint8_t *)source);
        SetFlag<HardEvent::S_MTE2>(event);
        WaitFlag<HardEvent::S_MTE2>(event);
        DataCopyPad(banks_[bank], input, {1, bytes, 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_MTE3>(event);
        WaitFlag<HardEvent::MTE2_MTE3>(event);
        DataCopyPad(destination, banks_[bank], {1, bytes, 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(event);
        busy_[bank] = true;
    }

    template <typename Policy>
    __aicore__ inline void Drain(Policy &policy)
    {
        for (uint32_t bank = 0; bank < 2; ++bank) {
            if (busy_[bank]) {
                const uint32_t event = bank == 0 ? EVENT_ID1 : EVENT_ID2;
                WaitFlag<HardEvent::MTE3_MTE2>(event);
                busy_[bank] = false;
            }
        }
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        for (uint32_t i = 0; i < submitted_; ++i) policy.Commit(Slot(i), Token(i));
        count_ = submitted_ = 0;
    }
private:
    TBuf<TPosition::VECCALC> firstBankBuf_, secondBankBuf_, pendingBuf_;
    LocalTensor<uint8_t> banks_[2];
    LocalTensor<int32_t> pending_;
    uint32_t count_ = 0, submitted_ = 0, nextBank_ = 0;
    bool busy_[2] = {false, false};
};
} // namespace LIQ950
#endif
