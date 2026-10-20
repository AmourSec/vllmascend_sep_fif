/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#ifndef LIGHTNING_INDEXER_QUANT_RESIDENT_MEMORY_H_
#define LIGHTNING_INDEXER_QUANT_RESIDENT_MEMORY_H_

#include "kernel_operator.h"
#include "../lightning_indexer_quant_resident_policy.h"

namespace LIQ950 {
using namespace AscendC;

// Persistent rows can share a scalar-cache line (C is not alignment constrained).
// Read/write them exclusively with exact-length DMA, so no stale scalar-cache
// line can ever write back over an adjacent core's row. The 32-byte UB source
// remains aligned even when the GM row or individual element is unaligned.
class ResidentMetadataMemory {
public:
    __aicore__ inline void Init(LocalTensor<int32_t> scratch) { scratch_ = scratch; }
    template <typename T>
    __aicore__ inline T Read(GlobalTensor<T> &tensor, uint64_t index)
    {
        auto local = scratch_.ReinterpretCast<T>();
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        DataCopyPad(local, tensor[index], {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        return local.GetValue(0);
    }
    template <typename T>
    __aicore__ inline void Write(GlobalTensor<T> &tensor, uint64_t index, T value)
    {
        auto local = scratch_.ReinterpretCast<T>();
        local.SetValue(0, value);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(tensor[index], local, {1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
private:
    LocalTensor<int32_t> scratch_;
};

class ResidentStorage {
public:
    __aicore__ inline void Init(GM_ADDR mapping, GM_ADDR lru, GM_ADDR request, GM_ADDR workspace,
                                LocalTensor<int32_t> candidates, ResidentMetadataMemory *memory,
                                uint32_t capacity, TPipe *pipe)
    {
        pipe->InitBuffer(readBuf_, 512);
        pipe->InitBuffer(writeBuf_, 512);
        read_ = readBuf_.Get<int32_t>();
        write_ = writeBuf_.Get<int32_t>();
        mapping_.SetGlobalBuffer((__gm__ int32_t *)mapping);
        lru_.SetGlobalBuffer((__gm__ int32_t *)lru);
        request_.SetGlobalBuffer((__gm__ int64_t *)request);
        workspace_.SetGlobalBuffer((__gm__ int32_t *)workspace);
        candidates_ = candidates;
        memory_ = memory;
        capacity_ = capacity;
        rowElements_ = ((static_cast<uint64_t>(capacity) * 12 + 511) / 512) * 128;
    }
    __aicore__ inline void Row(uint32_t row)
    {
        readCount_ = writeCount_ = 0;
        row_ = row;
        metadataOffset_ = static_cast<uint64_t>(row) * capacity_;
        workspaceOffset_ = static_cast<uint64_t>(row) * rowElements_;
    }
    __aicore__ inline int32_t Get(ResidentArray array, uint32_t index)
    {
        switch (array) {
            case ResidentArray::Mapping: return WindowRead(mapping_, array, index);
            case ResidentArray::Lru: FlushLru(); return WindowRead(lru_, array, index);
            case ResidentArray::OriginalLru: return workspace_.GetValue(workspaceOffset_ + index);
            case ResidentArray::Classes: return workspace_.GetValue(workspaceOffset_ + capacity_ + index);
            case ResidentArray::MissSlots: return workspace_.GetValue(workspaceOffset_ + 2ULL * capacity_ + index);
            case ResidentArray::Candidates: return candidates_.GetValue(index);
        }
        return -1;
    }
    __aicore__ inline void Set(ResidentArray array, uint32_t index, int32_t value)
    {
        switch (array) {
            case ResidentArray::Mapping:
                memory_->Write(mapping_, metadataOffset_ + index, value);
                if (readArray_ == array && index >= readStart_ && index - readStart_ < readCount_)
                    read_.SetValue(index - readStart_, value);
                break;
            case ResidentArray::Lru:
                if (writeCount_ > 0 && index != writeStart_ + writeCount_) FlushLru();
                if (writeCount_ == 0) writeStart_ = index;
                write_.SetValue(writeCount_++, value);
                if (writeCount_ == 128) FlushLru();
                break;
            case ResidentArray::OriginalLru: workspace_.SetValue(workspaceOffset_ + index, value); break;
            case ResidentArray::Classes: workspace_.SetValue(workspaceOffset_ + capacity_ + index, value); break;
            case ResidentArray::MissSlots: workspace_.SetValue(workspaceOffset_ + 2ULL * capacity_ + index, value); break;
            case ResidentArray::Candidates: candidates_.SetValue(index, value); break;
        }
    }
    __aicore__ inline void FinishRow()
    {
        FlushLru();
        // Only the private align512 workspace uses scalar stores. Flush its
        // cache lines before another graph replay assigns this row to a new core.
        for (uint64_t i = 0; i < rowElements_; i += 8) {
            auto line = workspace_[workspaceOffset_ + i];
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(line);
        }
    }
    __aicore__ inline int64_t Request() { return memory_->Read(request_, row_); }
    __aicore__ inline void SetRequest(int64_t request) { memory_->Write(request_, row_, request); }
private:
    __aicore__ inline int32_t WindowRead(GlobalTensor<int32_t> &tensor, ResidentArray array, uint32_t index)
    {
        if (array != readArray_ || index < readStart_ || index - readStart_ >= readCount_) {
            readArray_ = array;
            readStart_ = index / 128 * 128;
            readCount_ = capacity_ - readStart_ < 128 ? capacity_ - readStart_ : 128;
            SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
            DataCopyPad(read_, tensor[metadataOffset_ + readStart_], {1, readCount_ * 4, 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        return read_.GetValue(index - readStart_);
    }
    __aicore__ inline void FlushLru()
    {
        if (writeCount_ == 0) return;
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        DataCopyPad(lru_[metadataOffset_ + writeStart_], write_, {1, writeCount_ * 4, 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        writeCount_ = 0;
        if (readArray_ == ResidentArray::Lru) readCount_ = 0;
    }
    TBuf<TPosition::VECCALC> readBuf_, writeBuf_;
    LocalTensor<int32_t> read_, write_;
    ResidentArray readArray_ = ResidentArray::Mapping;
    uint32_t readStart_ = 0, readCount_ = 0, writeStart_ = 0, writeCount_ = 0;
    ResidentMetadataMemory *memory_;
    GlobalTensor<int32_t> mapping_, lru_, workspace_;
    GlobalTensor<int64_t> request_;
    LocalTensor<int32_t> candidates_;
    uint32_t row_ = 0, capacity_ = 0;
    uint64_t metadataOffset_ = 0, workspaceOffset_ = 0, rowElements_ = 0;
};
} // namespace LIQ950
#endif
