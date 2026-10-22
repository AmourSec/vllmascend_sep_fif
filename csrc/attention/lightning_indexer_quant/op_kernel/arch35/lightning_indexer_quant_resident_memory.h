/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#ifndef LIGHTNING_INDEXER_QUANT_RESIDENT_MEMORY_H_
#define LIGHTNING_INDEXER_QUANT_RESIDENT_MEMORY_H_

#include "kernel_operator.h"

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

} // namespace LIQ950
#endif
