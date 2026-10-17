/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full license text.
 */

#ifndef LIGHTNING_INDEXER_QUANT_TOPK_ARCH35_H
#define LIGHTNING_INDEXER_QUANT_TOPK_ARCH35_H

#include "kernel_operator.h"
#if __has_include("../../quant_lightning_indexer_v2/arch35/vf/vf_topk.h")
#include "../../quant_lightning_indexer_v2/arch35/vf/vf_topk.h"
#else
#include "../../../quant_lightning_indexer_v2/op_kernel/arch35/vf/vf_topk.h"
#endif

namespace LIQ950 {

class StreamingTopK {
public:
    __aicore__ inline void Init(AscendC::TPipe *pipe, uint32_t k)
    {
        pipe_ = pipe;
        k_ = k;
        capacity_ = Align(k_ + kOverflowSlack, kAlignment);
        pipe_->InitBuffer(inputKeyBuf_, capacity_ * sizeof(uint32_t));
        pipe_->InitBuffer(inputIdBuf_, capacity_ * sizeof(uint32_t));
        pipe_->InitBuffer(outputIdxBuf_, Align(k_ + kOutputSlack, kAlignment) * sizeof(uint32_t));
        pipe_->InitBuffer(outputKeyBuf_, Align(k_ + kOutputSlack, kAlignment) * sizeof(uint32_t));
        pipe_->InitBuffer(tmpIdxBuf_, Align(k_ + kOutputSlack, kAlignment) * sizeof(uint32_t));
        pipe_->InitBuffer(tmpKeyBuf_, Align(k_ + kOutputSlack, kAlignment) * sizeof(uint32_t));
        pipe_->InitBuffer(histogramBuf_, kHistogramSize * sizeof(uint32_t));
        pipe_->InitBuffer(idx0Buf_, kHistogramSize * sizeof(uint32_t));
        pipe_->InitBuffer(idx1Buf_, kHistogramSize * sizeof(uint32_t));
        pipe_->InitBuffer(idx2Buf_, kHistogramSize * sizeof(uint32_t));
        pipe_->InitBuffer(idx3Buf_, kHistogramSize * sizeof(uint32_t));
        pipe_->InitBuffer(nextKBuf_, kNextKSize * sizeof(uint32_t));
        pipe_->InitBuffer(selectedIdBuf_, Align(k_ + kOutputSlack, kAlignment) * sizeof(uint32_t));
        inputKeys_ = inputKeyBuf_.Get<uint32_t>();
        inputIds_ = inputIdBuf_.Get<uint32_t>();
        outputIndices_ = outputIdxBuf_.Get<uint32_t>();
        outputKeys_ = outputKeyBuf_.Get<uint32_t>();
        tmpIndices_ = tmpIdxBuf_.Get<uint32_t>();
        tmpKeys_ = tmpKeyBuf_.Get<uint32_t>();
        histograms_ = histogramBuf_.Get<uint32_t>();
        idx0_ = idx0Buf_.Get<uint32_t>();
        idx1_ = idx1Buf_.Get<uint32_t>();
        idx2_ = idx2Buf_.Get<uint32_t>();
        idx3_ = idx3Buf_.Get<uint32_t>();
        nextK_ = nextKBuf_.Get<uint32_t>();
        selectedIds_ = selectedIdBuf_.Get<uint32_t>();
        Reset();
    }

    __aicore__ inline void Reset()
    {
        count_ = 0;
        resultCount_ = 0;
        finished_ = false;
    }

    __aicore__ inline void Append(float score, uint32_t logicalId)
    {
        if (count_ == capacity_) {
            ReduceToK();
        }
        inputKeys_.SetValue(count_, FloatToOrderedKey(score));
        inputIds_.SetValue(count_, logicalId);
        ++count_;
        finished_ = false;
    }

    __aicore__ inline void Finish()
    {
        if (count_ > k_) {
            ReduceToK();
        } else {
            for (uint32_t i = 0; i < count_; ++i) {
                selectedIds_.SetValue(i, inputIds_.GetValue(i));
                outputKeys_.SetValue(i, inputKeys_.GetValue(i));
            }
            resultCount_ = count_;
        }
        finished_ = true;
    }

    __aicore__ inline uint32_t Count() const
    {
        return finished_ ? resultCount_ : count_;
    }

    __aicore__ inline uint32_t Index(uint32_t i) const
    {
        return selectedIds_.GetValue(i);
    }

    __aicore__ inline float Score(uint32_t i) const
    {
        return OrderedKeyToFloat(outputKeys_.GetValue(i));
    }

    __aicore__ inline float MinScore() const
    {
        if (resultCount_ == 0) {
            return OrderedKeyToFloat(0x007FFFFFU);
        }
        float minimum = OrderedKeyToFloat(outputKeys_.GetValue(0));
        for (uint32_t i = 1; i < resultCount_; ++i) {
            float score = OrderedKeyToFloat(outputKeys_.GetValue(i));
            if (score < minimum) {
                minimum = score;
            }
        }
        return minimum;
    }

private:
    static constexpr uint32_t kAlignment = 256;
    static constexpr uint32_t kOverflowSlack = 4096;
    static constexpr uint32_t kOutputSlack = 64;
    static constexpr uint32_t kHistogramSize = 256;
    static constexpr uint32_t kNextKSize = 64;

    __aicore__ static inline uint32_t Align(uint32_t value, uint32_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    __aicore__ static inline uint32_t FloatToOrderedKey(float score)
    {
        union FloatBits {
            float value;
            uint32_t bits;
        } bits = {score};
        uint32_t value = bits.bits;
        uint32_t exponent = value & 0x7F800000U;
        uint32_t mantissa = value & 0x007FFFFFU;
        if ((value & 0x7FFFFFFFU) == 0 || (exponent == 0x7F800000U && mantissa != 0)) {
            value = (exponent == 0x7F800000U && mantissa != 0) ? 0xFF800000U : 0U;
        }
        return (value & 0x80000000U) != 0 ? ~value : value ^ 0x80000000U;
    }

    __aicore__ static inline float OrderedKeyToFloat(uint32_t key)
    {
        uint32_t bits = (key & 0x80000000U) != 0 ? key ^ 0x80000000U : ~key;
        union FloatBits {
            float value;
            uint32_t bits;
        } result = {0.0f};
        result.bits = bits;
        return result.value;
    }

    __aicore__ inline void ReduceToK()
    {
        uint32_t alignedCount = Align(count_, kAlignment);
        for (uint32_t i = count_; i < alignedCount; ++i) {
            inputKeys_.SetValue(i, 0);
            inputIds_.SetValue(i, 0);
        }
        event_t scalarToVector = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(scalarToVector);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(scalarToVector);
        topkb32::LiTopKVF(outputIndices_, outputKeys_, inputKeys_, tmpIndices_, tmpKeys_, histograms_, idx0_, idx1_,
                          idx2_, idx3_, nextK_, k_, alignedCount);
        event_t vectorToScalar = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::V_S));
        AscendC::SetFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(vectorToScalar);
        for (uint32_t i = 0; i < k_; ++i) {
            uint32_t source = outputIndices_.GetValue(i);
            selectedIds_.SetValue(i, inputIds_.GetValue(source));
        }
        for (uint32_t i = 0; i < k_; ++i) {
            inputIds_.SetValue(i, selectedIds_.GetValue(i));
            inputKeys_.SetValue(i, outputKeys_.GetValue(i));
        }
        count_ = k_;
        resultCount_ = k_;
    }

    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputKeyBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputIdBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputIdxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputKeyBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpIdxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpKeyBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> histogramBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idx0Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idx1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idx2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idx3Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> nextKBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> selectedIdBuf_;
    AscendC::LocalTensor<uint32_t> inputKeys_;
    AscendC::LocalTensor<uint32_t> inputIds_;
    AscendC::LocalTensor<uint32_t> outputIndices_;
    AscendC::LocalTensor<uint32_t> outputKeys_;
    AscendC::LocalTensor<uint32_t> tmpIndices_;
    AscendC::LocalTensor<uint32_t> tmpKeys_;
    AscendC::LocalTensor<uint32_t> histograms_;
    AscendC::LocalTensor<uint32_t> idx0_;
    AscendC::LocalTensor<uint32_t> idx1_;
    AscendC::LocalTensor<uint32_t> idx2_;
    AscendC::LocalTensor<uint32_t> idx3_;
    AscendC::LocalTensor<uint32_t> nextK_;
    AscendC::LocalTensor<uint32_t> selectedIds_;
    uint32_t k_ = 0;
    uint32_t capacity_ = 0;
    uint32_t count_ = 0;
    uint32_t resultCount_ = 0;
    bool finished_ = false;
};

}

#endif
