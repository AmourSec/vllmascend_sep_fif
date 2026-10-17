/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full license text.
 */

#ifndef LIGHTNING_INDEXER_QUANT_SCORE_ARCH35_H
#define LIGHTNING_INDEXER_QUANT_SCORE_ARCH35_H

#include "kernel_operator.h"
#if __has_include("../../quant_lightning_indexer_v2/arch35/vf/quant_lightning_indexer_v2_vector1.h")
#include "../../quant_lightning_indexer_v2/arch35/vf/quant_lightning_indexer_v2_vector1.h"
#else
#include "../../../quant_lightning_indexer_v2/op_kernel/arch35/vf/quant_lightning_indexer_v2_vector1.h"
#endif

namespace LIQ950 {

namespace detail {

namespace Reg = AscendC::Reg;

constexpr uint16_t kQueryHeads = 64;
constexpr uint32_t kS2Tile = 128;
constexpr uint32_t kMBaseMax = 128;
constexpr uint32_t kQkHeadStride = 128;
constexpr uint32_t kQkSecondBankOffset = kMBaseMax * kS2Tile / 2;
constexpr float kInt32Scale = 1.0f / 1024.0f;

__simd_vf__ void ComputeScoresVF(__ubuf__ float *out, __ubuf__ int32_t *qk, __ubuf__ half *weights,
                                 __ubuf__ half *qScale, __ubuf__ half *kScale)
{
    using CastTrait = Reg::CastTrait;

    constexpr CastTrait kInt32ToFloat = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
    constexpr CastTrait kFloatToHalfRint = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    constexpr CastTrait kHalfToFloat = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                        Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    Reg::MaskReg maskF32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskF16 = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
    Reg::RegTensor<float> sum[2];
    Reg::RegTensor<float> qkFloat[2];
    Reg::RegTensor<float> weightedQk[2];
    Reg::RegTensor<int32_t> qkInt[2];
    Reg::RegTensor<half> qkHalf[2];
    Reg::RegTensor<float> weight;
    Reg::RegTensor<half> weightHalf;
    Reg::RegTensor<float> qScaleFloat;
    Reg::RegTensor<half> qScaleHalf;
    Reg::RegTensor<float> weightScaled;
    Reg::RegTensor<half> weightScaledHalf;
    Reg::RegTensor<float> weightBroadcast;
    Reg::RegTensor<float> keyScale[2];
    Reg::RegTensor<half> keyScaleHalf[2];

    Reg::Duplicate(sum[0], 0.0f, maskF32);
    Reg::Duplicate(sum[1], 0.0f, maskF32);

    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(weightHalf, weights);
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(qScaleHalf, qScale);
    Reg::Cast<float, half, kHalfToFloat>(weight, weightHalf, maskF16);
    Reg::Cast<float, half, kHalfToFloat>(qScaleFloat, qScaleHalf, maskF16);
    Reg::Mul(weightScaled, weight, qScaleFloat, maskF32);
    Reg::Cast<half, float, kFloatToHalfRint>(weightScaledHalf, weightScaled, maskF32);
    Reg::Cast<float, half, kHalfToFloat>(weightScaled, weightScaledHalf, maskF16);

    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(keyScaleHalf[0], kScale);
    Reg::LoadAlign<half, Reg::LoadDist::DIST_UNPACK_B16>(keyScaleHalf[1], kScale + kQueryHeads);
    Reg::Cast<float, half, kHalfToFloat>(keyScale[0], keyScaleHalf[0], maskF16);
    Reg::Cast<float, half, kHalfToFloat>(keyScale[1], keyScaleHalf[1], maskF16);

    for (uint16_t head = 0; head < kQueryHeads; ++head) {
        Reg::LoadAlign<int32_t>(qkInt[0], qk + head * kQkHeadStride);
        Reg::LoadAlign<int32_t>(qkInt[1], qk + kQkSecondBankOffset + head * kQkHeadStride);
        Reg::Cast<float, int32_t, kInt32ToFloat>(qkFloat[0], qkInt[0], maskF32);
        Reg::Cast<float, int32_t, kInt32ToFloat>(qkFloat[1], qkInt[1], maskF32);
        Reg::Muls(qkFloat[0], qkFloat[0], kInt32Scale, maskF32);
        Reg::Muls(qkFloat[1], qkFloat[1], kInt32Scale, maskF32);
        Reg::Cast<half, float, kFloatToHalfRint>(qkHalf[0], qkFloat[0], maskF32);
        Reg::Cast<half, float, kFloatToHalfRint>(qkHalf[1], qkFloat[1], maskF32);
        Reg::Cast<float, half, kHalfToFloat>(qkFloat[0], qkHalf[0], maskF16);
        Reg::Cast<float, half, kHalfToFloat>(qkFloat[1], qkHalf[1], maskF16);
        Reg::Relu(qkFloat[0], qkFloat[0], maskF32);
        Reg::Relu(qkFloat[1], qkFloat[1], maskF32);
        liV2Vector1::BroadcastLane(weightBroadcast, weightScaled, head);
        Reg::Mul(weightedQk[0], qkFloat[0], weightBroadcast, maskF32);
        Reg::Mul(weightedQk[1], qkFloat[1], weightBroadcast, maskF32);
        Reg::Add(sum[0], sum[0], weightedQk[0], maskF32);
        Reg::Add(sum[1], sum[1], weightedQk[1], maskF32);
    }

    Reg::Mul(sum[0], sum[0], keyScale[0], maskF32);
    Reg::Mul(sum[1], sum[1], keyScale[1], maskF32);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(out, sum[0], maskF32);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(out + kS2Tile / 2, sum[1], maskF32);
}

}

__aicore__ inline void ComputeScores(const AscendC::LocalTensor<float> &out,
                                     const AscendC::LocalTensor<int32_t> &qk,
                                     const AscendC::LocalTensor<half> &weights,
                                     const AscendC::LocalTensor<half> &qScale,
                                     const AscendC::LocalTensor<half> &kScale)
{
    detail::ComputeScoresVF((__ubuf__ float *)out.GetPhyAddr(), (__ubuf__ int32_t *)qk.GetPhyAddr(),
                            (__ubuf__ half *)weights.GetPhyAddr(), (__ubuf__ half *)qScale.GetPhyAddr(),
                            (__ubuf__ half *)kScale.GetPhyAddr());
}

}

#endif
