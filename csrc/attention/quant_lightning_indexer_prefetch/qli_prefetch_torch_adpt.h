/* Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#pragma once

#include <cstdint>

#include <torch/extension.h>

#include "../../aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {
namespace {
constexpr int64_t kHeads = 64;
constexpr int64_t kDim = 128;
constexpr int64_t kTopK = 2048;

void CheckQliPrefetchTensor(const at::Tensor& tensor, at::ScalarType dtype, const char* name,
                            const at::Device& device, bool contiguous)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device() == device, name, " must be on the same NPU device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
    if (contiguous) TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

inline void npu_qli_prefetch_stage(const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
                                   const at::Tensor& qscale, const at::Tensor& kscale, const at::Tensor& seqLens,
                                   const at::Tensor& blockTable, const at::Tensor& thresholds,
                                   const at::Tensor& stablePrefixLens, const at::Tensor& activeRows,
                                   const at::Tensor& scores, const at::Tensor& candidates,
                                   const at::Tensor& candidateCounts, const at::Tensor& exactIndices,
                                   const at::Tensor& kthScores, int64_t stage, int64_t budgetDivisor)
{
    const auto device = query.device();
    TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "query must be an NPU tensor");
    CheckQliPrefetchTensor(query, at::kFloat8_e4m3fn, "query", device, true);
    CheckQliPrefetchTensor(key, at::kFloat8_e4m3fn, "key", device, false);
    CheckQliPrefetchTensor(weights, at::kBFloat16, "weights", device, true);
    CheckQliPrefetchTensor(qscale, at::kFloat, "qscale", device, true);
    CheckQliPrefetchTensor(kscale, at::kFloat, "kscale", device, false);
    CheckQliPrefetchTensor(seqLens, at::kInt, "seq_lens", device, true);
    CheckQliPrefetchTensor(blockTable, at::kInt, "block_table", device, true);
    CheckQliPrefetchTensor(thresholds, at::kFloat, "thresholds", device, true);
    CheckQliPrefetchTensor(stablePrefixLens, at::kInt, "stable_prefix_lens", device, true);
    CheckQliPrefetchTensor(activeRows, at::kInt, "active_rows", device, true);
    CheckQliPrefetchTensor(scores, at::kShort, "scores", device, true);
    CheckQliPrefetchTensor(candidates, at::kInt, "candidates", device, true);
    CheckQliPrefetchTensor(candidateCounts, at::kInt, "candidate_counts", device, true);
    CheckQliPrefetchTensor(exactIndices, at::kInt, "exact_indices", device, true);
    CheckQliPrefetchTensor(kthScores, at::kFloat, "kth_scores", device, true);
    TORCH_CHECK(query.dim() == 3 && query.size(1) == kHeads && query.size(2) == kDim,
                "query must have shape [R,64,128]");
    TORCH_CHECK(key.dim() == 4 && key.size(0) > 0 && key.size(2) == 1 && key.size(3) == kDim &&
                    key.stride(3) == 1 && key.stride(2) >= kDim && key.stride(1) == kDim,
                "key must have shape [P,BS,1,128]");
    const int64_t rows = query.size(0);
    TORCH_CHECK(rows > 0 && weights.dim() == 2 && weights.size(0) == rows && weights.size(1) == kHeads &&
                    qscale.dim() == 2 && qscale.size(0) == rows && qscale.size(1) == kHeads && seqLens.dim() == 1 &&
                    seqLens.size(0) == rows && blockTable.dim() == 2 && blockTable.size(0) == rows &&
                    thresholds.dim() == 1 && thresholds.size(0) == rows && stablePrefixLens.dim() == 1 &&
                    stablePrefixLens.size(0) == rows && activeRows.dim() == 1 && activeRows.size(0) == 1 &&
                    scores.dim() == 2 && scores.size(0) == rows && candidates.dim() == 2 && candidates.size(0) == rows &&
                    candidates.size(1) == kTopK && candidateCounts.dim() == 1 && candidateCounts.size(0) == rows &&
                    exactIndices.dim() == 3 && exactIndices.size(0) == rows && exactIndices.size(1) == 1 &&
                    exactIndices.size(2) == kTopK && kthScores.dim() == 1 && kthScores.size(0) == rows,
                "QLI prefetch shape contract violated");
    TORCH_CHECK(key.size(1) >= 16 && key.size(1) <= 1024 && key.size(1) % 16 == 0,
                "key block size must be a multiple of 16 in [16,1024]");
    TORCH_CHECK(kscale.dim() == 2 && kscale.size(0) == key.size(0) && kscale.size(1) == key.size(1) &&
                    kscale.stride(1) == 1,
                "kscale must have shape [P,BS] and unit inner stride");
    TORCH_CHECK(stage == 0 || stage == 1, "stage must be 0 or 1");
    TORCH_CHECK(budgetDivisor > 0, "budget_divisor must be positive");
    for (const auto& output : {scores, candidates, candidateCounts, exactIndices, kthScores}) {
        for (const auto& input : {query, key, weights, qscale, kscale, seqLens, blockTable, thresholds,
                                  stablePrefixLens, activeRows}) {
            TORCH_CHECK(!output.is_alias_of(input), "mutable QLI output aliases a read-only input");
        }
    }
    TORCH_CHECK(!scores.is_alias_of(candidates) && !scores.is_alias_of(candidateCounts) &&
                    !scores.is_alias_of(exactIndices) && !scores.is_alias_of(kthScores) &&
                    !candidates.is_alias_of(candidateCounts) && !candidates.is_alias_of(exactIndices) &&
                    !candidates.is_alias_of(kthScores) && !candidateCounts.is_alias_of(exactIndices) &&
                    !candidateCounts.is_alias_of(kthScores) && !exactIndices.is_alias_of(kthScores),
                "mutable QLI outputs must not alias each other");
    const int64_t keyStride = key.stride(0);
    const int64_t scaleStride = kscale.stride(0);
    TORCH_CHECK(keyStride >= key.size(1) * kDim && scaleStride >= kscale.size(1),
                "key and kscale stride(0) are smaller than one page");
    EXEC_NPU_CMD(aclnnQuantLightningIndexerPrefetch, query, key, weights, qscale, kscale, seqLens, blockTable,
                 thresholds, stablePrefixLens, activeRows, scores, candidates, candidateCounts, exactIndices,
                 kthScores, stage, budgetDivisor, keyStride, scaleStride);
}

inline void npu_qli_prefetch_stage_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                         const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                         const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                         const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t)
{
}
}  // namespace vllm_ascend
