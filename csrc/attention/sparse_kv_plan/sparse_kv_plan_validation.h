/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef VLLM_ASCEND_SPARSE_KV_PLAN_VALIDATION_H
#define VLLM_ASCEND_SPARSE_KV_PLAN_VALIDATION_H

#include <cstdint>
#include <limits>

#include <ATen/ATen.h>

#include "op_kernel/sparse_kv_plan_config.h"

namespace vllm_ascend::sparse_kv_plan_detail {

inline void CheckPlanTensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype, int64_t rank,
                            const at::Device& device) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
  TORCH_CHECK(tensor.dim() == rank, name, " must have rank ", rank);
}

inline void CheckMutablePlanTensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype, int64_t rank,
                                   const at::Device& device) {
  CheckPlanTensor(tensor, name, dtype, rank, device);
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous because it is mutated in place");
}

inline void CheckPlanScalarTensor(const at::Tensor& tensor, const char* name, const at::Device& device) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " has an unsupported dtype");
  TORCH_CHECK(tensor.dim() == 0 || tensor.dim() == 1, name, " must be a scalar or rank 1 tensor");
  TORCH_CHECK(tensor.numel() >= 1, name, " must contain a value");
}

inline int64_t PlanHashCapacity(int64_t topk) {
  int64_t hashCapacity = 32;
  while (hashCapacity < 2 * topk) {
    hashCapacity <<= 1;
  }
  return hashCapacity;
}

inline void CheckPlanWorkspace(const at::Tensor& compactWorkspace, int64_t rows, int64_t rowElements,
                               const at::Device& device) {
  TORCH_CHECK(compactWorkspace.defined(), "compact_workspace must be defined");
  TORCH_CHECK(compactWorkspace.device() == device, "compact_workspace must be on ", device);
  TORCH_CHECK(compactWorkspace.scalar_type() == at::kInt, "compact_workspace has an unsupported dtype");
  TORCH_CHECK(compactWorkspace.dim() == 1 || compactWorkspace.dim() == 2, "compact_workspace must have rank 1 or 2");
  TORCH_CHECK(compactWorkspace.is_contiguous(), "compact_workspace must be contiguous because it is mutated in place");

  const bool usesUb = rowElements <= sparse_kv_plan_cfg::PLAN_ROW_UB_LIMIT_ELEMENTS;
  const int64_t fullElements = rows * rowElements;
  if (compactWorkspace.dim() == 1) {
    const bool isPlaceholder = compactWorkspace.numel() == 1;
    TORCH_CHECK((usesUb && isPlaceholder) || compactWorkspace.numel() >= fullElements,
                "compact_workspace is too small for planner storage");
    return;
  }

  const bool isPlaceholder = compactWorkspace.size(0) == 1 && compactWorkspace.size(1) == 1;
  const bool hasFullRows = compactWorkspace.size(0) >= rows;
  if (usesUb) {
    TORCH_CHECK(isPlaceholder || (hasFullRows && compactWorkspace.size(1) >= rowElements),
                "compact_workspace must be [1,1] or provide full planner rows");
  } else {
    TORCH_CHECK(hasFullRows && compactWorkspace.size(1) == rowElements,
                "compact_workspace must have enough rows and the exact GM row width");
  }
}

inline void ValidatePlanStateInputs(const at::Tensor& reqIds, const at::Tensor& stablePrefixLens,
                                    const at::Tensor& visibleSeqLens, const at::Tensor& tokenToReq,
                                    const at::Tensor& blockTable, const at::Tensor& activeRows,
                                    const at::Tensor& lastReqIds, const at::Tensor& slotToToken,
                                    const at::Tensor& lruSlots, const at::Tensor& currentSlots,
                                    const at::Tensor& missCount, const at::Tensor& missTokens,
                                    const at::Tensor& missSlots, const at::Tensor& compactWorkspace, int64_t rows,
                                    int64_t topk, int64_t capacity, int64_t maxToken, int64_t blockSize,
                                    int64_t hostNumBlocks) {
  TORCH_CHECK(reqIds.defined(), "req_ids must be defined");
  TORCH_CHECK(reqIds.is_privateuseone(), "req_ids must be an NPU tensor");
  const at::Device device = reqIds.device();
  TORCH_CHECK(rows > 0, "planner must contain at least one row");
  TORCH_CHECK(topk > 0 && capacity > 0, "topk and capacity must be positive");
  TORCH_CHECK(maxToken > 0 && blockSize > 0 && hostNumBlocks > 0,
              "max_token, block_size, and host_num_blocks must be positive");
  TORCH_CHECK(topk <= std::numeric_limits<int32_t>::max() / 2 && capacity <= std::numeric_limits<int32_t>::max() &&
                  maxToken <= std::numeric_limits<int32_t>::max() && blockSize <= std::numeric_limits<int32_t>::max(),
              "planner dimensions exceed int32 kernel limits");

  CheckPlanTensor(reqIds, "req_ids", at::kLong, 1, device);
  CheckPlanTensor(stablePrefixLens, "stable_prefix_lens", at::kInt, 1, device);
  CheckPlanTensor(visibleSeqLens, "visible_seq_lens", at::kInt, 1, device);
  CheckPlanTensor(tokenToReq, "token_to_req", at::kInt, 1, device);
  CheckPlanTensor(blockTable, "block_table", at::kInt, 2, device);
  CheckPlanScalarTensor(activeRows, "active_rows", device);

  TORCH_CHECK(reqIds.numel() > 0 && blockTable.size(0) == reqIds.numel() && blockTable.size(1) > 0,
              "block_table must have one nonempty row per request");
  TORCH_CHECK(stablePrefixLens.numel() >= rows && visibleSeqLens.numel() >= rows && tokenToReq.numel() >= rows,
              "planner row inputs must contain at least T entries");

  CheckMutablePlanTensor(lastReqIds, "last_req_ids", at::kLong, 1, device);
  CheckMutablePlanTensor(slotToToken, "slot_to_token", at::kInt, 2, device);
  CheckMutablePlanTensor(lruSlots, "lru_slots", at::kInt, 2, device);
  CheckMutablePlanTensor(currentSlots, "current_slots", at::kInt, 2, device);
  CheckMutablePlanTensor(missCount, "miss_count", at::kInt, 1, device);
  CheckMutablePlanTensor(missTokens, "miss_tokens", at::kInt, 2, device);
  CheckMutablePlanTensor(missSlots, "miss_slots", at::kInt, 2, device);
  TORCH_CHECK(lastReqIds.size(0) >= rows && slotToToken.size(0) >= rows && lruSlots.size(0) >= rows &&
                  currentSlots.size(0) >= rows && missCount.size(0) >= rows && missTokens.size(0) >= rows &&
                  missSlots.size(0) >= rows,
              "planner state tensors must contain at least T rows");
  TORCH_CHECK(slotToToken.size(1) == capacity && lruSlots.size(1) == capacity,
              "slot_to_token and lru_slots width must equal capacity");
  TORCH_CHECK(currentSlots.size(1) == topk && missTokens.size(1) == topk && missSlots.size(1) == topk,
              "current_slots, miss_tokens, and miss_slots width must equal topk");

  const int64_t rowElements = 3 * PlanHashCapacity(topk) + 2 * capacity + topk;
  TORCH_CHECK(rows <= std::numeric_limits<int64_t>::max() / rowElements,
              "compact_workspace required size exceeds int64 limits");
  CheckPlanWorkspace(compactWorkspace, rows, rowElements, device);
}

inline void ValidatePlanInputs(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                               const at::Tensor& stablePrefixLens, const at::Tensor& visibleSeqLens,
                               const at::Tensor& tokenToReq, const at::Tensor& blockTable, const at::Tensor& activeRows,
                               const at::Tensor& lastReqIds, const at::Tensor& slotToToken, const at::Tensor& lruSlots,
                               const at::Tensor& currentSlots, const at::Tensor& missCount,
                               const at::Tensor& missTokens, const at::Tensor& missSlots,
                               const at::Tensor& compactWorkspace, int64_t topk, int64_t capacity, int64_t maxToken,
                               int64_t blockSize, int64_t hostNumBlocks) {
  TORCH_CHECK(reqIds.defined(), "req_ids must be defined");
  CheckPlanTensor(topkIndices, "topk_indices", at::kInt, 2, reqIds.device());
  TORCH_CHECK(topkIndices.size(0) > 0 && topkIndices.size(1) == topk,
              "topk_indices must have shape [T,topk] with T > 0");
  ValidatePlanStateInputs(reqIds, stablePrefixLens, visibleSeqLens, tokenToReq, blockTable, activeRows, lastReqIds,
                          slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots, compactWorkspace,
                          topkIndices.size(0), topk, capacity, maxToken, blockSize, hostNumBlocks);
}

}  // namespace vllm_ascend::sparse_kv_plan_detail

#endif
