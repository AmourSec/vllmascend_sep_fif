/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef QUANT_LIGHTNING_INDEXER_SPARSE_KV_TORCH_ADPT_H
#define QUANT_LIGHTNING_INDEXER_SPARSE_KV_TORCH_ADPT_H

#include <cstdint>
#include <limits>

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"

#include "../../aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend::qli_sparse_kv {
namespace detail {

constexpr int64_t kQueryHeads = 32;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kSparseCount = 2048;
constexpr int64_t kHashCapacity = 4096;

inline void CheckTensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype, int64_t rank,
                        const at::Device& device) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device() == device, name, " must be on ", device);
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an unsupported dtype");
  TORCH_CHECK(tensor.dim() == rank, name, " must have rank ", rank);
}

inline void CheckMutableTensor(const at::Tensor& tensor, const char* name, at::ScalarType dtype, int64_t rank,
                               const at::Device& device) {
  CheckTensor(tensor, name, dtype, rank, device);
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous because it is mutated in place");
}

inline at::Tensor ValidateAndNormalizeKeyScale(const at::Tensor& keyDequantScale, const at::Tensor& key,
                                               const at::Device& device) {
  TORCH_CHECK(keyDequantScale.defined(), "key_dequant_scale must be defined");
  TORCH_CHECK(keyDequantScale.device() == device, "key_dequant_scale must be on ", device);
  TORCH_CHECK(keyDequantScale.scalar_type() == at::kFloat, "key_dequant_scale must be float32");
  TORCH_CHECK(keyDequantScale.dim() == 2 || keyDequantScale.dim() == 3,
              "key_dequant_scale must be [blocks,block_size] or [blocks,block_size,1]");
  TORCH_CHECK(keyDequantScale.size(0) == key.size(0) && keyDequantScale.size(1) == key.size(1),
              "key_dequant_scale leading dimensions must match key");
  TORCH_CHECK(keyDequantScale.dim() == 2 || keyDequantScale.size(2) == 1, "key_dequant_scale last dimension must be 1");
  TORCH_CHECK(keyDequantScale.stride(1) == 1 && (keyDequantScale.dim() == 2 || keyDequantScale.stride(2) == 1),
              "key_dequant_scale must be contiguous on all axes except axis 0");
  return keyDequantScale.dim() == 2 ? keyDequantScale.unsqueeze(-1) : keyDequantScale;
}

inline void ValidateInputs(const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights,
                           const at::Tensor& queryDequantScale, const at::Tensor& actualSeqLengthsQuery,
                           const at::Tensor& actualSeqLengthsKey, const at::Tensor& indexerBlockTable,
                           const at::Tensor& reqIds, const at::Tensor& stablePrefixLens,
                           const at::Tensor& visibleSeqLens, const at::Tensor& tokenToReq,
                           const at::Tensor& kvBlockTable, const at::Tensor& activeRows, const at::Tensor& lastReqIds,
                           const at::Tensor& slotToToken, const at::Tensor& lruSlots, const at::Tensor& currentSlots,
                           const at::Tensor& missCount, const at::Tensor& missTokens, const at::Tensor& missSlots,
                           const at::Tensor& compactWorkspace, int64_t capacity, int64_t maxToken, int64_t blockSize,
                           int64_t hostNumBlocks) {
  const at::Device device = query.device();
  TORCH_CHECK(query.is_privateuseone(), "query must be an NPU tensor");
  CheckTensor(query, "query", at::kFloat8_e4m3fn, 3, device);
  TORCH_CHECK(query.size(0) > 0 && query.size(1) == kQueryHeads && query.size(2) == kHeadDim,
              "query must have shape [T,32,128] with T > 0");
  const int64_t rows = query.size(0);

  CheckTensor(key, "key", at::kFloat8_e4m3fn, 4, device);
  TORCH_CHECK(key.size(0) > 0 && key.size(1) == blockSize && key.size(2) == 1 && key.size(3) == kHeadDim,
              "key must have shape [blocks,block_size,1,128]");
  TORCH_CHECK(key.stride(3) == 1 && key.stride(2) == kHeadDim && key.stride(1) == kHeadDim,
              "key must be contiguous on all axes except axis 0");

  CheckTensor(weights, "weights", at::kBFloat16, 2, device);
  TORCH_CHECK(weights.size(0) == rows && weights.size(1) == kQueryHeads, "weights must have shape [T,32]");
  CheckTensor(queryDequantScale, "query_dequant_scale", at::kFloat, 2, device);
  TORCH_CHECK(queryDequantScale.sizes() == weights.sizes(), "query_dequant_scale must have shape [T,32]");

  CheckTensor(actualSeqLengthsQuery, "actual_seq_lengths_query", at::kInt, 1, device);
  CheckTensor(actualSeqLengthsKey, "actual_seq_lengths_key", at::kInt, 1, device);
  TORCH_CHECK(actualSeqLengthsQuery.numel() > 0 && actualSeqLengthsQuery.numel() == actualSeqLengthsKey.numel(),
              "actual sequence length tensors must have the same nonzero length");
  CheckTensor(indexerBlockTable, "indexer_block_table", at::kInt, 2, device);
  TORCH_CHECK(indexerBlockTable.size(0) == actualSeqLengthsQuery.numel(),
              "indexer_block_table request count must match actual sequence lengths");

  CheckTensor(reqIds, "req_ids", at::kLong, 1, device);
  CheckTensor(stablePrefixLens, "stable_prefix_lens", at::kInt, 1, device);
  CheckTensor(visibleSeqLens, "visible_seq_lens", at::kInt, 1, device);
  CheckTensor(tokenToReq, "token_to_req", at::kInt, 1, device);
  CheckTensor(kvBlockTable, "kv_block_table", at::kInt, 2, device);
  CheckTensor(activeRows, "active_rows", at::kInt, 1, device);
  TORCH_CHECK(reqIds.numel() == kvBlockTable.size(0), "req_ids length must match kv_block_table request count");
  TORCH_CHECK(reqIds.numel() == actualSeqLengthsQuery.numel(), "planner and indexer request counts must match");
  TORCH_CHECK(stablePrefixLens.numel() >= rows && visibleSeqLens.numel() >= rows && tokenToReq.numel() >= rows &&
                  activeRows.numel() >= 1,
              "planner row inputs must contain at least T entries");

  TORCH_CHECK(capacity >= kSparseCount, "capacity must be at least 2048");
  TORCH_CHECK(maxToken > 0 && blockSize > 0 && hostNumBlocks > 0,
              "max_token, block_size, and host_num_blocks must be positive");
  TORCH_CHECK(capacity <= std::numeric_limits<int32_t>::max() && maxToken <= std::numeric_limits<int32_t>::max(),
              "capacity and max_token must fit int32");

  CheckMutableTensor(lastReqIds, "last_req_ids", at::kLong, 1, device);
  CheckMutableTensor(slotToToken, "slot_to_token", at::kInt, 2, device);
  CheckMutableTensor(lruSlots, "lru_slots", at::kInt, 2, device);
  CheckMutableTensor(currentSlots, "current_slots", at::kInt, 2, device);
  CheckMutableTensor(missCount, "miss_count", at::kInt, 1, device);
  CheckMutableTensor(missTokens, "miss_tokens", at::kInt, 2, device);
  CheckMutableTensor(missSlots, "miss_slots", at::kInt, 2, device);
  CheckMutableTensor(compactWorkspace, "compact_workspace", at::kInt, 2, device);
  TORCH_CHECK(lastReqIds.size(0) >= rows && slotToToken.size(0) >= rows && lruSlots.size(0) >= rows &&
                  currentSlots.size(0) >= rows && missCount.size(0) >= rows && missTokens.size(0) >= rows &&
                  missSlots.size(0) >= rows && compactWorkspace.size(0) >= rows,
              "planner state tensors must contain at least T rows");
  TORCH_CHECK(slotToToken.size(1) == capacity && lruSlots.size(1) == capacity,
              "slot_to_token and lru_slots width must equal capacity");
  TORCH_CHECK(
      currentSlots.size(1) == kSparseCount && missTokens.size(1) == kSparseCount && missSlots.size(1) == kSparseCount,
      "current_slots, miss_tokens, and miss_slots width must equal 2048");
  const int64_t workspaceWidth = 3 * kHashCapacity + 2 * capacity + kSparseCount;
  TORCH_CHECK(compactWorkspace.size(1) >= workspaceWidth, "compact_workspace is too small for capacity");
}

}
}

namespace vllm_ascend {

inline at::Tensor npu_quant_lightning_indexer_sparse_kv(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& weights, const at::Tensor& queryDequantScale,
    const at::Tensor& keyDequantScale, const at::Tensor& actualSeqLengthsQuery, const at::Tensor& actualSeqLengthsKey,
    const at::Tensor& indexerBlockTable, const at::Tensor& reqIds, const at::Tensor& stablePrefixLens,
    const at::Tensor& visibleSeqLens, const at::Tensor& tokenToReq, const at::Tensor& kvBlockTable,
    const at::Tensor& activeRows, const at::Tensor& lastReqIds, const at::Tensor& slotToToken,
    const at::Tensor& lruSlots, const at::Tensor& currentSlots, const at::Tensor& missCount,
    const at::Tensor& missTokens, const at::Tensor& missSlots, const at::Tensor& compactWorkspace, int64_t capacity,
    int64_t maxToken, int64_t blockSize, int64_t hostNumBlocks) {
  qli_sparse_kv::detail::ValidateInputs(
      query, key, weights, queryDequantScale, actualSeqLengthsQuery, actualSeqLengthsKey, indexerBlockTable, reqIds,
      stablePrefixLens, visibleSeqLens, tokenToReq, kvBlockTable, activeRows, lastReqIds, slotToToken, lruSlots,
      currentSlots, missCount, missTokens, missSlots, compactWorkspace, capacity, maxToken, blockSize, hostNumBlocks);
  at::Tensor canonicalKeyScale =
      qli_sparse_kv::detail::ValidateAndNormalizeKeyScale(keyDequantScale, key, query.device());

  const c10_npu::OptionalNPUGuard deviceGuard(query.device());
  at::Tensor sparseIndices =
      at::empty({query.size(0), 1, qli_sparse_kv::detail::kSparseCount}, query.options().dtype(at::kInt));
  int64_t queryQuantMode = 0;
  int64_t keyQuantMode = 0;
  char layoutQuery[] = "TND";
  char layoutKey[] = "PA_BSND";
  int64_t sparseCount = qli_sparse_kv::detail::kSparseCount;
  int64_t sparseMode = 3;
  int64_t preTokens = std::numeric_limits<int64_t>::max();
  int64_t nextTokens = std::numeric_limits<int64_t>::max();
  int64_t keyStride0 = key.stride(0);
  int64_t keyScaleStride0 = canonicalKeyScale.stride(0);

  EXEC_NPU_CMD(aclnnQuantLightningIndexerSparseKv, query, key, weights, queryDequantScale, canonicalKeyScale,
               actualSeqLengthsQuery, actualSeqLengthsKey, indexerBlockTable, reqIds, stablePrefixLens, visibleSeqLens,
               tokenToReq, kvBlockTable, activeRows, lastReqIds, slotToToken, lruSlots, currentSlots, missCount,
               missTokens, missSlots, compactWorkspace, queryQuantMode, keyQuantMode, layoutQuery, layoutKey,
               sparseCount, sparseMode, preTokens, nextTokens, keyStride0, keyScaleStride0, capacity, maxToken,
               blockSize, hostNumBlocks, sparseIndices);
  return sparseIndices;
}

}

#endif
