/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef VLLM_ASCEND_SPARSE_KV_LRU_TORCH_ADPT_H
#define VLLM_ASCEND_SPARSE_KV_LRU_TORCH_ADPT_H

#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"

#include "../../aclnn_torch_adapter/op_api_common.h"
#include "sparse_kv_plan_validation.h"

namespace vllm_ascend {
namespace {
inline void RunSparseKvTransfer(const at::Tensor& missCount, const at::Tensor& missTokens, const at::Tensor& missSlots,
                                const at::Tensor& tokenToReq, const at::Tensor& blockTable,
                                const at::Tensor& hostCacheBases, const at::Tensor& activeRows,
                                const at::Tensor& residentK, const at::Tensor& residentV, int64_t topk,
                                int64_t capacity, int64_t blockSize, int64_t hostNumBlocks, int64_t tokenSizeBytesK,
                                int64_t tokenSizeBytesV) {
  EXEC_NPU_CMD(aclnnSparseKvTransfer, missCount, missTokens, missSlots, tokenToReq, blockTable, hostCacheBases,
               activeRows, residentK, residentV, topk, capacity, blockSize, hostNumBlocks, tokenSizeBytesK,
               tokenSizeBytesV);
}
}  // namespace

inline void npu_sparse_kv_plan(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                               const at::Tensor& stablePrefixLens, const at::Tensor& visibleSeqLens,
                               const at::Tensor& tokenToReq, const at::Tensor& blockTable, const at::Tensor& activeRows,
                               const at::Tensor& lastReqIds, const at::Tensor& slotToToken, const at::Tensor& lruSlots,
                               const at::Tensor& currentSlots, const at::Tensor& missCount,
                               const at::Tensor& missTokens, const at::Tensor& missSlots,
                               const at::Tensor& compactWorkspace, int64_t topk, int64_t capacity, int64_t maxToken,
                               int64_t blockSize, int64_t hostNumBlocks) {
  sparse_kv_plan_detail::ValidatePlanInputs(reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq,
                                            blockTable, activeRows, lastReqIds, slotToToken, lruSlots, currentSlots,
                                            missCount, missTokens, missSlots, compactWorkspace, topk, capacity,
                                            maxToken, blockSize, hostNumBlocks);
  const c10_npu::OptionalNPUGuard deviceGuard(reqIds.device());
  EXEC_NPU_CMD(aclnnSparseKvPlan, reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, blockTable,
               activeRows, lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots,
               compactWorkspace, topk, capacity, maxToken, blockSize, hostNumBlocks);
}

inline void npu_sparse_kv_plan_transfer(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                                        const at::Tensor& stablePrefixLens, const at::Tensor& visibleSeqLens,
                                        const at::Tensor& tokenToReq, const at::Tensor& blockTable,
                                        const at::Tensor& activeRows, const at::Tensor& lastReqIds,
                                        const at::Tensor& slotToToken, const at::Tensor& lruSlots,
                                        const at::Tensor& currentSlots, const at::Tensor& missCount,
                                        const at::Tensor& missTokens, const at::Tensor& missSlots,
                                        const at::Tensor& compactWorkspace, const at::Tensor& hostCacheBases,
                                        const at::Tensor& residentK, const at::Tensor& residentV, int64_t topk,
                                        int64_t capacity, int64_t maxToken, int64_t blockSize, int64_t hostNumBlocks,
                                        int64_t tokenSizeBytesK, int64_t tokenSizeBytesV) {
  npu_sparse_kv_plan(reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, blockTable, activeRows,
                     lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots,
                     compactWorkspace, topk, capacity, maxToken, blockSize, hostNumBlocks);
  RunSparseKvTransfer(missCount, missTokens, missSlots, tokenToReq, blockTable, hostCacheBases, activeRows, residentK,
                      residentV, topk, capacity, blockSize, hostNumBlocks, tokenSizeBytesK, tokenSizeBytesV);
}

inline void npu_sparse_kv_transfer(const at::Tensor& missCount, const at::Tensor& missTokens,
                                   const at::Tensor& missSlots, const at::Tensor& tokenToReq,
                                   const at::Tensor& blockTable, const at::Tensor& hostCacheBases,
                                   const at::Tensor& activeRows, const at::Tensor& residentK,
                                   const at::Tensor& residentV, int64_t topk, int64_t capacity, int64_t blockSize,
                                   int64_t hostNumBlocks, int64_t tokenSizeBytesK, int64_t tokenSizeBytesV) {
  RunSparseKvTransfer(missCount, missTokens, missSlots, tokenToReq, blockTable, hostCacheBases, activeRows, residentK,
                      residentV, topk, capacity, blockSize, hostNumBlocks, tokenSizeBytesK, tokenSizeBytesV);
}
}  // namespace vllm_ascend

#endif
