/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef VLLM_ASCEND_SPARSE_KV_ECHO_TORCH_ADPT_H
#define VLLM_ASCEND_SPARSE_KV_ECHO_TORCH_ADPT_H

#include <torch/extension.h>

#include "../../aclnn_torch_adapter/op_api_common.h"

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

inline void npu_sparse_kv_echo_plan_transfer(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                                             const at::Tensor& stablePrefixLens, const at::Tensor& visibleSeqLens,
                                             const at::Tensor& tokenToReq, const at::Tensor& blockTable,
                                             const at::Tensor& activeRows, const at::Tensor& lastReqIds,
                                             const at::Tensor& slotToToken, const at::Tensor& slotPriorities,
                                             const at::Tensor& currentSlots, const at::Tensor& missCount,
                                             const at::Tensor& missTokens, const at::Tensor& missSlots,
                                             const at::Tensor& compactWorkspace, const at::Tensor& hostCacheBases,
                                             const at::Tensor& residentK, const at::Tensor& residentV, int64_t topk,
                                             int64_t capacity, int64_t maxToken, int64_t blockSize,
                                             int64_t hostNumBlocks, int64_t tokenSizeBytesK, int64_t tokenSizeBytesV) {
  EXEC_NPU_CMD(aclnnSparseKvEchoPlan, reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, blockTable,
               activeRows, lastReqIds, slotToToken, slotPriorities, currentSlots, missCount, missTokens, missSlots,
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
