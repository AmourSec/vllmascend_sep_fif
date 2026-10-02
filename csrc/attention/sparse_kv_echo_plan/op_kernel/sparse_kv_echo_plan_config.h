/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef VLLM_ASCEND_SPARSE_KV_ECHO_PLAN_CONFIG_H
#define VLLM_ASCEND_SPARSE_KV_ECHO_PLAN_CONFIG_H

#include <cstdint>

namespace sparse_kv_echo_plan {
constexpr uint32_t PLAN_THREADS = 2048U;
constexpr uint32_t PLAN_WARP_SIZE = 32U;
constexpr uint32_t PLAN_WARP_COUNT = PLAN_THREADS / PLAN_WARP_SIZE;
constexpr uint32_t PLAN_MAX_BLOCKS = 64U;
constexpr uint32_t PLAN_CONTROL_ELEMENTS = 8U;
constexpr uint32_t PLAN_SCAN_ELEMENTS = 2U * PLAN_WARP_COUNT + PLAN_CONTROL_ELEMENTS;
constexpr uint32_t PLAN_SCAN_BYTES = (PLAN_SCAN_ELEMENTS * sizeof(int32_t) + 31U) & ~31U;
constexpr uint64_t WORKSPACE_ALIGNMENT_ELEMENTS = 32U;

constexpr uint64_t WorkspaceBlockElements(uint64_t maxToken, uint64_t topk, uint64_t capacity) {
  return (maxToken + capacity + topk + WORKSPACE_ALIGNMENT_ELEMENTS - 1U) & ~(WORKSPACE_ALIGNMENT_ELEMENTS - 1U);
}

static_assert(PLAN_THREADS > 0U && PLAN_THREADS <= 2048U, "ECHO requires 1..2048 threads");
static_assert(PLAN_THREADS % PLAN_WARP_SIZE == 0U, "ECHO threads must contain whole warps");
static_assert(PLAN_SCAN_BYTES % 32U == 0U, "ECHO scan storage must be 32-byte aligned");
}  // namespace sparse_kv_echo_plan
#endif
