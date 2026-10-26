# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

set(
    quant_lightning_indexer_sparse_kv_depends
    "attention/sparse_kv_plan"
    CACHE INTERNAL
    "Kernel source dependencies for quant_lightning_indexer_sparse_kv"
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/
    DESTINATION ${IMPL_INSTALL_DIR}/ascendc/quant_lightning_indexer_sparse_kv/vendor
)
install(
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../../sparse_kv_plan/op_kernel/sparse_kv_plan_shared.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../../sparse_kv_plan/op_kernel/sparse_kv_plan_config.h
    DESTINATION ${IMPL_INSTALL_DIR}/ascendc/quant_lightning_indexer_sparse_kv/planner
)

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        quant_lightning_indexer_sparse_kv_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME QuantLightningIndexerSparseKv
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
            -mllvm -cce-aicore-dcci-before-kernel-end=false
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources_with_soc(OPTYPE quant_lightning_indexer_sparse_kv ACLNNTYPE aclnn)
endif()
