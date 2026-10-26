/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef QUANT_LIGHTNING_INDEXER_SPARSE_KV_TEMPLATE_TILING_KEY_H_
#define QUANT_LIGHTNING_INDEXER_SPARSE_KV_TEMPLATE_TILING_KEY_H_

#include "quant_lightning_indexer_sparse_kv_tiling_data.h"
#include "ascendc/host_api/tiling/template_argument.h"

#define QLI_TPL_INT32 3
#define QLI_TPL_FLOAT8_E4M3FN 36
#define QLI_TPL_FIXED_INPUT QLI_TPL_FLOAT8_E4M3FN
#define QLI_LAYOUT_TND 1
#define QLI_LAYOUT_PA_BSND 2
#define QLI_TPL_UINT_WIDTH 4

ASCENDC_TPL_ARGS_DECL(QuantLightningIndexerSparseKv, ASCENDC_TPL_DTYPE_DECL(DT_Q, QLI_TPL_FIXED_INPUT),
                      ASCENDC_TPL_DTYPE_DECL(DT_K, QLI_TPL_FIXED_INPUT), ASCENDC_TPL_DTYPE_DECL(DT_OUT, QLI_TPL_INT32),
                      ASCENDC_TPL_BOOL_DECL(PAGE_ATTENTION, 1, 0),
                      ASCENDC_TPL_UINT_DECL(Q_LAYOUT_T, QLI_TPL_UINT_WIDTH, ASCENDC_TPL_UI_LIST, QLI_LAYOUT_TND),
                      ASCENDC_TPL_UINT_DECL(K_LAYOUT_T, QLI_TPL_UINT_WIDTH, ASCENDC_TPL_UI_LIST, QLI_LAYOUT_PA_BSND), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT_Q, QLI_TPL_FIXED_INPUT),
                                     ASCENDC_TPL_DTYPE_SEL(DT_K, QLI_TPL_FIXED_INPUT),
                                     ASCENDC_TPL_DTYPE_SEL(DT_OUT, QLI_TPL_INT32),
                                     ASCENDC_TPL_BOOL_SEL(PAGE_ATTENTION, 1),
                                     ASCENDC_TPL_UINT_SEL(Q_LAYOUT_T, ASCENDC_TPL_UI_LIST, QLI_LAYOUT_TND),
                                     ASCENDC_TPL_UINT_SEL(K_LAYOUT_T, ASCENDC_TPL_UI_LIST, QLI_LAYOUT_PA_BSND), ));

#endif
