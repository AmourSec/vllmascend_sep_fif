/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShapeQuantLightningIndexerSparseKv(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* queryShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (queryShape == nullptr || outputShape == nullptr || queryShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    outputShape->SetDimNum(3);
    outputShape->SetDim(0, queryShape->GetDim(0));
    outputShape->SetDim(1, 1);
    outputShape->SetDim(2, 2048);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQuantLightningIndexerSparseKv(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantLightningIndexerSparseKv)
    .InferShape(InferShapeQuantLightningIndexerSparseKv)
    .InferDataType(InferDataTypeQuantLightningIndexerSparseKv);
}
