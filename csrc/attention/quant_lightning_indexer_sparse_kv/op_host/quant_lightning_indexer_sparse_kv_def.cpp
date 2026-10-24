/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "register/op_def_registry.h"

namespace ops {
namespace {
void AddInt32Input(OpDef& op, const char* name, bool optional = false)
{
    op.Input(name)
        .ParamType(optional ? OPTIONAL : REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
}

void AddInt64Input(OpDef& op, const char* name)
{
    op.Input(name)
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
}

void AddMutableInt32Input(OpDef& op, const char* name)
{
    op.Input(name)
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .IgnoreContiguous();
}

void AddMutableInt64Input(OpDef& op, const char* name)
{
    op.Input(name)
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .IgnoreContiguous();
}
}

class QuantLightningIndexerSparseKv : public OpDef {
public:
    explicit QuantLightningIndexerSparseKv(const char* name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E4M3FN})
            .Format({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("weights").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("query_dequant_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key_dequant_scale").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).IgnoreContiguous();
        AddInt32Input(*this, "actual_seq_lengths_query", true);
        AddInt32Input(*this, "actual_seq_lengths_key", true);
        AddInt32Input(*this, "indexer_block_table", true);
        AddInt64Input(*this, "req_ids");
        AddInt32Input(*this, "stable_prefix_lens");
        AddInt32Input(*this, "visible_seq_lens");
        AddInt32Input(*this, "token_to_req");
        AddInt32Input(*this, "kv_block_table");
        AddInt32Input(*this, "active_rows");
        AddMutableInt64Input(*this, "last_req_ids");
        AddMutableInt32Input(*this, "slot_to_token");
        AddMutableInt32Input(*this, "lru_slots");
        AddMutableInt32Input(*this, "current_slots");
        AddMutableInt32Input(*this, "miss_count");
        AddMutableInt32Input(*this, "miss_tokens");
        AddMutableInt32Input(*this, "miss_slots");
        AddMutableInt32Input(*this, "compact_workspace");

        this->Output("sparse_indices").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Attr("query_quant_mode").AttrType(REQUIRED).Int(0);
        this->Attr("key_quant_mode").AttrType(REQUIRED).Int(0);
        this->Attr("layout_query").AttrType(REQUIRED).String("TND");
        this->Attr("layout_key").AttrType(REQUIRED).String("PA_BSND");
        this->Attr("sparse_count").AttrType(REQUIRED).Int(2048);
        this->Attr("sparse_mode").AttrType(REQUIRED).Int(3);
        this->Attr("pre_tokens").AttrType(REQUIRED).Int(9223372036854775807);
        this->Attr("next_tokens").AttrType(REQUIRED).Int(9223372036854775807);
        this->Attr("key_stride0").AttrType(REQUIRED).Int(-1);
        this->Attr("key_dequant_scale_stride0").AttrType(REQUIRED).Int(-1);
        this->Attr("capacity").AttrType(REQUIRED).Int();
        this->Attr("max_token").AttrType(REQUIRED).Int();
        this->Attr("block_size").AttrType(REQUIRED).Int();
        this->Attr("host_num_blocks").AttrType(REQUIRED).Int();

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "quant_lightning_indexer_sparse_kv");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(QuantLightningIndexerSparseKv);
}
