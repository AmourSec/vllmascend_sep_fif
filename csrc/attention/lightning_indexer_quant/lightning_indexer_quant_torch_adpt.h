/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef LIGHTING_INDEXER_QUANT_VLLM_TORCH_ADPT_H
#define LIGHTING_INDEXER_QUANT_VLLM_TORCH_ADPT_H

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include "op_host/lightning_indexer_quant_host_contract.h"

namespace vllm_ascend {

namespace liq_detail {
inline void check_common(const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
                         const at::Tensor &query_scale, const at::Tensor &key_scale)
{
    TORCH_CHECK(query.device() == key.device() && query.device() == weights.device() &&
                    query.device() == query_scale.device() && query.device() == key_scale.device(),
                "LightningIndexerQuant inputs must be on the same device");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && weights.is_contiguous() &&
                    query_scale.is_contiguous() && key_scale.is_contiguous(),
                "LightningIndexerQuant inputs must be contiguous");
    TORCH_CHECK(query.scalar_type() == key.scalar_type(),
                "query and key must have the same dtype");
    const bool int8_inputs = query.scalar_type() == at::kChar;
    const bool fp8_inputs = query.scalar_type() == at::kFloat8_e4m3fn;
    TORCH_CHECK(int8_inputs || fp8_inputs,
                "query and key must be INT8 or FLOAT8_E4M3FN");
    const at::ScalarType parameter_dtype = int8_inputs ? at::kHalf : at::kFloat;
    TORCH_CHECK(weights.scalar_type() == parameter_dtype &&
                    query_scale.scalar_type() == parameter_dtype &&
                    key_scale.scalar_type() == parameter_dtype,
                int8_inputs ? "INT8 query/key require FP16 weights and dequant scales"
                            : "FLOAT8_E4M3FN query/key require FP32 weights and dequant scales");
}

inline at::Tensor empty_output(const at::Tensor &like, at::IntArrayRef shape, at::ScalarType dtype)
{
    return at::empty(shape, like.options().dtype(dtype));
}
}

at::Tensor npu_lightning_indexer_quant(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, int64_t query_quant_mode, int64_t key_quant_mode,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode)
{
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    const int SIZE = 8;
    const int DIM_0 = 0;
    const int DIM_1 = 1;
    const int DIM_2 = 2;
    const int DIM_3 = 3;

    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key's shape should be greater "
            "than 0, but shape[", i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);
    TORCH_CHECK((query_layout_str == "TND" && query.dim() == 3) ||
                    (query_layout_str == "BSND" && query.dim() == 4), "invalid query layout/rank");
    TORCH_CHECK((key_layout_str == "TND" && key.dim() == 3) ||
                    ((key_layout_str == "BSND" || key_layout_str == "PA_BSND") && key.dim() == 4),
                "invalid key layout/rank");
    int64_t keyHeadNum = (key_layout_str == "TND")? key.size(DIM_1) : key.size(DIM_2);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparse_count};
    } else {
        output_size = {query.size(DIM_0), keyHeadNum, sparse_count};
    }
    at::Tensor lightning_indexer_quant_output = at::empty(output_size, query.options().dtype(at::kInt));
    at::Tensor gathered_kv = liq_detail::empty_output(query, {query.size(0), sparse_count, 1}, at::kByte);
    at::Tensor prefetch_indices = liq_detail::empty_output(query, {query.size(0), 1}, at::kInt);
    at::Tensor prefetch_stats = liq_detail::empty_output(query, {query.size(0), 2}, at::kInt);
    at::Tensor next_threshold = liq_detail::empty_output(query, {query.size(0)}, at::kFloat);
    const c10::optional<at::Tensor> no_prefetch_tensor = c10::nullopt;
    const int64_t no_prefetch_bytes = 0;
    const int64_t no_prefetch_budget = 0;
    double default_ema_decay = 0.5;

    // convert str
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());

    EXEC_NPU_CMD(aclnnLightningIndexerQuant, 
                    query,
                    key, 
                    weights, 
                    query_dequant_scale, 
                    key_dequant_scale, 
                    actual_seq_lengths_query, 
                    actual_seq_lengths_key,
                    block_table, 
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    no_prefetch_tensor,
                    query_quant_mode,
                    key_quant_mode,
                    query_layout_ptr,
                    key_layout_ptr,
                    sparse_count,
                    sparse_mode,
                    no_prefetch_bytes,
                    no_prefetch_budget,
                    default_ema_decay,
                    no_prefetch_bytes,
                    no_prefetch_bytes,
                    lightning_indexer_quant_output,
                    gathered_kv,
                    prefetch_indices,
                    prefetch_stats,
                    next_threshold
                );

    return lightning_indexer_quant_output;

}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_lightning_indexer_quant_prefetch(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_key,
    const at::Tensor &block_table, const at::Tensor &kv_addresses,
    const at::Tensor &prefetch_threshold, int64_t kv_row_bytes, int64_t prefetch_budget,
    int64_t sparse_count, double ema_decay)
{
    liq_detail::check_common(query, key, weights, query_dequant_scale, key_dequant_scale);
    TORCH_CHECK(query.device().is_privateuseone(), "prefetch API requires an NPU tensor");
    TORCH_CHECK(actual_seq_lengths_query.device() == query.device() && actual_seq_lengths_key.device() == query.device() &&
                    block_table.device() == query.device() && kv_addresses.device() == query.device() &&
                    prefetch_threshold.device() == query.device(),
                "prefetch inputs must be on the same NPU device");
    TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && weights.dim() == 2 &&
                    query_dequant_scale.dim() == 2 && key_dequant_scale.dim() == 3,
                "prefetch API expects TND/PA_BSND ranks (3,4,2,2,3)");
    TORCH_CHECK((query.size(1) == 32 || query.size(1) == 64) && query.size(2) == 128,
                "query shape must be [T,H,128], H must be 32 or 64");
    TORCH_CHECK(key.size(2) == 1 && key.size(3) == 128 && key.size(1) > 0 &&
                    key.size(1) <= 1024 && key.size(1) % 16 == 0,
                "key shape must be [blocks,block_size,1,128], block_size multiple of 16 and <= 1024");
    TORCH_CHECK(weights.size(0) == query.size(0) && weights.size(1) == query.size(1) &&
                    query_dequant_scale.size(0) == query.size(0) && query_dequant_scale.size(1) == query.size(1),
                "weights and query_dequant_scale must be [T,H], matching query heads");
    TORCH_CHECK(key_dequant_scale.size(0) == key.size(0) && key_dequant_scale.size(1) == key.size(1) &&
                    key_dequant_scale.size(2) == 1,
                "key_dequant_scale must be [blocks,block_size,1]");
    TORCH_CHECK(actual_seq_lengths_query.dim() == 1 && actual_seq_lengths_key.dim() == 1 &&
                    block_table.dim() == 2 && kv_addresses.dim() == 2 && prefetch_threshold.dim() == 1,
                "sequence, block_table, address and threshold ranks are invalid");
    TORCH_CHECK(actual_seq_lengths_query.scalar_type() == at::kInt &&
                    actual_seq_lengths_key.scalar_type() == at::kInt &&
                    block_table.scalar_type() == at::kInt && kv_addresses.scalar_type() == at::kLong &&
                    prefetch_threshold.scalar_type() == at::kFloat,
                "sequence/block_table must be INT32, kv_addresses INT64 and threshold FLOAT32");
    TORCH_CHECK(actual_seq_lengths_query.is_contiguous() && actual_seq_lengths_key.is_contiguous() &&
                    block_table.is_contiguous() && kv_addresses.is_contiguous() &&
                    prefetch_threshold.is_contiguous(),
                "sequence, block_table, addresses and threshold must be contiguous");
    for (int64_t dim : query.sizes()) TORCH_CHECK(LIQHost::FitsIndex(dim), "query dimensions exceed INT32 indexing");
    for (int64_t dim : key.sizes()) TORCH_CHECK(LIQHost::FitsIndex(dim), "key dimensions exceed INT32 indexing");
    TORCH_CHECK(LIQHost::FitsIndex(actual_seq_lengths_query.size(0)) &&
                    LIQHost::FitsIndex(block_table.size(1)) &&
                    block_table.size(1) <= INT32_MAX / key.size(1), "request/block capacity exceeds INT32 indexing");
    const int64_t batch = query.size(0);
    TORCH_CHECK(batch > 0 && key.size(0) > 0 && block_table.size(1) > 0,
                "batch, physical blocks and block table capacity must be nonzero");
    TORCH_CHECK(actual_seq_lengths_query.size(0) == actual_seq_lengths_key.size(0) &&
                    block_table.size(0) == actual_seq_lengths_query.size(0) && prefetch_threshold.size(0) == batch,
                "sequence and block metadata must have length N; threshold must have length T");
    TORCH_CHECK(kv_addresses.size(0) == actual_seq_lengths_query.size(0) &&
                    kv_addresses.size(1) == block_table.size(1) * key.size(1),
                "kv_addresses must have exact [N,max_blocks*block_size] capacity");
    TORCH_CHECK(sparse_count >= 1 && sparse_count <= 2048, "sparse_count must be in [1,2048]");
    TORCH_CHECK(kv_row_bytes >= 1 && kv_row_bytes <= 4096, "kv_row_bytes must be in [1,4096]");
    TORCH_CHECK(prefetch_budget >= 0 && prefetch_budget <= 2048, "prefetch_budget must be in [0,2048]");
    TORCH_CHECK(std::isfinite(ema_decay) && ema_decay >= 0.0 && ema_decay < 1.0,
                "ema_decay must be finite and in [0,1)");

    at::Tensor sparse_indices = liq_detail::empty_output(query, {batch, 1, sparse_count}, at::kInt);
    at::Tensor gathered_kv = liq_detail::empty_output(query, {batch, sparse_count, std::max<int64_t>(1, kv_row_bytes)}, at::kByte);
    at::Tensor prefetch_indices = liq_detail::empty_output(query, {batch, std::max<int64_t>(1, prefetch_budget)}, at::kInt);
    at::Tensor prefetch_stats = liq_detail::empty_output(query, {batch, 2}, at::kInt);
    at::Tensor next_threshold = liq_detail::empty_output(query, {batch}, at::kFloat);

    std::string query_layout = "TND";
    std::string key_layout = "PA_BSND";
    char *query_layout_ptr = const_cast<char *>(query_layout.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout.c_str());
    const c10::optional<at::Tensor> no_prefetch_tensor = c10::nullopt;
    const int64_t zero_attr = 0;
    const int64_t quant_mode = 0;
    const int64_t sparse_mode = 3;
    EXEC_NPU_CMD(aclnnLightningIndexerQuant,
                 query, key, weights, query_dequant_scale, key_dequant_scale,
                 actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                 kv_addresses, prefetch_threshold,
                 no_prefetch_tensor, no_prefetch_tensor, no_prefetch_tensor, no_prefetch_tensor, no_prefetch_tensor,
                 no_prefetch_tensor, no_prefetch_tensor, no_prefetch_tensor, no_prefetch_tensor, no_prefetch_tensor, no_prefetch_tensor,
                 quant_mode, quant_mode, query_layout_ptr, key_layout_ptr, sparse_count, sparse_mode,
                 kv_row_bytes, prefetch_budget, ema_decay, zero_attr, zero_attr,
                 sparse_indices, gathered_kv, prefetch_indices, prefetch_stats, next_threshold);
    return {sparse_indices, gathered_kv, prefetch_indices, prefetch_stats, next_threshold};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_lightning_indexer_quant_prefetch_resident(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    const at::Tensor &actual_seq_lengths_query, const at::Tensor &actual_seq_lengths_key,
    const at::Tensor &block_table, const at::Tensor &prefetch_threshold,
    const at::Tensor &req_ids, const at::Tensor &token_to_req, const at::Tensor &stable_prefix_lens,
    const at::Tensor &host_block_table, const at::Tensor &host_cache_bases,
    const at::Tensor &resident_k_input, const at::Tensor &resident_v_input,
    const at::Tensor &last_req_ids, const at::Tensor &slot_to_token, const at::Tensor &lru_slots,
    const at::Tensor &active_rows, int64_t host_block_size, int64_t host_num_blocks,
    int64_t prefetch_budget, int64_t sparse_count, double ema_decay)
{
    liq_detail::check_common(query, key, weights, query_dequant_scale, key_dequant_scale);
    TORCH_CHECK(query.device().is_privateuseone(), "resident prefetch API requires an NPU tensor");
    const std::array<at::Tensor, 15> inputs = {actual_seq_lengths_query, actual_seq_lengths_key, block_table,
        prefetch_threshold, req_ids, token_to_req, stable_prefix_lens, host_block_table, host_cache_bases,
        resident_k_input, resident_v_input, last_req_ids, slot_to_token, lru_slots, active_rows};
    for (const auto &input : inputs) {
        TORCH_CHECK(input.device() == query.device() && input.is_contiguous(),
                    "resident prefetch inputs must be contiguous and on the same NPU device");
    }
    TORCH_CHECK(query.dim() == 3 && query.size(2) == 128 && key.dim() == 4 && key.size(2) == 1 && key.size(3) == 128,
                "resident prefetch expects query [T,H,128] and key [blocks,block,1,128]");
    TORCH_CHECK(weights.dim() == 2 && query_dequant_scale.dim() == 2 && key_dequant_scale.dim() == 3 &&
                    actual_seq_lengths_query.dim() == 1 && actual_seq_lengths_key.dim() == 1 &&
                    block_table.dim() == 2 && prefetch_threshold.dim() == 1 && req_ids.dim() == 1 &&
                    token_to_req.dim() == 1 && stable_prefix_lens.dim() == 1 && host_block_table.dim() == 2 &&
                    host_cache_bases.dim() == 1 && last_req_ids.dim() == 1 && slot_to_token.dim() == 2 &&
                    lru_slots.dim() == 2 && active_rows.dim() == 1, "resident input ranks are invalid");
    TORCH_CHECK((query.size(1) == 32 || query.size(1) == 64) && key.size(1) > 0 &&
                    key.size(1) <= 1024 && key.size(1) % 16 == 0,
                "resident query heads or key block size are invalid");
    TORCH_CHECK(weights.size(0) == query.size(0) && weights.size(1) == query.size(1) &&
                    query_dequant_scale.sizes() == weights.sizes() && key_dequant_scale.size(0) == key.size(0) &&
                    key_dequant_scale.size(1) == key.size(1) && key_dequant_scale.size(2) == 1,
                "resident weights and dequantization scales do not match query/key");
    for (const auto &input : inputs) {
        for (int64_t dim : input.sizes()) TORCH_CHECK(LIQHost::FitsIndex(dim), "resident dimensions exceed INT32 indexing");
    }
    for (int64_t dim : query.sizes()) TORCH_CHECK(LIQHost::FitsIndex(dim), "query dimensions exceed INT32 indexing");
    for (int64_t dim : key.sizes()) TORCH_CHECK(LIQHost::FitsIndex(dim), "key dimensions exceed INT32 indexing");
    TORCH_CHECK(block_table.size(1) <= INT32_MAX / key.size(1), "key capacity exceeds INT32 indexing");
    const auto as_bytes = [](const at::Tensor &value) {
        TORCH_CHECK(value.dim() == 3 || value.dim() == 4, "resident K/V must be rank 3 or 4");
        if (value.dim() == 4) {
            TORCH_CHECK(value.size(2) == 1, "native resident K/V must have singleton head dimension");
            return value.view(at::kByte).view({value.size(0), value.size(1), value.size(3) * value.element_size()});
        }
        TORCH_CHECK(value.scalar_type() == at::kByte, "flat resident K/V must use uint8");
        return value;
    };
    at::Tensor resident_k = as_bytes(resident_k_input);
    at::Tensor resident_v = as_bytes(resident_v_input);
    const int64_t rows = resident_k.size(0);
    TORCH_CHECK(resident_k.dim() == 3 && resident_v.dim() == 3 && rows == resident_v.size(0),
                "resident K/V must be [R,C,row_bytes] byte views");
    TORCH_CHECK(resident_k.size(1) == resident_v.size(1) && resident_k.size(1) > 0,
                "resident K/V capacity must match and be nonzero");
    TORCH_CHECK(resident_k.size(2) >= 1 && resident_k.size(2) <= 4096 &&
                    resident_v.size(2) >= 1 && resident_v.size(2) <= 4096,
                "resident K/V row bytes must be in [1,4096]");
    TORCH_CHECK(actual_seq_lengths_query.scalar_type() == at::kInt && actual_seq_lengths_key.scalar_type() == at::kInt &&
                    block_table.scalar_type() == at::kInt && prefetch_threshold.scalar_type() == at::kFloat &&
                    req_ids.scalar_type() == at::kLong && token_to_req.scalar_type() == at::kInt &&
                    stable_prefix_lens.scalar_type() == at::kInt && host_block_table.scalar_type() == at::kInt &&
                    host_cache_bases.scalar_type() == at::kLong && last_req_ids.scalar_type() == at::kLong &&
                    slot_to_token.scalar_type() == at::kInt && lru_slots.scalar_type() == at::kInt &&
                    active_rows.scalar_type() == at::kInt,
                "resident metadata has invalid dtype");
    const int64_t requests = actual_seq_lengths_query.size(0);
    const int64_t query_rows = query.size(0);
    TORCH_CHECK(actual_seq_lengths_key.size(0) == requests && block_table.size(0) == requests &&
                    req_ids.size(0) == requests && token_to_req.size(0) == query_rows &&
                    stable_prefix_lens.size(0) == query_rows && prefetch_threshold.size(0) == query_rows &&
                    last_req_ids.size(0) == rows && slot_to_token.size(0) == rows && lru_slots.size(0) == rows &&
                    active_rows.numel() == 1 && rows >= query_rows,
                "resident metadata does not match query/request rows");
    TORCH_CHECK(host_block_table.size(0) == requests && host_cache_bases.numel() == 2 &&
                    slot_to_token.size(1) == resident_k.size(1) && lru_slots.size(1) == resident_k.size(1),
                "resident host/cache metadata shape is invalid");
    TORCH_CHECK(LIQHost::FitsIndex(host_block_size) && LIQHost::FitsIndex(host_num_blocks) && prefetch_budget >= 0 &&
                    prefetch_budget <= std::min<int64_t>(2048, resident_k.size(1)) &&
                    sparse_count >= 1 && sparse_count <= std::min<int64_t>(2048, resident_k.size(1)) &&
                    std::isfinite(ema_decay) && ema_decay >= 0.0 && ema_decay < 1.0,
                "resident attributes are out of range");

    TORCH_CHECK(LIQHost::TensorSpan(host_num_blocks, host_block_size, resident_k.size(2)) &&
                    LIQHost::TensorSpan(host_num_blocks, host_block_size, resident_v.size(2)) &&
                    LIQHost::TensorSpan(rows, resident_k.size(1), resident_k.size(2)) &&
                    LIQHost::TensorSpan(rows, resident_v.size(1), resident_v.size(2)),
                "resident byte spans exceed INT64 address indexing");
    at::Tensor sparse_indices = liq_detail::empty_output(query, {query_rows, 1, sparse_count}, at::kInt);
    at::Tensor gathered_kv = liq_detail::empty_output(query, {query_rows, sparse_count, 1}, at::kByte);
    at::Tensor prefetch_indices = liq_detail::empty_output(query, {query_rows, std::max<int64_t>(1, prefetch_budget)}, at::kInt);
    at::Tensor prefetch_stats = liq_detail::empty_output(query, {query_rows, 2}, at::kInt);
    at::Tensor next_threshold = liq_detail::empty_output(query, {query_rows}, at::kFloat);
    std::string query_layout = "TND";
    std::string key_layout = "PA_BSND";
    char *query_layout_ptr = const_cast<char *>(query_layout.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout.c_str());
    const c10::optional<at::Tensor> no_prefetch_tensor = c10::nullopt;
    const int64_t zero_attr = 0;
    const int64_t quant_mode = 0;
    const int64_t sparse_mode = 3;
    EXEC_NPU_CMD(aclnnLightningIndexerQuant,
                 query, key, weights, query_dequant_scale, key_dequant_scale,
                 actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                 no_prefetch_tensor, prefetch_threshold, req_ids, token_to_req, stable_prefix_lens,
                 host_block_table, host_cache_bases, resident_k, resident_v, last_req_ids,
                 slot_to_token, lru_slots, active_rows,
                 quant_mode, quant_mode, query_layout_ptr, key_layout_ptr, sparse_count, sparse_mode,
                 zero_attr, prefetch_budget, ema_decay, host_block_size, host_num_blocks,
                 sparse_indices, gathered_kv, prefetch_indices, prefetch_stats, next_threshold);
    return {sparse_indices, prefetch_indices, prefetch_stats, next_threshold};
}
}
#endif
