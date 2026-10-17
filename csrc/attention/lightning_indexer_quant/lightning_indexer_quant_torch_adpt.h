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
#include <cmath>
#include <string>
#include <tuple>

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
    TORCH_CHECK(query.scalar_type() == at::kChar && key.scalar_type() == at::kChar,
                "query and key must be INT8");
    TORCH_CHECK(weights.scalar_type() == at::kHalf && query_scale.scalar_type() == at::kHalf &&
                    key_scale.scalar_type() == at::kHalf,
                "weights and dequant scales must be FP16");
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
                    query_quant_mode,
                    key_quant_mode,
                    query_layout_ptr,
                    key_layout_ptr,
                    sparse_count,
                    sparse_mode,
                    no_prefetch_bytes,
                    no_prefetch_budget,
                    0.5,
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
    TORCH_CHECK(query.size(1) == 64 && query.size(2) == 128,
                "query shape must be [T,64,128]");
    TORCH_CHECK(key.size(2) == 1 && key.size(3) == 128 && key.size(1) > 0 &&
                    key.size(1) <= 1024 && key.size(1) % 16 == 0,
                "key shape must be [blocks,block_size,1,128], block_size multiple of 16 and <= 1024");
    TORCH_CHECK(weights.size(0) == query.size(0) && weights.size(1) == query.size(1) &&
                    query_dequant_scale.size(0) == query.size(0) && query_dequant_scale.size(1) == query.size(1),
                "weights and query_dequant_scale must be [T,64]");
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
    const int64_t batch = query.size(0);
    TORCH_CHECK(batch > 0 && key.size(0) > 0 && block_table.size(1) > 0,
                "batch, physical blocks and block table capacity must be nonzero");
    TORCH_CHECK(actual_seq_lengths_query.size(0) == batch && actual_seq_lengths_key.size(0) == batch &&
                    block_table.size(0) == batch && prefetch_threshold.size(0) == batch,
                "one query/request is required: all batch vectors must have length T");
    TORCH_CHECK(kv_addresses.size(0) == batch &&
                    kv_addresses.size(1) == block_table.size(1) * key.size(1),
                "kv_addresses must have exact [B,max_blocks*block_size] capacity");
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
    const int64_t quant_mode = 0;
    const int64_t sparse_mode = 3;
    EXEC_NPU_CMD(aclnnLightningIndexerQuant,
                 query, key, weights, query_dequant_scale, key_dequant_scale,
                 actual_seq_lengths_query, actual_seq_lengths_key, block_table,
                 kv_addresses, prefetch_threshold,
                 quant_mode, quant_mode, query_layout_ptr, key_layout_ptr, sparse_count, sparse_mode,
                 kv_row_bytes, prefetch_budget, ema_decay,
                 sparse_indices, gathered_kv, prefetch_indices, prefetch_stats, next_threshold);
    return {sparse_indices, gathered_kv, prefetch_indices, prefetch_stats, next_threshold};
}
}
#endif
