/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#ifndef LIGHTNING_INDEXER_QUANT_RESIDENT_SIMT_POLICY_H_
#define LIGHTNING_INDEXER_QUANT_RESIDENT_SIMT_POLICY_H_

#include "kernel_operator.h"
#include "lightning_indexer_quant_resident_simt.h"

namespace LIQ950 {
using namespace AscendC;

__simt_vf__ __aicore__ LAUNCH_BOUND(sparse_kv_plan_cfg::PLAN_THREADS) inline void ResidentInitVf(
    __ubuf__ int32_t *scan, __ubuf__ int32_t *candidates, __gm__ volatile int32_t *mapping,
    __gm__ volatile int32_t *lru, __gm__ int32_t *scratch, uint32_t capacity, uint32_t budget,
    bool reuse, int32_t stable, uint32_t visible)
{
    ResidentSimt::InitRow<>(scan, candidates, mapping, lru, scratch, capacity, budget,
                          reuse, stable, visible, static_cast<uint32_t>(threadIdx.x));
}

__simt_vf__ __aicore__ LAUNCH_BOUND(sparse_kv_plan_cfg::PLAN_THREADS) inline void ResidentClassifyVf(
    __ubuf__ int32_t *scan, __ubuf__ int32_t *candidates, __ubuf__ int32_t *pending,
    __gm__ volatile int32_t *mapping, __gm__ int32_t *scratch, uint32_t capacity, uint32_t budget,
    uint32_t start, uint32_t end, uint32_t previousMisses)
{
    ResidentSimt::ClassifyTile<>(scan, candidates, pending, mapping, scratch, capacity, budget,
                               start, end, previousMisses, static_cast<uint32_t>(threadIdx.x));
}

__simt_vf__ __aicore__ LAUNCH_BOUND(sparse_kv_plan_cfg::PLAN_THREADS) inline void ResidentCommitVf(
    __ubuf__ int32_t *pending, __gm__ volatile int32_t *mapping, __gm__ int32_t *scratch,
    uint32_t capacity, uint32_t budget, uint32_t submitted)
{
    ResidentSimt::CommitTile<>(pending, mapping, scratch, capacity, budget, submitted,
                             static_cast<uint32_t>(threadIdx.x));
}

__simt_vf__ __aicore__ LAUNCH_BOUND(sparse_kv_plan_cfg::PLAN_THREADS) inline void ResidentFinishVf(
    __ubuf__ int32_t *scan, __gm__ volatile int32_t *lru, __gm__ int32_t *scratch,
    uint32_t capacity, uint32_t budget, uint32_t misses)
{
    ResidentSimt::FinishRow<>(scan, lru, scratch, capacity, budget, misses,
                            static_cast<uint32_t>(threadIdx.x));
}

struct ResidentSimtResult { int32_t admitted; int32_t copiedMisses; };

class ResidentSimtPolicy {
public:
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR mapping, GM_ADDR lru, GM_ADDR workspace,
                               LocalTensor<int32_t> candidates, LocalTensor<int32_t> pending,
                               uint32_t capacity, uint32_t budget)
    {
        pipe->InitBuffer(scanBuf_, sparse_kv_plan_cfg::PLAN_SCAN_BYTES);
        scan_ = scanBuf_.Get<int32_t>();
        candidates_ = candidates;
        pending_ = pending;
        mappingBase_ = (__gm__ volatile int32_t *)mapping;
        lruBase_ = (__gm__ volatile int32_t *)lru;
        workspace_ = (__gm__ int32_t *)workspace;
        capacity_ = capacity;
        budget_ = budget < capacity ? budget : capacity;
        rowElements_ = ResidentLayout::RowElements(capacity_, budget_);
    }

    __aicore__ inline void Row(uint32_t row, float threshold)
    {
        const uint64_t offset = static_cast<uint64_t>(row) * capacity_;
        mapping_ = mappingBase_ + offset;
        lru_ = lruBase_ + offset;
        scratch_ = workspace_ + static_cast<uint64_t>(row) * rowElements_;
        threshold_ = threshold;
    }

    __aicore__ inline bool Reset(bool reuse, int32_t stable, uint32_t visible, bool valid)
    {
        active_ = false;
        count_ = flushed_ = misses_ = 0;
        visible_ = visible;
        if (!valid) {
            for (uint32_t i = 0; i < (budget_ > 0 ? budget_ : 1); ++i) candidates_.SetValue(i, -1);
            return false;
        }
        BeginVf();
        asc_vf_call<ResidentInitVf>(dim3(sparse_kv_plan_cfg::PLAN_THREADS), Scan(), Candidates(),
                                   mapping_, lru_, scratch_, capacity_, budget_, reuse, stable, visible);
        EndVf();
        active_ = Control(1) == 0;
        return active_;
    }

    __aicore__ inline void Consider(uint32_t token, float score)
    {
        if (active_ && count_ < budget_ && token < visible_ && score > threshold_)
            candidates_.SetValue(count_++, static_cast<int32_t>(token));
    }

    __aicore__ inline uint32_t FlushTile()
    {
        if (!active_ || flushed_ == count_) return 0;
        BeginVf();
        asc_vf_call<ResidentClassifyVf>(dim3(sparse_kv_plan_cfg::PLAN_THREADS), Scan(), Candidates(), Pending(),
            mapping_, scratch_, capacity_, budget_, flushed_, count_, misses_);
        EndVf();
        count_ = flushed_ = static_cast<uint32_t>(Control(2));
        const uint32_t assigned = static_cast<uint32_t>(Control(3));
        misses_ += assigned;
        return assigned;
    }

    __aicore__ inline void CommitSubmitted(uint32_t submitted)
    {
        if (!active_ || submitted == 0) return;
        // Drain has already consumed both MTE3 completion events. S_V makes
        // that completion precede the VF which publishes the replacement IDs.
        BeginVf();
        asc_vf_call<ResidentCommitVf>(dim3(sparse_kv_plan_cfg::PLAN_THREADS), Pending(), mapping_, scratch_,
                                     capacity_, budget_, submitted);
        EndVf();
    }

    __aicore__ inline ResidentSimtResult Finish(bool validMetadata)
    {
        if (active_) {
            BeginVf();
            asc_vf_call<ResidentFinishVf>(dim3(sparse_kv_plan_cfg::PLAN_THREADS), Scan(), lru_, scratch_,
                                         capacity_, budget_, misses_);
            EndVf();
        }
        for (uint32_t i = count_; i < (budget_ > 0 ? budget_ : 1); ++i) candidates_.SetValue(i, -1);
        active_ = false;
        return validMetadata ? ResidentSimtResult{static_cast<int32_t>(count_), static_cast<int32_t>(misses_)}
                             : ResidentSimtResult{-1, -1};
    }

private:
    __aicore__ inline void BeginVf()
    {
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }
    __aicore__ inline void EndVf()
    {
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    __aicore__ inline __ubuf__ int32_t *Scan() { return (__ubuf__ int32_t *)scan_.GetPhyAddr(); }
    __aicore__ inline __ubuf__ int32_t *Candidates() { return (__ubuf__ int32_t *)candidates_.GetPhyAddr(); }
    __aicore__ inline __ubuf__ int32_t *Pending() { return (__ubuf__ int32_t *)pending_.GetPhyAddr(); }
    __aicore__ inline int32_t Control(uint32_t index)
    {
        return scan_.GetValue(sparse_kv_plan_simt::kControlOffset<> + index);
    }
    TBuf<TPosition::VECCALC> scanBuf_;
    LocalTensor<int32_t> scan_, candidates_, pending_;
    __gm__ volatile int32_t *mappingBase_, *lruBase_, *mapping_, *lru_;
    __gm__ int32_t *workspace_, *scratch_;
    uint64_t rowElements_ = 0;
    uint32_t capacity_ = 0, budget_ = 0, visible_ = 0, count_ = 0, flushed_ = 0, misses_ = 0;
    float threshold_ = 0.0F;
    bool active_ = false;
};
} // namespace LIQ950
#endif
