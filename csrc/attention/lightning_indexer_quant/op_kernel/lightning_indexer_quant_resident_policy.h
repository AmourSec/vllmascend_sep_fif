/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef LIGHTNING_INDEXER_QUANT_RESIDENT_POLICY_H_
#define LIGHTNING_INDEXER_QUANT_RESIDENT_POLICY_H_

#include <cstdint>

#if defined(__CCE_AICORE__)
#define LIQ_RESIDENT_INLINE __aicore__ inline
#else
#define LIQ_RESIDENT_INLINE inline
#endif

namespace LIQ950 {
enum class ResidentArray : uint32_t { Mapping, Lru, OriginalLru, Classes, MissSlots, Candidates };
struct ResidentPolicyResult { int32_t admitted; int32_t copiedMisses; };

// Scalar reference retained for regression tests; production uses ResidentSimtPolicy.
template <typename Storage>
class ResidentPolicy {
public:
    LIQ_RESIDENT_INLINE void Init(Storage &storage, uint32_t capacity, uint32_t budget, float threshold)
    {
        storage_ = &storage;
        capacity_ = capacity;
        budget_ = budget < capacity ? budget : capacity;
        threshold_ = threshold;
    }

    LIQ_RESIDENT_INLINE bool Reset(int64_t request, int32_t stablePrefix, uint32_t visible, bool valid)
    {
        active_ = false;
        count_ = flushed_ = misses_ = 0;
        visible_ = visible;
        for (uint32_t i = 0; i < (budget_ > 0 ? budget_ : 1); ++i) Set(ResidentArray::Candidates, i, -1);
        if (!valid) return false;
        const bool reuse = storage_->Request() != request;
        for (uint32_t i = 0; i < capacity_; ++i) Set(ResidentArray::Classes, i, 0);
        // Validate the permutation before changing caller-owned state. The same
        // scratch is then reused for untouched/hit/miss classes indexed by slot.
        for (uint32_t order = 0; order < capacity_; ++order) {
            const int32_t slot = reuse ? static_cast<int32_t>(order) : Get(ResidentArray::Lru, order);
            if (slot < 0 || static_cast<uint32_t>(slot) >= capacity_ || Get(ResidentArray::Classes, slot) != 0)
                return false;
            Set(ResidentArray::OriginalLru, order, slot);
            Set(ResidentArray::Classes, slot, 1);
        }
        const uint32_t stable = stablePrefix > 0 ? static_cast<uint32_t>(stablePrefix) : 0;
        for (uint32_t slot = 0; slot < capacity_; ++slot) {
            Set(ResidentArray::Classes, slot, kUntouched);
            const int32_t token = reuse ? -1 : Get(ResidentArray::Mapping, slot);
            if (reuse || (token >= 0 && (static_cast<uint32_t>(token) >= stable ||
                                        static_cast<uint32_t>(token) >= visible)))
                Set(ResidentArray::Mapping, slot, -1);
            if (reuse) Set(ResidentArray::Lru, slot, static_cast<int32_t>(slot));
        }
        if (reuse) storage_->SetRequest(request);
        active_ = true;
        return true;
    }

    LIQ_RESIDENT_INLINE void Consider(uint32_t token, float score)
    {
        // Ordered comparison rejects NaN in either operand, matching standalone.
        if (active_ && count_ < budget_ && token < visible_ && score > threshold_)
            Set(ResidentArray::Candidates, count_++, static_cast<int32_t>(token));
    }

    template <typename Transfers>
    LIQ_RESIDENT_INLINE void FlushTile(Transfers &transfers)
    {
        if (!active_ || flushed_ == count_) return;
        transfers.BeginTile(count_ - flushed_);
        // Protect every original resident copy of every hit before allocating a
        // single miss. SparseKvPlan likewise keeps duplicate resident hit slots.
        for (uint32_t slot = 0; slot < capacity_; ++slot) {
            const int32_t token = Get(ResidentArray::Mapping, slot);
            for (uint32_t i = flushed_; i < count_; ++i) {
                if (token == Get(ResidentArray::Candidates, i)) {
                    if (Get(ResidentArray::Classes, slot) == kUntouched)
                        Set(ResidentArray::Classes, slot, kHit);
                    transfers.Protect(i - flushed_);
                }
            }
        }
        uint32_t admitted = flushed_;
        for (uint32_t i = flushed_; i < count_; ++i) {
            const int32_t token = Get(ResidentArray::Candidates, i);
            if (!transfers.HasHit(i - flushed_)) {
                const int32_t slot = Allocate();
                if (slot < 0) continue;
                Set(ResidentArray::Classes, slot, kMiss);
                Set(ResidentArray::MissSlots, misses_++, slot);
                // Invalidate the victim immediately; publish the replacement only
                // through Commit after BOTH payload transfers have completed.
                Set(ResidentArray::Mapping, slot, -1);
                transfers.Enqueue(token, slot);
            }
            Set(ResidentArray::Candidates, admitted++, token);
        }
        count_ = flushed_ = admitted;
    }

    LIQ_RESIDENT_INLINE void Commit(int32_t slot, int32_t token)
    {
        if (active_ && slot >= 0 && static_cast<uint32_t>(slot) < capacity_ &&
            Get(ResidentArray::Classes, slot) == kMiss)
            Set(ResidentArray::Mapping, slot, token);
    }

    LIQ_RESIDENT_INLINE ResidentPolicyResult Finish(bool validMetadata)
    {
        for (uint32_t i = count_; i < (budget_ > 0 ? budget_ : 1); ++i)
            Set(ResidentArray::Candidates, i, -1);
        if (active_) {
            uint32_t output = 0;
            for (uint32_t order = 0; order < capacity_; ++order) {
                const uint32_t slot = Get(ResidentArray::OriginalLru, order);
                if (Get(ResidentArray::Classes, slot) == kUntouched)
                    Set(ResidentArray::Lru, output++, static_cast<int32_t>(slot));
            }
            for (uint32_t miss = 0; miss < misses_; ++miss)
                Set(ResidentArray::Lru, output++, Get(ResidentArray::MissSlots, miss));
            for (uint32_t order = 0; order < capacity_; ++order) {
                const uint32_t slot = Get(ResidentArray::OriginalLru, order);
                if (Get(ResidentArray::Classes, slot) == kHit)
                    Set(ResidentArray::Lru, output++, static_cast<int32_t>(slot));
            }
        }
        active_ = false;
        return validMetadata ? ResidentPolicyResult{static_cast<int32_t>(count_), static_cast<int32_t>(misses_)}
                             : ResidentPolicyResult{-1, -1};
    }

private:
    LIQ_RESIDENT_INLINE int32_t Get(ResidentArray array, uint32_t index) const { return storage_->Get(array, index); }
    LIQ_RESIDENT_INLINE void Set(ResidentArray array, uint32_t index, int32_t value) { storage_->Set(array, index, value); }
    LIQ_RESIDENT_INLINE int32_t Allocate() const
    {
        for (uint32_t order = 0; order < capacity_; ++order) {
            const uint32_t slot = Get(ResidentArray::OriginalLru, order);
            if (Get(ResidentArray::Classes, slot) == kUntouched) return static_cast<int32_t>(slot);
        }
        return -1;
    }

    static constexpr int32_t kUntouched = 0, kHit = 1, kMiss = 2;
    Storage *storage_ = nullptr;
    uint32_t capacity_ = 0, budget_ = 0, visible_ = 0, count_ = 0, flushed_ = 0, misses_ = 0;
    float threshold_ = 0.0F;
    bool active_ = false;
};
} // namespace LIQ950
#undef LIQ_RESIDENT_INLINE
#endif
