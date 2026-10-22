/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#ifndef LIGHTNING_INDEXER_QUANT_RESIDENT_SIMT_H_
#define LIGHTNING_INDEXER_QUANT_RESIDENT_SIMT_H_

#include <climits>
#include <cstdint>
#include "../lightning_indexer_quant_resident_layout.h"
#if __has_include("../../sparse_kv_plan/sparse_kv_plan_simt.h")
#include "../../sparse_kv_plan/sparse_kv_plan_simt.h"
#else
#include "../../../sparse_kv_plan/op_kernel/sparse_kv_plan_simt.h"
#endif

namespace LIQ950::ResidentSimt {

constexpr int32_t kInvalid = -1;
using namespace ResidentLayout;

template <uint32_t Threads = sparse_kv_plan_cfg::PLAN_THREADS>
__simt_callee__ __aicore__ inline __ubuf__ int32_t *Control(__ubuf__ int32_t *scan) {
    return scan + sparse_kv_plan_simt::kControlOffset<Threads>;
}

template <uint32_t Threads = sparse_kv_plan_cfg::PLAN_THREADS>
__simt_callee__ __aicore__ inline void InitRow(
    __ubuf__ int32_t *scan, __ubuf__ int32_t *candidates, volatile __gm__ int32_t *mapping,
    volatile __gm__ int32_t *lru, __gm__ int32_t *scratch, uint32_t capacity, uint32_t budget,
    bool reuse, int32_t stablePrefix, uint32_t visible, uint32_t thread) {
    __gm__ int32_t *original = scratch;
    __gm__ int32_t *classes = scratch + capacity;
    __ubuf__ int32_t *control = Control<Threads>(scan);
    const uint32_t candidateWidth = budget == 0U ? 1U : budget;
    for (uint32_t i = thread; i < candidateWidth; i += Threads) candidates[i] = kInvalid;
    if (thread == 0U) {
        control[1] = 0;
        control[2] = 0;
        control[3] = 0;
    }
    asc_syncthreads();
    for (uint32_t slot = thread; slot < capacity; slot += Threads) classes[slot] = kUntouched;
    asc_syncthreads();
    if (reuse) {
        for (uint32_t order = thread; order < capacity; order += Threads) original[order] = static_cast<int32_t>(order);
    } else {
        for (uint32_t order = thread; order < capacity; order += Threads) {
            const int32_t slot = lru[order];
            if (slot < 0 || static_cast<uint32_t>(slot) >= capacity) {
                asc_atomic_max(control + 1, 1);
            } else {
                const int32_t old = asc_atomic_cas(classes + slot, kUntouched, kHit);
                if (old != kUntouched) asc_atomic_max(control + 1, 1);
                original[order] = slot;
            }
        }
    }
    asc_syncthreads();
    if (control[1] != 0) return;
    const uint32_t stable = stablePrefix > 0 ? static_cast<uint32_t>(stablePrefix) : 0U;
    for (uint32_t slot = thread; slot < capacity; slot += Threads) {
        classes[slot] = kUntouched;
        const int32_t token = reuse ? kInvalid : mapping[slot];
        if (reuse || (token >= 0 && (static_cast<uint32_t>(token) >= stable || static_cast<uint32_t>(token) >= visible))) {
            mapping[slot] = kInvalid;
        }
        if (reuse) lru[slot] = static_cast<int32_t>(slot);
    }
    asc_syncthreads();
}

template <uint32_t Threads = sparse_kv_plan_cfg::PLAN_THREADS>
__simt_callee__ __aicore__ inline void ClassifyTile(
    __ubuf__ int32_t *scan, __ubuf__ int32_t *candidates, __ubuf__ int32_t *pending,
    volatile __gm__ int32_t *mapping, __gm__ int32_t *scratch, uint32_t capacity, uint32_t budget,
    uint32_t start, uint32_t end, uint32_t previousMisses, uint32_t thread) {
    __gm__ int32_t *original = scratch;
    __gm__ int32_t *classes = scratch + capacity;
    __gm__ int32_t *evict = scratch + 2ULL * capacity;
    __gm__ int32_t *hit = scratch + 3ULL * capacity;
    __gm__ int32_t *missSlots = scratch + 4ULL * capacity;
    __gm__ int32_t *keys = missSlots + (budget > 0 ? budget : 1);
    __gm__ int32_t *first = keys + kHashCapacity;
    __gm__ int32_t *owner = first + kHashCapacity;
    __ubuf__ int32_t *control = Control<Threads>(scan);
    const uint32_t count = end > start ? end - start : 0U;
    for (uint32_t i = thread; i < kHashCapacity; i += Threads) {
        keys[i] = sparse_kv_plan_simt::HASH_EMPTY;
        first[i] = sparse_kv_plan_simt::HASH_POSITION_EMPTY;
        owner[i] = kInvalid;
    }
    for (uint32_t i = thread; i < count; i += Threads) pending[i] = candidates[start + i];
    asc_syncthreads();
    for (uint32_t i = thread; i < count; i += Threads) {
        if (pending[i] >= 0) sparse_kv_plan_simt::InsertTopkToken(keys, first, pending[i], static_cast<int32_t>(i), kHashCapacity);
    }
    asc_syncthreads();
    int32_t evictCount = 0;
    for (uint32_t tile = 0; tile < capacity; tile += Threads) {
        const uint32_t order = tile + thread;
        const int32_t slot = order < capacity ? original[order] : kInvalid;
        int32_t evictFlag = 0;
        if (slot >= 0) {
            const int32_t token = mapping[slot];
            const uint32_t bucket = token >= 0 ? sparse_kv_plan_simt::LookupTopkToken(keys, token, kHashCapacity)
                                               : sparse_kv_plan_simt::HASH_BUCKET_INVALID;
            if (bucket != sparse_kv_plan_simt::HASH_BUCKET_INVALID) {
                if (classes[slot] == kUntouched) classes[slot] = kHit;
                asc_atomic_max(owner + bucket, static_cast<int32_t>(order));
            }
            evictFlag = classes[slot] == kUntouched;
        }
        const int32_t totals = sparse_kv_plan_simt::PartitionSlots<Threads>(
            scan, thread, slot, evictFlag, 0, evict, hit, evictCount, 0);
        evictCount += totals & sparse_kv_plan_simt::COUNT_MASK;
    }
    int32_t missing = 0;
    int32_t kept = 0;
    for (uint32_t tile = 0; tile < count; tile += Threads) {
        const uint32_t i = tile + thread;
        const int32_t token = i < count ? pending[i] : kInvalid;
        const uint32_t bucket = token >= 0 ? sparse_kv_plan_simt::LookupTopkToken(keys, token, kHashCapacity)
                                           : sparse_kv_plan_simt::HASH_BUCKET_INVALID;
        const bool isHit = bucket != sparse_kv_plan_simt::HASH_BUCKET_INVALID && owner[bucket] >= 0;
        const int32_t isMiss = token >= 0 && !isHit;
        const int32_t rank = sparse_kv_plan_simt::BlockExclusiveScan<Threads>(scan, isMiss, thread);
        const int32_t tileMisses = control[0];
        const int32_t missIndex = missing + rank;
        const bool assigned = isMiss != 0 && missIndex < evictCount;
        if (assigned) {
            const int32_t slot = evict[missIndex];
            mapping[slot] = kInvalid;
            classes[slot] = kMiss;
            missSlots[previousMisses + missIndex] = slot;
            pending[kPendingTokenOffset + missIndex] = token;
            pending[kPendingSlotOffset + missIndex] = slot;
        }
        const int32_t keep = isHit || assigned;
        const int32_t keepRank = sparse_kv_plan_simt::BlockExclusiveScan<Threads>(scan, keep, thread);
        const int32_t tileKept = control[0];
        if (keep != 0) candidates[start + kept + keepRank] = token;
        asc_syncthreads();
        missing += tileMisses;
        kept += tileKept;
    }
    if (thread == 0U) {
        control[2] = static_cast<int32_t>(start) + kept;
        control[3] = missing < evictCount ? missing : evictCount;
    }
    asc_syncthreads();
}

template <uint32_t Threads = sparse_kv_plan_cfg::PLAN_THREADS>
__simt_callee__ __aicore__ inline void CommitTile(
    __ubuf__ int32_t *pending, volatile __gm__ int32_t *mapping, __gm__ int32_t *scratch,
    uint32_t capacity, uint32_t, uint32_t submitted, uint32_t thread) {
    __gm__ int32_t *classes = scratch + capacity;
    for (uint32_t i = thread; i < submitted; i += Threads) {
        const int32_t slot = pending[kPendingSlotOffset + i];
        const int32_t token = pending[kPendingTokenOffset + i];
        if (slot >= 0 && static_cast<uint32_t>(slot) < capacity && classes[slot] == kMiss) mapping[slot] = token;
    }
    asc_syncthreads();
}

template <uint32_t Threads = sparse_kv_plan_cfg::PLAN_THREADS>
__simt_callee__ __aicore__ inline void FinishRow(
    __ubuf__ int32_t *scan, volatile __gm__ int32_t *lru, __gm__ int32_t *scratch,
    uint32_t capacity, uint32_t, uint32_t totalMisses, uint32_t thread) {
    __gm__ int32_t *original = scratch;
    __gm__ int32_t *classes = scratch + capacity;
    __gm__ int32_t *evict = scratch + 2ULL * capacity;
    __gm__ int32_t *hit = scratch + 3ULL * capacity;
    __gm__ int32_t *missSlots = scratch + 4ULL * capacity;
    int32_t untouched = 0;
    int32_t hitCount = 0;
    for (uint32_t tile = 0; tile < capacity; tile += Threads) {
        const uint32_t order = tile + thread;
        const int32_t slot = order < capacity ? original[order] : kInvalid;
        const int32_t value = slot >= 0 ? classes[slot] : kMiss;
        const int32_t totals = sparse_kv_plan_simt::PartitionSlots<Threads>(
            scan, thread, slot, value == kUntouched, value == kHit, evict + totalMisses, hit, untouched, hitCount);
        untouched += totals & sparse_kv_plan_simt::COUNT_MASK;
        hitCount += totals >> sparse_kv_plan_simt::COUNT_SHIFT;
    }
    sparse_kv_plan_simt::MergeLru<Threads>(evict, hit, missSlots, lru,
        untouched + static_cast<int32_t>(totalMisses), hitCount, static_cast<int32_t>(totalMisses), thread);
}

}

#endif
