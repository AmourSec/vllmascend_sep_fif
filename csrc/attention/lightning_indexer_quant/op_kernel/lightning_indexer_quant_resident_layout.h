/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#ifndef LIGHTNING_INDEXER_QUANT_RESIDENT_LAYOUT_H_
#define LIGHTNING_INDEXER_QUANT_RESIDENT_LAYOUT_H_

#include <cstdint>

#if defined(__CCE_AICORE__)
#define LIQ_LAYOUT_DEVICE __aicore__
#else
#define LIQ_LAYOUT_DEVICE
#endif

namespace LIQ950::ResidentLayout {

constexpr uint32_t kTileWidth = 128U;
constexpr uint32_t kHashCapacity = 256U;
constexpr uint64_t kAlignmentElements = 128U;

LIQ_LAYOUT_DEVICE constexpr uint64_t AlignElements(uint64_t elements) {
    return (elements + kAlignmentElements - 1U) & ~(kAlignmentElements - 1U);
}

struct RowLayout {
    uint64_t originalLru;
    uint64_t classes;
    uint64_t evict;
    uint64_t hit;
    uint64_t missSlots;
    uint64_t hashKeys;
    uint64_t hashFirstPos;
    uint64_t hashOwner;
    uint64_t elements;
};

LIQ_LAYOUT_DEVICE constexpr RowLayout MakeRowLayout(uint64_t capacity, uint64_t budget) {
    const uint64_t missWidth = budget == 0U ? 1U : budget;
    RowLayout layout{};
    layout.originalLru = 0U;
    layout.classes = layout.originalLru + capacity;
    layout.evict = layout.classes + capacity;
    layout.hit = layout.evict + capacity;
    layout.missSlots = layout.hit + capacity;
    layout.hashKeys = layout.missSlots + missWidth;
    layout.hashFirstPos = layout.hashKeys + kHashCapacity;
    layout.hashOwner = layout.hashFirstPos + kHashCapacity;
    layout.elements = AlignElements(layout.hashOwner + kHashCapacity);
    return layout;
}

LIQ_LAYOUT_DEVICE constexpr uint64_t RowElements(uint64_t capacity, uint64_t budget) {
    return MakeRowLayout(capacity, budget).elements;
}

constexpr uint32_t kPendingTokenOffset = kTileWidth;
constexpr uint32_t kPendingSlotOffset = 2U * kTileWidth;
constexpr uint32_t kPendingElements = 3U * kTileWidth;

constexpr int32_t kUntouched = 0U;
constexpr int32_t kHit = 1U;
constexpr int32_t kMiss = 2U;

}

#undef LIQ_LAYOUT_DEVICE
#endif
