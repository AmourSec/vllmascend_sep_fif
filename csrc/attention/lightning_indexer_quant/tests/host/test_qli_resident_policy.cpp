#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "../../op_kernel/lightning_indexer_quant_resident_policy.h"
#include "../../op_kernel/lightning_indexer_quant_resident_source.h"

using LIQ950::ResidentArray;

struct Storage {
    std::array<std::vector<int32_t>, 6> arrays;
    int64_t request = 7;
    explicit Storage(uint32_t capacity) {
        for (auto &array : arrays) array.resize(capacity, -1);
        std::iota(arrays[1].begin(), arrays[1].end(), 0);
    }
    int32_t Get(ResidentArray array, uint32_t index) const { return arrays[static_cast<uint32_t>(array)].at(index); }
    void Set(ResidentArray array, uint32_t index, int32_t value) { arrays[static_cast<uint32_t>(array)].at(index) = value; }
    int64_t Request() const { return request; }
    void SetRequest(int64_t value) { request = value; }
};

struct Pending {
    std::vector<std::array<int32_t, 2>> copies;
    std::array<bool, 128> hits{};
    void BeginTile(uint32_t count) { assert(count <= 128); hits.fill(false); }
    void Protect(uint32_t index) { hits.at(index) = true; }
    bool HasHit(uint32_t index) const { return hits.at(index); }
    void Enqueue(int32_t token, int32_t slot) { copies.push_back({token, slot}); }
    void Complete(LIQ950::ResidentPolicy<Storage> &policy) {
        for (auto copy : copies) policy.Commit(copy[1], copy[0]);
        copies.clear();
    }
};

static void HitProtectionAndPublication() {
    Storage store(4);
    store.arrays[0] = {2, 8, 10, 12};
    store.arrays[1] = {1, 0, 3, 2};
    LIQ950::ResidentPolicy<Storage> policy;
    policy.Init(store, 4, 4, 0.5F);
    assert(policy.Reset(7, 100, 100, true));
    // A miss precedes the oldest resident hit in this score tile.
    policy.Consider(3, 1.0F);
    policy.Consider(8, 1.0F);
    Pending pending;
    policy.FlushTile(pending);
    assert((pending.copies == std::vector<std::array<int32_t, 2>>{{3, 0}}));
    assert(store.arrays[0][0] == -1 && store.arrays[0][1] == 8);
    pending.Complete(policy);
    assert(store.arrays[0][0] == 3);
    policy.Consider(13, 1.0F);
    policy.FlushTile(pending);
    assert((pending.copies == std::vector<std::array<int32_t, 2>>{{13, 3}}));
    pending.Complete(policy);
    const auto result = policy.Finish(true);
    assert(result.admitted == 3 && result.copiedMisses == 2);
    assert((store.arrays[1] == std::vector<int32_t>{2, 0, 3, 1}));
    assert((store.arrays[5] == std::vector<int32_t>{3, 8, 13, -1}));
}

static void ResetTailAndInvalidRows() {
    Storage store(3);
    store.arrays[0] = {3, 8, 9};
    LIQ950::ResidentPolicy<Storage> policy;
    policy.Init(store, 3, 2, 0.0F);
    assert(policy.Reset(7, 8, 10, true));
    assert((store.arrays[0] == std::vector<int32_t>{3, -1, -1}));
    Pending pending;
    policy.Consider(8, 1.0F); // Speculative tail candidates remain eligible.
    policy.Consider(9, 1.0F);
    policy.Consider(10, 1.0F);
    policy.FlushTile(pending);
    pending.Complete(policy);
    assert(policy.Finish(true).copiedMisses == 2);
    assert((store.arrays[0] == std::vector<int32_t>{8, 9, -1}));
    assert(policy.Reset(8, 100, 10, true));
    assert(store.request == 8);
    assert((store.arrays[0] == std::vector<int32_t>{-1, -1, -1}));
    assert((store.arrays[1] == std::vector<int32_t>{0, 1, 2}));
    policy.Finish(true);
    const auto old = store.arrays;
    assert(!policy.Reset(9, 0, 10, false));
    policy.Consider(1, 2.0F);
    policy.FlushTile(pending);
    assert(policy.Finish(false).admitted == -1);
    assert(store.arrays[0] == old[0] && store.arrays[1] == old[1] && store.request == 8);
    assert(!policy.Reset(9, 0, 10, false));
    const auto inactive = policy.Finish(true);
    assert(inactive.admitted == 0 && inactive.copiedMisses == 0);
}

static void ThresholdsBudgetAndMalformedLru() {
    Storage store(3);
    LIQ950::ResidentPolicy<Storage> policy;
    Pending pending;
    policy.Init(store, 3, 0, -1.0F);
    assert(policy.Reset(7, 100, 10, true));
    policy.Consider(1, 5.0F);
    policy.FlushTile(pending);
    assert(policy.Finish(true).admitted == 0 && pending.copies.empty());
    policy.Init(store, 3, 2, std::numeric_limits<float>::quiet_NaN());
    assert(policy.Reset(7, 100, 10, true));
    policy.Consider(1, 5.0F);
    policy.FlushTile(pending);
    assert(policy.Finish(true).admitted == 0);
    policy.Init(store, 3, 2, 0.5F);
    assert(policy.Reset(7, -1, 10, true));
    policy.Consider(0, std::numeric_limits<float>::quiet_NaN());
    policy.Consider(1, 0.5F);
    policy.Consider(2, 0.6F);
    policy.Consider(3, 0.7F);
    policy.Consider(4, 0.8F);
    policy.FlushTile(pending);
    assert(pending.copies.size() == 2);
    pending.Complete(policy);
    assert(policy.Finish(true).admitted == 2);
    store.arrays[1] = {0, 0, 4};
    const auto before = store.arrays;
    assert(!policy.Reset(7, 0, 10, true));
    assert(store.arrays[0] == before[0] && store.arrays[1] == before[1]);
}

static void DuplicateResidentsAreAllProtected() {
    Storage store(4);
    store.arrays[0] = {2, 2, 4, 6};
    LIQ950::ResidentPolicy<Storage> policy;
    policy.Init(store, 4, 4, 0.0F);
    assert(policy.Reset(7, 100, 100, true));
    for (int32_t token : {1, 2, 3, 5}) policy.Consider(token, 1.0F);
    Pending pending;
    policy.FlushTile(pending);
    assert(pending.copies.size() == 2);
    pending.Complete(policy);
    const auto result = policy.Finish(true);
    assert(result.admitted == 3 && result.copiedMisses == 2);
    assert((store.arrays[0] == std::vector<int32_t>{2, 2, 1, 3}));
    assert((store.arrays[1] == std::vector<int32_t>{2, 3, 0, 1}));
    assert((store.arrays[5] == std::vector<int32_t>{1, 2, 3, -1}));
}

static void LargeCapacityKeepsPinnedSlots() {
    constexpr uint32_t capacity = 4099;
    Storage store(capacity);
    std::iota(store.arrays[0].begin(), store.arrays[0].end(), 0);
    LIQ950::ResidentPolicy<Storage> policy;
    policy.Init(store, capacity, 2048, -1.0F);
    assert(policy.Reset(7, 10000, 10000, true));
    Pending pending;
    for (uint32_t token = 0; token < 2048; ++token) {
        policy.Consider(token * 2, 0.0F);
        if (token % 128 == 127) {
            policy.FlushTile(pending);
            pending.Complete(policy);
        }
    }
    const auto result = policy.Finish(true);
    assert(result.admitted == 2048 && result.copiedMisses == 0);
    uint32_t output = 0;
    for (uint32_t slot = 0; slot < capacity; ++slot)
        if (slot >= 4096 || slot % 2) assert(store.arrays[1][output++] == static_cast<int32_t>(slot));
    for (uint32_t slot = 0; slot < 4096; slot += 2)
        assert(store.arrays[1][output++] == static_cast<int32_t>(slot));
    assert(output == capacity);
}

static void SourceAddressesAreBounded() {
    uint64_t address = 0;
    assert(LIQ950::ResidentSourceAddress(4096, 7, 12, 16, 33, address));
    assert(address == 4096 + (7 * 16 + 12) * 33);
    assert(LIQ950::ResidentSourceAddress(8192, 7, 12, 16, 4095, address));
    assert(address == 8192 + (7 * 16 + 12) * 4095);
    assert(!LIQ950::ResidentSourceAddress(0, 0, 0, 16, 33, address));
    assert(!LIQ950::ResidentSourceAddress(4096, 0, 16, 16, 33, address));
    assert(!LIQ950::ResidentSourceAddress(4096, 0, 0, 0, 33, address));
    assert(!LIQ950::ResidentSourceAddress(4096, 0, 0, 16, 4097, address));
    assert(!LIQ950::ResidentSourceAddress(UINT64_MAX - 8, 0, 0, 16, 33, address));
    assert(!LIQ950::ResidentSourceAddress(4096, UINT32_MAX, 0, UINT32_MAX, 4096, address));
    assert(LIQ950::ResidentSourceAddress(UINT64_MAX, 0, 0, 1, 1, address));
    assert(address == UINT64_MAX);
}

int main() {
    SourceAddressesAreBounded();
    HitProtectionAndPublication();
    ResetTailAndInvalidRows();
    ThresholdsBudgetAndMalformedLru();
    DuplicateResidentsAreAllProtected();
    LargeCapacityKeepsPinnedSlots();
}
