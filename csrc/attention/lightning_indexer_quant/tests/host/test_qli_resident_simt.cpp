#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "simt_api/asc_simt.h"
#include "../../op_kernel/arch35/lightning_indexer_quant_resident_simt.h"
#include "../../op_kernel/lightning_indexer_quant_resident_policy.h"

namespace {

using LIQ950::ResidentLayout::MakeRowLayout;

struct Row {
    explicit Row(uint32_t capacity, uint32_t budget)
        : mapping(capacity), lru(capacity), scratch(MakeRowLayout(capacity, budget).elements, -1),
          candidates(std::max(1U, budget), -1), pending(3U * 128U, -1), scan(64, 0) {
        std::iota(lru.begin(), lru.end(), 0);
    }

    std::vector<int32_t> mapping;
    std::vector<int32_t> lru;
    std::vector<int32_t> scratch;
    std::vector<int32_t> candidates;
    std::vector<int32_t> pending;
    std::vector<int32_t> scan;
};

struct OracleStorage {
    std::vector<int32_t> mapping;
    std::vector<int32_t> lru;
    std::vector<int32_t> original;
    std::vector<int32_t> classes;
    std::vector<int32_t> misses;
    std::vector<int32_t> candidates;
    explicit OracleStorage(uint32_t capacity, uint32_t budget)
        : mapping(capacity), lru(capacity), original(capacity), classes(capacity), misses(capacity),
          candidates(std::max(1U, budget), -1) {
        std::iota(lru.begin(), lru.end(), 0);
    }
    int32_t Get(LIQ950::ResidentArray array, uint32_t index) const {
        const std::vector<int32_t> *values = nullptr;
        switch (array) {
        case LIQ950::ResidentArray::Mapping: values = &mapping; break;
        case LIQ950::ResidentArray::Lru: values = &lru; break;
        case LIQ950::ResidentArray::OriginalLru: values = &original; break;
        case LIQ950::ResidentArray::Classes: values = &classes; break;
        case LIQ950::ResidentArray::MissSlots: values = &misses; break;
        case LIQ950::ResidentArray::Candidates: values = &candidates; break;
        }
        return values->at(index);
    }
    void Set(LIQ950::ResidentArray array, uint32_t index, int32_t value) {
        switch (array) {
        case LIQ950::ResidentArray::Mapping: mapping.at(index) = value; break;
        case LIQ950::ResidentArray::Lru: lru.at(index) = value; break;
        case LIQ950::ResidentArray::OriginalLru: original.at(index) = value; break;
        case LIQ950::ResidentArray::Classes: classes.at(index) = value; break;
        case LIQ950::ResidentArray::MissSlots: misses.at(index) = value; break;
        case LIQ950::ResidentArray::Candidates: candidates.at(index) = value; break;
        }
    }
    int64_t Request() const { return 7; }
    void SetRequest(int64_t) {}
};

struct OracleTransfers {
    std::vector<std::pair<int32_t, int32_t>> copies;
    std::array<bool, 128> hits{};
    void BeginTile(uint32_t) { hits.fill(false); }
    void Protect(uint32_t index) { hits.at(index) = true; }
    bool HasHit(uint32_t index) const { return hits.at(index); }
    void Enqueue(int32_t token, int32_t slot) { copies.emplace_back(token, slot); }
};

template <uint32_t Threads>
void Start(Row &, uint32_t, bool = false, int32_t = 1000, uint32_t = 1000);

template <uint32_t Threads>
void OneTile(Row &, uint32_t, uint32_t, std::initializer_list<int32_t>, uint32_t = 0);

template <uint32_t Threads>
void Commit(Row &, uint32_t, uint32_t, uint32_t);

template <uint32_t Threads>
void Finish(Row &, uint32_t, uint32_t, uint32_t);

template <uint32_t Threads>
void CompareWithOracle() {
    Row row(4, 2);
    row.mapping = {10, 20, 30, 40};
    Start<Threads>(row, 2, false, 1000, 1000);
    OneTile<Threads>(row, 4, 2, {30}, 0);
    OneTile<Threads>(row, 4, 2, {50}, 0);
    Commit<Threads>(row, 4, 2, 1);
    Finish<Threads>(row, 4, 2, 1);

    OracleStorage storage(4, 2);
    storage.mapping = {10, 20, 30, 40};
    LIQ950::ResidentPolicy<OracleStorage> oracle;
    oracle.Init(storage, 4, 2, -1.0F);
    assert(oracle.Reset(7, 1000, 1000, true));
    OracleTransfers transfers;
    oracle.Consider(30, 0.0F);
    oracle.FlushTile(transfers);
    for (auto [token, slot] : transfers.copies) oracle.Commit(slot, token);
    transfers.copies.clear();
    oracle.Consider(50, 0.0F);
    oracle.FlushTile(transfers);
    for (auto [token, slot] : transfers.copies) oracle.Commit(slot, token);
    const auto result = oracle.Finish(true);
    assert(result.admitted == 2 && result.copiedMisses == 1);
    assert(row.mapping == storage.mapping);
    assert(row.lru == storage.lru);
    assert(row.candidates[0] == 30 && row.candidates[1] == 50);
}

template <uint32_t Threads, typename Fn>
void Parallel(Fn &&function) {
    liq_host_simt::Configure(Threads);
    std::vector<std::thread> workers;
    workers.reserve(Threads);
    for (uint32_t thread = 0; thread < Threads; ++thread) {
        workers.emplace_back([thread, &function] {
            if (thread % 7U == 0U) std::this_thread::yield();
            liq_host_simt::SetThread(thread);
            function(thread);
        });
    }
    for (auto &worker : workers) worker.join();
}

template <uint32_t Threads>
void Start(Row &row, uint32_t budget, bool reuse, int32_t stable, uint32_t visible) {
    Parallel<Threads>([&](uint32_t thread) {
        LIQ950::ResidentSimt::InitRow<Threads>(row.scan.data(), row.candidates.data(), row.mapping.data(), row.lru.data(),
                                                row.scratch.data(), static_cast<uint32_t>(row.mapping.size()), budget,
                                                reuse, stable, visible, thread);
    });
}

template <uint32_t Threads>
void StreamTile(Row &row, uint32_t capacity, uint32_t budget, const std::vector<int32_t> &values,
                uint32_t previousMisses) {
    const uint32_t count = static_cast<uint32_t>(values.size());
    const uint32_t controlOffset = 2U * (Threads / 32U);
    const uint32_t start = static_cast<uint32_t>(row.scan[controlOffset + 2]);
    std::copy(values.begin(), values.end(), row.candidates.begin() + start);
    Parallel<Threads>([&](uint32_t thread) {
        LIQ950::ResidentSimt::ClassifyTile<Threads>(row.scan.data(), row.candidates.data(), row.pending.data(),
                                                     row.mapping.data(), row.scratch.data(), capacity, budget, start,
                                                     start + count, previousMisses, thread);
    });
}

template <uint32_t Threads>
void OneTile(Row &row, uint32_t capacity, uint32_t budget, std::initializer_list<int32_t> values,
             uint32_t previousMisses) {
    StreamTile<Threads>(row, capacity, budget, std::vector<int32_t>(values), previousMisses);
}

template <uint32_t Threads>
void Commit(Row &row, uint32_t capacity, uint32_t budget, uint32_t submitted) {
    Parallel<Threads>([&](uint32_t thread) {
        LIQ950::ResidentSimt::CommitTile<Threads>(row.pending.data(), row.mapping.data(), row.scratch.data(), capacity,
                                                   budget, submitted, thread);
    });
}

template <uint32_t Threads>
void Finish(Row &row, uint32_t capacity, uint32_t budget, uint32_t misses) {
    Parallel<Threads>([&](uint32_t thread) {
        LIQ950::ResidentSimt::FinishRow<Threads>(row.scan.data(), row.lru.data(), row.scratch.data(), capacity, budget,
                                                  misses, thread);
    });
}

template <uint32_t Threads>
void TestPublishAfterCommit() {
    Row row(4, 2);
    row.mapping = {10, 20, 30, 40};
    Start<Threads>(row, 2);
    OneTile<Threads>(row, 4, 2, {50});
    assert(row.mapping[0] == -1);
    const int32_t victim = row.pending[2 * 128];
    assert(victim >= 0 && row.pending[128] == 50);
    Commit<Threads>(row, 4, 2, 0);
    assert(row.mapping[0] == -1);
    Commit<Threads>(row, 4, 2, 1);
    assert(row.mapping[victim] == 50);
}

template <uint32_t Threads>
void TestDuplicateHitAndCrossTileProtection() {
    Row row(4, 2);
    row.mapping = {7, 7, 8, 9};
    Start<Threads>(row, 2);
    OneTile<Threads>(row, 4, 2, {7});
    assert(row.pending[0] == 7);
    Finish<Threads>(row, 4, 2, 0);
    assert(row.lru[0] == 2 && row.lru[1] == 3);

    Row cross(4, 2);
    cross.mapping = {10, 20, 30, 40};
    Start<Threads>(cross, 2);
    OneTile<Threads>(cross, 4, 2, {30});
    OneTile<Threads>(cross, 4, 2, {50}, 0);
    assert(cross.mapping[0] == -1);
    Commit<Threads>(cross, 4, 2, 1);
    assert(cross.mapping[0] == 50);
    Finish<Threads>(cross, 4, 2, 1);
    assert(cross.lru[0] == 1 && cross.lru[1] == 3 && cross.lru[2] == 0 && cross.lru[3] == 2);
}

template <uint32_t Threads>
void TestBudgetAndInvalidMetadata() {
    Row zero(4, 0);
    zero.mapping = {1, 2, 3, 4};
    Start<Threads>(zero, 0);
    OneTile<Threads>(zero, 4, 0, {});

    Row invalid(4, 2);
    invalid.mapping = {1, 2, 3, 4};
    invalid.lru = {0, 1, 1, 3};
    const auto beforeMapping = invalid.mapping;
    const auto beforeLru = invalid.lru;
    Start<Threads>(invalid, 2);
    assert(invalid.mapping == beforeMapping);
    assert(invalid.lru == beforeLru);
}

template <uint32_t Threads>
void TestStableTailAndIndependentRows() {
    Row row(4, 2);
    row.mapping = {4, 8, 12, 16};
    Start<Threads>(row, 2, false, 10, 14);
    assert(row.mapping == std::vector<int32_t>({4, 8, -1, -1}));

    Row first(2053, 1);
    Row second(2053, 1);
    first.mapping[0] = 101;
    second.mapping[0] = 202;
    Start<Threads>(first, 1);
    Start<Threads>(second, 1);
    OneTile<Threads>(first, 2053, 1, {303});
    OneTile<Threads>(second, 2053, 1, {404});
    Commit<Threads>(first, 2053, 1, 1);
    Commit<Threads>(second, 2053, 1, 1);
    assert(first.mapping[0] == 303);
    assert(second.mapping[0] == 404);
}

template <uint32_t Threads>
void TestWideStreamingAndExhaustion() {
    Row row(65, 65);
    std::iota(row.mapping.begin(), row.mapping.end(), 0);
    Start<Threads>(row, 65);
    std::vector<int32_t> first(64);
    std::iota(first.begin(), first.end(), 10000);
    StreamTile<Threads>(row, 65, 65, first, 0);
    const uint32_t offset = 2U * (Threads / 32U);
    assert(row.scan[offset + 2] == 64 && row.scan[offset + 3] == 64);
    Commit<Threads>(row, 65, 65, 64);
    StreamTile<Threads>(row, 65, 65, {11000}, 64);
    assert(row.scan[offset + 2] == 65 && row.scan[offset + 3] == 1);
    Commit<Threads>(row, 65, 65, 1);
    Finish<Threads>(row, 65, 65, 65);
    for (uint32_t index = 0; index < 64; ++index)
        assert(row.candidates[index] == static_cast<int32_t>(10000 + index));
    assert(row.candidates[64] == 11000);

    Row exhausted(4, 4);
    exhausted.mapping = {0, 0, 0, 9};
    Start<Threads>(exhausted, 4);
    OneTile<Threads>(exhausted, 4, 4, {0, 1, 2, 3});
    assert(exhausted.scan[offset + 2] == 2 && exhausted.scan[offset + 3] == 1);
    Commit<Threads>(exhausted, 4, 4, 1);
    assert(exhausted.mapping[3] == 1);
}

template <uint32_t Threads>
void CompareStreamingWithOracle(uint32_t capacity, uint32_t budget, uint32_t seed) {
    Row row(capacity, budget);
    std::mt19937 random(seed);
    for (auto &token : row.mapping) token = static_cast<int32_t>(random() % (budget + 200U));
    std::shuffle(row.lru.begin(), row.lru.end(), random);
    OracleStorage storage(capacity, budget);
    storage.mapping = row.mapping;
    storage.lru = row.lru;
    LIQ950::ResidentPolicy<OracleStorage> oracle;
    oracle.Init(storage, capacity, budget, -1.0F);
    assert(oracle.Reset(7, 128, budget + 256, true));
    Start<Threads>(row, budget, false, 128, budget + 256);
    assert(row.mapping == storage.mapping);
    constexpr uint32_t offset = 2U * (Threads / 32U);
    uint32_t totalMisses = 0;
    for (uint32_t base = 0; base < budget + 128; base += 128) {
        const uint32_t admitted = row.scan[offset + 2];
        const uint32_t count = std::min(128U, budget - admitted);
        if (count == 0) break;
        std::vector<int32_t> tokens(count);
        std::iota(tokens.begin(), tokens.end(), static_cast<int32_t>(base));
        OracleTransfers transfers;
        for (const auto token : tokens) oracle.Consider(token, 0.0F);
        oracle.FlushTile(transfers);
        StreamTile<Threads>(row, capacity, budget, tokens, totalMisses);
        const uint32_t assigned = row.scan[offset + 3];
        assert(assigned == transfers.copies.size());
        assert(row.mapping == storage.mapping);
        for (uint32_t i = 0; i < assigned; ++i) {
            assert(row.pending[128 + i] == transfers.copies[i].first);
            assert(row.pending[256 + i] == transfers.copies[i].second);
            oracle.Commit(transfers.copies[i].second, transfers.copies[i].first);
        }
        Commit<Threads>(row, capacity, budget, assigned);
        assert(row.mapping == storage.mapping);
        totalMisses += assigned;
        for (uint32_t i = 0; i < static_cast<uint32_t>(row.scan[offset + 2]); ++i)
            assert(row.candidates[i] == storage.candidates[i]);
    }
    const auto result = oracle.Finish(true);
    const uint32_t admitted = row.scan[offset + 2];
    for (uint32_t i = admitted; i < std::max(1U, budget); ++i) row.candidates[i] = -1;
    Finish<Threads>(row, capacity, budget, totalMisses);
    assert(result.admitted == static_cast<int32_t>(admitted));
    assert(result.copiedMisses == static_cast<int32_t>(totalMisses));
    assert(row.mapping == storage.mapping);
    assert(row.lru == storage.lru);
    assert(row.candidates == storage.candidates);
}

template <uint32_t Threads>
void TestPartialCommitAndReuse() {
    Row row(8, 4);
    std::fill(row.mapping.begin(), row.mapping.end(), 99);
    Start<Threads>(row, 4);
    OneTile<Threads>(row, 8, 4, {0, 1, 2, 3});
    Commit<Threads>(row, 8, 4, 2);
    assert(row.mapping[0] == 0 && row.mapping[1] == 1);
    assert(row.mapping[2] == -1 && row.mapping[3] == -1);
    Finish<Threads>(row, 8, 4, 4);
    assert(row.lru == std::vector<int32_t>({4, 5, 6, 7, 0, 1, 2, 3}));
    std::fill(row.lru.begin(), row.lru.end(), -2);
    Start<Threads>(row, 4, true);
    assert(row.mapping == std::vector<int32_t>(8, -1));
    assert(row.lru == std::vector<int32_t>({0, 1, 2, 3, 4, 5, 6, 7}));
    assert(row.scan[2U * (Threads / 32U) + 1] == 0);
}

}

int main() {
    CompareStreamingWithOracle<32>(67, 65, 17);
    CompareStreamingWithOracle<64>(2053, 129, 29);
    CompareStreamingWithOracle<64>(4099, 2048, 71);
    TestPartialCommitAndReuse<32>();
    TestPartialCommitAndReuse<64>();
    CompareWithOracle<32>();
    CompareWithOracle<64>();
    TestPublishAfterCommit<32>();
    TestPublishAfterCommit<64>();
    TestDuplicateHitAndCrossTileProtection<32>();
    TestDuplicateHitAndCrossTileProtection<64>();
    TestBudgetAndInvalidMetadata<32>();
    TestBudgetAndInvalidMetadata<64>();
    TestStableTailAndIndependentRows<32>();
    TestStableTailAndIndependentRows<64>();
    TestWideStreamingAndExhaustion<32>();
    TestWideStreamingAndExhaustion<64>();
}
