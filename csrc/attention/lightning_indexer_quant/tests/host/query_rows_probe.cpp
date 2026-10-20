#include <cassert>
#include <cstdint>
#include <vector>

#include "../../op_kernel/arch35/lightning_indexer_quant_query_rows.h"

struct Lengths {
    std::vector<int32_t> values;
    int32_t GetValue(uint32_t index) const { return values.at(index); }
};

int main()
{
    LIQ950::QueryRow result{};
    const Lengths query{{2, 5, 6}}, key{{10, 20, 3}};
    const uint32_t requests[] = {0, 0, 1, 1, 1, 2};
    const uint32_t visible[] = {9, 10, 18, 19, 20, 3};
    for (uint32_t row = 0; row < 6; ++row) {
        assert(LIQ950::ResolveQueryRow(row, 3, 6, 6, 32, query, key, result));
        assert(result.request == requests[row]);
        assert(result.visible == visible[row]);
    }

    const Lengths emptyRequests{{0, 2, 2, 3}}, emptyKeys{{0, 6, 0, 4}};
    assert(LIQ950::ResolveQueryRow(0, 4, 5, 3, 32, emptyRequests, emptyKeys, result));
    assert(result.request == 1 && result.visible == 5);
    assert(LIQ950::ResolveQueryRow(2, 4, 5, 3, 32, emptyRequests, emptyKeys, result));
    assert(result.request == 3 && result.visible == 4);
    assert(!LIQ950::ResolveQueryRow(3, 4, 5, 3, 32, emptyRequests, emptyKeys, result));
    assert(!LIQ950::ResolveQueryRow(0, 4, 2, 3, 32, emptyRequests, emptyKeys, result));
    assert(!LIQ950::ResolveQueryRow(0, 4, 5, 4, 32, emptyRequests, emptyKeys, result));

    const Lengths single{{1}}, zero{{0}}, tooLong{{33}}, negative{{-1}};
    assert(LIQ950::ResolveQueryRow(0, 1, 1, 1, 32, single, zero, result));
    assert(result.visible == 0);
    assert(!LIQ950::ResolveQueryRow(0, 1, 1, 1, 32, single, tooLong, result));
    assert(!LIQ950::ResolveQueryRow(0, 1, 1, 1, 32, single, negative, result));
    assert(!LIQ950::ResolveQueryRow(0, 1, 1, 1, 32, negative, single, result));
    assert(!LIQ950::ResolveQueryRow(0, 0, 1, 1, 32, single, single, result));
    assert(!LIQ950::ResolveQueryRow(1, 1, 1, 1, 32, single, single, result));

    const Lengths decreasing{{2, 1, 3}}, shortKeys{{1, 2, 3}};
    assert(!LIQ950::ResolveQueryRow(0, 3, 3, 3, 32, decreasing, shortKeys, result));
    assert(!LIQ950::ResolveQueryRow(0, 3, 6, 6, 32, query, shortKeys, result));
    return 0;
}
