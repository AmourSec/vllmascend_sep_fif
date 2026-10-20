#ifndef LIGHTNING_INDEXER_QUANT_QUERY_ROWS_H
#define LIGHTNING_INDEXER_QUANT_QUERY_ROWS_H

#include <cstdint>

#if defined(__CCE_AICORE__)
#define LIQ_QUERY_INLINE __aicore__ inline
#else
#define LIQ_QUERY_INLINE inline
#endif

namespace LIQ950 {
struct QueryRow {
    uint32_t request = 0;
    uint32_t visible = 0;
};

template <typename QueryLengths, typename KeyLengths>
LIQ_QUERY_INLINE bool ResolveQueryRow(uint32_t row, uint32_t requests, uint32_t queryRows,
    uint32_t activeRows, uint32_t maxKeyLength, const QueryLengths &queryLengths,
    const KeyLengths &keyLengths, QueryRow &result)
{
    result = {};
    if (requests == 0 || activeRows > queryRows || row >= activeRows) {
        return false;
    }
    int64_t begin = 0;
    int64_t selectedBegin = 0;
    int64_t selectedEnd = 0;
    bool found = false;
    for (uint32_t request = 0; request < requests; ++request) {
        int64_t end = queryLengths.GetValue(request);
        if (end < begin || end > activeRows) {
            return false;
        }
        if (row >= begin && row < end) {
            result.request = request;
            selectedBegin = begin;
            selectedEnd = end;
            found = true;
        }
        begin = end;
    }
    if (!found || begin != activeRows) {
        return false;
    }
    int64_t finalLength = keyLengths.GetValue(result.request);
    int64_t queryCount = selectedEnd - selectedBegin;
    if (finalLength < 0 || finalLength > maxKeyLength ||
        (finalLength < queryCount && !(finalLength == 0 && queryCount == 1))) {
        return false;
    }
    int64_t visible = finalLength - queryCount + (row - selectedBegin) + 1;
    if (visible < 0 || visible > maxKeyLength) {
        return false;
    }
    result.visible = static_cast<uint32_t>(visible);
    return true;
}
}
#undef LIQ_QUERY_INLINE
#endif
