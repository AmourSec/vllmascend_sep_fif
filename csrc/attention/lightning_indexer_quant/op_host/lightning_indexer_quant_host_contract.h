#ifndef LIGHTNING_INDEXER_QUANT_HOST_CONTRACT_H
#define LIGHTNING_INDEXER_QUANT_HOST_CONTRACT_H
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace LIQHost {
enum class Mode { Invalid, Legacy, Standalone, Resident };
inline Mode ClassifyMode(bool addresses, bool threshold, unsigned residentInputs)
{
    if (residentInputs != 0) {
        return residentInputs == 11 && !addresses && threshold ? Mode::Resident : Mode::Invalid;
    }
    if (addresses || threshold) {
        return addresses && threshold ? Mode::Standalone : Mode::Invalid;
    }
    return Mode::Legacy;
}
inline bool FitsIndex(int64_t value)
{
    return value > 0 && value <= std::numeric_limits<int32_t>::max();
}
inline bool Add(size_t left, size_t right, size_t &result)
{
    if (left > std::numeric_limits<size_t>::max() - right) return false;
    result = left + right;
    return true;
}
inline bool Multiply(size_t left, size_t right, size_t &result)
{
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right) return false;
    result = left * right;
    return true;
}
inline bool Align(size_t value, size_t alignment, size_t &result)
{
    if (!Add(value, alignment - 1, result)) return false;
    result &= ~(alignment - 1);
    return true;
}
inline bool TensorSpan(size_t rows, size_t columns, size_t bytes)
{
    size_t span = 0;
    return Multiply(rows, columns, span) && Multiply(span, bytes, span) &&
           span <= static_cast<size_t>(std::numeric_limits<int64_t>::max());
}
inline bool WorkspaceBytes(Mode mode, size_t rows, size_t capacity, size_t rowBytes,
                           size_t budget, size_t runtime, size_t &result)
{
    size_t stride = 0;
    if (mode == Mode::Resident) {
        size_t elements = 0;
        if (!Multiply(capacity, 4, elements) ||
            !Add(elements, std::max<size_t>(1, budget), elements) ||
            !Add(elements, 3 * 256, elements) ||
            !Multiply(elements, sizeof(int32_t), stride)) {
            return false;
        }
    } else if (mode == Mode::Standalone) {
        if (!Align(rowBytes, 32, stride) || !Multiply(stride, std::max<size_t>(1, budget), stride)) return false;
    } else {
        return false;
    }
    return Align(stride, 512, stride) && Multiply(rows, stride, result) &&
           Add(result, 512, result) && Add(result, runtime, result);
}
// Mirrors the AIV TPipe allocations: score/scale, streaming TopK, then prefetch.
// Capacity-dependent resident arrays live in GM and do not consume UB.
inline size_t UbBytes(size_t sparse, size_t budget, size_t rowBytes, bool resident)
{
    const auto align = [](size_t value, size_t unit) { return (value + unit - 1) / unit * unit; };
    const size_t score = 65536 + 256 * sizeof(float) + 512;
    const size_t topk = 2 * align(sparse + 4096, 256) * 4 +
                        5 * align(sparse + 64, 256) * 4 + 5 * 256 * 4 + 64 * 4;
    const size_t prefetch = 4096 + align(std::max<size_t>(1, rowBytes), 32) +
                            align(std::max<size_t>(1, budget) * 4, 32) + align(sparse * 4, 32) + 32;
    return score + topk + (resident ? 8192 + align(std::max<size_t>(1, budget) * 4, 32) +
                                      align(sparse * 4, 32) + 32 + 1536 + 544 : prefetch);
}
}
#endif
