/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0. */
#ifndef LIGHTNING_INDEXER_QUANT_RESIDENT_SOURCE_H_
#define LIGHTNING_INDEXER_QUANT_RESIDENT_SOURCE_H_
#include <cstdint>
namespace LIQ950 {
#if defined(__CCE_AICORE__)
__aicore__
#endif
inline bool ResidentSourceAddress(uint64_t base, uint32_t block, uint32_t offset,
                                  uint32_t blockSize, uint32_t bytes, uint64_t &address)
{
    if (base == 0 || blockSize == 0 || offset >= blockSize || bytes == 0 || bytes > 4096) return false;
    const uint64_t limit = ~uint64_t{0};
    const uint64_t linear = static_cast<uint64_t>(block) * blockSize + offset;
    if (base > limit - (bytes - 1) || linear > (limit - base - (bytes - 1)) / bytes) return false;
    address = base + linear * bytes;
    return true;
}
} // namespace LIQ950
#endif
