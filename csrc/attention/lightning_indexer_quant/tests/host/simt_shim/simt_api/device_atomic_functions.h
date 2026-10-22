#ifndef LIQ_HOST_SIMT_DEVICE_ATOMIC_FUNCTIONS_H_
#define LIQ_HOST_SIMT_DEVICE_ATOMIC_FUNCTIONS_H_

#include "host_simt_runtime.h"

inline int32_t asc_atomic_cas(int32_t *address, int32_t expected, int32_t desired) {
    std::lock_guard<std::mutex> lock(liq_host_simt::atomicMutex);
    const int32_t old = *address;
    if (old == expected) *address = desired;
    return old;
}

inline int32_t asc_atomic_min(int32_t *address, int32_t value) {
    std::lock_guard<std::mutex> lock(liq_host_simt::atomicMutex);
    const int32_t old = *address;
    if (value < *address) *address = value;
    return old;
}

inline int32_t asc_atomic_max(int32_t *address, int32_t value) {
    std::lock_guard<std::mutex> lock(liq_host_simt::atomicMutex);
    const int32_t old = *address;
    if (value > *address) *address = value;
    return old;
}

#endif
