#ifndef LIQ_HOST_SIMT_ASC_SIMT_H_
#define LIQ_HOST_SIMT_ASC_SIMT_H_

#include "host_simt_runtime.h"

#define __aicore__
#define __simt_callee__
#define __ubuf__
#define __gm__

inline void asc_syncthreads() { liq_host_simt::barrier.Wait(); }

inline int32_t asc_shfl_up(int32_t value, uint32_t delta, uint32_t width) {
    liq_host_simt::ShuffleWrite(value);
    const uint32_t lane = liq_host_simt::thread % width;
    return lane >= delta ? liq_host_simt::ShuffleRead(liq_host_simt::thread - delta) :
                            liq_host_simt::ShuffleRead(liq_host_simt::thread);
}

#endif
