#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Common utilities and constants

#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024

// Helper for warp reductions
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

#endif // COMMON_CUH

