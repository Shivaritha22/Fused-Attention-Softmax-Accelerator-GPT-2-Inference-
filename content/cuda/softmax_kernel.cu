#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

/**
 * Fused softmax kernel: scale + causal mask + softmax in one pass
 * 
 * Kernel design:
 * - Each thread handles one or more elements in a row
 * - Warp-level reductions for max and sum
 * - Numerically stable (subtract row max before exp)
 * - Supports causal masking
 */

// Forward declaration
template<int VEC_SIZE>
__device__ void load_vectorized(float* dst, const float* src, int idx, int n);

template<int VEC_SIZE>
__device__ void store_vectorized(float* dst, const float* src, int idx, int n);

// Specialization for VEC_SIZE=1 (scalar)
template<>
__device__ void load_vectorized<1>(float* dst, const float* src, int idx, int n) {
    if (idx < n) dst[0] = src[idx];
}

template<>
__device__ void store_vectorized<1>(float* dst, const float* src, int idx, int n) {
    if (idx < n) dst[idx] = src[0];
}

// Specialization for VEC_SIZE=4 (float4)
template<>
__device__ void load_vectorized<4>(float* dst, const float* src, int idx, int n) {
    if (idx + 3 < n) {
        float4 vec = *reinterpret_cast<const float4*>(&src[idx]);
        dst[0] = vec.x; dst[1] = vec.y; dst[2] = vec.z; dst[3] = vec.w;
    } else {
        // Handle remainder
        for (int i = 0; i < 4 && idx + i < n; i++) {
            dst[i] = src[idx + i];
        }
    }
}

template<>
__device__ void store_vectorized<4>(float* dst, const float* src, int idx, int n) {
    if (idx + 3 < n) {
        float4 vec;
        vec.x = src[0]; vec.y = src[1]; vec.z = src[2]; vec.w = src[3];
        *reinterpret_cast<float4*>(&dst[idx]) = vec;
    } else {
        // Handle remainder
        for (int i = 0; i < 4 && idx + i < n; i++) {
            dst[idx + i] = src[i];
        }
    }
}

/**
 * Fused softmax kernel v0: stable softmax (FP32)
 * 
 * @param output Output tensor [batch, seq_len]
 * @param input Input tensor [batch, seq_len]
 * @param scale Scale factor (typically 1/sqrt(d_k))
 * @param batch_size Batch dimension
 * @param seq_len Sequence length
 * @param causal_mask If true, apply causal masking
 */
template<int VEC_SIZE, bool CAUSAL_MASK>
__global__ void fused_softmax_kernel_v0(
    float* output,
    const float* input,
    float scale,
    int batch_size,
    int seq_len
) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int vec_tid = tid * VEC_SIZE;
    
    if (batch_idx >= batch_size) return;
    
    const float* row_input = input + batch_idx * seq_len;
    float* row_output = output + batch_idx * seq_len;
    
    // Shared memory for row max and sum (one per warp)
    __shared__ float s_max[32];  // Max 32 warps per block
    __shared__ float s_sum[32];
    
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Step 1: Load, scale, apply mask, and find row max
    float local_max = -INFINITY;
    float local_vals[VEC_SIZE];
    
    for (int i = vec_tid; i < seq_len; i += blockDim.x * VEC_SIZE) {
        load_vectorized<VEC_SIZE>(local_vals, row_input, i, seq_len);
        
        for (int v = 0; v < VEC_SIZE && i + v < seq_len; v++) {
            float val = local_vals[v] * scale;
            
            // Apply causal mask
            if (CAUSAL_MASK) {
                int pos_in_seq = i + v;
                int pos_in_batch = batch_idx;
                // For causal mask: if col >= row, set to -inf
                // Assuming row-major: row = batch_idx, col = pos_in_seq
                // For attention: if query_pos >= key_pos, mask
                // Simplified: mask if col_idx >= row_idx
                if (pos_in_seq > pos_in_batch) {  // This will be adjusted for proper attention
                    val = -INFINITY;
                }
            }
            
            local_vals[v] = val;
            local_max = fmaxf(local_max, val);
        }
    }
    
    // Warp-level max reduction
    local_max = warp_reduce_max(local_max);
    
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
    }
    __syncthreads();
    
    // Block-level max reduction
    if (warp_id == 0) {
        float block_max = (lane_id < blockDim.x / WARP_SIZE) ? s_max[lane_id] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane_id == 0) {
            s_max[0] = block_max;
        }
    }
    __syncthreads();
    
    float row_max = s_max[0];
    
    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    
    for (int i = vec_tid; i < seq_len; i += blockDim.x * VEC_SIZE) {
        for (int v = 0; v < VEC_SIZE && i + v < seq_len; v++) {
            float val = local_vals[v];
            float exp_val = expf(val - row_max);
            local_vals[v] = exp_val;
            local_sum += exp_val;
        }
    }
    
    // Warp-level sum reduction
    local_sum = warp_reduce_sum(local_sum);
    
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Block-level sum reduction
    if (warp_id == 0) {
        float block_sum = (lane_id < blockDim.x / WARP_SIZE) ? s_sum[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane_id == 0) {
            s_sum[0] = block_sum;
        }
    }
    __syncthreads();
    
    float row_sum = s_sum[0];
    
    // Step 3: Normalize and write output
    for (int i = vec_tid; i < seq_len; i += blockDim.x * VEC_SIZE) {
        for (int v = 0; v < VEC_SIZE && i + v < seq_len; v++) {
            local_vals[v] /= row_sum;
        }
        store_vectorized<VEC_SIZE>(row_output, local_vals, i, seq_len);
    }
}

/**
 * Fused softmax kernel v1: Proper causal mask for attention
 * 
 * For attention matrices, causal mask means: mask[i][j] = (j > i) ? -inf : 0
 * This kernel handles 2D attention scores [batch, seq_len, seq_len]
 */
template<int VEC_SIZE>
__global__ void fused_softmax_kernel_v1_attention(
    float* output,
    const float* input,
    float scale,
    int batch_size,
    int seq_len,
    bool causal_mask
) {
    const int batch_idx = blockIdx.x;
    const int row_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int vec_tid = tid * VEC_SIZE;
    
    if (batch_idx >= batch_size || row_idx >= seq_len) return;
    
    const float* row_input = input + (batch_idx * seq_len + row_idx) * seq_len;
    float* row_output = output + (batch_idx * seq_len + row_idx) * seq_len;
    
    __shared__ float s_max[32];
    __shared__ float s_sum[32];
    
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Step 1: Load, scale, apply causal mask, find max
    float local_max = -INFINITY;
    float local_vals[VEC_SIZE];
    
    for (int col_idx = vec_tid; col_idx < seq_len; col_idx += blockDim.x * VEC_SIZE) {
        load_vectorized<VEC_SIZE>(local_vals, row_input, col_idx, seq_len);
        
        for (int v = 0; v < VEC_SIZE && col_idx + v < seq_len; v++) {
            float val = local_vals[v] * scale;
            
            // Causal mask: mask positions where key_pos > query_pos
            if (causal_mask && (col_idx + v) > row_idx) {
                val = -INFINITY;
            }
            
            local_vals[v] = val;
            local_max = fmaxf(local_max, val);
        }
    }
    
    // Reductions (same as v0)
    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) {
        s_max[warp_id] = local_max;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float block_max = (lane_id < blockDim.x / WARP_SIZE) ? s_max[lane_id] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if (lane_id == 0) {
            s_max[0] = block_max;
        }
    }
    __syncthreads();
    
    float row_max = s_max[0];
    
    // Step 2: Compute exp and sum
    float local_sum = 0.0f;
    
    for (int col_idx = vec_tid; col_idx < seq_len; col_idx += blockDim.x * VEC_SIZE) {
        for (int v = 0; v < VEC_SIZE && col_idx + v < seq_len; v++) {
            float val = local_vals[v];
            float exp_val = expf(val - row_max);
            local_vals[v] = exp_val;
            local_sum += exp_val;
        }
    }
    
    local_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float block_sum = (lane_id < blockDim.x / WARP_SIZE) ? s_sum[lane_id] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane_id == 0) {
            s_sum[0] = block_sum;
        }
    }
    __syncthreads();
    
    float row_sum = s_sum[0];
    
    // Step 3: Normalize and write
    for (int col_idx = vec_tid; col_idx < seq_len; col_idx += blockDim.x * VEC_SIZE) {
        for (int v = 0; v < VEC_SIZE && col_idx + v < seq_len; v++) {
            local_vals[v] /= row_sum;
        }
        store_vectorized<VEC_SIZE>(row_output, local_vals, col_idx, seq_len);
    }
}

// C wrapper function to launch the kernel
// Using extern "C" to ensure proper linking between .cu and .cpp files
extern "C" {
    void launch_fused_softmax_kernel(
        float* output,
        const float* input,
        float scale,
        int batch_size,
        int seq_len,
        bool causal_mask,
        int block_dim,
        cudaStream_t stream
    ) {
        // Launch one block per row of attention matrix
        dim3 grid(batch_size, seq_len);
        dim3 block(block_dim);
        
        // Use VEC_SIZE=1 for now (can be extended to 4 later)
        fused_softmax_kernel_v1_attention<1><<<grid, block, 0, stream>>>(
            output, input, scale, batch_size, seq_len, causal_mask
        );
    }
}

