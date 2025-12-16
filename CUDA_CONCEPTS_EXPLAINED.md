 CUDA Concepts Explained: Blocks, Warp Reductions, and Vectorized Loads

 Question 1: How Many CUDA Blocks Are Used?

 Answer: We define it! It's batch_size × seq_len blocks

Looking at the kernel launch code (lines 304-306):

cpp
dim3 grid(batch_size, seq_len);  // Grid dimensions
dim3 block(block_dim);            // Block dimensions (e.g., 256)


 Block Count Calculation

Number of blocks = batch_size × seq_len

Example:
- Input tensor: [batch_size=4, seq_len=1024, seq_len=1024]
- Number of blocks: 4 × 1024 = 4,096 blocks

 Can We Define It?

Yes! The programmer defines the number of blocks. It's not automatically determined by the GPU.

In this code:
- We launch one block per row of the attention matrix
- Each block processes one row independently
- Total blocks = number of rows = batch_size × seq_len

 Does It Vary Based on GPU?

Partially, but not in the way you might think:

1. We define the grid size (number of blocks) - this is fixed by our code
2. GPU limits determine if it can handle our grid:
   - Maximum blocks per grid dimension: varies by GPU (typically 65,535 per dimension)
   - Maximum total blocks: 65535 × 65535 × 65535 (for 3D grids)
   - Our 2D grid: batch_size × seq_len must be ≤ 65,535 × 65,535

3. GPU execution:
   - GPU has Streaming Multiprocessors (SMs) - typically 20-108 SMs per GPU
   - Each SM can run multiple blocks simultaneously
   - GPU automatically schedules blocks to available SMs
   - If we launch 4,096 blocks but GPU has 80 SMs, blocks execute in waves

 Example: GPU Execution


Our launch: 4,096 blocks, 256 threads per block
GPU: NVIDIA A100 with 108 SMs

Execution:
- Each SM can run multiple blocks (limited by registers/shared memory)
- GPU scheduler distributes blocks to SMs
- Blocks execute in waves until all 4,096 complete
- All blocks run the same kernel code, but on different data


 Block Size (Threads Per Block)

We also define this:
- block_dim parameter (default 256 in the code)
- Can be adjusted: 128, 256, 512, 1024 (max)
- Must be multiple of 32 (warp size)

Why 256?
- Good balance between parallelism and resource usage
- 256 threads = 8 warps per block
- Leaves room for shared memory and registers



 Question 2: What Are Warp Reductions? Have We Done That?

 Answer: Yes! We use warp reductions extensively for max and sum

 What is a Warp?

Warp = 32 threads that execute in lockstep

- GPU threads are organized into warps (groups of 32)
- Threads in a warp execute the same instruction simultaneously
- Warp is the fundamental unit of execution on GPU

 What is a Warp Reduction?

Reduction = combining values from multiple threads into one result

Examples:
- Sum reduction: Add up values from all 32 threads → one sum
- Max reduction: Find maximum value across all 32 threads → one max

 Our Implementation (in common.cuh)

cpp
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


 How It Works: Binary Tree Reduction

Visual Example (8 threads, finding sum):


Initial: Each thread has a value
Thread:  0    1    2    3    4    5    6    7
Value:   1    2    3    4    5    6    7    8

Step 1 (offset=4): Thread i gets value from thread i+4
Thread 0: 1 + 5 = 6
Thread 1: 2 + 6 = 8
Thread 2: 3 + 7 = 10
Thread 3: 4 + 8 = 12
Thread 4-7: (not used)

Step 2 (offset=2): Thread i gets value from thread i+2
Thread 0: 6 + 10 = 16
Thread 1: 8 + 12 = 20
Thread 2-7: (not used)

Step 3 (offset=1): Thread i gets value from thread i+1
Thread 0: 16 + 20 = 36  ← Final sum!
Thread 1-7: (not used)


Key instruction: __shfl_down_sync
- Shuffle instruction: threads can read values from other threads in the warp
- No shared memory needed! Direct register-to-register communication
- Very fast: Single cycle operation

 Where We Use Warp Reductions

In the softmax kernel, we use them in 4 places:

 1. Max Reduction (Step 1) - Line 125
cpp
// Each thread has local_max (max value it saw)
local_max = warp_reduce_max(local_max);
// After this, thread 0 in the warp has the max of all threads in warp


 2. Block-Level Max Reduction (Step 1) - Line 135
cpp
// Combine max values from different warps
if (warp_id == 0) {
    float block_max = ...;
    block_max = warp_reduce_max(block_max);
    // Thread 0 now has max of entire block
}


 3. Sum Reduction (Step 2) - Line 157
cpp
// Each thread has local_sum (sum of exp values it computed)
local_sum = warp_reduce_sum(local_sum);
// After this, thread 0 has sum of all threads in warp


 4. Block-Level Sum Reduction (Step 2) - Line 167
cpp
// Combine sum values from different warps
if (warp_id == 0) {
    float block_sum = ...;
    block_sum = warp_reduce_sum(block_sum);
    // Thread 0 now has sum of entire block
}


 Why Use Warp Reductions?

Advantages:
1. Fast: Uses shuffle instructions (register-to-register), no memory access
2. Efficient: No shared memory needed for warp-level reductions
3. Low latency: Single cycle operations

Alternative (slower):
cpp
// Without warp reduction (slower):
__shared__ float s_data[256];
s_data[tid] = local_max;
__syncthreads();
// Then manually reduce in shared memory (many memory accesses)


 Two-Stage Reduction Pattern

We use a two-stage reduction:

1. Stage 1: Warp reduction (fast, uses shuffle)
   - Each warp reduces its 32 values → 1 value
   - Result stored in shared memory by warp leader

2. Stage 2: Block reduction (combines warps)
   - Warp 0 collects results from all warps
   - Uses another warp reduction to combine them
   - Final result in shared memory

Why two stages?
- Blocks can have multiple warps (e.g., 256 threads = 8 warps)
- Warp reduction only works within a warp (32 threads)
- Need second stage to combine results from multiple warps



 Question 3: What Are Vectorized Loads?

 Answer: Loading multiple elements in one memory transaction

 Scalar Load (Normal)

cpp
float val1 = src[0];  // One memory transaction
float val2 = src[1];  // Another memory transaction
float val3 = src[2];  // Another memory transaction
float val4 = src[3];  // Another memory transaction


4 separate memory transactions to load 4 floats.

 Vectorized Load (Optimized)

cpp
float4 vec = *reinterpret_cast<const float4*>(&src[0]);
// One memory transaction loads 4 floats!
float val1 = vec.x;
float val2 = vec.y;
float val3 = vec.z;
float val4 = vec.w;


1 memory transaction to load 4 floats!

 Our Implementation

In softmax_kernel.cu (lines 36-46):

cpp
template<>
__device__ void load_vectorized<4>(float* dst, const float* src, int idx, int n) {
    if (idx + 3 < n) {
        // Vectorized load: 4 floats in one transaction
        float4 vec = *reinterpret_cast<const float4*>(&src[idx]);
        dst[0] = vec.x; 
        dst[1] = vec.y; 
        dst[2] = vec.z; 
        dst[3] = vec.w;
    } else {
        // Handle remainder (if less than 4 elements left)
        for (int i = 0; i < 4 && idx + i < n; i++) {
            dst[i] = src[idx + i];
        }
    }
}


 How It Works

1. float4 type: CUDA built-in type for 4 floats
   cpp
   struct float4 {
       float x, y, z, w;
   };
   

2. Type casting: reinterpret_cast treats the memory address as a float4*
   - Memory must be aligned (address divisible by 16 bytes for float4)
   - GPU memory is typically aligned, so this is safe

3. Single transaction: GPU loads 16 bytes (4 × 4 bytes) in one go

 Benefits

Performance improvement:
- Fewer memory transactions: 4x reduction in transaction count
- Better memory bandwidth utilization: GPU memory bus is optimized for wide loads
- Reduced overhead: Less address calculation and transaction setup

Example:
- Loading 1024 floats:
  - Scalar: 1024 transactions
  - Vectorized (VEC_SIZE=4): 256 transactions
  - 4x fewer transactions!

 When Vectorized Loads Work Best

Requirements:
1. Alignment: Memory address must be aligned (divisible by 16 for float4)
2. Contiguous data: Elements must be consecutive in memory
3. Multiple of vector size: Best when number of elements is multiple of 4

Our code handles edge cases:
cpp
if (idx + 3 < n) {
    // Safe to use vectorized load
    float4 vec = *reinterpret_cast<const float4*>(&src[idx]);
} else {
    // Fallback to scalar for remainder
    for (int i = 0; i < 4 && idx + i < n; i++) {
        dst[i] = src[idx + i];
    }
}


 Vectorized Store (Same Concept)

Lines 49-59: We also have vectorized stores:

cpp
template<>
__device__ void store_vectorized<4>(float* dst, const float* src, int idx, int n) {
    if (idx + 3 < n) {
        float4 vec;
        vec.x = src[0]; vec.y = src[1]; vec.z = src[2]; vec.w = src[3];
        *reinterpret_cast<float4*>(&dst[idx]) = vec;  // One transaction!
    } else {
        // Scalar fallback
    }
}


 Current Usage in Code

Note: The kernel currently uses VEC_SIZE=1 (line 309):
cpp
fused_softmax_kernel_v1_attention<1><<<grid, block, 0, stream>>>


This means scalar loads are used. To enable vectorized loads, change to:
cpp
fused_softmax_kernel_v1_attention<4><<<grid, block, 0, stream>>>


Why not use VEC_SIZE=4?
- May require additional alignment checks
- Code is written to support it, but default is conservative (VEC_SIZE=1)



 Summary

1. CUDA Blocks: We define batch_size × seq_len blocks. GPU schedules them across SMs. Not automatically determined by GPU.

2. Warp Reductions: Yes! We use them for max and sum reductions. Fast because they use shuffle instructions (no shared memory). Used in 4 places in the kernel.

3. Vectorized Loads: Yes! We have the code for it (load_vectorized<4>), but currently use VEC_SIZE=1. Vectorized loads use float4 to load 4 floats in one memory transaction, reducing memory traffic.

