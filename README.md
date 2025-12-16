# Fused-Attention-Softmax-Accelerator-GPT-2-Inference-
Custom CUDA C++ kernel that fuses scale, causal mask, and softmax into a single pass to cut global memory traffic and speed up GPT-2 inference. Algorithmic outputs stay identical; only latency and tokens/sec change.


 Fused Softmax CUDA Kernel for GPT-2 Attention

 Project Overview

One-line summary: Build a CUDA kernel that fuses scale + causal mask + softmax into one pass, then show speedups on a microbenchmark and on GPT-2 text generation.

What we're proving: Not "better text," but better engineering. This project optimizes a hot path in transformers (attention softmax) by making it use the GPU's memory system efficiently. We cut global memory trips from ~4 passes to 1, so a bandwidth-bound step runs meaningfully faster.

Why it matters:
- Demonstrates CUDA skills: warp reductions, memory coalescing, vectorization, launch configuration
- Shows HPC thinking: bytes moved → GB/s → latency; p50/p5/p95 percentiles; roofline analysis
- Produces a reusable PyTorch op that can drop into any attention layer, not just GPT-2
- Clear resume metric: "optimized attention softmax; +X% tokens/sec on GPT-2"

Why GPT-2:
- Standard, easy-to-run transformer
- Causal masking matches the fused optimization path
- Tokens/sec is a clear "before vs after" metric that anyone can understand
- We're not changing what GPT-2 says; we're making it say it faster

 High-Level Tasks

1. Baseline GPT-2 inference and timing - Establish performance baseline
2. Microbenchmark for the unfused softmax chain - Measure current performance
3. PyTorch CUDA extension skeleton that compiles - Set up the extension framework
4. CUDA kernel v0: stable softmax (FP32), correctness tests - Basic kernel with numerical stability
5. CUDA kernel v1: fuse scale + causal mask, retest - Add fusion optimizations
6. Integrate the fused kernel into GPT-2 attention - Replace PyTorch operations
7. Measure and report: p50 latency, speedup, GB/s, bytes moved, tokens/sec - Performance analysis

 Requirements

- Google Colab (with GPU runtime)
- PyTorch
- CUDA toolkit
- GPT-2 model (Hugging Face transformers)

 Setup

This project is designed to run on Google Colab with GPU runtime enabled.

 Implementation Plan

 Phase 1: Baseline & Microbenchmark
- Load GPT-2 model
- Implement timing for attention operations
- Create microbenchmark: synthetic N×N tensors on GPU (no dataset), measure latency/bandwidth

 Phase 2: CUDA Extension Setup
- Create PyTorch CUDA extension structure
- Set up compilation pipeline for Colab
- Verify extension compiles and loads

 Phase 3: CUDA Kernel Development
- v0: Implement numerically stable softmax kernel (FP32)
- Add correctness tests against PyTorch
- v1: Fuse scale and causal mask operations
- Retest correctness and performance

 Phase 4: Integration & Benchmarking
- Integrate fused kernel into GPT-2 attention module
- Run comprehensive benchmarks
- Generate performance report with metrics

 Deliverables

1. Fused CUDA kernel - Scale + causal mask + softmax in one pass
2. Microbenchmark results - Synthetic N×N tensors on GPU (no dataset), measure latency, speedup, GB/s, and memory comparison table (4-DRAM vs 1-DRAM bytes)
3. GPT-2 demo - Real prompts/dataset, same outputs, higher tokens/sec and lower per-token latency
4. Performance report - Key results and how to reproduce

 Expected Metrics

- Latency: p50 (median), p5, p95 percentiles
- Speedup: vs PyTorch baseline
- Memory bandwidth: GB/s achieved
- Bytes moved: Comparison of unfused (4 passes) vs fused (1 pass)
- Throughput: Tokens/sec improvement

 Key Optimization

The unfused PyTorch path typically does:
1. Scale (multiply by 1/√d_k)
2. Causal mask (set future positions to -inf)
3. Softmax (with max reduction for stability)
4. Write output

Each step reads/writes from global memory. The fused kernel does all of this in a single pass, dramatically reducing memory traffic for this bandwidth-bound operation.

 Usage

 Google Colab Setup

1. Upload project files to Colab:
   - Upload the entire project structure to /content/ in Colab
   - Or clone from repository if hosted on GitHub

2. Open the notebook:
   - Open gpt2_fused_softmax.ipynb in Colab
   - Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)

3. Run cells sequentially:
   - The notebook will:
     - Check GPU availability
     - Install dependencies
     - Compile CUDA extension
     - Run correctness tests
     - Execute microbenchmark
     - Run GPT-2 benchmark

 Project Structure


/content/
  cuda/
    softmax_kernel.cu       CUDA kernel implementation
    softmax_binding.cpp      PyTorch C++ binding
    common.cuh               Common utilities
  python/
    baseline_softmax.py      Unfused PyTorch baseline
    fused_softmax.py         CUDA extension wrapper
    bench_micro.py           Microbenchmark harness
    bench_gpt2.py            GPT-2 tokens/sec benchmark
gpt2_fused_softmax.ipynb     Main Colab notebook


 Configuration

Key parameters (defined in benchmark files):
- GPT2_SIZE = "distilgpt2" - Fast and small model
- SEQ_LENGTHS = [512, 1024] - Sequence lengths to test
- BATCH_SIZES = [1] - Batch sizes
- KERNEL_BLOCKDIM = 256 - CUDA block dimension
- WARMUP_KERNEL = 5, RUNS_KERNEL = 100 - Microbench iterations
- WARMUP_GPT2 = 1, RUNS_GPT2 = 3 - GPT-2 benchmark iterations

 Running Benchmarks

Microbenchmark:
python
from bench_micro import run_microbenchmark
results = run_microbenchmark()


GPT-2 Benchmark:
python
from bench_gpt2 import run_gpt2_benchmark
gpt2_results = run_gpt2_benchmark()


Expected Output:
three txt files

The benchmarks will produce:
- Latency statistics (p5, p50, p95)
- Memory bandwidth (GB/s)
- Speedup vs baseline
- Memory comparison table (4 passes → 1 pass)
- Tokens/sec improvement for GPT-2

