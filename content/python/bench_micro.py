"""
Microbenchmark harness for fused softmax kernel
Creates synthetic N×N tensors on GPU and measures performance
"""

import torch
import time
import numpy as np
from typing import List, Dict
import statistics

# Configuration
GPT2_SIZE = "distilgpt2"
SEQ_LENGTHS = [512, 1024]  # add 2048 only if GPU allows
BATCH_SIZES = [1]  # optionally [1, 2]
DTYPE = "fp32"
MASKS_MICRO = ["none", "causal"]
WARMUP_KERNEL = 5
RUNS_KERNEL = 100
KERNEL_BLOCKDIM = 256
KERNEL_VEC = 1


def percentile(data, p):
    """Compute percentile"""
    return np.percentile(data, p)


def benchmark_kernel(
    batch_size: int,
    seq_len: int,
    causal_mask: bool,
    use_fused: bool = True,
    block_dim: int = 256,
    warmup: int = 5,
    runs: int = 100
) -> Dict:
    """
    Benchmark a single kernel configuration
    
    Returns:
        Dictionary with timing and performance metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA not available")
    
    # Create synthetic tensor
    input_tensor = torch.randn(
        batch_size, seq_len, seq_len,
        device=device,
        dtype=torch.float32
    )
    
    scale = 1.0 / np.sqrt(seq_len)  # Typical attention scale
    
    # Import implementations
    if use_fused:
        try:
            from fused_softmax import fused_softmax_forward
        except ImportError:
            from .fused_softmax import fused_softmax_forward
        fn = lambda: fused_softmax_forward(input_tensor, scale, causal_mask, block_dim)
    else:
        try:
            from baseline_softmax import baseline_softmax_forward
        except ImportError:
            from .baseline_softmax import baseline_softmax_forward
        fn = lambda: baseline_softmax_forward(input_tensor, scale, causal_mask)
    
    # Warmup
    for _ in range(warmup):
        _ = fn()
    
    torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    # Compute statistics
    times_ms = np.array(times)
    
    # Compute bytes moved
    if use_fused:
        try:
            from fused_softmax import compute_bytes_moved_fused
        except ImportError:
            from .fused_softmax import compute_bytes_moved_fused
        bytes_moved = compute_bytes_moved_fused(batch_size, seq_len)
    else:
        try:
            from baseline_softmax import compute_bytes_moved_unfused
        except ImportError:
            from .baseline_softmax import compute_bytes_moved_unfused
        bytes_moved = compute_bytes_moved_unfused(batch_size, seq_len)
    
    # Compute bandwidth (GB/s)
    mean_time_s = np.mean(times_ms) / 1000.0
    bandwidth_gbs = (bytes_moved / (1024**3)) / mean_time_s
    
    return {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'causal_mask': causal_mask,
        'use_fused': use_fused,
        'p5_latency_ms': percentile(times_ms, 5),
        'p50_latency_ms': percentile(times_ms, 50),
        'p95_latency_ms': percentile(times_ms, 95),
        'mean_latency_ms': np.mean(times_ms),
        'std_latency_ms': np.std(times_ms),
        'bytes_moved': bytes_moved,
        'bandwidth_gbs': bandwidth_gbs,
        'times_ms': times_ms
    }


def run_microbenchmark():
    """
    Run full microbenchmark suite
    """
    print("=" * 80)
    print("Microbenchmark: Fused Softmax Kernel")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  SEQ_LENGTHS: {SEQ_LENGTHS}")
    print(f"  BATCH_SIZES: {BATCH_SIZES}")
    print(f"  MASKS: {MASKS_MICRO}")
    print(f"  WARMUP: {WARMUP_KERNEL}, RUNS: {RUNS_KERNEL}")
    print(f"  BLOCK_DIM: {KERNEL_BLOCKDIM}")
    print()
    
    results = []
    
    for batch_size in BATCH_SIZES:
        for seq_len in SEQ_LENGTHS:
            for mask_type in MASKS_MICRO:
                causal_mask = (mask_type == "causal")
                
                print(f"Testing: batch={batch_size}, seq_len={seq_len}, mask={mask_type}")
                
                # Baseline
                print("  Running baseline...")
                baseline_result = benchmark_kernel(
                    batch_size, seq_len, causal_mask,
                    use_fused=False,
                    warmup=WARMUP_KERNEL,
                    runs=RUNS_KERNEL
                )
                results.append(baseline_result)
                
                # Fused
                print("  Running fused kernel...")
                fused_result = benchmark_kernel(
                    batch_size, seq_len, causal_mask,
                    use_fused=True,
                    block_dim=KERNEL_BLOCKDIM,
                    warmup=WARMUP_KERNEL,
                    runs=RUNS_KERNEL
                )
                results.append(fused_result)
                
                # Compute speedup
                speedup = baseline_result['p50_latency_ms'] / fused_result['p50_latency_ms']
                print(f"  Speedup: {speedup:.2f}x")
                print()
    
    # Print summary table
    print("=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"{'Batch':<6} {'SeqLen':<8} {'Mask':<8} {'Type':<10} {'p50(ms)':<10} {'GB/s':<10} {'Bytes':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for i in range(0, len(results), 2):
        baseline = results[i]
        fused = results[i + 1]
        speedup = baseline['p50_latency_ms'] / fused['p50_latency_ms']
        
        print(f"{baseline['batch_size']:<6} {baseline['seq_len']:<8} {baseline['causal_mask']!s:<8} {'Baseline':<10} "
              f"{baseline['p50_latency_ms']:<10.3f} {baseline['bandwidth_gbs']:<10.2f} "
              f"{baseline['bytes_moved']:<15} {'-':<10}")
        print(f"{'':<6} {'':<8} {'':<8} {'Fused':<10} "
              f"{fused['p50_latency_ms']:<10.3f} {fused['bandwidth_gbs']:<10.2f} "
              f"{fused['bytes_moved']:<15} {speedup:<10.2f}x")
        print()
    
    # Memory comparison table
    print("=" * 80)
    print("Memory Comparison: Unfused (4 passes) vs Fused (1 pass)")
    print("=" * 80)
    print(f"{'Batch':<6} {'SeqLen':<8} {'Unfused Bytes':<20} {'Fused Bytes':<20} {'Reduction':<15}")
    print("-" * 80)
    
    for batch_size in BATCH_SIZES:
        for seq_len in SEQ_LENGTHS:
            try:
                from baseline_softmax import compute_bytes_moved_unfused
                from fused_softmax import compute_bytes_moved_fused
            except ImportError:
                from .baseline_softmax import compute_bytes_moved_unfused
                from .fused_softmax import compute_bytes_moved_fused
            
            unfused_bytes = compute_bytes_moved_unfused(batch_size, seq_len)
            fused_bytes = compute_bytes_moved_fused(batch_size, seq_len)
            reduction = (1 - fused_bytes / unfused_bytes) * 100
            
            print(f"{batch_size:<6} {seq_len:<8} {unfused_bytes:<20} {fused_bytes:<20} {reduction:<15.1f}%")
    
    # Generate comprehensive report
    try:
        from generate_results_report import generate_comprehensive_report, save_report_to_file
        print("\n" + "=" * 80)
        print("Generating Comprehensive Performance Report...")
        print("=" * 80)
        report = generate_comprehensive_report(results, block_dim=KERNEL_BLOCKDIM)
        print(report)
        save_report_to_file(report, "microbenchmark_report.txt")
    except Exception as e:
        print(f"Warning: Could not generate comprehensive report: {e}")
    
    return results


if __name__ == "__main__":
    results = run_microbenchmark()

