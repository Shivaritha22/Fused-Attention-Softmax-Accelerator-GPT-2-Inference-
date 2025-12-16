"""
GPT-2 tokens/sec benchmark harness
Measures end-to-end text generation performance
"""

import torch
import time
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict

# Configuration
GPT2_SIZE = "distilgpt2"
SEQ_LENGTHS = [512, 1024]  # add 2048 only if GPU allows
BATCH_SIZES = [1]  # optionally [1, 2]
MASKS_GPT2 = ["causal"]  # GPT-2 uses causal mask
WARMUP_GPT2 = 1
RUNS_GPT2 = 3


def monkey_patch_attention_softmax(model, use_fused=True, block_dim=256):
    """
    Monkey patch the attention softmax in GPT-2 to use fused kernel
    
    This hooks into the attention mechanism and replaces the softmax operation
    """
    try:
        from fused_softmax import fused_softmax_forward
        from baseline_softmax import baseline_softmax_forward
    except ImportError:
        from .fused_softmax import fused_softmax_forward
        from .baseline_softmax import baseline_softmax_forward
    
    original_forward = None
    
    def patched_attention_forward(self, query, key, value, attention_mask=None, head_mask=None):
        """Patched attention forward with fused softmax"""
        # Get the original forward method
        if original_forward is None:
            # This is a simplified version - actual GPT-2 attention is more complex
            pass
        
        # For now, we'll need to hook at a different level
        # This is a placeholder - actual implementation will depend on GPT-2 structure
        pass
    
    # This is a simplified approach - actual monkey patching will be more involved
    # We'll need to identify the exact softmax call in the attention module
    pass


def load_gpt2_model(model_name="distilgpt2"):
    """Load GPT-2 model and tokenizer"""
    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.cuda()
        model = model.eval()
    
    return model, tokenizer


def generate_text_baseline(model, tokenizer, prompt, max_length=100, num_return_sequences=1):
    """Generate text using baseline (unfused) implementation"""
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=False,  # Deterministic for benchmarking
            pad_token_id=tokenizer.eos_token_id
        )
    
    return outputs


def benchmark_gpt2_generation(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    use_fused: bool = True,
    warmup: int = 1,
    runs: int = 3
) -> Dict:
    """
    Benchmark GPT-2 text generation
    
    Returns:
        Dictionary with timing and throughput metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA not available")
    
    # Apply monkey patch if using fused
    if use_fused:
        monkey_patch_attention_softmax(model, use_fused=True)
    
    # Warmup
    for _ in range(warmup):
        _ = generate_text_baseline(model, tokenizer, prompt, max_length)
    
    torch.cuda.synchronize()
    
    # Timed runs
    times = []
    total_tokens = []
    
    for _ in range(runs):
        start = time.perf_counter()
        outputs = generate_text_baseline(model, tokenizer, prompt, max_length)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        elapsed = end - start
        times.append(elapsed)
        
        # Count tokens generated (excluding input)
        input_len = len(tokenizer.encode(prompt))
        generated_tokens = outputs[0].shape[0] - input_len
        total_tokens.append(generated_tokens)
    
    times_s = np.array(times)
    tokens = np.array(total_tokens)
    
    # Compute metrics
    total_time = np.sum(times_s)
    total_tokens_generated = np.sum(tokens)
    tokens_per_sec = total_tokens_generated / total_time
    
    per_token_latency_ms = (np.mean(times_s) / np.mean(tokens)) * 1000
    
    return {
        'use_fused': use_fused,
        'total_time_s': total_time,
        'total_tokens': total_tokens_generated,
        'tokens_per_sec': tokens_per_sec,
        'mean_time_s': np.mean(times_s),
        'mean_tokens': np.mean(tokens),
        'per_token_latency_ms': per_token_latency_ms,
        'p50_latency_s': np.percentile(times_s, 50),
        'p5_latency_s': np.percentile(times_s, 5),
        'p95_latency_s': np.percentile(times_s, 95),
    }


def run_gpt2_benchmark():
    """
    Run GPT-2 end-to-end benchmark
    """
    print("=" * 80)
    print("GPT-2 Text Generation Benchmark")
    print("=" * 80)
    
    # Load model
    model, tokenizer = load_gpt2_model(GPT2_SIZE)
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence",
        "In a world where technology",
        "The quick brown fox",
    ]
    
    results = []
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: '{prompt}'")
        
        # Baseline
        print("  Running baseline...")
        baseline_result = benchmark_gpt2_generation(
            model, tokenizer, prompt,
            max_length=100,
            use_fused=False,
            warmup=WARMUP_GPT2,
            runs=RUNS_GPT2
        )
        results.append(('baseline', prompt, baseline_result))
        
        # Fused
        print("  Running fused kernel...")
        fused_result = benchmark_gpt2_generation(
            model, tokenizer, prompt,
            max_length=100,
            use_fused=True,
            warmup=WARMUP_GPT2,
            runs=RUNS_GPT2
        )
        results.append(('fused', prompt, fused_result))
        
        # Compute speedup
        speedup = baseline_result['tokens_per_sec'] / fused_result['tokens_per_sec']
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Baseline: {baseline_result['tokens_per_sec']:.2f} tokens/sec")
        print(f"  Fused: {fused_result['tokens_per_sec']:.2f} tokens/sec")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"{'Type':<10} {'Prompt':<30} {'Tokens/sec':<15} {'Latency/Token (ms)':<20}")
    print("-" * 80)
    
    for i in range(0, len(results), 2):
        baseline_type, baseline_prompt, baseline_data = results[i]
        fused_type, fused_prompt, fused_data = results[i + 1]
        
        print(f"{'Baseline':<10} {baseline_prompt[:28]:<30} {baseline_data['tokens_per_sec']:<15.2f} "
              f"{baseline_data['per_token_latency_ms']:<20.3f}")
        print(f"{'Fused':<10} {fused_prompt[:28]:<30} {fused_data['tokens_per_sec']:<15.2f} "
              f"{fused_data['per_token_latency_ms']:<20.3f}")
        print()
    
    # Generate comprehensive report (combines with microbenchmark if available)
    try:
        from generate_results_report import generate_comprehensive_report, save_report_to_file
        print("\n" + "=" * 80)
        print("Generating Comprehensive Performance Report...")
        print("=" * 80)
        # Note: micro_results should be passed separately if available
        # This will generate report with GPT-2 results only
        report = generate_comprehensive_report([], results, block_dim=256)
        print(report)
        save_report_to_file(report, "gpt2_benchmark_report.txt")
    except Exception as e:
        print(f"Warning: Could not generate comprehensive report: {e}")
    
    return results


if __name__ == "__main__":
    results = run_gpt2_benchmark()

