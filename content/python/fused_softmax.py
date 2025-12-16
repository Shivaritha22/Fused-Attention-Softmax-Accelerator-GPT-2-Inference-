"""
Fused softmax CUDA extension wrapper
"""

import torch
from torch.utils.cpp_extension import load
import os

# Path to CUDA files
CUDA_DIR = os.path.join(os.path.dirname(__file__), '..', 'cuda')

# Try to load the extension (will compile if needed)
try:
    _fused_softmax = load(
        name='fused_softmax',
        sources=[
            os.path.join(CUDA_DIR, 'softmax_binding.cpp'),
            os.path.join(CUDA_DIR, 'softmax_kernel.cu'),
        ],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=True
    )
    EXTENSION_LOADED = True
except Exception as e:
    print(f"Warning: Could not load CUDA extension: {e}")
    print("Falling back to baseline implementation")
    EXTENSION_LOADED = False
    _fused_softmax = None


def fused_softmax_forward(input_tensor, scale, causal_mask=True, block_dim=256):
    """
    Fused softmax forward: scale + causal mask + softmax in one pass
    
    Args:
        input_tensor: [batch, seq_len, seq_len] attention scores
        scale: Scale factor (typically 1/sqrt(d_k))
        causal_mask: If True, apply causal masking
        block_dim: CUDA block dimension (default 256)
    
    Returns:
        Softmax output [batch, seq_len, seq_len]
    """
    if not EXTENSION_LOADED or not input_tensor.is_cuda:
        # Fallback to baseline
        try:
            from baseline_softmax import baseline_softmax_forward
        except ImportError:
            from .baseline_softmax import baseline_softmax_forward
        return baseline_softmax_forward(input_tensor, scale, causal_mask)
    
    return _fused_softmax.forward(input_tensor, scale, causal_mask, block_dim)


def compute_bytes_moved_fused(batch_size, seq_len, dtype_size=4):
    """
    Compute total bytes moved in fused path (1 pass):
    - Read input once
    - Write output once
    - Shared memory for reductions (not counted as global memory)
    
    Args:
        batch_size: Batch dimension
        seq_len: Sequence length
        dtype_size: Size of float (4 bytes for FP32)
    
    Returns:
        Total bytes moved
    """
    tensor_size = batch_size * seq_len * seq_len * dtype_size
    
    # Single pass: read input, write output
    total_bytes = tensor_size * 2
    
    return total_bytes

