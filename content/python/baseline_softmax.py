"""
Baseline unfused softmax chain: scale -> mask -> softmax
This represents the PyTorch baseline that we're optimizing against.
"""

import torch
import torch.nn.functional as F


def baseline_softmax_forward(input_tensor, scale, causal_mask=True):
    """
    Unfused PyTorch baseline: scale -> causal mask -> softmax
    
    Args:
        input_tensor: [batch, seq_len, seq_len] attention scores
        scale: Scale factor (typically 1/sqrt(d_k))
        causal_mask: If True, apply causal masking
    
    Returns:
        Softmax output [batch, seq_len, seq_len]
    """
    # Step 1: Scale
    scaled = input_tensor * scale
    
    # Step 2: Causal mask
    if causal_mask:
        batch_size, seq_len, _ = scaled.shape
        # Create causal mask: mask[i][j] = (j > i) ? -inf : 0
        mask = torch.triu(torch.ones(seq_len, seq_len, device=scaled.device, dtype=scaled.dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        scaled = scaled + mask.unsqueeze(0)  # Add batch dimension
    
    # Step 3: Softmax (numerically stable)
    output = F.softmax(scaled, dim=-1)
    
    return output


def compute_bytes_moved_unfused(batch_size, seq_len, dtype_size=4):
    """
    Compute total bytes moved in unfused path (4 passes):
    1. Read input, write scaled
    2. Read scaled, write masked
    3. Read masked (for max), write max
    4. Read masked, write softmax output
    
    Args:
        batch_size: Batch dimension
        seq_len: Sequence length
        dtype_size: Size of float (4 bytes for FP32)
    
    Returns:
        Total bytes moved
    """
    tensor_size = batch_size * seq_len * seq_len * dtype_size
    
    # Pass 1: input -> scaled
    bytes_1 = tensor_size * 2  # read input, write scaled
    
    # Pass 2: scaled -> masked (in-place with mask addition)
    bytes_2 = tensor_size * 2  # read scaled, write masked
    
    # Pass 3: masked -> max reduction (read all, write max per row)
    bytes_3 = tensor_size + batch_size * seq_len * dtype_size  # read all, write max
    
    # Pass 4: masked -> softmax output
    bytes_4 = tensor_size * 2  # read masked, write output
    
    total_bytes = bytes_1 + bytes_2 + bytes_3 + bytes_4
    return total_bytes

