The 4 Stages of Attention Softmax Explained
Context: Scaled Dot-Product Attention

In transformer models like GPT-2, the attention mechanism computes:
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

The softmax operation goes through these 4 stages:

Stage 1: Scale (Multiply by 1/√d_k)

What it is:
- Multiplies all attention scores by a scaling factor: scale = 1 / √d_k
- Where d_k is the dimension of the key vectors (typically 64, 128, 256, etc.)

Why it's needed:
- Prevents gradient vanishing: Without scaling, when d_k  is large, the dot products QK^T can become very large
- Stabilizes training: Large values cause the softmax to saturate (become very close to 0 or 1), making gradients vanish
- Standard practice: This is part of the "scaled dot-product attention" formula from the original Transformer paper

Example:
python
Input: attention scores [batch, seq_len, seq_len]
If d_k = 64, then scale = 1/√64 = 1/8 = 0.125
scaled = input_tensor * 0.125


Memory trip:
- Read: Full input tensor from global memory
- Write: Scaled tensor to global memory

Stage 2: Causal Mask (Set future positions to -∞)

What it is:
- Sets attention scores for "future" positions to negative infinity (-∞)
- For a position i, masks all positions j > i (positions that come after in the sequence)

Why it's needed:
- Autoregressive models: GPT-2 generates text one token at a time, left-to-right
- Prevents cheating: The model shouldn't see future tokens when predicting the current token
- After softmax: -∞ becomes 0 (since exp(-∞) = 0), so masked positions get zero attention weight

Visual example:
For a sequence of length 4, the causal mask looks like:

[  0    -∞   -∞   -∞ ]  ← Position 0 can only attend to position 0
[  0     0   -∞   -∞ ]  ← Position 1 can attend to positions 0, 1
[  0     0    0   -∞ ]  ← Position 2 can attend to positions 0, 1, 2
[  0     0    0    0 ]  ← Position 3 can attend to all positions


Code:
python
Create upper triangular mask (1s above diagonal)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
mask = mask.masked_fill(mask == 1, float('-inf'))
scaled = scaled + mask   Add -∞ to future positions
 
Memory trip:
- Read: Scaled tensor from global memory
- Write: Masked tensor to global memory

---

 Stage 3: Softmax Max Reduction (Find maximum per row)

 What it is:
- First pass of numerically stable softmax
- Finds the maximum value in each row of the attention matrix
- This is done before computing exponentials

 Why it's needed:
- Numerical stability: Computing exp(x) for large values causes overflow (becomes infinity)
- Stable softmax formula: Instead of exp(x) / sum(exp(x)), we use:
  
  softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
  
- Subtracting max: Shifts all values to be ≤ 0, preventing overflow

 Example:
python
 Row values: [10.0, 12.0, 8.0, 11.0]
 Max = 12.0
 After subtracting max: [-2.0, 0.0, -4.0, -1.0]
 Now exp() won't overflow!


 Memory trip:
- Read: Full masked tensor from global memory (to scan for max)
- Write: Max values per row to global memory (much smaller: [batch, seq_len] instead of [batch, seq_len, seq_len])

---

 Stage 4: Softmax Normalization (Compute exp and normalize)

 What it is:
- Second pass of softmax
- Computes exp(x - max) for each element
- Sums the exponentials per row
- Normalizes: divides each exp(x - max) by the sum

 Why it's needed:
- Softmax property: Each row must sum to 1.0 (probability distribution)
- Attention weights: The output represents how much each position should "attend" to each other position
- Final step: Produces the actual attention weights used to weight the value vectors

 Formula:

For each row:
  1. Compute: exp_vals = exp(x - max)
  2. Compute: row_sum = sum(exp_vals)
  3. Normalize: output = exp_vals / row_sum


 Example:
python
 After max reduction: values = [-2.0, 0.0, -4.0, -1.0], max = 12.0
 Step 1: exp_vals = [exp(-2.0), exp(0.0), exp(-4.0), exp(-1.0)]
         = [0.135, 1.0, 0.018, 0.368]
 Step 2: row_sum = 0.135 + 1.0 + 0.018 + 0.368 = 1.521
 Step 3: output = [0.089, 0.658, 0.012, 0.242]  ← Sums to 1.0!


 Memory trip:
- Read: Masked tensor from global memory (again) + max values
- Write: Final softmax output to global memory

Summary: Why 4 Memory Passes?

In the unfused PyTorch implementation:

1. Pass 1: Read input → Scale → Write scaled
2. Pass 2: Read scaled → Apply mask → Write masked  
3. Pass 3: Read masked → Find max → Write max values
4. Pass 4: Read masked + max → Compute softmax → Write output

Each pass requires reading/writing the full tensor (or most of it) from/to global GPU memory.

The Optimization

The fused CUDA kernel combines all 4 stages into 1 memory pass:
- Reads input once
- Performs scale, mask, max reduction, and normalization in shared memory
- Writes output once

This reduces memory traffic by ~4×, which is critical because attention softmax is bandwidth-bound (limited by memory speed, not computation speed).


Unfused (4 passes):
Input → [Scale] → Scaled → [Mask] → Masked → [Max] → Max → [Norm] → Output
  ↓        ↓         ↓        ↓         ↓       ↓      ↓       ↓        ↓
Read    Write     Read    Write     Read   Write   Read   Write    Write

Fused (1 pass):
Input → [Scale + Mask + Max + Norm] → Output
  ↓                                        ↓
Read                                    Write


