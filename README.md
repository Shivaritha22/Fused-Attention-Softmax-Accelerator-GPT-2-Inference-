# Fused-Attention-Softmax-Accelerator-GPT-2-Inference-
Custom CUDA C++ kernel that fuses scale, causal mask, and softmax into a single pass to cut global memory traffic and speed up GPT-2 inference. Algorithmic outputs stay identical; only latency and tokens/sec change.
