#include <torch/extension.h>
#include <cuda_runtime.h>

// ✅ Use c10 CUDA stream API (works across many PyTorch versions)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// Forward declaration with C linkage (matches extern "C" in .cu file)
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
    );
}

torch::Tensor fused_softmax_forward(
    torch::Tensor input,
    float scale,
    bool causal_mask,
    int block_dim = 256
) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(input.dim() == 3, "Input tensor must be 3D [batch, seq_len, seq_len]");

    // Ensure contiguous (kernel likely assumes contiguous layout)
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    const int batch_size = static_cast<int>(input.size(0));
    const int seq_len     = static_cast<int>(input.size(1));
    TORCH_CHECK(input.size(2) == seq_len, "Input must be square [batch, seq, seq]");

    // Create output tensor
    auto output = torch::empty_like(input);

    // ✅ Set device + get current stream on that device
    c10::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel via C++ wrapper function (defined in .cu file)
    launch_fused_softmax_kernel(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        scale,
        batch_size,
        seq_len,
        causal_mask,
        block_dim,
        stream
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_softmax_forward, "Fused softmax forward (CUDA)");
}
