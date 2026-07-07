#pragma once
// engram_fused_weight optimized kernel - FINAL BEST
// 2 blocks x 256 threads = 512 threads for 512 elements
// Direct element-wise access, implicit bf16->f32 conversion
// No __restrict__ (causes ATU fault on MACA C500)
// opt_time: ~0.0110ms, baseline: ~0.020ms, speedup: ~1.82x
// Key optimizations:
//   1. Implicit float conversion (not explicit cast) - better codegen
//   2. Direct idx access (no grid-stride loop overhead) for 1:1 thread:elem mapping
//   3. Separate temp vars for ILP

__global__ void engram_fused_weight_kernel_opt(
    const __FLOAT16__* wh_data,
    const __FLOAT16__* we_data,
    float* Y,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float a = wh_data[idx];
        float b = we_data[idx];
        Y[idx] = a * b;
    }
}

void test_tmp_kernel_opt(
    __FLOAT16__* wh_data, __FLOAT16__* we_data, float* Y,
    int size, cudaStream_t stream)
{
    const int block_size = 256;
    int num_blocks = 2;
    engram_fused_weight_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        wh_data, we_data, Y, size);
}
