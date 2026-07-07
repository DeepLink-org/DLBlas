#pragma once
// engram_fused_weight baseline kernel
// Element-wise multiply: Y[i] = wh_data[i] * we_data[i]
// Input:  two bf16 tensors [hc_mult, hidden_size]
// Output: one f32 tensor [hc_mult, hidden_size]

__global__ void engram_fused_weight_kernel_ori(
    const __FLOAT16__* wh_data,
    const __FLOAT16__* we_data,
    float* Y,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        Y[idx] = (float)(wh_data[idx]) * (float)(we_data[idx]);
    }
}

void test_tmp_kernel_ori(
    __FLOAT16__* wh_data, __FLOAT16__* we_data, float* Y,
    int size, cudaStream_t stream)
{
    const int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    engram_fused_weight_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        wh_data, we_data, Y, size);
}
