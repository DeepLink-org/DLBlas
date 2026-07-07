#pragma once
// Best version: float4 vectorized loads/stores with block_size=256
// Semantics: output = sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
// Key optimizations: float4 (MHC=4 naturally aligned), block_size=256

__global__ void head_compute_mix_fwd_kernel_opt(
    const float* __restrict__ input_mix,
    const float* __restrict__ mhc_scale,
    const float* __restrict__ mhc_base,
    const float* __restrict__ mhc_pre_eps_d,
    float* __restrict__ output,
    int total,
    int MHC
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 >= total) {
        // Tail: scalar path for border elements (at most 3 elements)
        int base = (total / 4) * 4;
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (base + tid < total) {
            int m = (base + tid) % MHC;
            float val = __ldg(input_mix + base + tid) * __ldg(mhc_scale) + mhc_base[m];
            output[base + tid] = 1.0f / (1.0f + expf(-val)) + __ldg(mhc_pre_eps_d);
        }
        return;
    }
    
    float4 in = *((const float4*)(input_mix + idx));
    float scale = __ldg(mhc_scale);
    float eps = __ldg(mhc_pre_eps_d);
    
    float4 out;
    out.x = 1.0f / (1.0f + expf(-(in.x * scale + mhc_base[0]))) + eps;
    out.y = 1.0f / (1.0f + expf(-(in.y * scale + mhc_base[1]))) + eps;
    out.z = 1.0f / (1.0f + expf(-(in.z * scale + mhc_base[2]))) + eps;
    out.w = 1.0f / (1.0f + expf(-(in.w * scale + mhc_base[3]))) + eps;
    
    *((float4*)(output + idx)) = out;
}

template <typename T>
void test_tmp_kernel_opt(
    T* input_mix, T* mhc_scale, T* mhc_base, T* mhc_pre_eps_d, T* output,
    int total, int MHC, cudaStream_t stream
) {
    const int block_size = 256;
    int num_blocks = (total/4 + block_size - 1) / block_size;
    head_compute_mix_fwd_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        (const float*)input_mix, (const float*)mhc_scale, (const float*)mhc_base,
        (const float*)mhc_pre_eps_d, (float*)output, total, MHC);
}
