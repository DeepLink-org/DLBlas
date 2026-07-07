#pragma once
// head_compute_mix_fwd baseline kernel
// Semantics: output = sigmoid(input_mix * mhc_scale + mhc_base) + mhc_pre_eps
// dtype: float32

__global__ void head_compute_mix_fwd_kernel_ori(
    const float* __restrict__ input_mix,
    const float* __restrict__ mhc_scale,
    const float* __restrict__ mhc_base,
    const float* __restrict__ mhc_pre_eps_d,
    float* __restrict__ output,
    int total,
    int MHC
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int m = idx % MHC;
    float val = __ldg(input_mix + idx) * __ldg(mhc_scale) + __ldg(mhc_base + m);
    output[idx] = 1.0f / (1.0f + expf(-val)) + __ldg(mhc_pre_eps_d);
}

template <typename T>
void test_tmp_kernel_ori(
    T* input_mix, T* mhc_scale, T* mhc_base, T* mhc_pre_eps_d, T* output,
    int total, int MHC, cudaStream_t stream
) {
    const int block_size = 512;
    int num_blocks = (total + block_size - 1) / block_size;
    head_compute_mix_fwd_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        (const float*)input_mix, (const float*)mhc_scale, (const float*)mhc_base,
        (const float*)mhc_pre_eps_d, (float*)output, total, MHC);
}
