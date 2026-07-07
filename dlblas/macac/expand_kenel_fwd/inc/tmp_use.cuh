#pragma once
// v20: Best pattern — 2D grid 320 threads, float4 load+4 explicit stores, no loop
__global__ void expand_kenel_fwd_kernel_opt(
    const float* __restrict__ input, float* __restrict__ output,
    int batch_size, int seq_len, int mhc_mult, int hidden_size
) {
    int row = blockIdx.x;
    if (row >= batch_size * seq_len) return;
    int tid = threadIdx.x;
    int stride4 = hidden_size / 4;
    const float4* __restrict__ in4 = (const float4*)(input + row * hidden_size);
    float4* __restrict__ out4 = (float4*)(output + row * mhc_mult * hidden_size);
    float4 val = in4[tid];
    out4[0 * stride4 + tid] = val;
    out4[1 * stride4 + tid] = val;
    out4[2 * stride4 + tid] = val;
    out4[3 * stride4 + tid] = val;
}
template <typename T>
void test_tmp_kernel_opt(T* input, T* output, int batch_size, int seq_len, int mhc_mult, int hidden_size, cudaStream_t stream) {
    expand_kenel_fwd_kernel_opt<<<batch_size * seq_len, 320, 0, stream>>>((const float*)input, (float*)output, batch_size, seq_len, mhc_mult, hidden_size);
}
