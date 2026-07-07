#pragma once
// expand_kenel_fwd baseline kernel
// Semantics: input (B, S, H) -> unsqueeze(-2) -> expand to (B, S, M, H) -> contiguous
// y[b, s, m, h] = x[b, s, h]
// dtype: float32, layout: contiguous

__global__ void expand_kenel_fwd_kernel_ori(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int mhc_mult,
    int hidden_size
) {
    int total = batch_size * seq_len * mhc_mult * hidden_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int h = idx % hidden_size;
    int rest = idx / hidden_size;
    int m = rest % mhc_mult;
    rest /= mhc_mult;
    int s = rest % seq_len;
    int b = rest / seq_len;

    int src_idx = (b * seq_len + s) * hidden_size + h;
    output[idx] = __ldg(input + src_idx);
}

template <typename T>
void test_tmp_kernel_ori(
    T* input,
    T* output,
    int batch_size,
    int seq_len,
    int mhc_mult,
    int hidden_size,
    cudaStream_t stream
) {
    int total = batch_size * seq_len * mhc_mult * hidden_size;
    const int block_size = 512;
    int num_blocks = (total + block_size - 1) / block_size;
    expand_kenel_fwd_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        (const float*)input, (float*)output,
        batch_size, seq_len, mhc_mult, hidden_size
    );
}
