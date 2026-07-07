#ifndef TMP_USE_CUH
#define TMP_USE_CUH

// head_compute_mix_bwd optimized kernel - FINAL (Iteration 8)
// Best configuration: block_size=128, warp shuffle for scale reduction,
// cross-warp reduction via shared memory. 64 blocks for good SM coverage.

#define WARP_SIZE 64

__global__ void head_compute_mix_bwd_kernel_opt(
    const float* __restrict__ input_mix,
    const float* __restrict__ mhc_scale,
    const float* __restrict__ mhc_base,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_input_mix,
    float* __restrict__ grad_mhc_scale,
    float* __restrict__ grad_mhc_base,
    int batch0, int batch1, int mhc_mult)
{
    int total_elems = batch0 * batch1 * mhc_mult;
    float scale = mhc_scale[0];

    float local_base0 = 0.0f, local_base1 = 0.0f, local_base2 = 0.0f, local_base3 = 0.0f;
    float local_scale = 0.0f;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;

    for (int idx = blockIdx.x * block_size + tid; idx < total_elems; idx += block_size * gridDim.x) {
        int mhc_idx = idx % mhc_mult;

        float x = input_mix[idx];
        float go = grad_out[idx];
        float base = mhc_base[mhc_idx];

        float z = x * scale + base;
        float sig = 1.0f / (1.0f + expf(-z));
        float grad_z = go * sig * (1.0f - sig);

        grad_input_mix[idx] = grad_z * scale;

        if (mhc_idx == 0) local_base0 += grad_z;
        else if (mhc_idx == 1) local_base1 += grad_z;
        else if (mhc_idx == 2) local_base2 += grad_z;
        else local_base3 += grad_z;
        local_scale += grad_z * x;
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_scale += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, local_scale, offset);
    }

    // Cross-warp reduction (only 2 warps with block_size=128)
    extern __shared__ float warp_partials[];
    int num_warps = block_size / WARP_SIZE;
    if (lane == 0) {
        warp_partials[warp_id] = local_scale;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (tid < num_warps) ? warp_partials[tid] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, val, offset);
        }
        if (lane == 0 && val != 0.0f) {
            atomicAdd(grad_mhc_scale, val);
        }
    }

    if (local_base0 != 0.0f) atomicAdd(&grad_mhc_base[0], local_base0);
    if (local_base1 != 0.0f) atomicAdd(&grad_mhc_base[1], local_base1);
    if (local_base2 != 0.0f) atomicAdd(&grad_mhc_base[2], local_base2);
    if (local_base3 != 0.0f) atomicAdd(&grad_mhc_base[3], local_base3);
}

template <typename T>
void test_tmp_kernel_opt(
    T* input_mix, T* mhc_scale, T* mhc_base, T* grad_out,
    T* grad_input_mix, T* grad_mhc_scale, T* grad_mhc_base,
    int batch0, int batch1, int mhc_mult, cudaStream_t stream)
{
    int total_elems = batch0 * batch1 * mhc_mult;
    int block_size = 128;
    int num_blocks = (total_elems + block_size - 1) / block_size;
    if (num_blocks > 104 * 2) num_blocks = 104 * 2;
    int num_warps = block_size / WARP_SIZE;
    int shared_mem = num_warps * sizeof(float);

    cudaMemsetAsync(grad_mhc_base, 0, mhc_mult * sizeof(float), stream);
    float zero = 0.0f;
    cudaMemcpyAsync(grad_mhc_scale, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);

    head_compute_mix_bwd_kernel_opt<<<num_blocks, block_size, shared_mem, stream>>>(
        input_mix, mhc_scale, mhc_base, grad_out,
        grad_input_mix, grad_mhc_scale, grad_mhc_base,
        batch0, batch1, mhc_mult);
}

#endif
