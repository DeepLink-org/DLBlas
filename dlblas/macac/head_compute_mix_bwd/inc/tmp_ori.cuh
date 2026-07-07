#ifndef TMP_ORI_CUH
#define TMP_ORI_CUH

// head_compute_mix_bwd baseline kernel
// Computes:
//   z = input_mix * mhc_scale + mhc_base[mhc_idx]
//   sig = sigmoid(z)
//   grad_z = grad_out * sig * (1 - sig)
//   grad_input_mix = grad_z * mhc_scale
//   grad_mhc_base[mhc_idx] = sum over batch0,batch1 of grad_z[mhc_idx]
//   grad_mhc_scale = sum over all elements of grad_z * input_mix

__global__ void head_compute_mix_bwd_kernel_ori(
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

    // Shared memory layout:
    // For each mhc channel (0..mhc_mult-1): blockDim.x floats for partial reduction
    // Last channel: grad_mhc_scale partial reduction
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Initialize shared memory for all channels including the scale channel
    for (int c = 0; c <= mhc_mult; c++) {
        sdata[c * block_size + tid] = 0.0f;
    }
    __syncthreads();

    // Grid-stride loop: each thread processes multiple elements
    for (int idx = blockIdx.x * block_size + tid; idx < total_elems; idx += block_size * gridDim.x) {
        int mhc_idx = idx % mhc_mult;

        float x = input_mix[idx];
        float go = grad_out[idx];
        float base = mhc_base[mhc_idx];

        // Forward intermediate: z = x * scale + base
        float z = x * scale + base;

        // Sigmoid backward: sigmoid(z) * (1 - sigmoid(z))
        float sig = 1.0f / (1.0f + expf(-z));
        float grad_z = go * sig * (1.0f - sig);

        // grad_input_mix
        grad_input_mix[idx] = grad_z * scale;

        // Accumulate partial sums in shared memory
        sdata[mhc_idx * block_size + tid] += grad_z;
        sdata[mhc_mult * block_size + tid] += grad_z * x;
    }
    __syncthreads();

    // Tree reduction for each channel
    for (int c = 0; c <= mhc_mult; c++) {
        float* chan_sdata = &sdata[c * block_size];
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                chan_sdata[tid] += chan_sdata[tid + s];
            }
            __syncthreads();
        }
    }

    // Write block results to global via atomics
    if (tid < mhc_mult) {
        float val = sdata[tid * block_size];
        if (val != 0.0f) {
            atomicAdd(&grad_mhc_base[tid], val);
        }
    }
    if (tid == 0) {
        float val = sdata[mhc_mult * block_size];
        if (val != 0.0f) {
            atomicAdd(grad_mhc_scale, val);
        }
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* input_mix, T* mhc_scale, T* mhc_base, T* grad_out,
    T* grad_input_mix, T* grad_mhc_scale, T* grad_mhc_base,
    int batch0, int batch1, int mhc_mult, cudaStream_t stream)
{
    int total_elems = batch0 * batch1 * mhc_mult;
    int block_size = 512;
    int num_blocks = (total_elems + block_size - 1) / block_size;
    if (num_blocks > 104 * 8) num_blocks = 104 * 8;
    // Shared memory: (mhc_mult + 1) channels * block_size floats
    int shared_mem = (mhc_mult + 1) * block_size * sizeof(float);

    // Zero the reduction outputs before launch
    cudaMemsetAsync(grad_mhc_base, 0, mhc_mult * sizeof(float), stream);
    float zero = 0.0f;
    cudaMemcpyAsync(grad_mhc_scale, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);

    head_compute_mix_bwd_kernel_ori<<<num_blocks, block_size, shared_mem, stream>>>(
        input_mix, mhc_scale, mhc_base, grad_out,
        grad_input_mix, grad_mhc_scale, grad_mhc_base,
        batch0, batch1, mhc_mult);
}

#endif
