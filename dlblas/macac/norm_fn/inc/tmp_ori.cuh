// tmp_ori.cuh - Baseline norm_fn kernel
// Computes: output[i,j] = dot(residual[i,:], mhc_fn[j,:]) * rsqrt(sqrsum[i]/N + eps)
// Simple baseline: one block per (row, mix) pair, parallel reduction

__global__ void norm_fn_kernel_ori(
    const float* __restrict__ residual,  // [num_rows * rms_group_size]
    const float* __restrict__ mhc_fn,    // [num_mixes * rms_group_size]
    float* __restrict__ output,          // [num_rows * num_mixes]
    int num_rows,
    int num_mixes,
    int rms_group_size,
    float eps
) {
    int row = blockIdx.y;
    int mix = blockIdx.x;

    if (row >= num_rows || mix >= num_mixes) return;

    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    const float* res_row = residual + row * rms_group_size;
    const float* mhc_row = mhc_fn + mix * rms_group_size;

    // Each thread accumulates its portion of dot and sqrsum
    float dot = 0.0f;
    float sqrsum = 0.0f;

    for (int k = tid; k < rms_group_size; k += bdim) {
        float r_val = res_row[k];
        dot += r_val * mhc_row[k];
        sqrsum += r_val * r_val;
    }

    // Tree reduction for dot product
    sdata[tid] = dot;
    __syncthreads();
    for (int s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    dot = sdata[0];

    // Tree reduction for sqrsum
    sdata[tid] = sqrsum;
    __syncthreads();
    for (int s = bdim >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    sqrsum = sdata[0];

    // Only thread 0 writes the final result
    if (tid == 0) {
        float rms_norm = rsqrtf(sqrsum / (float)rms_group_size + eps);
        output[row * num_mixes + mix] = dot * rms_norm;
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* residual,
    T* mhc_fn,
    T* output,
    int num_rows,
    int num_mixes,
    int rms_group_size,
    float eps,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_mixes, num_rows);
    norm_fn_kernel_ori<<<grid, block, 0, stream>>>(
        residual, mhc_fn, output,
        num_rows, num_mixes, rms_group_size, eps
    );
}
