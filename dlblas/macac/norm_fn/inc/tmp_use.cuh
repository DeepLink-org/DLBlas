// tmp_use.cuh - Final best version: warp-shuffle + float4 + inv_K precompute
// Key optimizations:
// 1. float4 vectorized loads (4 floats per load)
// 2. Warp-level reduction via __shfl_down_sync (no shared mem until cross-warp)
// 3. Precompute inv_group_size to replace division with multiplication
// 4. Grid: 312 blocks (24 mixes × 13 rows), Block: 256 threads (4 warps × 64 lanes)
__global__ void norm_fn_kernel_opt(
    const float* __restrict__ residual, const float* __restrict__ mhc_fn,
    float* __restrict__ output, int num_rows, int num_mixes, int rms_group_size, float eps
) {
    int row = blockIdx.y, mix = blockIdx.x;
    if (row >= num_rows || mix >= num_mixes) return;

    __shared__ float2 s_warp[4];
    int tid = threadIdx.x, bdim = blockDim.x;
    int lane = tid & 63, warp_id = tid >> 6;

    const float* res_row = residual + row * rms_group_size;
    const float* mhc_row = mhc_fn + mix * rms_group_size;

    float dot = 0.0f, sqrsum = 0.0f;
    for (int k = tid * 4; k < rms_group_size; k += bdim * 4) {
        float4 r_vec = *((const float4*)(res_row + k));
        float4 m_vec = *((const float4*)(mhc_row + k));
        dot   += r_vec.x * m_vec.x + r_vec.y * m_vec.y + r_vec.z * m_vec.z + r_vec.w * m_vec.w;
        sqrsum += r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z + r_vec.w * r_vec.w;
    }

    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        dot    += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, dot,    offset);
        sqrsum += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, sqrsum, offset);
    }

    if (lane == 0) s_warp[warp_id] = make_float2(dot, sqrsum);
    __syncthreads();

    if (tid == 0) {
        float dot_sum = s_warp[0].x + s_warp[1].x + s_warp[2].x + s_warp[3].x;
        float sqr_sum = s_warp[0].y + s_warp[1].y + s_warp[2].y + s_warp[3].y;
        float inv_K = 1.0f / (float)rms_group_size;
        output[row * num_mixes + mix] = dot_sum * rsqrtf(sqr_sum * inv_K + eps);
    }
}

template <typename T>
void test_tmp_kernel_opt(T* residual, T* mhc_fn, T* output,
    int num_rows, int num_mixes, int rms_group_size, float eps, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(num_mixes, num_rows);
    norm_fn_kernel_opt<<<grid, block, 0, stream>>>(
        residual, mhc_fn, output, num_rows, num_mixes, rms_group_size, eps);
}
