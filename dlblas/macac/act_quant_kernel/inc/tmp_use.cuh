// Iter 5: block_size=256 — all 256 threads active (vs 512 with half idle)
// Fewer cross-warp reductions (4 warps vs 8). Vectorized uint32_t loads + warp shuffle.

#define WARP_SIZE 64

__global__ __launch_bounds__(256, 2) void act_quant_kernel_opt(
    const __FLOAT16__* __restrict__ x,
    __FLOAT16__* __restrict__ x_q,
    float* __restrict__ x_s,
    int B,
    int D,
    int group_size,
    float fp8_max,
    float fp8_min)
{
    int row = blockIdx.x;
    if (row >= B) return;

    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int num_pairs = group_size / 2;  // 256
    const uint32_t* x_u32 = (const uint32_t*)(x + row * D);

    // Step 1: Each thread processes exactly 1 pair (256 threads, 256 pairs)
    float local_max = 0.0f;
    if (tid < num_pairs) {
        uint32_t pair = x_u32[tid];
        local_max = fmaxf(local_max, fabsf((float)(*(__FLOAT16__*)&pair)));
        local_max = fmaxf(local_max, fabsf((float)(*((__FLOAT16__*)&pair + 1))));
    }

    // Warp-level reduction (64-lane warp, 4 warps)
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFFFFFFFFF, local_max, 32, WARP_SIZE));
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFFFFFFFFF, local_max, 16, WARP_SIZE));
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFFFFFFFFF, local_max, 8, WARP_SIZE));
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFFFFFFFFF, local_max, 4, WARP_SIZE));
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFFFFFFFFF, local_max, 2, WARP_SIZE));
    local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFFFFFFFFFF, local_max, 1, WARP_SIZE));

    // Cross-warp reduction: only 4 warps, use thread 0 for sequential read
    __shared__ float warp_maxs[4];
    if (lane == 0) warp_maxs[tid / WARP_SIZE] = local_max;
    __syncthreads();

    float amax;
    if (tid == 0) {
        amax = warp_maxs[0];
        amax = fmaxf(amax, warp_maxs[1]);
        amax = fmaxf(amax, warp_maxs[2]);
        amax = fmaxf(amax, warp_maxs[3]);
        float final_amax = fmaxf(amax, 1e-10f);
        float scale = final_amax / fp8_max;
        x_s[row] = scale;
        warp_maxs[0] = 1.0f / scale;  // store inv_scale for broadcast
    }
    __syncthreads();

    float inv_scale = warp_maxs[0];
    uint32_t* x_q_u32 = (uint32_t*)(x_q + row * D);

    if (tid < num_pairs) {
        uint32_t pair = x_u32[tid];
        float v0 = fminf(fmaxf((float)(*(__FLOAT16__*)&pair) * inv_scale, fp8_min), fp8_max);
        float v1 = fminf(fmaxf((float)(*((__FLOAT16__*)&pair + 1)) * inv_scale, fp8_min), fp8_max);
        __FLOAT16__ q0 = (__FLOAT16__)v0;
        __FLOAT16__ q1 = (__FLOAT16__)v1;
        x_q_u32[tid] = (uint32_t)(*(uint16_t*)&q0) | ((uint32_t)(*(uint16_t*)&q1) << 16);
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* x, T* x_q, float* x_s,
    int B, int D, int group_size,
    float fp8_max, float fp8_min,
    cudaStream_t stream)
{
    int block_size = 256;
    int shared_mem = 4 * sizeof(float);

    act_quant_kernel_opt<<<B, block_size, shared_mem, stream>>>(
        x, x_q, x_s, B, D, group_size, fp8_max, fp8_min);
}
