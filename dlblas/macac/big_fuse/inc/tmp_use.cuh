#pragma once
// big_fuse optimized kernel v2: warp-shuffle reduction + optimized Sinkhorn
// Key optimizations:
// 1. Warp-level reduction using __shfl_xor_sync (C500 warp=64), eliminating 24 separate block reductions
// 2. Only 4 cross-warp values stored to shared memory (96 floats vs 6144)
// 3. Reduced __syncthreads from ~192 to ~3

__global__ void big_fuse_kernel_opt(
    const __FLOAT16__* __restrict__ residual,
    const float* __restrict__ fn,
    const float* __restrict__ mhc_scale,
    const float* __restrict__ mhc_base,
    float* __restrict__ post_mix,
    float* __restrict__ comb_mix,
    __FLOAT16__* __restrict__ layer_input,
    int seq_len,
    int mhc_mult,
    int hidden_size,
    int mhc_mult3,
    int mhc_rgs,
    float rms_eps,
    float pre_eps,
    float sinkhorn_eps,
    float post_mult_val,
    int sinkhorn_repeat
) {
    int seq_idx = blockIdx.x;
    if (seq_idx >= seq_len) return;

    int tid = threadIdx.x;
    int warp_id = tid >> 6;   // warp 0..3 for 256 threads, warp_size=64
    int lane_id = tid & 63;

    // Shared memory: only for cross-warp reduction (4 warps × 24 outputs) + broadcast
    __shared__ float smem[256];

    const __FLOAT16__* res_slice = residual + seq_idx * mhc_rgs;

    // ===== Stage 1: Matmul — compute all 24 dot products =====
    float partial[24];
    #pragma unroll
    for (int j = 0; j < 24; j++) partial[j] = 0.0f;

    for (int k = tid; k < mhc_rgs; k += blockDim.x) {
        float r = (float)res_slice[k];
        #pragma unroll
        for (int j = 0; j < mhc_mult3; j++) {
            partial[j] += r * fn[j * mhc_rgs + k];
        }
    }

    // Warp-level reduction (C500 warp=64, full mask)
    unsigned long long full_mask = 0xFFFFFFFFFFFFFFFFULL;
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < mhc_mult3; j++) {
            partial[j] += __shfl_xor_sync(full_mask, partial[j], offset, 64);
        }
    }

    // Store warp results to shared memory (only 1 thread per warp)
    if (lane_id == 0) {
        #pragma unroll
        for (int j = 0; j < mhc_mult3; j++) {
            smem[warp_id * 24 + j] = partial[j];
        }
    }
    __syncthreads();

    // First warp combines 4 warp results
    float mixes[24];
    if (warp_id == 0) {
        #pragma unroll
        for (int j = 0; j < mhc_mult3; j++) {
            mixes[j] = (lane_id < 4) ? smem[lane_id * 24 + j] : 0.0f;
        }
        // Reduce 4 warp results within warp 0 (only lanes 0..3 have data)
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int j = 0; j < mhc_mult3; j++) {
                mixes[j] += __shfl_xor_sync(full_mask, mixes[j], offset, 64);
            }
        }
        // Store final result to shared memory for all warps
        if (lane_id == 0) {
            #pragma unroll
            for (int j = 0; j < mhc_mult3; j++) {
                smem[j] = mixes[j];
            }
        }
    }
    __syncthreads();
    // All threads read the final mixes
    #pragma unroll
    for (int j = 0; j < mhc_mult3; j++) {
        mixes[j] = smem[j];
    }

    // ===== Stage 1b: RMS normalize =====
    float sqrsum = 0.0f;
    #pragma unroll
    for (int j = 0; j < mhc_mult3; j++) {
        sqrsum += mixes[j] * mixes[j];
    }
    float inv_rms = rsqrtf(sqrsum / (float)mhc_rgs + rms_eps);
    #pragma unroll
    for (int j = 0; j < mhc_mult3; j++) {
        mixes[j] *= inv_rms;
    }

    // ===== Stage 2: Split mixes with scaling + sigmoid =====
    float scale_0 = mhc_scale[0];
    float scale_1 = mhc_scale[1];
    float scale_2 = mhc_scale[2];

    float pre_mix_val[4];
    float post_mix_val[4];
    float comb_mix_val[16];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float v = mixes[i] * scale_0 + mhc_base[i];
        pre_mix_val[i] = 1.0f / (1.0f + expf(-v)) + pre_eps;
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float v = mixes[4 + i] * scale_1 + mhc_base[4 + i];
        post_mix_val[i] = 1.0f / (1.0f + expf(-v)) * post_mult_val;
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        comb_mix_val[i] = mixes[8 + i] * scale_2 + mhc_base[8 + i];
    }

    // ===== Stage 3: Sinkhorn normalize (warp-level, no block sync needed) =====
    {
        // First iteration: softmax rows + eps, then normalize cols
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            float row_max = -1e30f;
            #pragma unroll
            for (int c = 0; c < 4; c++) row_max = fmaxf(row_max, comb_mix_val[r * 4 + c]);
            float row_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float e = expf(comb_mix_val[r * 4 + c] - row_max) + sinkhorn_eps;
                comb_mix_val[r * 4 + c] = e;
                row_sum += e;
            }
            #pragma unroll
            for (int c = 0; c < 4; c++) comb_mix_val[r * 4 + c] /= row_sum;
        }
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float col_sum = 0.0f;
            #pragma unroll
            for (int r = 0; r < 4; r++) col_sum += comb_mix_val[r * 4 + c] + sinkhorn_eps;
            #pragma unroll
            for (int r = 0; r < 4; r++) comb_mix_val[r * 4 + c] /= col_sum;
        }

        for (int iter = 1; iter < sinkhorn_repeat; iter++) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                float row_sum = sinkhorn_eps * 4.0f;
                #pragma unroll
                for (int c = 0; c < 4; c++) row_sum += comb_mix_val[r * 4 + c];
                #pragma unroll
                for (int c = 0; c < 4; c++) comb_mix_val[r * 4 + c] /= row_sum;
            }
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float col_sum = sinkhorn_eps * 4.0f;
                #pragma unroll
                for (int r = 0; r < 4; r++) col_sum += comb_mix_val[r * 4 + c];
                #pragma unroll
                for (int r = 0; r < 4; r++) comb_mix_val[r * 4 + c] /= col_sum;
            }
        }
    }

    // ===== Stage 4: Write outputs =====
    if (tid < 4)  post_mix[seq_idx * 4 + tid] = post_mix_val[tid];
    if (tid < 16) comb_mix[seq_idx * 16 + tid] = comb_mix_val[tid];

    // ===== Stage 5: Apply mix =====
    __syncthreads();
    for (int h = tid; h < hidden_size; h += blockDim.x) {
        float val = 0.0f;
        #pragma unroll
        for (int m = 0; m < 4; m++) {
            val += (float)res_slice[m * hidden_size + h] * pre_mix_val[m];
        }
        layer_input[seq_idx * hidden_size + h] = (__FLOAT16__)val;
    }
}

template <typename T>
void test_tmp_kernel_opt(
    T* residual, float* fn, float* mhc_scale, float* mhc_base,
    float* post_mix, float* comb_mix, T* layer_input,
    int seq_len, int mhc_mult, int hidden_size, int mhc_mult3, int mhc_rgs,
    float rms_eps, float pre_eps, float sinkhorn_eps, float post_mult_val, int sinkhorn_repeat,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid = seq_len;
    big_fuse_kernel_opt<<<grid, block_size, 0, stream>>>(
        (const __FLOAT16__*)residual, fn, mhc_scale, mhc_base,
        post_mix, comb_mix, (__FLOAT16__*)layer_input,
        seq_len, mhc_mult, hidden_size, mhc_mult3, mhc_rgs,
        rms_eps, pre_eps, sinkhorn_eps, post_mult_val, sinkhorn_repeat
    );
}
