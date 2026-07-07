#pragma once
// big_fuse baseline kernel: MHC pre-processing fused kernel
// Fuses: RMS-norm linear projection + split mixes + Sinkhorn normalize + weighted sum
// Input: residual [1, seq_len, mhc_mult, hidden_size] bf16
// Outputs: post_mix [seq_len, mhc_mult] f32, comb_mix [seq_len, mhc_mult, mhc_mult] f32,
//          layer_input [seq_len, hidden_size] bf16

__global__ void big_fuse_kernel_ori(
    const __FLOAT16__* __restrict__ residual,   // [seq_len, mhc_rgs] bf16 (flattened)
    const float* __restrict__ fn,               // [mhc_mult3, mhc_rgs] f32
    const float* __restrict__ mhc_scale,         // [3] f32
    const float* __restrict__ mhc_base,          // [mhc_mult3] f32
    float* __restrict__ post_mix,                // [seq_len, mhc_mult] f32
    float* __restrict__ comb_mix,                // [seq_len, mhc_mult, mhc_mult] f32
    __FLOAT16__* __restrict__ layer_input,       // [seq_len, hidden_size] bf16
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

    __shared__ float sdata[256];

    const __FLOAT16__* res_slice = residual + seq_idx * mhc_rgs;

    // ===== Stage 1: Compute all 24 dot products (matmul + rms norm) =====
    // Each thread accumulates 24 partial dot products across its stride of K=5120
    float partial[24];
    #pragma unroll
    for (int j = 0; j < 24; j++) partial[j] = 0.0f;

    // Single pass: accumulate all 24 dot products
    for (int k = tid; k < mhc_rgs; k += blockDim.x) {
        float r = (float)res_slice[k];
        #pragma unroll
        for (int j = 0; j < mhc_mult3; j++) {
            partial[j] += r * fn[j * mhc_rgs + k];
        }
    }

    // Reduce each of the 24 partials across threads
    float mixes[24];
    #pragma unroll
    for (int j = 0; j < mhc_mult3; j++) {
        sdata[tid] = partial[j];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) mixes[j] = sdata[0];
        __syncthreads();
    }

    // RMS normalize: mixes *= rsqrt(sum(mixes^2) / mhc_rgs + eps)
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
    // scale layout: [scale[0]*4, scale[1]*4, scale[2]*16]
    float scale_0 = mhc_scale[0];
    float scale_1 = mhc_scale[1];
    float scale_2 = mhc_scale[2];

    // Apply scale + base, then sigmoid for pre/post
    float pre_mix_val[4];
    float post_mix_val[4];
    float comb_mix_val[16];

    // Pre: mixes[0:4]
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float v = mixes[i] * scale_0 + mhc_base[i];
        pre_mix_val[i] = 1.0f / (1.0f + expf(-v)) + pre_eps;
    }
    // Post: mixes[4:8]
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float v = mixes[4 + i] * scale_1 + mhc_base[4 + i];
        post_mix_val[i] = 1.0f / (1.0f + expf(-v)) * post_mult_val;
    }
    // Comb: mixes[8:24]
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        comb_mix_val[i] = mixes[8 + i] * scale_2 + mhc_base[8 + i];
    }

    // ===== Stage 3: Sinkhorn doubly-stochastic normalization =====
    // Input: comb_mix_val[16] as [4, 4] matrix
    // Algorithm: softmax(-1) + eps, then iteratively normalize rows/cols
    {
        // First: softmax along last dim (rows) + eps
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            float row_max = -1e30f;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                row_max = fmaxf(row_max, comb_mix_val[r * 4 + c]);
            }
            float row_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float e = expf(comb_mix_val[r * 4 + c] - row_max) + sinkhorn_eps;
                comb_mix_val[r * 4 + c] = e;
                row_sum += e;
            }
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                comb_mix_val[r * 4 + c] /= row_sum;
            }
        }

        // Then normalize cols
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float col_sum = 0.0f;
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                col_sum += comb_mix_val[r * 4 + c] + sinkhorn_eps;
            }
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                comb_mix_val[r * 4 + c] /= col_sum;
            }
        }

        // Remaining iterations: alternate row/col normalization
        for (int iter = 1; iter < sinkhorn_repeat; iter++) {
            // Normalize rows
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                float row_sum = sinkhorn_eps * 4.0f;
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    row_sum += comb_mix_val[r * 4 + c];
                }
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    comb_mix_val[r * 4 + c] /= row_sum;
                }
            }
            // Normalize cols
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                float col_sum = sinkhorn_eps * 4.0f;
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    col_sum += comb_mix_val[r * 4 + c];
                }
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    comb_mix_val[r * 4 + c] /= col_sum;
                }
            }
        }
    }

    // ===== Stage 4: Write outputs =====
    // post_mix[seq_idx * 4 + i]
    if (tid < 4) {
        post_mix[seq_idx * 4 + tid] = post_mix_val[tid];
    }

    // comb_mix[seq_idx * 16 + i]
    if (tid < 16) {
        comb_mix[seq_idx * 16 + tid] = comb_mix_val[tid];
    }

    // ===== Stage 5: Apply mix - layer_input = (residual * pre_mix).sum(dim=-2) =====
    // residual shape: [mhc_mult, hidden_size] = [4, 1280]
    // result: sum over mhc_mult of residual[m, h] * pre_mix[m]
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
void test_tmp_kernel_ori(
    T* residual, float* fn, float* mhc_scale, float* mhc_base,
    float* post_mix, float* comb_mix, T* layer_input,
    int seq_len, int mhc_mult, int hidden_size, int mhc_mult3, int mhc_rgs,
    float rms_eps, float pre_eps, float sinkhorn_eps, float post_mult_val, int sinkhorn_repeat,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid = seq_len;
    big_fuse_kernel_ori<<<grid, block_size, 0, stream>>>(
        (const __FLOAT16__*)residual, fn, mhc_scale, mhc_base,
        post_mix, comb_mix, (__FLOAT16__*)layer_input,
        seq_len, mhc_mult, hidden_size, mhc_mult3, mhc_rgs,
        rms_eps, pre_eps, sinkhorn_eps, post_mult_val, sinkhorn_repeat
    );
}
