#ifndef ENGRAM_GATE_W_REDUCE_OPT_CUH
#define ENGRAM_GATE_W_REDUCE_OPT_CUH

#include "common.h"

// Best kernel: 256 threads + unroll by 4 + __ldg + pointer arithmetic
// runtime_ratio ~0.80 (20% improvement over baseline)
__global__ void engram_gate_w_reduce_kernel_opt(
    const float* __restrict__ grad_w_partial,
    const __FLOAT16__* __restrict__ weight_hidden,
    const __FLOAT16__* __restrict__ weight_embed,
    const float* __restrict__ grad_wh_ref,
    const float* __restrict__ grad_we_ref,
    float* __restrict__ grad_wh_out,
    float* __restrict__ grad_we_out,
    int B, int C, int H)
{
    int total_outs = C * H;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_outs) return;

    int c = tid / H;
    int h = tid % H;
    int base = c * H + h;
    int stride = C * H;

    float sum = 0.0f;
    const float* src = grad_w_partial + base;
    int b = 0;
    for (; b + 3 < B; b += 4) {
        sum += __ldg(src) + __ldg(src + stride) + __ldg(src + 2*stride) + __ldg(src + 3*stride);
        src += 4 * stride;
    }
    for (; b < B; b++) {
        sum += __ldg(src);
        src += stride;
    }

    float wh = (float)weight_hidden[base];
    float we = (float)weight_embed[base];
    grad_wh_out[base] = grad_wh_ref[base] + sum * we;
    grad_we_out[base] = grad_we_ref[base] + sum * wh;
}

void test_tmp_kernel_opt(
    float* grad_w_partial,
    __FLOAT16__* weight_hidden,
    __FLOAT16__* weight_embed,
    float* grad_wh_ref,
    float* grad_we_ref,
    float* grad_wh_out,
    float* grad_we_out,
    int B, int C, int H,
    cudaStream_t stream)
{
    int total_outs = C * H;
    const int block_size = 256;
    int num_blocks = (total_outs + block_size - 1) / block_size;
    engram_gate_w_reduce_kernel_opt<<<num_blocks, block_size, 0, stream>>>(
        grad_w_partial, weight_hidden, weight_embed,
        grad_wh_ref, grad_we_ref,
        grad_wh_out, grad_we_out,
        B, C, H);
}
#endif
