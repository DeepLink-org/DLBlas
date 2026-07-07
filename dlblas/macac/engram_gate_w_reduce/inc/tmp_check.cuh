#ifndef ENGRAM_GATE_W_REDUCE_ORI_CUH
#define ENGRAM_GATE_W_REDUCE_ORI_CUH

#include "common.h"

// engram_gate_w_reduce baseline kernel
// Operation:
//   1. Reduce sum grad_w_partial along dim 0: [B, C, H] -> [C, H]
//   2. grad_wh_out = grad_wh_ref + sum * weight_embed
//   3. grad_we_out = grad_we_ref + sum * weight_hidden
// Shape: B=108, C=4 (hc_mult), H=4096 (hidden_size)

__global__ void engram_gate_w_reduce_kernel_ori(
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

    // Reduce sum over B dimension at position (c, h)
    float sum = 0.0f;
    int base_offset = c * H + h;
    int stride = C * H;
    const float* src = grad_w_partial + base_offset;
    for (int b = 0; b < B; b++) {
        sum += *src;
        src += stride;
    }

    // Elementwise: multiply-add with bf16 weights (cast to float)
    float wh = (float)weight_hidden[base_offset];
    float we = (float)weight_embed[base_offset];

    grad_wh_out[base_offset] = grad_wh_ref[base_offset] + sum * we;
    grad_we_out[base_offset] = grad_we_ref[base_offset] + sum * wh;
}

void test_tmp_kernel_ori(
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
    engram_gate_w_reduce_kernel_ori<<<num_blocks, block_size, 0, stream>>>(
        grad_w_partial, weight_hidden, weight_embed,
        grad_wh_ref, grad_we_ref,
        grad_wh_out, grad_we_out,
        B, C, H);
}
#endif
