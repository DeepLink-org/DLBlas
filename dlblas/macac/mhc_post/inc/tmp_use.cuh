#ifndef MHC_POST_USE_CUH
#define MHC_POST_USE_CUH

#include "common.h"

// Iteration 9: grid=total_bs (one block per batch_seq), block=256, __ldg(), compact
// Eliminates grid-stride loop over batch_seq for simpler indexing

__global__ void mhc_post_kernel_opt(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post_layer_mix,
    const float* __restrict__ comb_res_mix,
    __nv_bfloat16* __restrict__ output,
    int n0, int n1, int h, int mhc_mult
) {
    int bs = blockIdx.y * gridDim.x + blockIdx.x;
    int total_bs = n0 * n1;
    if (bs >= total_bs) return;

    int stride_h = mhc_mult * h;
    int stride_crm = mhc_mult * mhc_mult;

    const __nv_bfloat16* __restrict__ x_bs = x + bs * h;
    const __nv_bfloat16* __restrict__ residual_bs = residual + bs * stride_h;
    const float* __restrict__ plm_bs = post_layer_mix + bs * mhc_mult;
    const float* __restrict__ crm_bs = comb_res_mix + bs * stride_crm;
    __nv_bfloat16* __restrict__ output_bs = output + bs * stride_h;

    float crm00 = __ldg(crm_bs),     crm01 = __ldg(crm_bs+1),  crm02 = __ldg(crm_bs+2),  crm03 = __ldg(crm_bs+3);
    float crm10 = __ldg(crm_bs+4),   crm11 = __ldg(crm_bs+5),  crm12 = __ldg(crm_bs+6),  crm13 = __ldg(crm_bs+7);
    float crm20 = __ldg(crm_bs+8),   crm21 = __ldg(crm_bs+9),  crm22 = __ldg(crm_bs+10), crm23 = __ldg(crm_bs+11);
    float crm30 = __ldg(crm_bs+12),  crm31 = __ldg(crm_bs+13), crm32 = __ldg(crm_bs+14), crm33 = __ldg(crm_bs+15);
    float plm0 = __ldg(plm_bs), plm1 = __ldg(plm_bs+1), plm2 = __ldg(plm_bs+2), plm3 = __ldg(plm_bs+3);

    for (int hi = threadIdx.x; hi < h; hi += blockDim.x) {
        float xv = __bfloat162float(__ldg(x_bs + hi));
        float r0 = __bfloat162float(__ldg(residual_bs + hi));
        float r1 = __bfloat162float(__ldg(residual_bs + h + hi));
        float r2 = __bfloat162float(__ldg(residual_bs + 2 * h + hi));
        float r3 = __bfloat162float(__ldg(residual_bs + 3 * h + hi));

        output_bs[hi]       = __float2bfloat16(xv * plm0 + crm00*r0 + crm01*r1 + crm02*r2 + crm03*r3);
        output_bs[h + hi]   = __float2bfloat16(xv * plm1 + crm10*r0 + crm11*r1 + crm12*r2 + crm13*r3);
        output_bs[2*h + hi] = __float2bfloat16(xv * plm2 + crm20*r0 + crm21*r1 + crm22*r2 + crm23*r3);
        output_bs[3*h + hi] = __float2bfloat16(xv * plm3 + crm30*r0 + crm31*r1 + crm32*r2 + crm33*r3);
    }
}

void test_tmp_kernel_opt(
    __nv_bfloat16* x, __nv_bfloat16* residual,
    float* post_layer_mix, float* comb_res_mix,
    __nv_bfloat16* output,
    int n0, int n1, int h, int mhc_mult,
    cudaStream_t stream
) {
    int total_bs = n0 * n1;
    int gy = (total_bs + 255) / 256;
    int gx = 256;
    dim3 grid(gx, gy);
    int block = 256;
    mhc_post_kernel_opt<<<grid, block, 0, stream>>>(
        x, residual, post_layer_mix, comb_res_mix, output,
        n0, n1, h, mhc_mult
    );
}
#endif
