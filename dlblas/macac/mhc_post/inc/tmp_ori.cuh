#ifndef MHC_POST_ORI_CUH
#define MHC_POST_ORI_CUH

#include "common.h"

// Baseline kernel for mhc_post operator
// Computes: term2 = einsum('abmn,abmc->abnc', comb_res_mix, residual)
//           output = bf16(x * post_layer_mix + term2)
// Shapes: x(n0,n1,h), residual(n0,n1,mhc_mult,h),
//         post_layer_mix(n0,n1,mhc_mult,1), comb_res_mix(n0,n1,mhc_mult,mhc_mult)
//         output(n0,n1,mhc_mult,h)

__global__ void mhc_post_kernel_ori(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post_layer_mix,
    const float* __restrict__ comb_res_mix,
    __nv_bfloat16* __restrict__ output,
    int n0, int n1, int h, int mhc_mult
) {
    int total_bs = n0 * n1;
    int stride_h = mhc_mult * h;       // stride per batch_seq for residual/output
    int stride_crm = mhc_mult * mhc_mult; // stride per batch_seq for comb_res_mix

    for (int bs = blockIdx.x; bs < total_bs; bs += gridDim.x) {
        const __nv_bfloat16* x_bs = x + bs * h;
        const __nv_bfloat16* residual_bs = residual + bs * stride_h;
        const float* plm_bs = post_layer_mix + bs * mhc_mult;
        const float* crm_bs = comb_res_mix + bs * stride_crm;
        __nv_bfloat16* output_bs = output + bs * stride_h;

        for (int hi = threadIdx.x; hi < h; hi += blockDim.x) {
            float x_val = __bfloat162float(x_bs[hi]);

            // Load residual for all k channels at this h position
            float res0 = __bfloat162float(residual_bs[hi]);           // k=0
            float res1 = __bfloat162float(residual_bs[h + hi]);        // k=1
            float res2 = __bfloat162float(residual_bs[2 * h + hi]);    // k=2
            float res3 = __bfloat162float(residual_bs[3 * h + hi]);    // k=3

            // Load comb_res_mix for this batch_seq (4x4, register-cached)
            float crm00 = crm_bs[0],  crm01 = crm_bs[1],  crm02 = crm_bs[2],  crm03 = crm_bs[3];
            float crm10 = crm_bs[4],  crm11 = crm_bs[5],  crm12 = crm_bs[6],  crm13 = crm_bs[7];
            float crm20 = crm_bs[8],  crm21 = crm_bs[9],  crm22 = crm_bs[10], crm23 = crm_bs[11];
            float crm30 = crm_bs[12], crm31 = crm_bs[13], crm32 = crm_bs[14], crm33 = crm_bs[15];

            // Load post_layer_mix
            float plm0 = plm_bs[0];
            float plm1 = plm_bs[1];
            float plm2 = plm_bs[2];
            float plm3 = plm_bs[3];

            // m=0: term2 = crm00*res0 + crm01*res1 + crm02*res2 + crm03*res3
            float term2_0 = crm00 * res0 + crm01 * res1 + crm02 * res2 + crm03 * res3;
            output_bs[hi] = __float2bfloat16(x_val * plm0 + term2_0);

            // m=1
            float term2_1 = crm10 * res0 + crm11 * res1 + crm12 * res2 + crm13 * res3;
            output_bs[h + hi] = __float2bfloat16(x_val * plm1 + term2_1);

            // m=2
            float term2_2 = crm20 * res0 + crm21 * res1 + crm22 * res2 + crm23 * res3;
            output_bs[2 * h + hi] = __float2bfloat16(x_val * plm2 + term2_2);

            // m=3
            float term2_3 = crm30 * res0 + crm31 * res1 + crm32 * res2 + crm33 * res3;
            output_bs[3 * h + hi] = __float2bfloat16(x_val * plm3 + term2_3);
        }
    }
}

void test_tmp_kernel_ori(
    __nv_bfloat16* x, __nv_bfloat16* residual,
    float* post_layer_mix, float* comb_res_mix,
    __nv_bfloat16* output,
    int n0, int n1, int h, int mhc_mult,
    cudaStream_t stream
) {
    int total_bs = n0 * n1;
    int grid = (total_bs < 832) ? total_bs : 832;
    int block = 512;
    mhc_post_kernel_ori<<<grid, block, 0, stream>>>(
        x, residual, post_layer_mix, comb_res_mix, output,
        n0, n1, h, mhc_mult
    );
}
#endif
