#pragma once
#include "helpers.cuh"

__global__ __launch_bounds__(512, 2)
void expand_kenel_bwd_kernel_opt(
    const float* __restrict__ x, float* __restrict__ out,
    int N0, int N1, int M, int H)
{
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int total_groups = N0 * N1 * (H / 4);
    if (idx4 >= total_groups) return;

    int N1H4 = N1 * (H / 4);
    int H4 = H / 4;
    int n0i = idx4 / N1H4;
    int rem = idx4 - n0i * N1H4;
    int n1j = rem / H4;
    int h4 = rem - n1j * H4;
    int h = h4 * 4;

    int out_base = ((n0i * N1 + n1j) * H) + h;
    int in_base = ((n0i * N1 + n1j) * M) * H + h;

    float4 s0 = __ldg((const float4*)(x + in_base));
    float4 s1 = __ldg((const float4*)(x + in_base + H));
    float4 s2 = __ldg((const float4*)(x + in_base + 2*H));
    float4 s3 = __ldg((const float4*)(x + in_base + 3*H));

    float4 result;
    result.x = s0.x + s1.x + s2.x + s3.x;
    result.y = s0.y + s1.y + s2.y + s3.y;
    result.z = s0.z + s1.z + s2.z + s3.z;
    result.w = s0.w + s1.w + s2.w + s3.w;

    *(float4*)(out + out_base) = result;
}

static void test_tmp_kernel_opt(const float* x, float* out,
                                 int N0, int N1, int M, int H,
                                 cudaStream_t s)
{
    int total_groups = N0 * N1 * (H / 4);
    int bs = 512;
    int g = (total_groups + bs - 1) / bs;
    expand_kenel_bwd_kernel_opt<<<g, bs, 0, s>>>(x, out, N0, N1, M, H);
}
