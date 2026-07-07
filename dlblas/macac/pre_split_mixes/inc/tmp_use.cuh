#pragma once
#include "helpers.cuh"

__global__ void pre_split_mixes_kernel_opt(const float* __restrict__ x, const float* __restrict__ scale, const float* __restrict__ base,
    float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb,
    int B, int N, int M, float eps, float post_mult) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * N;
    if (idx >= total) return;

    int M2 = M * 2;
    int M3 = M2 + M * M;
    const float* row = x + idx * M3;
    float* pre_row = pre + idx * M;
    float* post_row = post + idx * M;
    float* comb_row = comb + idx * M * M;

    float s0 = scale[0], s1 = scale[1], s2 = scale[2];
    // Use __expf (device-native faster exp) for sigmoid
    #pragma unroll
    for (int m = 0; m < M; m++) {
        float v0 = row[m] * s0 + base[m];
        pre_row[m] = __fdividef(1.f, 1.f + __expf(-v0)) + eps;
    }
    #pragma unroll
    for (int m = 0; m < M; m++) {
        float v1 = row[M + m] * s1 + base[M + m];
        post_row[m] = __fdividef(1.f, 1.f + __expf(-v1)) * post_mult;
    }
    #pragma unroll
    for (int m = 0; m < M * M; m++) {
        comb_row[m] = row[M2 + m] * s2 + base[M2 + m];
    }
}
static void test_tmp_kernel_opt(const float* x, const float* scale, const float* base,
    float* pre, float* post, float* comb,
    int B, int N, int M, float eps, float post_mult, cudaStream_t s) {
    int total = B * N, bs = 32, g = (total + bs - 1) / bs;
    pre_split_mixes_kernel_opt<<<g, bs, 0, s>>>(x, scale, base, pre, post, comb, B, N, M, eps, post_mult);
}
