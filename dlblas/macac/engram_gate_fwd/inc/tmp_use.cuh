#pragma once
#include "helpers.cuh"

__global__ void engram_gate_fwd_kernel_opt(const uint16_t* __restrict__ x, const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    const uint16_t* __restrict__ wh, const uint16_t* __restrict__ we,
    uint16_t* __restrict__ out, float* __restrict__ dot_out, float* __restrict__ gate_out,
    float* __restrict__ rstd_x_out, float* __restrict__ rstd_k_out,
    int T, int M, int D, float scalar, float clamp_val, float eps) {

    int idx = blockIdx.x;
    int tidx = idx / M;
    int m = idx % M;
    if (tidx >= T) return;

    const uint16_t* __restrict__ x_tm = x + (tidx * M + m) * D;
    const uint16_t* __restrict__ k_tm = k + (tidx * M + m) * D;
    const uint16_t* __restrict__ wh_m = wh + m * D;
    const uint16_t* __restrict__ we_m = we + m * D;
    const uint16_t* __restrict__ v_t = v + tidx * D;
    uint16_t* __restrict__ out_tm = out + (tidx * M + m) * D;

    int tid = threadIdx.x;
    int bs = blockDim.x;
    extern __shared__ float smem[];
    float* dot_smem = smem;
    float* x2_smem = smem + bs;
    float* k2_smem = smem + 2 * bs;

    // 8x unrolled reduction pass
    float partial_dot = 0.f, partial_x2 = 0.f, partial_k2 = 0.f;
    int d = tid;
    for (; d + 7*bs < D; d += 8*bs) {
        int d1=d+bs,d2=d+2*bs,d3=d+3*bs,d4=d+4*bs,d5=d+5*bs,d6=d+6*bs,d7=d+7*bs;
        float x[8], k_[8], w[8];
        x[0]=bf(x_tm[d]);k_[0]=bf(k_tm[d]);w[0]=bf(wh_m[d])*bf(we_m[d]);
        x[1]=bf(x_tm[d1]);k_[1]=bf(k_tm[d1]);w[1]=bf(wh_m[d1])*bf(we_m[d1]);
        x[2]=bf(x_tm[d2]);k_[2]=bf(k_tm[d2]);w[2]=bf(wh_m[d2])*bf(we_m[d2]);
        x[3]=bf(x_tm[d3]);k_[3]=bf(k_tm[d3]);w[3]=bf(wh_m[d3])*bf(we_m[d3]);
        x[4]=bf(x_tm[d4]);k_[4]=bf(k_tm[d4]);w[4]=bf(wh_m[d4])*bf(we_m[d4]);
        x[5]=bf(x_tm[d5]);k_[5]=bf(k_tm[d5]);w[5]=bf(wh_m[d5])*bf(we_m[d5]);
        x[6]=bf(x_tm[d6]);k_[6]=bf(k_tm[d6]);w[6]=bf(wh_m[d6])*bf(we_m[d6]);
        x[7]=bf(x_tm[d7]);k_[7]=bf(k_tm[d7]);w[7]=bf(wh_m[d7])*bf(we_m[d7]);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            partial_dot += x[i] * k_[i] * w[i];
            partial_x2 += x[i] * x[i];
            partial_k2 += k_[i] * k_[i];
        }
    }
    for (; d < D; d += bs) {
        float xv=bf(x_tm[d]),kv=bf(k_tm[d]),wv=bf(wh_m[d])*bf(we_m[d]);
        partial_dot += xv * kv * wv;
        partial_x2 += xv * xv;
        partial_k2 += kv * kv;
    }

    dot_smem[tid] = partial_dot;
    x2_smem[tid] = partial_x2;
    k2_smem[tid] = partial_k2;
    __syncthreads();

    for (int s = bs/2; s > 0; s >>= 1) {
        if (tid < s) {
            dot_smem[tid] += dot_smem[tid + s];
            x2_smem[tid] += x2_smem[tid + s];
            k2_smem[tid] += k2_smem[tid + s];
        }
        __syncthreads();
    }

    float raw_dot = dot_smem[0];
    float rstd_x = rsqrtf(x2_smem[0] / D + eps);
    float rstd_k = rsqrtf(k2_smem[0] / D + eps);
    float dot = raw_dot * rstd_x * rstd_k * scalar;
    float signed_sqrt = copysignf(sqrtf(fmaxf(fabsf(dot), clamp_val)), dot);
    float gate = 1.f / (1.f + expf(-signed_sqrt));

    dot_out[idx] = dot;
    gate_out[idx] = gate;
    rstd_x_out[idx] = rstd_x;
    rstd_k_out[idx] = rstd_k;

    // 8x unrolled output
    d = tid;
    for (; d + 7*bs < D; d += 8*bs) {
        int d1=d+bs,d2=d+2*bs,d3=d+3*bs,d4=d+4*bs,d5=d+5*bs,d6=d+6*bs,d7=d+7*bs;
        out_tm[d]=fb(bf(x_tm[d])+gate*bf(v_t[d]));
        out_tm[d1]=fb(bf(x_tm[d1])+gate*bf(v_t[d1]));
        out_tm[d2]=fb(bf(x_tm[d2])+gate*bf(v_t[d2]));
        out_tm[d3]=fb(bf(x_tm[d3])+gate*bf(v_t[d3]));
        out_tm[d4]=fb(bf(x_tm[d4])+gate*bf(v_t[d4]));
        out_tm[d5]=fb(bf(x_tm[d5])+gate*bf(v_t[d5]));
        out_tm[d6]=fb(bf(x_tm[d6])+gate*bf(v_t[d6]));
        out_tm[d7]=fb(bf(x_tm[d7])+gate*bf(v_t[d7]));
    }
    for (; d < D; d += bs) {
        out_tm[d]=fb(bf(x_tm[d])+gate*bf(v_t[d]));
    }
}
static void test_tmp_kernel_opt(const uint16_t* x,const uint16_t* k,const uint16_t* v,const uint16_t* wh,const uint16_t* we,
    uint16_t* out,float* dot_out,float* gate_out,float* rstd_x_out,float* rstd_k_out,
    int T,int M,int D,float scalar,float clamp_val,float eps,cudaStream_t s){
    int total = T * M, bs = 64;
    int shared_mem = 3 * bs * sizeof(float);
    engram_gate_fwd_kernel_opt<<<total, bs, shared_mem, s>>>(x,k,v,wh,we,out,dot_out,gate_out,rstd_x_out,rstd_k_out,T,M,D,scalar,clamp_val,eps);
}