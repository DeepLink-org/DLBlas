#pragma once
#include "helpers.cuh"

__global__ void engram_gate_bwd_kernel_opt(
    const uint16_t* __restrict__ grad_out, const uint16_t* __restrict__ x_data,
    const uint16_t* __restrict__ k_data, const uint16_t* __restrict__ v_data,
    const uint16_t* __restrict__ wh_data, const uint16_t* __restrict__ we_data,
    uint16_t* __restrict__ grad_x, uint16_t* __restrict__ grad_k,
    float* __restrict__ grad_v, float* __restrict__ grad_wh, float* __restrict__ grad_we,
    int T, int H, int D, float scalar, float eps) {

    int idx = blockIdx.x;
    int t = idx / H;
    int h = idx % H;
    if (t >= T) return;

    int tid = threadIdx.x;  // 0..63, 1 warp
    extern __shared__ float smem[];  // 0 bytes used, but declare for compatibility

    const uint16_t* go_th = grad_out + idx * D;
    const uint16_t* x_th  = x_data   + idx * D;
    const uint16_t* k_th  = k_data   + idx * D;
    const uint16_t* v_t   = v_data   + t * D;
    const uint16_t* wh_h  = wh_data  + h * D;
    const uint16_t* we_h  = we_data  + h * D;
    uint16_t* gx_th = grad_x + idx * D;
    uint16_t* gk_th = grad_k + idx * D;

    float invD = 1.0f / D;
    unsigned long long wmask = 0xFFFFFFFFFFFFFFFFULL;

    // Each thread handles element tid and tid+64 (matching cross-warp layout)
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    {
        float xv0 = bf(__ldg(x_th + tid)), kv0 = bf(__ldg(k_th + tid));
        float whv0 = bf(__ldg(wh_h + tid)), wev0 = bf(__ldg(we_h + tid));
        float gov0 = bf(__ldg(go_th + tid)), vtv0 = bf(__ldg(v_t + tid));
        s0 += xv0 * xv0; s1 += kv0 * kv0;
        s2 += xv0 * whv0 * kv0 * wev0; s3 += gov0 * vtv0;
    }
    {
        float xv1 = bf(__ldg(x_th + tid + 64)), kv1 = bf(__ldg(k_th + tid + 64));
        float whv1 = bf(__ldg(wh_h + tid + 64)), wev1 = bf(__ldg(we_h + tid + 64));
        float gov1 = bf(__ldg(go_th + tid + 64)), vtv1 = bf(__ldg(v_t + tid + 64));
        s0 += xv1 * xv1; s1 += kv1 * kv1;
        s2 += xv1 * whv1 * kv1 * wev1; s3 += gov1 * vtv1;
    }

    // Single-warp shuffle reduction (no __syncthreads needed!)
    for (int s = 32; s > 0; s >>= 1) {
        s0 += __shfl_down_sync(wmask, s0, s);
        s1 += __shfl_down_sync(wmask, s1, s);
        s2 += __shfl_down_sync(wmask, s2, s);
        s3 += __shfl_down_sync(wmask, s3, s);
    }

    // Thread 0 computes scalars and broadcasts via shuffle
    float sum_x2 = __shfl_sync(wmask, s0, 0);
    float sum_k2 = __shfl_sync(wmask, s1, 0);
    float raw_dot = __shfl_sync(wmask, s2, 0);
    float grad_gate = __shfl_sync(wmask, s3, 0);

    float rstd_x = rsqrtf(sum_x2 * invD + eps);
    float rstd_k = rsqrtf(sum_k2 * invD + eps);
    float dot = raw_dot * rstd_x * rstd_k * scalar;

    float abs_dot = fabsf(dot);
    float ssqrt = copysignf(sqrtf(fmaxf(abs_dot, 1e-6f)), dot);
    float gate = 1.f / (1.f + expf(-ssqrt));

    float gsg = grad_gate * gate * (1.f - gate);
    float dmask = (abs_dot >= 1e-6f) ? 1.f : 0.f;
    float gdot = gsg * dmask * 0.5f / sqrtf(fmaxf(abs_dot, 1e-6f));
    float grow = gdot * rstd_x * rstd_k * scalar;
    float grx = gdot * raw_dot * rstd_k * scalar;
    float grk = gdot * raw_dot * rstd_x * scalar;

    float rx3 = rstd_x * rstd_x * rstd_x;
    float rk3 = rstd_k * rstd_k * rstd_k;
    float neg_invD = -invD;

    // Broadcast remaining scalars from thread 0
    gate = __shfl_sync(wmask, gate, 0);
    grow = __shfl_sync(wmask, grow, 0);
    grx = __shfl_sync(wmask, grx, 0);
    grk = __shfl_sync(wmask, grk, 0);
    rx3 = __shfl_sync(wmask, rx3, 0);
    rk3 = __shfl_sync(wmask, rk3, 0);

    // Compute outputs for element tid and tid+64
    {
        float xv = bf(x_th[tid]), kv = bf(k_th[tid]);
        float whv = bf(wh_h[tid]), wev = bf(we_h[tid]);
        float gov = bf(go_th[tid]);
        gx_th[tid] = fb(gov + grow * whv * kv * wev + grx * xv * neg_invD * rx3);
        gk_th[tid] = fb(grow * wev * xv * whv + grk * kv * neg_invD * rk3);
    }
    {
        float xv = bf(x_th[tid + 64]), kv = bf(k_th[tid + 64]);
        float whv = bf(wh_h[tid + 64]), wev = bf(we_h[tid + 64]);
        float gov = bf(go_th[tid + 64]);
        gx_th[tid + 64] = fb(gov + grow * whv * kv * wev + grx * xv * neg_invD * rx3);
        gk_th[tid + 64] = fb(grow * wev * xv * whv + grk * kv * neg_invD * rk3);
    }

    // Atomics
    {
        float gov0 = bf(go_th[tid]);
        atomicAdd(&grad_v[t * D + tid], gov0 * gate);
        float xv0 = bf(x_th[tid]), kv0 = bf(k_th[tid]);
        float whv0 = bf(wh_h[tid]), wev0 = bf(we_h[tid]);
        atomicAdd(&grad_wh[h * D + tid], grow * kv0 * wev0 * xv0);
        atomicAdd(&grad_we[h * D + tid], grow * xv0 * whv0 * kv0);
    }
    {
        float gov1 = bf(go_th[tid + 64]);
        atomicAdd(&grad_v[t * D + tid + 64], gov1 * gate);
        float xv1 = bf(x_th[tid + 64]), kv1 = bf(k_th[tid + 64]);
        float whv1 = bf(wh_h[tid + 64]), wev1 = bf(we_h[tid + 64]);
        atomicAdd(&grad_wh[h * D + tid + 64], grow * kv1 * wev1 * xv1);
        atomicAdd(&grad_we[h * D + tid + 64], grow * xv1 * whv1 * kv1);
    }
}
static void test_tmp_kernel_opt(
    const uint16_t* go,const uint16_t* xd,const uint16_t* kd,const uint16_t* vd,
    const uint16_t* wh,const uint16_t* we,
    uint16_t* gx,uint16_t* gk,float* gv,float* gwh,float* gwe,
    int T,int H,int D,float scalar,float eps,cudaStream_t s){
    int total = T * H;
    // 64 threads = 1 warp, zero shared memory
    engram_gate_bwd_kernel_opt<<<total, 64, 0, s>>>(go,xd,kd,vd,wh,we,gx,gk,gv,gwh,gwe,T,H,D,scalar,eps);
}
