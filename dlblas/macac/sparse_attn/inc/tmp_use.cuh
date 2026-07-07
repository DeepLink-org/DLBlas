#pragma once
#include "helpers.cuh"
__global__ void sparse_attn_kernel_opt(const uint16_t* __restrict__ q, const uint16_t* __restrict__ kv,
    const int* __restrict__ topk_idxs, const float* __restrict__ attn_sink,
    uint16_t* __restrict__ out, int B, int M, int H, int D, int N, int K, float scale) {
    int bmh=blockIdx.x,total=B*M*H;if(bmh>=total)return;
    int b=bmh/(M*H),rem=bmh%(M*H),m_idx=rem/H,h_idx=rem%H;
    int q_offs=((b*M+m_idx)*H+h_idx)*D,topk_offs=(b*M+m_idx)*K,tid=threadIdx.x;
    float sink=attn_sink[h_idx];
    const uint16_t* q_row=q+q_offs;uint16_t* out_row=out+q_offs;
    extern __shared__ float smem[];
    // Stage 1: compute all 16 scores using thread 0..15
    if(tid<K){int kv_idx=topk_idxs[topk_offs+tid];
        if(kv_idx>=0){const uint16_t* kv_row=kv+(b*N+kv_idx)*D;
            // Load q through __ldg for better cache behavior
            float p[32];
            for(int i=0;i<32;i++){
                float q0=bf(__ldg(q_row+i)), q1=bf(__ldg(q_row+i+32));
                float k0=bf(__ldg(kv_row+i)), k1=bf(__ldg(kv_row+i+32));
                p[i]=q0*k0+q1*k1;
            }
            for(int s=16;s>0;s>>=1)for(int i=0;i<s;i++)p[i]+=p[i+s];
            smem[tid]=p[0]*scale;}else smem[tid]=-1e30f;}
    __syncthreads();
    float scores[16],max_score=sink;
    for(int k=0;k<K;k++){scores[k]=smem[k];if(scores[k]>max_score)max_score=scores[k];}
    float exp_sum=expf(sink-max_score),attn_w[16];
    for(int k=0;k<K;k++){if(topk_idxs[topk_offs+k]>=0){attn_w[k]=expf(scores[k]-max_score);exp_sum+=attn_w[k];}else attn_w[k]=0.f;}
    float inv_sum=1.f/exp_sum;for(int k=0;k<K;k++)attn_w[k]*=inv_sum;
    // Stage 4: weighted sum with __ldg
    for(int d=tid;d<D;d+=blockDim.x){float oval=0.f;
        for(int k=0;k<K;k++){int kv_idx=topk_idxs[topk_offs+k];
            if(kv_idx>=0)oval+=attn_w[k]*bf(__ldg(kv+(b*N+kv_idx)*D+d));}
        out_row[d]=fb(oval);}
}
static void test_tmp_kernel_opt(const uint16_t* q,const uint16_t* kv,const int* topk,const float* sink,
    uint16_t* out,int B,int M,int H,int D,int N,int K,float scale,cudaStream_t s){
    int total=B*M*H,bs=64;sparse_attn_kernel_opt<<<total,bs,K*sizeof(float),s>>>(q,kv,topk,sink,out,B,M,H,D,N,K,scale);}
