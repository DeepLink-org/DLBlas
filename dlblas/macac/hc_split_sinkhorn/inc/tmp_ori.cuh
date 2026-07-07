#pragma once
#include "helpers.cuh"

__global__ void hc_split_sinkhorn_kernel_ori(const float* __restrict__ mixes, const float* __restrict__ scale, const float* __restrict__ base,
    float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb,
    int B, int S, int HC, int iters, float eps) {
    int row = blockIdx.x;
    int total = B * S;
    if (row >= total) return;

    const float* mx = mixes + row * ((2+HC)*HC);
    float* pr = pre + row * HC;
    float* po = post + row * HC;
    float* cb = comb + row * HC * HC;

    float s0=scale[0], s1=scale[1], s2=scale[2];

    // pre: sigmoid(x[:HC]*s0 + base[:HC]) + eps
    for(int i=0;i<HC;i++){
        float v=mx[i]*s0+base[i];
        pr[i]=1.f/(1.f+expf(-v))+eps;
    }
    // post: 2*sigmoid(x[HC:2*HC]*s1 + base[HC:2*HC])
    for(int i=0;i<HC;i++){
        float v=mx[HC+i]*s1+base[HC+i];
        po[i]=2.f/(1.f+expf(-v));
    }
    // comb: softmax + sinkhorn on x[2*HC:]
    int off=2*HC;
    float tmp[16]; // HC*HC max 4*4=16
    float maxv=-1e30f;
    for(int i=0;i<HC*HC;i++){
        tmp[i]=mx[off+i]*s2+base[off+i];
        maxv=fmaxf(maxv,tmp[i]);
    }
    float sum=0.f;
    for(int i=0;i<HC*HC;i++){tmp[i]=expf(tmp[i]-maxv);sum+=tmp[i];}
    for(int i=0;i<HC*HC;i++)tmp[i]=tmp[i]/sum+eps;
    // col normalize
    for(int j=0;j<HC;j++){
        float cs=0.f;
        for(int i=0;i<HC;i++)cs+=tmp[i*HC+j];
        float inv=1.f/(cs+eps);
        for(int i=0;i<HC;i++)tmp[i*HC+j]*=inv;
    }
    // sinkhorn iterations
    for(int r=0;r<iters-1;r++){
        for(int i=0;i<HC;i++){
            float rs=0.f;
            for(int j=0;j<HC;j++)rs+=tmp[i*HC+j];
            float inv=1.f/(rs+eps);
            for(int j=0;j<HC;j++)tmp[i*HC+j]*=inv;
        }
        for(int j=0;j<HC;j++){
            float cs=0.f;
            for(int i=0;i<HC;i++)cs+=tmp[i*HC+j];
            float inv=1.f/(cs+eps);
            for(int i=0;i<HC;i++)tmp[i*HC+j]*=inv;
        }
    }
    for(int i=0;i<HC*HC;i++)cb[i]=tmp[i];
}
static void test_tmp_kernel_ori(const float* mx,const float* sc,const float* ba,
    float* pr,float* po,float* cb,int B,int S,int HC,int iters,float eps,cudaStream_t s){
    int total=B*S;
    hc_split_sinkhorn_kernel_ori<<<total,1,0,s>>>(mx,sc,ba,pr,po,cb,B,S,HC,iters,eps);
}