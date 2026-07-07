#pragma once
#include "helpers.cuh"
__global__ void apply_mix_kernel_ori(const uint16_t* __restrict__ x, const float* __restrict__ mix, uint16_t* __restrict__ out, int n0, int n1, int mhc, int h){
    int row=blockIdx.x; if(row>=n0*n1)return;
    int tid=threadIdx.x;
    for(int hi=tid;hi<h;hi+=blockDim.x){
        float sum=0.f;
        for(int m=0;m<mhc;m++) sum+=bf(x[((row*mhc)+m)*h+hi])*mix[row*mhc+m];
        out[row*h+hi]=fb(sum);
    }
}
static void test_tmp_kernel_ori(const uint16_t* x,const float* mix,uint16_t* out,int n0,int n1,int mhc,int h,cudaStream_t s){
    apply_mix_kernel_ori<<<n0*n1,256,0,s>>>(x,mix,out,n0,n1,mhc,h);
}
