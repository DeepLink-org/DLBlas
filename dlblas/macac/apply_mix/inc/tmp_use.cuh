#pragma once
#include "helpers.cuh"
__global__ __launch_bounds__(256,8) void apply_mix_kernel_opt(const uint16_t* __restrict__ x, const float* __restrict__ mix, uint16_t* __restrict__ out, int n0, int n1, int mhc, int h){
    int row=blockIdx.x; if(row>=n0*n1)return;
    int tid=threadIdx.x;
    const uint32_t* __restrict__ x_row2 = (const uint32_t*)(x + row * mhc * h);
    const float* __restrict__ mix_row = mix + row * mhc;
    float m0=mix_row[0], m1=mix_row[1], m2=mix_row[2], m3=mix_row[3];
    int h2=h/2; uint32_t* __restrict__ out2=(uint32_t*)(out+row*h);
    int hi2a=tid, hi2b=tid+256, hi2c=tid+512;
    bool has_c = (hi2c < h2);
    {uint32_t v0=x_row2[hi2a], v1=x_row2[h2+hi2a], v2=x_row2[2*h2+hi2a], v3=x_row2[3*h2+hi2a];
     out2[hi2a]=((uint32_t)fb(bf((uint16_t)v0)*m0+bf((uint16_t)v1)*m1+bf((uint16_t)v2)*m2+bf((uint16_t)v3)*m3))|
                 (((uint32_t)fb(bf((uint16_t)(v0>>16))*m0+bf((uint16_t)(v1>>16))*m1+bf((uint16_t)(v2>>16))*m2+bf((uint16_t)(v3>>16))*m3))<<16);}
    if(hi2b < h2){
        uint32_t v0=x_row2[hi2b], v1=x_row2[h2+hi2b], v2=x_row2[2*h2+hi2b], v3=x_row2[3*h2+hi2b];
        out2[hi2b]=((uint32_t)fb(bf((uint16_t)v0)*m0+bf((uint16_t)v1)*m1+bf((uint16_t)v2)*m2+bf((uint16_t)v3)*m3))|
                    (((uint32_t)fb(bf((uint16_t)(v0>>16))*m0+bf((uint16_t)(v1>>16))*m1+bf((uint16_t)(v2>>16))*m2+bf((uint16_t)(v3>>16))*m3))<<16);
    }
    if(has_c){
        uint32_t v0=x_row2[hi2c], v1=x_row2[h2+hi2c], v2=x_row2[2*h2+hi2c], v3=x_row2[3*h2+hi2c];
        out2[hi2c]=((uint32_t)fb(bf((uint16_t)v0)*m0+bf((uint16_t)v1)*m1+bf((uint16_t)v2)*m2+bf((uint16_t)v3)*m3))|
                    (((uint32_t)fb(bf((uint16_t)(v0>>16))*m0+bf((uint16_t)(v1>>16))*m1+bf((uint16_t)(v2>>16))*m2+bf((uint16_t)(v3>>16))*m3))<<16);
    }
}
static void test_tmp_kernel_opt(const uint16_t* x,const float* mix,uint16_t* out,int n0,int n1,int mhc,int h,cudaStream_t s){
    apply_mix_kernel_opt<<<n0*n1,256,0,s>>>(x,mix,out,n0,n1,mhc,h);
}
