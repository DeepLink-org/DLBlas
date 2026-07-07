#pragma once
#include "helpers.cuh"
__global__ void hc_split_sinkhorn_kernel_opt(const float* __restrict__ mixes, const float* __restrict__ scale, const float* __restrict__ base,
    float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb,
    int B, int S, int HC, int iters, float eps) {
    int row=blockIdx.x;if(row>=B*S)return;
    int M3=(2+HC)*HC;const float*mx=mixes+row*M3;
    float*pr=pre+row*HC,*po=post+row*HC,*cb=comb+row*HC*HC;
    float s0=scale[0],s1=scale[1],s2=scale[2];
    float a0=mx[0]*s0+base[0],a1=mx[1]*s0+base[1],a2=mx[2]*s0+base[2],a3=mx[3]*s0+base[3];
    float b0=mx[4]*s1+base[4],b1=mx[5]*s1+base[5],b2=mx[6]*s1+base[6],b3=mx[7]*s1+base[7];
    float e0=expf(-a0),e1=expf(-a1),e2=expf(-a2),e3=expf(-a3);
    float f0=expf(-b0),f1=expf(-b1),f2=expf(-b2),f3=expf(-b3);
    pr[0]=1.f/(1.f+e0)+eps;pr[1]=1.f/(1.f+e1)+eps;pr[2]=1.f/(1.f+e2)+eps;pr[3]=1.f/(1.f+e3)+eps;
    po[0]=2.f/(1.f+f0);po[1]=2.f/(1.f+f1);po[2]=2.f/(1.f+f2);po[3]=2.f/(1.f+f3);
    float t0=mx[8]*s2+base[8],t1=mx[9]*s2+base[9],t2=mx[10]*s2+base[10],t3=mx[11]*s2+base[11];
    float t4=mx[12]*s2+base[12],t5=mx[13]*s2+base[13],t6=mx[14]*s2+base[14],t7=mx[15]*s2+base[15];
    float t8=mx[16]*s2+base[16],t9=mx[17]*s2+base[17],t10=mx[18]*s2+base[18],t11=mx[19]*s2+base[19];
    float t12=mx[20]*s2+base[20],t13=mx[21]*s2+base[21],t14=mx[22]*s2+base[22],t15=mx[23]*s2+base[23];
    float maxv=t0;maxv=fmaxf(maxv,t1);maxv=fmaxf(maxv,t2);maxv=fmaxf(maxv,t3);
    maxv=fmaxf(maxv,t4);maxv=fmaxf(maxv,t5);maxv=fmaxf(maxv,t6);maxv=fmaxf(maxv,t7);
    maxv=fmaxf(maxv,t8);maxv=fmaxf(maxv,t9);maxv=fmaxf(maxv,t10);maxv=fmaxf(maxv,t11);
    maxv=fmaxf(maxv,t12);maxv=fmaxf(maxv,t13);maxv=fmaxf(maxv,t14);maxv=fmaxf(maxv,t15);
    t0=expf(t0-maxv);t1=expf(t1-maxv);t2=expf(t2-maxv);t3=expf(t3-maxv);
    t4=expf(t4-maxv);t5=expf(t5-maxv);t6=expf(t6-maxv);t7=expf(t7-maxv);
    t8=expf(t8-maxv);t9=expf(t9-maxv);t10=expf(t10-maxv);t11=expf(t11-maxv);
    t12=expf(t12-maxv);t13=expf(t13-maxv);t14=expf(t14-maxv);t15=expf(t15-maxv);
    float inv_sum=1.f/(t0+t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11+t12+t13+t14+t15);
    t0=t0*inv_sum+eps;t1=t1*inv_sum+eps;t2=t2*inv_sum+eps;t3=t3*inv_sum+eps;
    t4=t4*inv_sum+eps;t5=t5*inv_sum+eps;t6=t6*inv_sum+eps;t7=t7*inv_sum+eps;
    t8=t8*inv_sum+eps;t9=t9*inv_sum+eps;t10=t10*inv_sum+eps;t11=t11*inv_sum+eps;
    t12=t12*inv_sum+eps;t13=t13*inv_sum+eps;t14=t14*inv_sum+eps;t15=t15*inv_sum+eps;
    float ir0=1.f/(t0+t4+t8+t12+eps),ir1=1.f/(t1+t5+t9+t13+eps),ir2=1.f/(t2+t6+t10+t14+eps),ir3=1.f/(t3+t7+t11+t15+eps);
    t0*=ir0;t4*=ir0;t8*=ir0;t12*=ir0;t1*=ir1;t5*=ir1;t9*=ir1;t13*=ir1;
    t2*=ir2;t6*=ir2;t10*=ir2;t14*=ir2;t3*=ir3;t7*=ir3;t11*=ir3;t15*=ir3;
    for(int r=0;r<iters-1;r++){
        ir0=1.f/(t0+t1+t2+t3);ir1=1.f/(t4+t5+t6+t7);ir2=1.f/(t8+t9+t10+t11);ir3=1.f/(t12+t13+t14+t15);
        t0*=ir0;t1*=ir0;t2*=ir0;t3*=ir0;t4*=ir1;t5*=ir1;t6*=ir1;t7*=ir1;
        t8*=ir2;t9*=ir2;t10*=ir2;t11*=ir2;t12*=ir3;t13*=ir3;t14*=ir3;t15*=ir3;
        ir0=1.f/(t0+t4+t8+t12);ir1=1.f/(t1+t5+t9+t13);ir2=1.f/(t2+t6+t10+t14);ir3=1.f/(t3+t7+t11+t15);
        t0*=ir0;t4*=ir0;t8*=ir0;t12*=ir0;t1*=ir1;t5*=ir1;t9*=ir1;t13*=ir1;
        t2*=ir2;t6*=ir2;t10*=ir2;t14*=ir2;t3*=ir3;t7*=ir3;t11*=ir3;t15*=ir3;
    }
    cb[0]=t0;cb[1]=t1;cb[2]=t2;cb[3]=t3;cb[4]=t4;cb[5]=t5;cb[6]=t6;cb[7]=t7;
    cb[8]=t8;cb[9]=t9;cb[10]=t10;cb[11]=t11;cb[12]=t12;cb[13]=t13;cb[14]=t14;cb[15]=t15;
}
static void test_tmp_kernel_opt(const float* mx,const float* sc,const float* ba,
    float* pr,float* po,float* cb,int B,int S,int HC,int iters,float eps,cudaStream_t s){
    int total=B*S;hc_split_sinkhorn_kernel_opt<<<total,1,0,s>>>(mx,sc,ba,pr,po,cb,B,S,HC,iters,eps);
}
