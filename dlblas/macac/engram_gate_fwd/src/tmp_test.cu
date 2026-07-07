#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"
struct Res{float ot,pt,rate;bool ok;};
__global__ void warm_up(){}
Res test(int T,int M,int D,int wu,int tc,int em){
    Res r={0}; CUDA_INIT();
    float scalar=powf((float)D,-0.5f), clamp=1e-6f, eps=1e-20f;
    int xn=T*M*D, vn=T*D, wn=M*D, dn=T*M, on=T*M*D;
    uint16_t *x,*k,*v,*wh,*we,*o1,*o2,*xc,*kc,*vc,*whc,*wec,*oc1,*oc2;
    float *dot1,*gate1,*rx1,*rk1,*dot2,*gate2,*rx2,*rk2;
    float *dotc1,*gatec1,*rxc1,*rkc1,*dotc2,*gatec2,*rxc2,*rkc2;
    cudaMalloc(&x,xn*2);cudaMalloc(&k,xn*2);cudaMalloc(&v,vn*2);cudaMalloc(&wh,wn*2);cudaMalloc(&we,wn*2);
    cudaMalloc(&o1,on*2);cudaMalloc(&o2,on*2);
    cudaMalloc(&dot1,dn*4);cudaMalloc(&gate1,dn*4);cudaMalloc(&rx1,dn*4);cudaMalloc(&rk1,dn*4);
    cudaMalloc(&dot2,dn*4);cudaMalloc(&gate2,dn*4);cudaMalloc(&rx2,dn*4);cudaMalloc(&rk2,dn*4);
    xc=(uint16_t*)malloc(xn*2);kc=(uint16_t*)malloc(xn*2);vc=(uint16_t*)malloc(vn*2);
    whc=(uint16_t*)malloc(wn*2);wec=(uint16_t*)malloc(wn*2);oc1=(uint16_t*)malloc(on*2);oc2=(uint16_t*)malloc(on*2);
    dotc1=(float*)malloc(dn*4);gatec1=(float*)malloc(dn*4);rxc1=(float*)malloc(dn*4);rkc1=(float*)malloc(dn*4);
    dotc2=(float*)malloc(dn*4);gatec2=(float*)malloc(dn*4);rxc2=(float*)malloc(dn*4);rkc2=(float*)malloc(dn*4);
    for(int i=0;i<xn;i++){float f=((i*7)%127)/127.f;union{float f;uint32_t u;}t;t.f=f;xc[i]=kc[i]=(uint16_t)((t.u+((t.u>>16)&1)+0x7FFF)>>16);}
    for(int i=0;i<vn;i++){float f=((i*13)%127)/127.f;union{float f;uint32_t u;}t;t.f=f;vc[i]=(uint16_t)((t.u+((t.u>>16)&1)+0x7FFF)>>16);}
    for(int i=0;i<wn;i++){float f=1.f+0.01f*((i*7)%127);union{float f;uint32_t u;}t;t.f=f;whc[i]=wec[i]=(uint16_t)((t.u+((t.u>>16)&1)+0x7FFF)>>16);}
    memset(oc1,0,on*2);memset(oc2,0,on*2);memset(dotc1,0,dn*4);memset(dotc2,0,dn*4);
    cudaMemcpy(x,xc,xn*2,cudaMemcpyHostToDevice);cudaMemcpy(k,kc,xn*2,cudaMemcpyHostToDevice);cudaMemcpy(v,vc,vn*2,cudaMemcpyHostToDevice);
    cudaMemcpy(wh,whc,wn*2,cudaMemcpyHostToDevice);cudaMemcpy(we,wec,wn*2,cudaMemcpyHostToDevice);
    cudaMemcpy(o1,oc1,on*2,cudaMemcpyHostToDevice);cudaMemcpy(o2,oc2,on*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dot1,dotc1,dn*4,cudaMemcpyHostToDevice);cudaMemcpy(dot2,dotc2,dn*4,cudaMemcpyHostToDevice);
    cudaMemcpy(gate1,gatec1,dn*4,cudaMemcpyHostToDevice);cudaMemcpy(gate2,gatec2,dn*4,cudaMemcpyHostToDevice);
    cudaMemcpy(rx1,rxc1,dn*4,cudaMemcpyHostToDevice);cudaMemcpy(rx2,rxc2,dn*4,cudaMemcpyHostToDevice);
    cudaMemcpy(rk1,rkc1,dn*4,cudaMemcpyHostToDevice);cudaMemcpy(rk2,rkc2,dn*4,cudaMemcpyHostToDevice);
    cudaEvent_t st,en;
    if(em==0||em==1){float tt=0;cudaEventCreate(&st);cudaEventCreate(&en);
        for(int i=0;i<wu;i++)test_tmp_kernel_ori(x,k,v,wh,we,o1,dot1,gate1,rx1,rk1,T,M,D,scalar,clamp,eps,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(st,0);test_tmp_kernel_ori(x,k,v,wh,we,o1,dot1,gate1,rx1,rk1,T,M,D,scalar,clamp,eps,stream);cudaEventRecord(en,0);cudaEventSynchronize(en);float el=0;cudaEventElapsedTime(&el,st,en);tt+=el;}
        r.ot=tt/tc;printf("ori:%f ms\n",r.ot);cudaEventDestroy(st);cudaEventDestroy(en);}
    if(em==0||em==2){float tt=0;cudaEventCreate(&st);cudaEventCreate(&en);
        for(int i=0;i<wu;i++)test_tmp_kernel_opt(x,k,v,wh,we,o2,dot2,gate2,rx2,rk2,T,M,D,scalar,clamp,eps,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(st,0);test_tmp_kernel_opt(x,k,v,wh,we,o2,dot2,gate2,rx2,rk2,T,M,D,scalar,clamp,eps,stream);cudaEventRecord(en,0);cudaEventSynchronize(en);float el=0;cudaEventElapsedTime(&el,st,en);tt+=el;}
        r.pt=tt/tc;printf("opt:%f ms\n",r.pt);cudaEventDestroy(st);cudaEventDestroy(en);}
    cudaDeviceSynchronize();
    cudaMemcpy(oc1,o1,on*2,cudaMemcpyDeviceToHost);cudaMemcpy(oc2,o2,on*2,cudaMemcpyDeviceToHost);
    if(em==0){r.rate=r.ot>0?r.pt/r.ot:0;r.ok=true;int d=0;for(int i=0;i<on;i++)if(oc1[i]!=oc2[i]){d++;if(d<=5)printf("M%d:r=%d t=%d\n",i,(int)oc1[i],(int)oc2[i]);}r.ok=(d==0);printf(r.ok?"[ok]\n":"[%d mismatches]\n",d);}
    else{r.rate=0;r.ok=true;}
    free(xc);free(kc);free(vc);free(whc);free(wec);free(oc1);free(oc2);free(dotc1);free(gatec1);free(rxc1);free(rkc1);free(dotc2);free(gatec2);free(rxc2);free(rkc2);
    cudaFree(x);cudaFree(k);cudaFree(v);cudaFree(wh);cudaFree(we);cudaFree(o1);cudaFree(o2);cudaFree(dot1);cudaFree(gate1);cudaFree(rx1);cudaFree(rk1);cudaFree(dot2);cudaFree(gate2);cudaFree(rx2);cudaFree(rk2);
    return r;
}
int main(int a,char**v){
    int wu=a>1?atoi(v[1]):5,tc=a>2?atoi(v[2]):1000,em=a>3?atoi(v[3]):0;
    printf("<warm_up_count>%d</warm_up_count>\n",wu);printf("<test_count>%d</test_count>\n",tc);printf("<exec_mode>%d</exec_mode>\n",em);
    Res r=test(4096,4,4096,wu,tc,em);
    printf("<time_before_opt>%f ms</time_before_opt>\n",r.ot);printf("<time_after_opt>%f ms</time_after_opt>\n",r.pt);
    printf("<runtime_ratio> %f</runtime_ratio>\n",r.rate);printf("<precision>%s</precision>\n",r.ok?"True":"False");
}
