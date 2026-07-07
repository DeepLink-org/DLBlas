#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"
struct Res{float ot,pt,rate;bool ok;};
__global__ void warm_up(){}
Res test(int n0,int n1,int mhc,int h,int wu,int tc,int em){
    Res r={0}; CUDA_INIT();
    int ie=n0*n1*mhc*h, oe=n0*n1*h;
    uint16_t *x,*o1,*o2,*xc,*oc1,*oc2; float *mix,*mc;
    cudaMalloc(&x,ie*2); cudaMalloc(&mix,n0*n1*mhc*4); cudaMalloc(&o1,oe*2); cudaMalloc(&o2,oe*2);
    xc=(uint16_t*)malloc(ie*2); mc=(float*)malloc(n0*n1*mhc*4); oc1=(uint16_t*)malloc(oe*2); oc2=(uint16_t*)malloc(oe*2);
    for(int i=0;i<ie;i++){float v=((i*7)%127)/127.f;union{float f;uint32_t u;}t;t.f=v;xc[i]=(uint16_t)((t.u+((t.u>>16)&1)+0x7FFF)>>16);}
    for(int i=0;i<n0*n1*mhc;i++)mc[i]=((i*3)%100)/100.f+0.5f;
    memset(oc1,0,oe*2); memset(oc2,0,oe*2);
    cudaMemcpy(x,xc,ie*2,cudaMemcpyHostToDevice); cudaMemcpy(mix,mc,n0*n1*mhc*4,cudaMemcpyHostToDevice);
    cudaMemcpy(o1,oc1,oe*2,cudaMemcpyHostToDevice); cudaMemcpy(o2,oc2,oe*2,cudaMemcpyHostToDevice);
    cudaEvent_t s,e;
    if(em==0||em==1){float tt=0;cudaEventCreate(&s);cudaEventCreate(&e);
        for(int i=0;i<wu;i++)test_tmp_kernel_ori(x,mix,o1,n0,n1,mhc,h,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(s,0);test_tmp_kernel_ori(x,mix,o1,n0,n1,mhc,h,stream);cudaEventRecord(e,0);cudaEventSynchronize(e);float el=0;cudaEventElapsedTime(&el,s,e);tt+=el;}
        r.ot=tt/tc; printf("ori avg: %f ms\n",r.ot); cudaEventDestroy(s);cudaEventDestroy(e);}
    if(em==0||em==2){float tt=0;cudaEventCreate(&s);cudaEventCreate(&e);
        for(int i=0;i<wu;i++)test_tmp_kernel_opt(x,mix,o2,n0,n1,mhc,h,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(s,0);test_tmp_kernel_opt(x,mix,o2,n0,n1,mhc,h,stream);cudaEventRecord(e,0);cudaEventSynchronize(e);float el=0;cudaEventElapsedTime(&el,s,e);tt+=el;}
        r.pt=tt/tc; printf("opt avg: %f ms\n",r.pt); cudaEventDestroy(s);cudaEventDestroy(e);}
    cudaDeviceSynchronize();
    cudaMemcpy(oc1,o1,oe*2,cudaMemcpyDeviceToHost); cudaMemcpy(oc2,o2,oe*2,cudaMemcpyDeviceToHost);
    if(em==0){r.rate=r.ot>0?r.pt/r.ot:0; r.ok=true; int d=0;for(int i=0;i<oe;i++)if(oc1[i]!=oc2[i]){d++;if(d<=3)printf("M%d:r=%d t=%d\n",i,(int)oc1[i],(int)oc2[i]);} r.ok=(d==0); printf(r.ok?"[out] ok\n":"[out] %d mismatches\n",d);}
    else{r.rate=0;r.ok=true;}
    free(xc);free(mc);free(oc1);free(oc2);cudaFree(x);cudaFree(mix);cudaFree(o1);cudaFree(o2);return r;
}
int main(int a,char**v){
    int wu=a>1?atoi(v[1]):5,tc=a>2?atoi(v[2]):1000,em=a>3?atoi(v[3]):0;
    printf("<warm_up_count>%d</warm_up_count>\n",wu);printf("<test_count>%d</test_count>\n",tc);printf("<exec_mode>%d</exec_mode>\n",em);
    Res r=test(2,1024,4,1280,wu,tc,em);
    printf("<time_before_opt>%f ms</time_before_opt>\n",r.ot);printf("<time_after_opt>%f ms</time_after_opt>\n",r.pt);
    printf("<runtime_ratio> %f</runtime_ratio>\n",r.rate);printf("<precision>%s</precision>\n",r.ok?"True":"False");
}
