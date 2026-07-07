#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"
struct Res{float ot,pt,rate;bool ok;};
__global__ void warm_up(){}
Res test(int N0,int N1,int M,int H,int wu,int tc,int em){
    Res r={0}; CUDA_INIT(); int ni=N0*N1*M*H, no=N0*N1*H;
    float *x,*o1,*o2,*xc,*oc1,*oc2;
    cudaMalloc(&x,ni*4); cudaMalloc(&o1,no*4); cudaMalloc(&o2,no*4);
    xc=(float*)malloc(ni*4); oc1=(float*)malloc(no*4); oc2=(float*)malloc(no*4);
    for(int i=0;i<ni;i++)xc[i]=((i*7)%127)/127.f;
    memset(oc1,0,no*4); memset(oc2,0,no*4);
    cudaMemcpy(x,xc,ni*4,cudaMemcpyHostToDevice); cudaMemcpy(o1,oc1,no*4,cudaMemcpyHostToDevice); cudaMemcpy(o2,oc2,no*4,cudaMemcpyHostToDevice);
    cudaEvent_t st,en;
    if(em==0||em==1){float tt=0;cudaEventCreate(&st);cudaEventCreate(&en);
        for(int i=0;i<wu;i++)test_tmp_kernel_ori(x,o1,N0,N1,M,H,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(st,0);test_tmp_kernel_ori(x,o1,N0,N1,M,H,stream);cudaEventRecord(en,0);cudaEventSynchronize(en);float el=0;cudaEventElapsedTime(&el,st,en);tt+=el;}
        r.ot=tt/tc;printf("ori:%f ms\n",r.ot);cudaEventDestroy(st);cudaEventDestroy(en);}
    if(em==0||em==2){float tt=0;cudaEventCreate(&st);cudaEventCreate(&en);
        for(int i=0;i<wu;i++)test_tmp_kernel_opt(x,o2,N0,N1,M,H,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(st,0);test_tmp_kernel_opt(x,o2,N0,N1,M,H,stream);cudaEventRecord(en,0);cudaEventSynchronize(en);float el=0;cudaEventElapsedTime(&el,st,en);tt+=el;}
        r.pt=tt/tc;printf("opt:%f ms\n",r.pt);cudaEventDestroy(st);cudaEventDestroy(en);}
    cudaDeviceSynchronize();cudaMemcpy(oc1,o1,no*4,cudaMemcpyDeviceToHost);cudaMemcpy(oc2,o2,no*4,cudaMemcpyDeviceToHost);
    if(em==0){r.rate=r.ot>0?r.pt/r.ot:0;r.ok=checkresult<float>(oc1,oc2,no);}else{r.rate=0;r.ok=true;}
    free(xc);free(oc1);free(oc2);cudaFree(x);cudaFree(o1);cudaFree(o2);return r;
}
int main(int a,char**v){
    int wu=a>1?atoi(v[1]):5,tc=a>2?atoi(v[2]):1000,em=a>3?atoi(v[3]):0;
    printf("<warm_up_count>%d</warm_up_count>\n",wu);printf("<test_count>%d</test_count>\n",tc);printf("<exec_mode>%d</exec_mode>\n",em);
    Res r=test(2,1024,4,1280,wu,tc,em);
    printf("<time_before_opt>%f ms</time_before_opt>\n",r.ot);printf("<time_after_opt>%f ms</time_after_opt>\n",r.pt);
    printf("<runtime_ratio> %f</runtime_ratio>\n",r.rate);printf("<precision>%s</precision>\n",r.ok?"True":"False");
}
