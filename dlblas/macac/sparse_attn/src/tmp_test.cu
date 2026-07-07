#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"
struct Res{float ot,pt,rate;bool ok;};
__global__ void warm_up(){}
Res test(int B,int M,int H,int D,int N,int K,int wu,int tc,int em){
    Res r={0}; CUDA_INIT(); float sc=powf((float)D,-0.5f);
    int qn=B*M*H*D, kvn=B*N*D, tkn=B*M*K, on=B*M*H*D;
    uint16_t *q,*kv,*o1,*o2,*qc,*kvc,*oc1,*oc2;
    int *topk,*topkc; float *sink,*sinkc;
    cudaMalloc(&q,qn*2);cudaMalloc(&kv,kvn*2);cudaMalloc(&topk,tkn*4);cudaMalloc(&sink,H*4);
    cudaMalloc(&o1,on*2);cudaMalloc(&o2,on*2);
    qc=(uint16_t*)malloc(qn*2);kvc=(uint16_t*)malloc(kvn*2);topkc=(int*)malloc(tkn*4);sinkc=(float*)malloc(H*4);
    oc1=(uint16_t*)malloc(on*2);oc2=(uint16_t*)malloc(on*2);
    for(int i=0;i<qn;i++){float v=((i*7)%127)/127.f;union{float f;uint32_t u;}t;t.f=v;qc[i]=(uint16_t)((t.u+((t.u>>16)&1)+0x7FFF)>>16);}
    for(int i=0;i<kvn;i++){float v=((i*13)%127)/127.f;union{float f;uint32_t u;}t;t.f=v;kvc[i]=(uint16_t)((t.u+((t.u>>16)&1)+0x7FFF)>>16);}
    for(int i=0;i<tkn;i++)topkc[i]=(i*3)%N;
    for(int i=0;i<H;i++)sinkc[i]=0.f;
    memset(oc1,0,on*2);memset(oc2,0,on*2);
    cudaMemcpy(q,qc,qn*2,cudaMemcpyHostToDevice);cudaMemcpy(kv,kvc,kvn*2,cudaMemcpyHostToDevice);
    cudaMemcpy(topk,topkc,tkn*4,cudaMemcpyHostToDevice);cudaMemcpy(sink,sinkc,H*4,cudaMemcpyHostToDevice);
    cudaMemcpy(o1,oc1,on*2,cudaMemcpyHostToDevice);cudaMemcpy(o2,oc2,on*2,cudaMemcpyHostToDevice);
    cudaEvent_t st,en;
    if(em==0||em==1){float tt=0;cudaEventCreate(&st);cudaEventCreate(&en);
        for(int i=0;i<wu;i++)test_tmp_kernel_ori(q,kv,topk,sink,o1,B,M,H,D,N,K,sc,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(st,0);test_tmp_kernel_ori(q,kv,topk,sink,o1,B,M,H,D,N,K,sc,stream);cudaEventRecord(en,0);cudaEventSynchronize(en);float el=0;cudaEventElapsedTime(&el,st,en);tt+=el;}
        r.ot=tt/tc;printf("ori:%f ms\n",r.ot);cudaEventDestroy(st);cudaEventDestroy(en);}
    if(em==0||em==2){float tt=0;cudaEventCreate(&st);cudaEventCreate(&en);
        for(int i=0;i<wu;i++)test_tmp_kernel_opt(q,kv,topk,sink,o2,B,M,H,D,N,K,sc,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(st,0);test_tmp_kernel_opt(q,kv,topk,sink,o2,B,M,H,D,N,K,sc,stream);cudaEventRecord(en,0);cudaEventSynchronize(en);float el=0;cudaEventElapsedTime(&el,st,en);tt+=el;}
        r.pt=tt/tc;printf("opt:%f ms\n",r.pt);cudaEventDestroy(st);cudaEventDestroy(en);}
    cudaDeviceSynchronize();cudaMemcpy(oc1,o1,on*2,cudaMemcpyDeviceToHost);cudaMemcpy(oc2,o2,on*2,cudaMemcpyDeviceToHost);
    if(em==0){r.rate=r.ot>0?r.pt/r.ot:0;r.ok=true;int d=0;for(int i=0;i<on;i++)if(oc1[i]!=oc2[i]){d++;if(d<=5)printf("M%d\n",i);}r.ok=(d==0);printf(r.ok?"[ok]\n":"[%d]\n",d);}
    else{r.rate=0;r.ok=true;}
    free(qc);free(kvc);free(topkc);free(sinkc);free(oc1);free(oc2);
    cudaFree(q);cudaFree(kv);cudaFree(topk);cudaFree(sink);cudaFree(o1);cudaFree(o2);return r;
}
int main(int a,char**v){
    int wu=a>1?atoi(v[1]):5,tc=a>2?atoi(v[2]):1000,em=a>3?atoi(v[3]):0;
    printf("<warm_up_count>%d</warm_up_count>\n",wu);printf("<test_count>%d</test_count>\n",tc);printf("<exec_mode>%d</exec_mode>\n",em);
    Res r=test(2,16,8,64,32,16,wu,tc,em);
    printf("<time_before_opt>%f ms</time_before_opt>\n",r.ot);printf("<time_after_opt>%f ms</time_after_opt>\n",r.pt);
    printf("<runtime_ratio> %f</runtime_ratio>\n",r.rate);printf("<precision>%s</precision>\n",r.ok?"True":"False");
}
