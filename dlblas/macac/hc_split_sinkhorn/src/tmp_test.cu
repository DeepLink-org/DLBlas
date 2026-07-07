#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"
struct Res{float ot,pt,rate;bool ok;};
__global__ void warm_up(){}
Res test(int B,int S,int HC,int iters,float eps,int wu,int tc,int em){
    Res r={0}; CUDA_INIT(); int M3=(2+HC)*HC;
    int in_n=B*S*M3, pre_n=B*S*HC, post_n=B*S*HC, comb_n=B*S*HC*HC;
    float *mx,*sc,*ba,*pr1,*po1,*cb1,*pr2,*po2,*cb2;
    float *mxc,*scc,*bac,*prc1,*poc1,*cbc1,*prc2,*poc2,*cbc2;
    cudaMalloc(&mx,in_n*4);cudaMalloc(&sc,3*4);cudaMalloc(&ba,M3*4);
    cudaMalloc(&pr1,pre_n*4);cudaMalloc(&po1,post_n*4);cudaMalloc(&cb1,comb_n*4);
    cudaMalloc(&pr2,pre_n*4);cudaMalloc(&po2,post_n*4);cudaMalloc(&cb2,comb_n*4);
    mxc=(float*)malloc(in_n*4);scc=(float*)malloc(3*4);bac=(float*)malloc(M3*4);
    prc1=(float*)malloc(pre_n*4);poc1=(float*)malloc(post_n*4);cbc1=(float*)malloc(comb_n*4);
    prc2=(float*)malloc(pre_n*4);poc2=(float*)malloc(post_n*4);cbc2=(float*)malloc(comb_n*4);
    for(int i=0;i<in_n;i++)mxc[i]=((i*7)%127)/127.f;
    scc[0]=0.5f;scc[1]=0.25f;scc[2]=1.0f;
    for(int i=0;i<M3;i++)bac[i]=0.1f*((i*3)%100)/100.f;
    memset(prc1,0,pre_n*4);memset(poc1,0,post_n*4);memset(cbc1,0,comb_n*4);
    memset(prc2,0,pre_n*4);memset(poc2,0,post_n*4);memset(cbc2,0,comb_n*4);
    cudaMemcpy(mx,mxc,in_n*4,cudaMemcpyHostToDevice);cudaMemcpy(sc,scc,3*4,cudaMemcpyHostToDevice);cudaMemcpy(ba,bac,M3*4,cudaMemcpyHostToDevice);
    cudaMemcpy(pr1,prc1,pre_n*4,cudaMemcpyHostToDevice);cudaMemcpy(po1,poc1,post_n*4,cudaMemcpyHostToDevice);cudaMemcpy(cb1,cbc1,comb_n*4,cudaMemcpyHostToDevice);
    cudaMemcpy(pr2,prc2,pre_n*4,cudaMemcpyHostToDevice);cudaMemcpy(po2,poc2,post_n*4,cudaMemcpyHostToDevice);cudaMemcpy(cb2,cbc2,comb_n*4,cudaMemcpyHostToDevice);
    cudaEvent_t st,en;
    if(em==0||em==1){float tt=0;cudaEventCreate(&st);cudaEventCreate(&en);
        for(int i=0;i<wu;i++)test_tmp_kernel_ori(mx,sc,ba,pr1,po1,cb1,B,S,HC,iters,eps,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(st,0);test_tmp_kernel_ori(mx,sc,ba,pr1,po1,cb1,B,S,HC,iters,eps,stream);cudaEventRecord(en,0);cudaEventSynchronize(en);float el=0;cudaEventElapsedTime(&el,st,en);tt+=el;}
        r.ot=tt/tc;printf("ori:%f ms\n",r.ot);cudaEventDestroy(st);cudaEventDestroy(en);}
    if(em==0||em==2){float tt=0;cudaEventCreate(&st);cudaEventCreate(&en);
        for(int i=0;i<wu;i++)test_tmp_kernel_opt(mx,sc,ba,pr2,po2,cb2,B,S,HC,iters,eps,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(st,0);test_tmp_kernel_opt(mx,sc,ba,pr2,po2,cb2,B,S,HC,iters,eps,stream);cudaEventRecord(en,0);cudaEventSynchronize(en);float el=0;cudaEventElapsedTime(&el,st,en);tt+=el;}
        r.pt=tt/tc;printf("opt:%f ms\n",r.pt);cudaEventDestroy(st);cudaEventDestroy(en);}
    cudaDeviceSynchronize();
    cudaMemcpy(prc1,pr1,pre_n*4,cudaMemcpyDeviceToHost);cudaMemcpy(poc1,po1,post_n*4,cudaMemcpyDeviceToHost);cudaMemcpy(cbc1,cb1,comb_n*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(prc2,pr2,pre_n*4,cudaMemcpyDeviceToHost);cudaMemcpy(poc2,po2,post_n*4,cudaMemcpyDeviceToHost);cudaMemcpy(cbc2,cb2,comb_n*4,cudaMemcpyDeviceToHost);
    if(em==0){r.rate=r.ot>0?r.pt/r.ot:0;
        bool pok=checkresult<float>(prc1,prc2,pre_n);
        bool ook=checkresult<float>(poc1,poc2,post_n);
        bool cok=checkresult<float>(cbc1,cbc2,comb_n);
        r.ok=pok&&ook&&cok;
    }else{r.rate=0;r.ok=true;}
    free(mxc);free(scc);free(bac);free(prc1);free(poc1);free(cbc1);free(prc2);free(poc2);free(cbc2);
    cudaFree(mx);cudaFree(sc);cudaFree(ba);cudaFree(pr1);cudaFree(po1);cudaFree(cb1);cudaFree(pr2);cudaFree(po2);cudaFree(cb2);
    return r;
}
int main(int a,char**v){
    int wu=a>1?atoi(v[1]):5,tc=a>2?atoi(v[2]):1000,em=a>3?atoi(v[3]):0;
    printf("<warm_up_count>%d</warm_up_count>\n",wu);printf("<test_count>%d</test_count>\n",tc);printf("<exec_mode>%d</exec_mode>\n",em);
    Res r=test(2,8,4,20,1e-06f,wu,tc,em);
    printf("<time_before_opt>%f ms</time_before_opt>\n",r.ot);printf("<time_after_opt>%f ms</time_after_opt>\n",r.pt);
    printf("<runtime_ratio> %f</runtime_ratio>\n",r.rate);printf("<precision>%s</precision>\n",r.ok?"True":"False");
}
