#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"
struct Res { float ot, pt, rate; bool ok; };
__global__ void warm_up() {}
Res test(int B, int N, int M, float eps, float pm, int wu, int tc, int em) {
    Res r = {0}; CUDA_INIT();
    int M3 = M*2+M*M;
    int in_elems = B * N * M3;
    int pre_elems = B * N * M;
    int post_elems = B * N * M;
    int comb_elems = B * N * M * M;

    float *x, *sc, *bs, *pre1, *post1, *comb1, *pre2, *post2, *comb2;
    float *xc, *scc, *bsc, *pr1, *po1, *co1, *pr2, *po2, *co2;
    cudaMalloc(&x, in_elems*4); cudaMalloc(&sc, 3*4); cudaMalloc(&bs, M3*4);
    cudaMalloc(&pre1, pre_elems*4); cudaMalloc(&post1, post_elems*4); cudaMalloc(&comb1, comb_elems*4);
    cudaMalloc(&pre2, pre_elems*4); cudaMalloc(&post2, post_elems*4); cudaMalloc(&comb2, comb_elems*4);
    xc=(float*)malloc(in_elems*4); scc=(float*)malloc(3*4); bsc=(float*)malloc(M3*4);
    pr1=(float*)malloc(pre_elems*4); po1=(float*)malloc(post_elems*4); co1=(float*)malloc(comb_elems*4);
    pr2=(float*)malloc(pre_elems*4); po2=(float*)malloc(post_elems*4); co2=(float*)malloc(comb_elems*4);

    for(int i=0;i<in_elems;i++) xc[i]=((i*7)%127)/127.f;
    for(int i=0;i<3;i++) scc[i]=0.1f*(i+1);
    for(int i=0;i<M3;i++) bsc[i]=0.1f*i;
    memset(pr1,0,pre_elems*4); memset(po1,0,post_elems*4); memset(co1,0,comb_elems*4);
    memset(pr2,0,pre_elems*4); memset(po2,0,post_elems*4); memset(co2,0,comb_elems*4);

    cudaMemcpy(x,xc,in_elems*4,cudaMemcpyHostToDevice); cudaMemcpy(sc,scc,3*4,cudaMemcpyHostToDevice); cudaMemcpy(bs,bsc,M3*4,cudaMemcpyHostToDevice);
    cudaMemcpy(pre1,pr1,pre_elems*4,cudaMemcpyHostToDevice); cudaMemcpy(post1,po1,post_elems*4,cudaMemcpyHostToDevice); cudaMemcpy(comb1,co1,comb_elems*4,cudaMemcpyHostToDevice);
    cudaMemcpy(pre2,pr2,pre_elems*4,cudaMemcpyHostToDevice); cudaMemcpy(post2,po2,post_elems*4,cudaMemcpyHostToDevice); cudaMemcpy(comb2,co2,comb_elems*4,cudaMemcpyHostToDevice);

    cudaEvent_t st, en;
    if (em==0||em==1) { float tt=0; cudaEventCreate(&st); cudaEventCreate(&en);
        for(int i=0;i<wu;i++) test_tmp_kernel_ori(x,sc,bs,pre1,post1,comb1,B,N,M,eps,pm,stream);
        for(int i=0;i<tc;i++){ cudaEventRecord(st,0); test_tmp_kernel_ori(x,sc,bs,pre1,post1,comb1,B,N,M,eps,pm,stream); cudaEventRecord(en,0); cudaEventSynchronize(en); float el=0; cudaEventElapsedTime(&el,st,en); tt+=el; }
        r.ot=tt/tc; printf("ori:%f ms\n",r.ot); cudaEventDestroy(st); cudaEventDestroy(en); }
    if (em==0||em==2) { float tt=0; cudaEventCreate(&st); cudaEventCreate(&en);
        for(int i=0;i<wu;i++) test_tmp_kernel_opt(x,sc,bs,pre2,post2,comb2,B,N,M,eps,pm,stream);
        for(int i=0;i<tc;i++){ cudaEventRecord(st,0); test_tmp_kernel_opt(x,sc,bs,pre2,post2,comb2,B,N,M,eps,pm,stream); cudaEventRecord(en,0); cudaEventSynchronize(en); float el=0; cudaEventElapsedTime(&el,st,en); tt+=el; }
        r.pt=tt/tc; printf("opt:%f ms\n",r.pt); cudaEventDestroy(st); cudaEventDestroy(en); }
    cudaDeviceSynchronize();
    cudaMemcpy(pr1,pre1,pre_elems*4,cudaMemcpyDeviceToHost); cudaMemcpy(pr2,pre2,pre_elems*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(po1,post1,post_elems*4,cudaMemcpyDeviceToHost); cudaMemcpy(po2,post2,post_elems*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(co1,comb1,comb_elems*4,cudaMemcpyDeviceToHost); cudaMemcpy(co2,comb2,comb_elems*4,cudaMemcpyDeviceToHost);
    if (em==0) { r.rate=r.ot>0?r.pt/r.ot:0;
        bool p_ok=checkresult<float>(pr1,pr2,pre_elems);
        bool po_ok=checkresult<float>(po1,po2,post_elems);
        bool c_ok=checkresult<float>(co1,co2,comb_elems);
        r.ok = p_ok && po_ok && c_ok;
    } else { r.rate=0; r.ok=true; }
    free(xc);free(scc);free(bsc);free(pr1);free(po1);free(co1);free(pr2);free(po2);free(co2);
    cudaFree(x);cudaFree(sc);cudaFree(bs);cudaFree(pre1);cudaFree(post1);cudaFree(comb1);cudaFree(pre2);cudaFree(post2);cudaFree(comb2);
    return r;
}
int main(int a, char** v) {
    int wu=a>1?atoi(v[1]):5, tc=a>2?atoi(v[2]):1000, em=a>3?atoi(v[3]):0;
    printf("<warm_up_count>%d</warm_up_count>\n",wu); printf("<test_count>%d</test_count>\n",tc); printf("<exec_mode>%d</exec_mode>\n",em);
    Res r = test(1,1024,4,1e-2f,2.0f,wu,tc,em);
    printf("<time_before_opt>%f ms</time_before_opt>\n",r.ot); printf("<time_after_opt>%f ms</time_after_opt>\n",r.pt);
    printf("<runtime_ratio> %f</runtime_ratio>\n",r.rate); printf("<precision>%s</precision>\n",r.ok?"True":"False");
}
