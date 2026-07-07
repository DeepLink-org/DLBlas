#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "common.h"
#include "tmp_ori.cuh"
#include "tmp_use.cuh"
struct Res{float ot,pt,rate;bool ok;};
__global__ void warm_up(){}
Res test(int T,int H,int D,int wu,int tc,int em){
    Res r={0}; CUDA_INIT();
    float scalar=powf((float)D,-0.5f), eps=1e-20f;
    int xn=T*H*D, vn=T*D, wn=H*D, out_n=T*H*D, gv_n=T*D;
    uint16_t *go,*xd,*kd,*vd,*wh,*we,*gx1,*gk1,*gx2,*gk2;
    uint16_t *goc,*xdc,*kdc,*vdc,*whc,*wec,*gxc1,*gkc1,*gxc2,*gkc2;
    float *gv1,*gwh1,*gwe1,*gv2,*gwh2,*gwe2,*gvc1,*gwhc1,*gwec1,*gvc2,*gwhc2,*gwec2;
    cudaMalloc(&go,xn*2);cudaMalloc(&xd,xn*2);cudaMalloc(&kd,xn*2);cudaMalloc(&vd,vn*2);
    cudaMalloc(&wh,wn*2);cudaMalloc(&we,wn*2);
    cudaMalloc(&gx1,out_n*2);cudaMalloc(&gk1,out_n*2);cudaMalloc(&gv1,gv_n*4);cudaMalloc(&gwh1,wn*4);cudaMalloc(&gwe1,wn*4);
    cudaMalloc(&gx2,out_n*2);cudaMalloc(&gk2,out_n*2);cudaMalloc(&gv2,gv_n*4);cudaMalloc(&gwh2,wn*4);cudaMalloc(&gwe2,wn*4);
    goc=(uint16_t*)malloc(xn*2);xdc=(uint16_t*)malloc(xn*2);kdc=(uint16_t*)malloc(xn*2);vdc=(uint16_t*)malloc(vn*2);
    whc=(uint16_t*)malloc(wn*2);wec=(uint16_t*)malloc(wn*2);
    gxc1=(uint16_t*)malloc(out_n*2);gkc1=(uint16_t*)malloc(out_n*2);gvc1=(float*)calloc(gv_n,4);gwhc1=(float*)calloc(wn,4);gwec1=(float*)calloc(wn,4);
    gxc2=(uint16_t*)malloc(out_n*2);gkc2=(uint16_t*)malloc(out_n*2);gvc2=(float*)calloc(gv_n,4);gwhc2=(float*)calloc(wn,4);gwec2=(float*)calloc(wn,4);
    for(int i=0;i<xn;i++){float v=((i*7)%127)/127.f;union{float f;uint32_t u;}t;t.f=v;goc[i]=xdc[i]=kdc[i]=(uint16_t)((t.u+((t.u>>16)&1)+0x7FFF)>>16);}
    for(int i=0;i<vn;i++){float v=((i*13)%127)/127.f;union{float f;uint32_t u;}t;t.f=v;vdc[i]=(uint16_t)((t.u+((t.u>>16)&1)+0x7FFF)>>16);}
    for(int i=0;i<wn;i++){float v=1.f+0.01f*((i*7)%127);union{float f;uint32_t u;}t;t.f=v;whc[i]=wec[i]=(uint16_t)((t.u+((t.u>>16)&1)+0x7FFF)>>16);}
    memset(gxc1,0,out_n*2);memset(gkc1,0,out_n*2);memset(gxc2,0,out_n*2);memset(gkc2,0,out_n*2);
    cudaMemcpy(go,goc,xn*2,cudaMemcpyHostToDevice);cudaMemcpy(xd,xdc,xn*2,cudaMemcpyHostToDevice);
    cudaMemcpy(kd,kdc,xn*2,cudaMemcpyHostToDevice);cudaMemcpy(vd,vdc,vn*2,cudaMemcpyHostToDevice);
    cudaMemcpy(wh,whc,wn*2,cudaMemcpyHostToDevice);cudaMemcpy(we,wec,wn*2,cudaMemcpyHostToDevice);
    cudaMemcpy(gx1,gxc1,out_n*2,cudaMemcpyHostToDevice);cudaMemcpy(gk1,gkc1,out_n*2,cudaMemcpyHostToDevice);
    cudaMemcpy(gv1,gvc1,gv_n*4,cudaMemcpyHostToDevice);cudaMemcpy(gwh1,gwhc1,wn*4,cudaMemcpyHostToDevice);cudaMemcpy(gwe1,gwec1,wn*4,cudaMemcpyHostToDevice);
    cudaMemcpy(gx2,gxc2,out_n*2,cudaMemcpyHostToDevice);cudaMemcpy(gk2,gkc2,out_n*2,cudaMemcpyHostToDevice);
    cudaMemcpy(gv2,gvc2,gv_n*4,cudaMemcpyHostToDevice);cudaMemcpy(gwh2,gwhc2,wn*4,cudaMemcpyHostToDevice);cudaMemcpy(gwe2,gwec2,wn*4,cudaMemcpyHostToDevice);

    cudaEvent_t st,en;
    if(em==0||em==1){float tt=0;cudaEventCreate(&st);cudaEventCreate(&en);
        cudaMemcpy(gv1,gvc1,gv_n*4,cudaMemcpyHostToDevice);cudaMemcpy(gwh1,gwhc1,wn*4,cudaMemcpyHostToDevice);cudaMemcpy(gwe1,gwec1,wn*4,cudaMemcpyHostToDevice);
        for(int i=0;i<wu;i++)test_tmp_kernel_ori(go,xd,kd,vd,wh,we,gx1,gk1,gv1,gwh1,gwe1,T,H,D,scalar,eps,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(st,0);test_tmp_kernel_ori(go,xd,kd,vd,wh,we,gx1,gk1,gv1,gwh1,gwe1,T,H,D,scalar,eps,stream);cudaEventRecord(en,0);cudaEventSynchronize(en);float el=0;cudaEventElapsedTime(&el,st,en);tt+=el;}
        r.ot=tt/tc;printf("ori:%f ms\n",r.ot);cudaEventDestroy(st);cudaEventDestroy(en);}
    if(em==0||em==2){float tt=0;cudaEventCreate(&st);cudaEventCreate(&en);
        cudaMemcpy(gv2,gvc2,gv_n*4,cudaMemcpyHostToDevice);cudaMemcpy(gwh2,gwhc2,wn*4,cudaMemcpyHostToDevice);cudaMemcpy(gwe2,gwec2,wn*4,cudaMemcpyHostToDevice);
        for(int i=0;i<wu;i++)test_tmp_kernel_opt(go,xd,kd,vd,wh,we,gx2,gk2,gv2,gwh2,gwe2,T,H,D,scalar,eps,stream);
        for(int i=0;i<tc;i++){cudaEventRecord(st,0);test_tmp_kernel_opt(go,xd,kd,vd,wh,we,gx2,gk2,gv2,gwh2,gwe2,T,H,D,scalar,eps,stream);cudaEventRecord(en,0);cudaEventSynchronize(en);float el=0;cudaEventElapsedTime(&el,st,en);tt+=el;}
        r.pt=tt/tc;printf("opt:%f ms\n",r.pt);cudaEventDestroy(st);cudaEventDestroy(en);}
    cudaDeviceSynchronize();
    cudaMemcpy(gxc1,gx1,out_n*2,cudaMemcpyDeviceToHost);cudaMemcpy(gkc1,gk1,out_n*2,cudaMemcpyDeviceToHost);
    cudaMemcpy(gxc2,gx2,out_n*2,cudaMemcpyDeviceToHost);cudaMemcpy(gkc2,gk2,out_n*2,cudaMemcpyDeviceToHost);
    if(em==0){r.rate=r.ot>0?r.pt/r.ot:0;
        bool xok=true;int d=0;for(int i=0;i<out_n;i++)if(gxc1[i]!=gxc2[i]){d++;if(d<=5)printf("gx M%d\n",i);}xok=(d==0);
        bool kok=true;d=0;for(int i=0;i<out_n;i++)if(gkc1[i]!=gkc2[i]){d++;if(d<=5)printf("gk M%d\n",i);}kok=(d==0);
        r.ok=xok&&kok;printf(r.ok?"[ok]\n":"[gx:%d gk:%d]\n",d,d);
    }else{r.rate=0;r.ok=true;}
    free(goc);free(xdc);free(kdc);free(vdc);free(whc);free(wec);free(gxc1);free(gkc1);free(gvc1);free(gwhc1);free(gwec1);free(gxc2);free(gkc2);free(gvc2);free(gwhc2);free(gwec2);
    cudaFree(go);cudaFree(xd);cudaFree(kd);cudaFree(vd);cudaFree(wh);cudaFree(we);
    cudaFree(gx1);cudaFree(gk1);cudaFree(gv1);cudaFree(gwh1);cudaFree(gwe1);
    cudaFree(gx2);cudaFree(gk2);cudaFree(gv2);cudaFree(gwh2);cudaFree(gwe2);
    return r;
}
int main(int a,char**v){
    int wu=a>1?atoi(v[1]):5,tc=a>2?atoi(v[2]):1000,em=a>3?atoi(v[3]):0;
    printf("<warm_up_count>%d</warm_up_count>\n",wu);printf("<test_count>%d</test_count>\n",tc);printf("<exec_mode>%d</exec_mode>\n",em);
    Res r=test(14,4,128,wu,tc,em);
    printf("<time_before_opt>%f ms</time_before_opt>\n",r.ot);printf("<time_after_opt>%f ms</time_after_opt>\n",r.pt);
    printf("<runtime_ratio> %f</runtime_ratio>\n",r.rate);printf("<precision>%s</precision>\n",r.ok?"True":"False");
}
