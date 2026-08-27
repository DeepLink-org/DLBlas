#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "my_compute.cuh"
#include "builtin/fa_params.h"
using K = myflash::xcore1000::MyTraits;
auto kp = &myflash::xcore1000::my_flash_fwd_kernel<mcFlashAttn::Flash_fwd_params>;
int main(int argc,char**argv){
    int B=1,H=32,SQ=(argc>1)?atoi(argv[1]):512,SK=SQ,D=128;
    int wu=(argc>2)?atoi(argv[2]):5,it=(argc>3)?atoi(argv[3]):30;
    int dump=(argc>4)?atoi(argv[4]):0;
    size_t qe=(size_t)B*H*SQ*D,le=(size_t)B*H*SQ;
    half*q,*k,*v,*o;float*lse;
    cudaMalloc(&q,qe*2);cudaMalloc(&k,qe*2);cudaMalloc(&v,qe*2);cudaMalloc(&o,qe*2);cudaMalloc(&lse,le*4);
    half*qh=(half*)malloc(qe*2),*kh=(half*)malloc(qe*2),*vh=(half*)malloc(qe*2);
    srand(42);for(size_t i=0;i<qe;i++){qh[i]=__float2half((float)((rand()%2000-1000))*0.0005f);kh[i]=__float2half((float)((rand()%2000-1000))*0.0005f);vh[i]=__float2half((float)((rand()%2000-1000))*0.0005f);}
    cudaMemcpy(q,qh,qe*2,cudaMemcpyHostToDevice);cudaMemcpy(k,kh,qe*2,cudaMemcpyHostToDevice);cudaMemcpy(v,vh,qe*2,cudaMemcpyHostToDevice);
    cudaMemset(o,0,qe*2);cudaMemset(lse,0,le*4);
    mcFlashAttn::Flash_fwd_params p;memset(&p,0,sizeof(p));
    p.q_ptr=q;p.k_ptr=k;p.v_ptr=v;p.o_ptr=o;p.softmax_lse_ptr=lse;
    p.q_batch_stride=(int64_t)H*SQ*D;p.k_batch_stride=(int64_t)H*SK*D;p.v_batch_stride=(int64_t)H*SK*D;
    p.q_row_stride=D;p.k_row_stride=D;p.v_row_stride=D;
    p.q_head_stride=SQ*D;p.k_head_stride=SK*D;p.v_head_stride=SK*D;
    p.o_batch_stride=(int64_t)H*SQ*D;p.o_row_stride=D;p.o_head_stride=SQ*D;
    p.b=B;p.seqlen_q=SQ;p.seqlen_k=SK;p.d=D;
    p.seqlen_q_rounded=(SQ+127)/128*128;p.seqlen_k_rounded=(SK+63)/64*64;
    p.h=H;p.h_k=H;p.h_h_k_ratio=1;
    p.scale_softmax=1.0f/sqrtf(128.f);p.scale_softmax_log2=p.scale_softmax*1.4426950408889634f;
    p.is_causal=false;p.num_splits=1;p.has_attn_mask=false;p.arch=1000;
    int nmb=(SQ+K::kBlockM-1)/K::kBlockM;dim3 grid(nmb,B,H);size_t sm=K::kSmemSize;
    if(sm>=32768)cudaFuncSetAttribute(kp,cudaFuncAttributeMaxDynamicSharedMemorySize,sm);
    for(int i=0;i<wu;i++)kp<<<grid,K::kNThreads,sm,0>>>(p,nmb,1);cudaDeviceSynchronize();
    cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);cudaEventRecord(s);
    for(int i=0;i<it;i++)kp<<<grid,K::kNThreads,sm,0>>>(p,nmb,1);
    cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);
    printf("myimpl S=%d: %.6f ms\n",SQ,ms/it);
    if(dump){
        half*oh=(half*)malloc(qe*2);cudaMemcpy(oh,o,qe*2,cudaMemcpyDeviceToHost);
        const char* dd=getenv("FA_DUMP_DIR");if(!dd)dd=".";
        char path[256];snprintf(path,256,"%s/fa_my_%d.bin",dd,SQ);
        FILE*f=fopen(path,"wb");
        int b=B,h=H,sq=SQ,d=D;fwrite(&b,4,1,f);fwrite(&h,4,1,f);fwrite(&sq,4,1,f);fwrite(&d,4,1,f);
        fwrite(qh,2,qe,f);fwrite(kh,2,qe,f);fwrite(vh,2,qe,f);fwrite(oh,2,qe,f);fclose(f);
        printf("dumped %s\n",path);free(oh);
    }
    cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(o);cudaFree(lse);free(qh);free(kh);free(vh);
    return 0;
}
