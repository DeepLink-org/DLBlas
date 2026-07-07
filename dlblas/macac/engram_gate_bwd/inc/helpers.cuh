#pragma once
static __device__ __forceinline__ float bf(uint16_t v){union{uint32_t u;float f;}t;t.u=((uint32_t)v)<<16;return t.f;}
static __device__ __forceinline__ uint16_t fb(float v){union{float f;uint32_t u;}t;t.f=v;uint32_t b=t.u;return(uint16_t)((b+((b>>16)&1)+0x7FFF)>>16);}
