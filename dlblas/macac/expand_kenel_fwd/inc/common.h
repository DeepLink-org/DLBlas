#include <cuda.h>
#include <cuda_fp16.h>
// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include<stdio.h>
#ifdef USE_MACA
typedef _Float16 __FLOAT16__;
// typedef half __FLOAT16__;
#else//!USE_MACA
typedef half __FLOAT16__;
#endif//USE_MACA

template <typename T>
struct ViaTypeMap {
    typedef T ViaT;
};
#define MIN(a,b) ((a) < (b) ? (a):(b))

#define CUDA_INIT() \
  cudaSetDevice(0); \
  cudaStream_t stream; \
  cudaStreamCreate(&stream);

#define MAX_DIMENSION 7 // should be acquired from ppl.common
#include <vector>
#include <stdint.h>
#include <assert.h>
