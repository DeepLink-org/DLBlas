# Copyright (c) 2025, DeepLink.
#include <cuda.h>
#include <cuda_fp8.h>
#include <iostream>

#include "fp8_convert.cuh"

extern "C" void launch(void* __raw_in, void* __raw_out) {
    auto in = reinterpret_cast<cutlass::float_e5m2_t*>(__raw_in);
    auto out = reinterpret_cast<cutlass::float_e4m3_t*>(__raw_out);
    cutlass::float_e5m2_t tmp_in;
    cutlass::float_e4m3_t tmp_out;
    cudaMemcpy(&tmp_in, in, 8, cudaMemcpyHostToHost);
    convert(tmp_out, tmp_in);
    cudaMemcpy(out, &tmp_out, 8, cudaMemcpyHostToHost);
}
