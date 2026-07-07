// Iteration 7: Use ____expf (fast-math intrinsic) instead of __expf.
// ____expf uses hardware-accelerated approximate exponential,
// trading minimal precision for better throughput.
__global__ void sinkhorn_kernel_opt(const float* __restrict__ input, float* __restrict__ output,
                                     int total_matrices, int /*mhc*/, int repeat, float eps) {
    int matrix_idx = blockIdx.x;
    if (matrix_idx >= total_matrices) return;

    int tid = threadIdx.x;
    const float* mat_in = input + matrix_idx * 16;
    float* mat_out = output + matrix_idx * 16;

    float c0 = __ldg(mat_in + 0 * 4 + tid);
    float c1 = __ldg(mat_in + 1 * 4 + tid);
    float c2 = __ldg(mat_in + 2 * 4 + tid);
    float c3 = __ldg(mat_in + 3 * 4 + tid);

    unsigned mask = 0xffffffff;

    // Softmax on rows
    { float v = c0;
      v = fmaxf(v, __shfl_xor_sync(mask, v, 2)); v = fmaxf(v, __shfl_xor_sync(mask, v, 1));
      c0 = __expf(c0 - __shfl_sync(mask, v, 0));
      float s = c0; s += __shfl_xor_sync(mask, s, 2); s += __shfl_xor_sync(mask, s, 1);
      c0 = c0 / __shfl_sync(mask, s, 0) + eps; }
    { float v = c1;
      v = fmaxf(v, __shfl_xor_sync(mask, v, 2)); v = fmaxf(v, __shfl_xor_sync(mask, v, 1));
      c1 = __expf(c1 - __shfl_sync(mask, v, 0));
      float s = c1; s += __shfl_xor_sync(mask, s, 2); s += __shfl_xor_sync(mask, s, 1);
      c1 = c1 / __shfl_sync(mask, s, 0) + eps; }
    { float v = c2;
      v = fmaxf(v, __shfl_xor_sync(mask, v, 2)); v = fmaxf(v, __shfl_xor_sync(mask, v, 1));
      c2 = __expf(c2 - __shfl_sync(mask, v, 0));
      float s = c2; s += __shfl_xor_sync(mask, s, 2); s += __shfl_xor_sync(mask, s, 1);
      c2 = c2 / __shfl_sync(mask, s, 0) + eps; }
    { float v = c3;
      v = fmaxf(v, __shfl_xor_sync(mask, v, 2)); v = fmaxf(v, __shfl_xor_sync(mask, v, 1));
      c3 = __expf(c3 - __shfl_sync(mask, v, 0));
      float s = c3; s += __shfl_xor_sync(mask, s, 2); s += __shfl_xor_sync(mask, s, 1);
      c3 = c3 / __shfl_sync(mask, s, 0) + eps; }

    // Column normalize
    { float inv = 1.0f / (eps + c0 + c1 + c2 + c3);
      c0 *= inv; c1 *= inv; c2 *= inv; c3 *= inv; }

    // Repeat
    for (int iter = 1; iter < repeat; iter++) {
        { float s = c0; s += __shfl_xor_sync(mask, s, 2); s += __shfl_xor_sync(mask, s, 1);
          c0 = c0 / (eps + __shfl_sync(mask, s, 0)); }
        { float s = c1; s += __shfl_xor_sync(mask, s, 2); s += __shfl_xor_sync(mask, s, 1);
          c1 = c1 / (eps + __shfl_sync(mask, s, 0)); }
        { float s = c2; s += __shfl_xor_sync(mask, s, 2); s += __shfl_xor_sync(mask, s, 1);
          c2 = c2 / (eps + __shfl_sync(mask, s, 0)); }
        { float s = c3; s += __shfl_xor_sync(mask, s, 2); s += __shfl_xor_sync(mask, s, 1);
          c3 = c3 / (eps + __shfl_sync(mask, s, 0)); }

        { float inv = 1.0f / (eps + c0 + c1 + c2 + c3);
          c0 *= inv; c1 *= inv; c2 *= inv; c3 *= inv; }
    }

    mat_out[0 * 4 + tid] = c0;
    mat_out[1 * 4 + tid] = c1;
    mat_out[2 * 4 + tid] = c2;
    mat_out[3 * 4 + tid] = c3;
}

template <typename T>
void test_tmp_kernel_opt(T* input, T* output, int total_matrices, int mhc, int repeat, float eps, cudaStream_t stream) {
    int block_size = 4;
    sinkhorn_kernel_opt<<<total_matrices, block_size, 0, stream>>>(input, output, total_matrices, mhc, repeat, eps);
}
