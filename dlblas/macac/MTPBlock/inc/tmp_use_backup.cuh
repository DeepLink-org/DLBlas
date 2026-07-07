static __device__ __forceinline__ float hc_weight_opt(int i, int j, int seed) {
    float v = sinf((float)(i * 127 + j * 31 + seed * 13) * 0.0174533f);
    v += cosf((float)(i * 73 + j * 17 - seed * 29) * 0.0174533f);
    return v * 0.01f;
}
__global__ void mtpblock_hc_fused_kernel_opt(const float* __restrict__ x, float* __restrict__ y, int B, int S, int HC, int D, float eps, int sinkhorn_iters) {
    int row = blockIdx.x; int rows = B * S; if (row >= rows) return;
    int HC_D = HC * D;
    extern __shared__ float smem[]; float* x_norm = smem; float* mixes = smem + HC_D;
    const float* x_row = x + row * HC_D;
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < HC_D; i += blockDim.x) { float v = __ldg(x_row + i); x_norm[i] = v; sum_sq += v * v; }
    for (int offset = 32; offset > 0; offset >>= 1) sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    float rms_r = rsqrtf(sum_sq / (float)HC_D + eps);
    for (int i = threadIdx.x; i < HC_D; i += blockDim.x) x_norm[i] *= rms_r;
    __syncthreads();
    // Use x_norm area for weight cache (no longer needed for x_norm values after mix computation)
    // But we need x_norm for weighted sum later! Cannot reuse.
    // Instead, cache one row of weights per thread iteration
    for (int i = threadIdx.x; i < HC; i += blockDim.x) {
        float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
        // Precompute 4 weight values per inner iteration
        for (int j = 0; j + 3 < HC_D; j += 4) {
            float w0 = hc_weight_opt(i, j, 0);
            float w1 = hc_weight_opt(i, j+1, 0);
            float w2 = hc_weight_opt(i, j+2, 0);
            float w3 = hc_weight_opt(i, j+3, 0);
            sum0 += x_norm[j] * w0;
            sum1 += x_norm[j+1] * w1;
            sum2 += x_norm[j+2] * w2;
            sum3 += x_norm[j+3] * w3;
        }
        mixes[i] = sum0 + sum1 + sum2 + sum3;
    }
    __syncthreads();
    float pre_vals[16];
    for (int h = threadIdx.x; h < HC; h += blockDim.x) pre_vals[h] = 1.f / (1.f + expf(-mixes[h])) + eps;
    float* y_row = y + row * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        y_row[d] = pre_vals[0]*x_norm[d] + pre_vals[1]*x_norm[D+d] + pre_vals[2]*x_norm[2*D+d] + pre_vals[3]*x_norm[3*D+d];
    }
}
template <typename T> void test_tmp_kernel_opt(T* input, T* output, int in_batch, int in_height, int in_channels, int in_width, int out_batch, int out_height, int out_channels, int out_width, int in_elems, int out_elems, cudaStream_t stream) {
    int B=in_batch,S=in_height,HC=in_channels,D=in_width,rows=B*S; float eps=1e-6f; int MIX_HC=(2+HC)*HC;
    int smem_size=(HC*D+MIX_HC)*sizeof(float); int block_size=256;
    while(block_size>512) block_size>>=1; if(block_size>HC*D) block_size=HC*D;
    mtpblock_hc_fused_kernel_opt<<<rows,block_size,smem_size,stream>>>((const float*)input,(float*)output,B,S,HC,D,eps,20);
}
