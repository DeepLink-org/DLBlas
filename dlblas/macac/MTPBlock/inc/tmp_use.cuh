// Iteration 10: Precompute HC weight table in shared memory
// Each block precomputes the weight table once, then all threads reuse it
// This eliminates sinf/cosf per element, replacing with a shared memory lookup

static __device__ __forceinline__ float hc_weight_opt(int i, int j, int seed) {
    float v = sinf((float)(i * 127 + j * 31 + seed * 13) * 0.0174533f);
    v += cosf((float)(i * 73 + j * 17 - seed * 29) * 0.0174533f);
    return v * 0.01f;
}

__global__ void mtpblock_hc_fused_kernel_opt(
    const float* __restrict__ x, float* __restrict__ y,
    int B, int S, int HC, int D, float eps, int sinkhorn_iters)
{
    int row = blockIdx.x;
    int rows = B * S;
    if (row >= rows) return;

    int HC_D = HC * D;
    int MIX_HC = (2 + HC) * HC;
    
    extern __shared__ float smem[];
    // Layout:
    // [0, HC_D): x_norm row
    // [HC_D, HC_D + MIX_HC): mixes
    // [HC_D + MIX_HC, HC_D + MIX_HC + HC * HC_D): weight table (HC rows of HC_D cols)
    
    float* x_norm = smem;
    float* mixes = smem + HC_D;
    float* weight_tbl = smem + HC_D + MIX_HC;
    
    const float* x_row = x + row * HC_D;
    
    // Step 1: Load x into shared memory + compute sum of squares
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < HC_D; i += blockDim.x) {
        float v = __ldg(x_row + i);
        x_norm[i] = v;
        sum_sq += v * v;
    }
    for (int offset = 32; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    
    float rms_r = rsqrtf(sum_sq / (float)HC_D + eps);
    for (int i = threadIdx.x; i < HC_D; i += blockDim.x)
        x_norm[i] *= rms_r;
    
    // Step 2: Precompute weight table in shared memory (once per block!)
    // Each thread computes a portion of the weight table
    int total_weights = HC * HC_D;
    for (int idx = threadIdx.x; idx < total_weights; idx += blockDim.x) {
        int i = idx / HC_D;
        int j = idx % HC_D;
        // Precompute the weight deterministically
        weight_tbl[idx] = hc_weight_opt(i, j, 0);
    }
    __syncthreads();
    
    // Step 3: Compute mixes using precomputed weight table
    for (int i = threadIdx.x; i < HC; i += blockDim.x) {
        float sum = 0.f;
        const float* wt = weight_tbl + i * HC_D;
        for (int j = 0; j < HC_D; j++) {
            sum += x_norm[j] * wt[j];
        }
        mixes[i] = sum;
    }
    __syncthreads();
    
    // Step 4: Compute pre values (sigmoid)
    float pre_vals[16];
    for (int h = threadIdx.x; h < HC; h += blockDim.x)
        pre_vals[h] = 1.f / (1.f + expf(-mixes[h])) + eps;
    
    // Step 5: Weighted sum output
    float* y_row = y + row * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        y_row[d] = pre_vals[0] * x_norm[d]
                 + pre_vals[1] * x_norm[D + d]
                 + pre_vals[2] * x_norm[2 * D + d]
                 + pre_vals[3] * x_norm[3 * D + d];
    }
}

template <typename T>
void test_tmp_kernel_opt(T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems, cudaStream_t stream)
{
    int B = in_batch, S = in_height, HC = in_channels, D = in_width;
    int rows = B * S;
    int HC_D = HC * D;
    int MIX_HC = (2 + HC) * HC;
    float eps = 1e-6f;
    
    // smem = x_norm(HC_D) + mixes(MIX_HC) + weight_table(HC * HC_D)
    int smem_size = (HC_D + MIX_HC + HC * HC_D) * sizeof(float);
    
    int block_size = 256;
    if (block_size > HC_D) block_size = HC_D;
    
    mtpblock_hc_fused_kernel_opt<<<rows, block_size, smem_size, stream>>>(
        (const float*)input, (float*)output, B, S, HC, D, eps, 20);
}
