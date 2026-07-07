// MTPBlock HC (Hyper-Connection) baseline kernels
// Implements hc_pre: x[b,s,hc,d] -> mixes -> (pre, post, comb) -> y[b,s,d]
// All HC weight parameters are generated deterministically inside the kernel
// to ensure reproducibility and correctness verification.

// ---------------------------------------------------------------------------
// Kernel 1: HC RMS Norm + Linear + Sinkhorn (fused baseline)
// Input:  x [B*S, HC, D] flattened as [B*S*HC*D]
// Output: y [B*S*D] — result of HC pre
//
// Math:
//   x_flat = x.reshape(B*S, HC*D)
//   rms = sqrt(mean(x_flat^2, dim=-1) + eps)
//   x_norm = x_flat * (1/rms)
//   mixes = x_norm @ hc_fn^T   [B*S, MIX_HC]
//   pre  = sigmoid(mixes[:,:HC]) + eps
//   post = 2*sigmoid(mixes[:,HC:2*HC])
//   comb = softmax(mixes[:,2*HC:].reshape(B*S,HC,HC), dim=-1) + eps
//   comb = sinkhorn(comb, iters)
//   y[b,d] = sum_hc pre[b,h] * x[b,h,d]
// ---------------------------------------------------------------------------

// Helper: deterministic weight generator
static __device__ __forceinline__ float hc_weight_ori(int i, int j, int seed) {
    // Deterministic weight generation using trigonometric mixing
    float v = sinf((float)(i * 127 + j * 31 + seed * 13) * 0.0174533f);
    v += cosf((float)(i * 73 + j * 17 - seed * 29) * 0.0174533f);
    return v * 0.01f;  // small initialization like nn.init.normal_(std=1e-4)
}

__global__ void mtpblock_hc_fused_kernel_ori(
    const float* __restrict__ x,
    float* __restrict__ y,
    int B, int S, int HC, int D,
    float eps, int sinkhorn_iters)
{
    int row = blockIdx.x;
    int rows = B * S;
    if (row >= rows) return;

    int HC_D = HC * D;
    int MIX_HC = (2 + HC) * HC;
    int HC2 = HC * HC;

    extern __shared__ float smem[];
    // smem layout:
    //   [0, HC_D)          : x_norm row
    //   [HC_D, HC_D+MIX_HC): mixes
    //   [HC_D+MIX_HC, ... ): comb matrix workspace

    float* x_norm = smem;
    float* mixes  = smem + HC_D;
    float* comb   = smem + HC_D + MIX_HC;  // [HC, HC] workspace

    // Step 1: Copy row + compute rms
    const float* x_row = x + row * HC_D;
    float sum_sq = 0.f;
    for (int i = threadIdx.x; i < HC_D; i += blockDim.x) {
        float v = x_row[i];
        x_norm[i] = v;
        sum_sq += v * v;
    }

    // Warp-level reduction for sum_sq
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    // Broadcast to all threads in warp 0
    float rms_r = rsqrtf(sum_sq / (float)HC_D + eps);

    // Apply RMS normalization
    for (int i = threadIdx.x; i < HC_D; i += blockDim.x) {
        x_norm[i] *= rms_r;
    }
    __syncthreads();

    // Step 2: Linear projection x_norm @ hc_fn^T -> mixes [MIX_HC]
    // hc_fn is [MIX_HC, HC_D], generated deterministically
    for (int i = threadIdx.x; i < MIX_HC; i += blockDim.x) {
        float sum = 0.f;
        for (int j = 0; j < HC_D; j++) {
            sum += x_norm[j] * hc_weight_ori(i, j, 0);
        }
        mixes[i] = sum;
    }
    __syncthreads();

    // Step 3: Compute pre (sigmoid + eps)
    float pre_vals[16];  // max HC=16
    for (int h = threadIdx.x; h < HC; h += blockDim.x) {
        float val = 1.f / (1.f + expf(-mixes[h])) + eps;
        pre_vals[h] = val;
    }

    // Step 4: Compute comb matrix: softmax then sinkhorn
    // Copy comb input from mixes[2*HC:]
    for (int i = threadIdx.x; i < HC2; i += blockDim.x) {
        comb[i] = mixes[2 * HC + i];
    }
    __syncthreads();

    // Softmax per row of comb
    for (int r = threadIdx.x; r < HC; r += blockDim.x) {
        // Find max
        float maxv = -1e30f;
        for (int c = 0; c < HC; c++) maxv = fmaxf(maxv, comb[r * HC + c]);
        // Exp sum
        float esum = 0.f;
        for (int c = 0; c < HC; c++) {
            float ev = expf(comb[r * HC + c] - maxv);
            comb[r * HC + c] = ev;
            esum += ev;
        }
        // Normalize
        for (int c = 0; c < HC; c++) {
            comb[r * HC + c] = comb[r * HC + c] / esum + eps;
        }
    }
    __syncthreads();

    // Sinkhorn iterations: alternate column-norm and row-norm
    for (int iter = 0; iter < sinkhorn_iters - 1; iter++) {
        // Column normalize
        for (int c = threadIdx.x; c < HC; c += blockDim.x) {
            float csum = 0.f;
            for (int r = 0; r < HC; r++) csum += comb[r * HC + c];
            csum += eps;
            for (int r = 0; r < HC; r++) comb[r * HC + c] /= csum;
        }
        __syncthreads();
        // Row normalize
        for (int r = threadIdx.x; r < HC; r += blockDim.x) {
            float rsum = 0.f;
            for (int c = 0; c < HC; c++) rsum += comb[r * HC + c];
            rsum += eps;
            for (int c = 0; c < HC; c++) comb[r * HC + c] /= rsum;
        }
        __syncthreads();
    }

    // Step 5: Weighted combination: y[row, d] = sum_h pre[row, h] * x[row, h, d]
    float* y_row = y + row * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float sum = 0.f;
        for (int h = 0; h < HC; h++) {
            sum += pre_vals[h] * x_norm[h * D + d];
        }
        y_row[d] = sum;
    }
}

// ---------------------------------------------------------------------------
// Test harness entry point: launches the fused HC kernel
// Input shape:  [B, S, HC, D]  -> pack as [B, S, HC*D, 1] or interpret through dims
// Output shape: [B, S, D]       -> pack as [B, S, D, 1]
//
// We use the test_tmp_* interface:
//   in_batch=B, in_height=S, in_channels=HC, in_width=D
//   out_batch=B, out_height=S, out_channels=D, out_width=1
// ---------------------------------------------------------------------------
template <typename T>
void test_tmp_kernel_ori(
    T* input, T* output,
    int in_batch, int in_height, int in_channels, int in_width,
    int out_batch, int out_height, int out_channels, int out_width,
    int in_elems, int out_elems,
    cudaStream_t stream)
{
    int B  = in_batch;
    int S  = in_height;
    int HC = in_channels;
    int D  = in_width;
    int rows = B * S;
    int HC_D = HC * D;

    float eps = 1e-6f;
    int sinkhorn_iters = 20;

    // Shared memory: HC_D floats for x_norm + MIX_HC floats for mixes + HC*HC for comb
    int MIX_HC = (2 + HC) * HC;
    int HC2 = HC * HC;
    int smem_size = (HC_D + MIX_HC + HC2) * sizeof(float);

    // Launch one block per row (B*S blocks)
    int block_size = 256;
    // Clamp block_size to next power of 2 <= 512
    while (block_size > 512) block_size >>= 1;
    if (block_size > HC_D) block_size = HC_D;

    mtpblock_hc_fused_kernel_ori<<<rows, block_size, smem_size, stream>>>(
        (const float*)input, (float*)output, B, S, HC, D, eps, sinkhorn_iters);
}
