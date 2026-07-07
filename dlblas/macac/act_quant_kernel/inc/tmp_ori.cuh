// act_quant_kernel_ori: Per-group activation quantization baseline
// Computes: x_q = clamp(x / scale, fp8_min, fp8_max)
// where scale = max(|x|, per group) / fp8_max
// Input:  x     [B, D]   bf16 (__FLOAT16__)
// Output: x_q   [B, D]   bf16 (__FLOAT16__)  – quantized to fp8 range
//         x_s   [B, G]   float               – per-group scales, G = D/group_size
// Params: B, D, group_size, fp8_max, fp8_min

#define WARP_SIZE 64
#define MAX_BLOCK_DIM 512

__global__ void act_quant_kernel_ori(
    const __FLOAT16__* __restrict__ x,
    __FLOAT16__* __restrict__ x_q,
    float* __restrict__ x_s,
    int B,
    int D,
    int group_size,
    float fp8_max,
    float fp8_min)
{
    int row = blockIdx.x;
    if (row >= B) return;

    // Each block handles one row.
    // group_size == D when D % group_size == 0 and only one group per row.
    // General: G = D / group_size groups per row.
    int G = D / group_size;
    extern __shared__ float smem[];  // shared memory for reduction

    // For each group within this row, compute scale and quantize
    for (int g = 0; g < G; g++) {
        int tid = threadIdx.x;
        int group_start = g * group_size;

        // Step 1: Find max absolute value in this group (fp32 accumulation)
        float local_max = 0.0f;
        for (int i = tid; i < group_size; i += blockDim.x) {
            int idx = group_start + i;
            float val = (float)x[row * D + idx];
            float abs_val = fabsf(val);
            local_max = fmaxf(local_max, abs_val);
        }

        // Shared memory reduction
        smem[tid] = local_max;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smem[tid] = fmaxf(smem[tid], smem[tid + s]);
            }
            __syncthreads();
        }

        float amax = smem[0];
        amax = fmaxf(amax, 1e-10f);  // eps clamp

        // Step 2: Compute scale = amax / fp8_max
        float scale = amax / fp8_max;
        x_s[row * G + g] = scale;

        // Step 3: Quantize elements in this group
        for (int i = tid; i < group_size; i += blockDim.x) {
            int idx = group_start + i;
            float val = (float)x[row * D + idx];
            float scaled = val / scale;
            // Clamp to fp8 range
            if (scaled > fp8_max) scaled = fp8_max;
            if (scaled < fp8_min) scaled = fp8_min;
            x_q[row * D + idx] = (__FLOAT16__)scaled;
        }
        __syncthreads();  // ensure all writes complete before next group
    }
}

template <typename T>
void test_tmp_kernel_ori(
    T* x, T* x_q, float* x_s,
    int B, int D, int group_size,
    float fp8_max, float fp8_min,
    cudaStream_t stream)
{
    int G = D / group_size;
    int block_size = (group_size < MAX_BLOCK_DIM) ? group_size : MAX_BLOCK_DIM;
    // Round to next power of 2 for reduction
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = (bs > MAX_BLOCK_DIM) ? MAX_BLOCK_DIM : bs;
    int shared_mem = block_size * sizeof(float);

    act_quant_kernel_ori<<<B, block_size, shared_mem, stream>>>(
        x, x_q, x_s, B, D, group_size, fp8_max, fp8_min);
}
