__global__ void sinkhorn_kernel_ori(const float* __restrict__ input, float* __restrict__ output,
                                     int total_matrices, int mhc, int repeat, float eps) {
    int matrix_idx = blockIdx.x;
    if (matrix_idx >= total_matrices) return;

    extern __shared__ float sdata[];
    // sdata layout: [mhc * mhc] matrix, then [blockDim.x] workspace
    float* smat = sdata;                    // mhc*mhc for matrix storage
    float* swork = sdata + mhc * mhc;       // blockDim.x for reduction workspace

    int tid = threadIdx.x;
    int elems_per_mat = mhc * mhc;
    const float* mat_in = input + matrix_idx * elems_per_mat;
    float* mat_out = output + matrix_idx * elems_per_mat;

    // Load matrix into shared memory
    if (tid < elems_per_mat) {
        smat[tid] = mat_in[tid];
    }
    __syncthreads();

    // === Step 1: softmax on rows (dim=-1) ===
    for (int r = 0; r < mhc; r++) {
        int row_offset = r * mhc;
        // Find max in this row
        float max_val = -1e30f;
        for (int c = tid; c < mhc; c += blockDim.x) {
            max_val = fmaxf(max_val, smat[row_offset + c]);
        }
        swork[tid] = max_val;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) swork[tid] = fmaxf(swork[tid], swork[tid + s]);
            __syncthreads();
        }
        float row_max = swork[0];
        __syncthreads();

        // Compute exp sum
        float sum_val = 0.f;
        for (int c = tid; c < mhc; c += blockDim.x) {
            sum_val += expf(smat[row_offset + c] - row_max);
        }
        swork[tid] = sum_val;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) swork[tid] += swork[tid + s];
            __syncthreads();
        }
        float row_sum = swork[0];
        __syncthreads();

        // Normalize + add eps
        for (int c = tid; c < mhc; c += blockDim.x) {
            smat[row_offset + c] = expf(smat[row_offset + c] - row_max) / row_sum + eps;
        }
        __syncthreads();
    }

    // === Step 2: column normalize (sum along dim=-2) ===
    for (int c = 0; c < mhc; c++) {
        float col_sum = eps;
        for (int r = tid; r < mhc; r += blockDim.x) {
            col_sum += smat[r * mhc + c];
        }
        swork[tid] = col_sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) swork[tid] += swork[tid + s];
            __syncthreads();
        }
        float col_sum_total = swork[0];
        __syncthreads();

        for (int r = tid; r < mhc; r += blockDim.x) {
            smat[r * mhc + c] /= col_sum_total;
        }
        __syncthreads();
    }

    // === Step 3: repeat (row-norm then col-norm) for (repeat-1) times ===
    for (int iter = 1; iter < repeat; iter++) {
        // Row normalize (sum along dim=-1)
        for (int r = 0; r < mhc; r++) {
            int row_offset = r * mhc;
            float row_sum = eps;
            for (int c = tid; c < mhc; c += blockDim.x) {
                row_sum += smat[row_offset + c];
            }
            swork[tid] = row_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) swork[tid] += swork[tid + s];
                __syncthreads();
            }
            float row_sum_total = swork[0];
            __syncthreads();

            for (int c = tid; c < mhc; c += blockDim.x) {
                smat[row_offset + c] /= row_sum_total;
            }
            __syncthreads();
        }

        // Column normalize (sum along dim=-2)
        for (int c = 0; c < mhc; c++) {
            float col_sum = eps;
            for (int r = tid; r < mhc; r += blockDim.x) {
                col_sum += smat[r * mhc + c];
            }
            swork[tid] = col_sum;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) swork[tid] += swork[tid + s];
                __syncthreads();
            }
            float col_sum_total = swork[0];
            __syncthreads();

            for (int r = tid; r < mhc; r += blockDim.x) {
                smat[r * mhc + c] /= col_sum_total;
            }
            __syncthreads();
        }
    }

    // Write back
    if (tid < elems_per_mat) {
        mat_out[tid] = smat[tid];
    }
}

template <typename T>
void test_tmp_kernel_ori(T* input, T* output, int total_matrices, int mhc, int repeat, float eps, cudaStream_t stream) {
    int block_size = mhc * mhc;
    // round up to power of 2 for reduction
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = bs;
    int shared_mem = (mhc * mhc + block_size) * sizeof(float);
    sinkhorn_kernel_ori<<<total_matrices, block_size, shared_mem, stream>>>(input, output, total_matrices, mhc, repeat, eps);
}
