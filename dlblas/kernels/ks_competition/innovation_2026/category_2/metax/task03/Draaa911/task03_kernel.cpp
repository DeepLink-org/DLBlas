// Task03 norm_fn kernel for the fixed input shape.
#include <mc_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kRows = 13;
constexpr int kK = 5120;
constexpr int kHidden = 1280;
constexpr int kM = 24;
constexpr int kThreads = 512;
constexpr float kEps = 1.0e-6f;

__device__ inline float bf16_to_float(uint16_t raw) {
    union Bits {
        uint32_t u;
        float f;
    } bits;
    bits.u = static_cast<uint32_t>(raw) << 16;
    return bits.f;
}

extern "C" __global__ void task03_kernel(
    const uint16_t* __restrict__ residual_bf16,
    const float* __restrict__ mhc_fn,
    float* __restrict__ out)
{
    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int m0 = 3 * blockIdx.x;
    const int m1 = m0 + 1;
    const int m2 = m0 + 2;
    const int row = blockIdx.y;

    __shared__ float partial[8][4];

    float dot0[2] = {0.0f, 0.0f};
    float dot1[2] = {0.0f, 0.0f};
    float dot2[2] = {0.0f, 0.0f};
    float sq[2] = {0.0f, 0.0f};

#pragma unroll
    for (int round = 0; round < 5; ++round) {
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int k = 1024 * round + 512 * part + 64 * wave + lane;
            const int group = k / kHidden;
            const int hidden = k - group * kHidden;
            const int residual_index = row + group * 16640 + hidden * kRows;
            const float r = bf16_to_float(residual_bf16[residual_index]);
            dot0[part] += r * mhc_fn[m0 * kK + k];
            dot1[part] += r * mhc_fn[m1 * kK + k];
            dot2[part] += r * mhc_fn[m2 * kK + k];
            sq[part] += r * r;
        }
    }

    float wave_dot0 = dot0[0] + dot0[1];
    float wave_dot1 = dot1[0] + dot1[1];
    float wave_dot2 = dot2[0] + dot2[1];
    float wave_sq = sq[0] + sq[1];

    wave_dot0 += __shfl_down_sync(UINT64_MAX, wave_dot0, 32, 64);
    wave_dot1 += __shfl_down_sync(UINT64_MAX, wave_dot1, 32, 64);
    wave_dot2 += __shfl_down_sync(UINT64_MAX, wave_dot2, 32, 64);
    wave_sq += __shfl_down_sync(UINT64_MAX, wave_sq, 32, 64);
    wave_dot0 += __shfl_down_sync(UINT64_MAX, wave_dot0, 16, 64);
    wave_dot1 += __shfl_down_sync(UINT64_MAX, wave_dot1, 16, 64);
    wave_dot2 += __shfl_down_sync(UINT64_MAX, wave_dot2, 16, 64);
    wave_sq += __shfl_down_sync(UINT64_MAX, wave_sq, 16, 64);
    wave_dot0 += __shfl_down_sync(UINT64_MAX, wave_dot0, 8, 64);
    wave_dot1 += __shfl_down_sync(UINT64_MAX, wave_dot1, 8, 64);
    wave_dot2 += __shfl_down_sync(UINT64_MAX, wave_dot2, 8, 64);
    wave_sq += __shfl_down_sync(UINT64_MAX, wave_sq, 8, 64);
    wave_dot0 += __shfl_down_sync(UINT64_MAX, wave_dot0, 4, 64);
    wave_dot1 += __shfl_down_sync(UINT64_MAX, wave_dot1, 4, 64);
    wave_dot2 += __shfl_down_sync(UINT64_MAX, wave_dot2, 4, 64);
    wave_sq += __shfl_down_sync(UINT64_MAX, wave_sq, 4, 64);
    wave_dot0 += __shfl_down_sync(UINT64_MAX, wave_dot0, 2, 64);
    wave_dot1 += __shfl_down_sync(UINT64_MAX, wave_dot1, 2, 64);
    wave_dot2 += __shfl_down_sync(UINT64_MAX, wave_dot2, 2, 64);
    wave_sq += __shfl_down_sync(UINT64_MAX, wave_sq, 2, 64);
    wave_dot0 += __shfl_down_sync(UINT64_MAX, wave_dot0, 1, 64);
    wave_dot1 += __shfl_down_sync(UINT64_MAX, wave_dot1, 1, 64);
    wave_dot2 += __shfl_down_sync(UINT64_MAX, wave_dot2, 1, 64);
    wave_sq += __shfl_down_sync(UINT64_MAX, wave_sq, 1, 64);

    if (lane == 0) {
        partial[wave][0] = wave_dot0;
        partial[wave][1] = wave_dot1;
        partial[wave][2] = wave_dot2;
        partial[wave][3] = wave_sq;
    }
    __syncthreads();

    if (tid == 0) {
        const float dot0_total = ((partial[0][0] + partial[1][0]) +
                                  (partial[2][0] + partial[3][0])) +
                                 ((partial[4][0] + partial[5][0]) +
                                  (partial[6][0] + partial[7][0]));
        const float dot1_total = ((partial[0][1] + partial[1][1]) +
                                  (partial[2][1] + partial[3][1])) +
                                 ((partial[4][1] + partial[5][1]) +
                                  (partial[6][1] + partial[7][1]));
        const float dot2_total = ((partial[0][2] + partial[1][2]) +
                                  (partial[2][2] + partial[3][2])) +
                                 ((partial[4][2] + partial[5][2]) +
                                  (partial[6][2] + partial[7][2]));
        const float sq_total = ((partial[0][3] + partial[1][3]) +
                                (partial[2][3] + partial[3][3])) +
                               ((partial[4][3] + partial[5][3]) +
                                (partial[6][3] + partial[7][3]));
        const float rms_inv = 1.0f / sqrtf(sq_total / static_cast<float>(kK) + kEps);
        float* row_out = out + row * kM;
        row_out[m0] = dot0_total * rms_inv;
        row_out[m1] = dot1_total * rms_inv;
        row_out[m2] = dot2_total * rms_inv;
    }
}

}  // namespace

extern "C" void task03_run(
    const uint16_t* residual_bf16,
    const float* mhc_fn,
    float* out)
{
    static int device_set = 0;
    if (!device_set) {
        mcSetDevice(0);
        device_set = 1;
    }
    dim3 grid(8, 13);
    task03_kernel<<<grid, kThreads>>>(residual_bf16, mhc_fn, out);
}
