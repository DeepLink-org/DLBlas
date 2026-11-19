#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kRows = 13;
constexpr int kMixes = 24;
constexpr int kWidth = 5120;
constexpr int kWave = 64;

// Ablation established the real cost: reading mhc_fn dominates at ~28.8 us
// while the strided residual read is only ~2.2 us. The 13-block designs made
// every block stream the whole 492 KB fn matrix, so 6.4 MB moved instead of
// 492 KB.
//
// Here one block owns one (row, mix) pair and touches exactly one 20 KB fn
// row, restoring total fn traffic to 492 KB across 312 blocks. The residual
// is read at its true interleaved address, k * kRows + row -- the layout that
// v5 validated -- so no transpose is needed.
//
// out[r][m] = (sum_k res[r][k] * fn[m][k]) * rsqrt(sumsq[r]/W + eps)
template <int Threads>
__global__ void __launch_bounds__(Threads) norm_fn_v6_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ mhc_fn,
    float* __restrict__ output,
    float eps) {
  const int row = blockIdx.x / kMixes;
  const int mix = blockIdx.x - row * kMixes;
  const float* fn_row = mhc_fn + mix * kWidth;
  const __nv_bfloat16* residual_base = residual + row;

  float dot = 0.0f;
  float sq = 0.0f;
  for (int k = threadIdx.x; k < kWidth; k += Threads) {
    const float v = __bfloat162float(residual_base[k * kRows]);
    dot += v * fn_row[k];
    sq += v * v;
  }

#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    dot += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, dot, offset);
    sq += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, sq, offset);
  }

  constexpr int kWaves = Threads / kWave;
  if constexpr (kWaves == 1) {
    if (threadIdx.x == 0) {
      const float inv = rsqrtf(sq / static_cast<float>(kWidth) + eps);
      output[row * kMixes + mix] = dot * inv;
    }
  } else {
    __shared__ float share_dot[kWaves];
    __shared__ float share_sq[kWaves];
    const int wave = threadIdx.x / kWave;
    if ((threadIdx.x & (kWave - 1)) == 0) {
      share_dot[wave] = dot;
      share_sq[wave] = sq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float total_dot = 0.0f;
      float total_sq = 0.0f;
#pragma unroll
      for (int w = 0; w < kWaves; ++w) {
        total_dot += share_dot[w];
        total_sq += share_sq[w];
      }
      const float inv = rsqrtf(total_sq / static_cast<float>(kWidth) + eps);
      output[row * kMixes + mix] = total_dot * inv;
    }
  }
}

// Wider variant: each block still owns one row but splits the mixes across
// its waves, so fn traffic stays at one row per (row, mix) while cutting the
// block count for launch-bound shapes.
template <int Threads, int MixPerBlock>
__global__ void __launch_bounds__(Threads) norm_fn_v6_multi_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ mhc_fn,
    float* __restrict__ output,
    float eps) {
  constexpr int kGroups = kMixes / MixPerBlock;
  const int row = blockIdx.x / kGroups;
  const int mix_base = (blockIdx.x - row * kGroups) * MixPerBlock;
  const __nv_bfloat16* residual_base = residual + row;

  float dots[MixPerBlock];
#pragma unroll
  for (int m = 0; m < MixPerBlock; ++m) dots[m] = 0.0f;
  float sq = 0.0f;

  for (int k = threadIdx.x; k < kWidth; k += Threads) {
    const float v = __bfloat162float(residual_base[k * kRows]);
    sq += v * v;
#pragma unroll
    for (int m = 0; m < MixPerBlock; ++m) {
      dots[m] += v * mhc_fn[(mix_base + m) * kWidth + k];
    }
  }

#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset >>= 1) {
    sq += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, sq, offset);
#pragma unroll
    for (int m = 0; m < MixPerBlock; ++m) {
      dots[m] += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, dots[m], offset);
    }
  }

  constexpr int kWaves = Threads / kWave;
  __shared__ float share_dot[MixPerBlock][kWaves];
  __shared__ float share_sq[kWaves];
  const int wave = threadIdx.x / kWave;
  if ((threadIdx.x & (kWave - 1)) == 0) {
    share_sq[wave] = sq;
#pragma unroll
    for (int m = 0; m < MixPerBlock; ++m) share_dot[m][wave] = dots[m];
  }
  __syncthreads();
  if (threadIdx.x < MixPerBlock) {
    float total_dot = 0.0f;
    float total_sq = 0.0f;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
      total_dot += share_dot[threadIdx.x][w];
      total_sq += share_sq[w];
    }
    const float inv = rsqrtf(total_sq / static_cast<float>(kWidth) + eps);
    output[row * kMixes + mix_base + threadIdx.x] = total_dot * inv;
  }
}

void run(const torch::Tensor& residual, const torch::Tensor& mhc_fn,
         double eps, torch::Tensor output, int64_t method) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto* r = reinterpret_cast<const __nv_bfloat16*>(
      residual.data_ptr<at::BFloat16>());
  const float* f = mhc_fn.data_ptr<float>();
  float* o = output.data_ptr<float>();
  const float e = static_cast<float>(eps);
  constexpr int kPairs = kRows * kMixes;

  switch (method) {
    case 0:
      norm_fn_v6_kernel<256><<<kPairs, 256, 0, stream>>>(r, f, o, e);
      break;
    case 1:
      norm_fn_v6_kernel<128><<<kPairs, 128, 0, stream>>>(r, f, o, e);
      break;
    case 2:
      norm_fn_v6_kernel<64><<<kPairs, 64, 0, stream>>>(r, f, o, e);
      break;
    case 3:
      norm_fn_v6_kernel<512><<<kPairs, 512, 0, stream>>>(r, f, o, e);
      break;
    case 4:
      norm_fn_v6_multi_kernel<256, 2><<<kPairs / 2, 256, 0, stream>>>(
          r, f, o, e);
      break;
    case 5:
      norm_fn_v6_multi_kernel<256, 4><<<kPairs / 4, 256, 0, stream>>>(
          r, f, o, e);
      break;
    case 6:
      norm_fn_v6_multi_kernel<128, 2><<<kPairs / 2, 128, 0, stream>>>(
          r, f, o, e);
      break;
    default:
      TORCH_CHECK(false, "unknown method ", method);
  }
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "norm_fn_v6 launch failed: ",
              cudaGetErrorString(error));
}

torch::Tensor forward_fixed(const torch::Tensor& residual,
                            const torch::Tensor& mhc_fn, double eps,
                            torch::Tensor output) {
  run(residual, mhc_fn, eps, output, 0);
  return output;
}

torch::Tensor forward_method(const torch::Tensor& residual,
                             const torch::Tensor& mhc_fn, double eps,
                             torch::Tensor output, int64_t method) {
  run(residual, mhc_fn, eps, output, method);
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_fixed", &forward_fixed, "norm_fn v6 pair-parallel");
  m.def("forward_method", &forward_method, "norm_fn v6 variant selector");
}
