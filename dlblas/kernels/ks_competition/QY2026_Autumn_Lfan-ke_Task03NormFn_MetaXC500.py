"""KernelSwift Task03 norm_fn for MetaX C500: one fused kernel, JG outputs of a row per block."""

import os

import torch
from torch.utils.cpp_extension import load_inline

n1 = 13
mhc_mult = 4
hidden_size = 1280
generate_normw = False

DEFAULT_BLOCK = 256


def _cuda_source():
    return r"""
#include <torch/extension.h>
#include <cstdint>

// MACA bf16 has no float conversion; widening the bits is exact.
__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
  union { uint32_t u; float f; } c;
  c.u = (uint32_t)b << 16;
  return c.f;
}

// JG outputs share one read of residual; JG = 6 is fastest on C500.
template <int JG>
__global__ void norm_fn_kernel(
    const uint16_t* __restrict__ residual,   // [rows, K] raw bf16 bits
    const float* __restrict__ fn,            // [M, K] fp32
    float* __restrict__ out,                 // [rows, M] fp32
    int M, int K, float eps, float inv_K) {
  extern __shared__ float red[];

  const int ng = M / JG;
  const int row = blockIdx.x / ng;
  const int j0 = (blockIdx.x - row * ng) * JG;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  const uint2* r4 = reinterpret_cast<const uint2*>(residual + (long)row * K);
  const float4* f4[JG];
#pragma unroll
  for (int u = 0; u < JG; ++u) f4[u] = reinterpret_cast<const float4*>(fn + (long)(j0 + u) * K);

  float acc[JG];
#pragma unroll
  for (int u = 0; u < JG; ++u) acc[u] = 0.f;
  float sq = 0.f;

  const int K4 = K >> 2;
  for (int k = tid; k < K4; k += nt) {
    const uint2 rb = r4[k];
    const float r0 = bf16_to_f32((uint16_t)(rb.x & 0xffffu));
    const float r1 = bf16_to_f32((uint16_t)(rb.x >> 16));
    const float r2 = bf16_to_f32((uint16_t)(rb.y & 0xffffu));
    const float r3 = bf16_to_f32((uint16_t)(rb.y >> 16));
    sq = fmaf(r0, r0, sq); sq = fmaf(r1, r1, sq);
    sq = fmaf(r2, r2, sq); sq = fmaf(r3, r3, sq);
#pragma unroll
    for (int u = 0; u < JG; ++u) {
      const float4 fv = f4[u][k];
      acc[u] = fmaf(r0, fv.x, acc[u]);
      acc[u] = fmaf(r1, fv.y, acc[u]);
      acc[u] = fmaf(r2, fv.z, acc[u]);
      acc[u] = fmaf(r3, fv.w, acc[u]);
    }
  }

#pragma unroll
  for (int u = 0; u < JG; ++u) red[u * nt + tid] = acc[u];
  red[JG * nt + tid] = sq;
  __syncthreads();
  for (int s = nt >> 1; s > 0; s >>= 1) {
    if (tid < s) {
#pragma unroll
      for (int u = 0; u <= JG; ++u) red[u * nt + tid] += red[u * nt + tid + s];
    }
    __syncthreads();
  }
  if (tid < JG) out[(long)row * M + j0 + tid] = red[tid * nt] * rsqrtf(red[JG * nt] * inv_K + eps);
}

torch::Tensor norm_fn(torch::Tensor residual_, torch::Tensor fn_, double eps,
                      int64_t block) {
  auto residual = residual_.is_contiguous() ? residual_ : residual_.contiguous();
  auto fn = fn_.is_contiguous() ? fn_ : fn_.contiguous();

  // residual: [n0, n1, mhc_mult, hidden]  ->  [n0*n1, K]
  const int n0 = (int)residual.size(0);
  const int n1v = (int)residual.size(1);
  const int rows = n0 * n1v;
  const int M = (int)fn.size(0);
  const int K = (int)fn.size(1);

  TORCH_CHECK((K & 3) == 0, "K must be a multiple of 4");

  auto out = torch::empty({rows, M}, fn.options());
  const int threads = (int)block;

  static const int kCand[] = {6, 4, 3, 2};
  int JG = 1;
  for (int i = 0; i < 4; ++i) {
    if (M % kCand[i] == 0) { JG = kCand[i]; break; }
  }
  const size_t shmem = (size_t)(JG + 1) * threads * sizeof(float);
  const int grid = rows * (M / JG);
  auto pr = (const uint16_t*)residual.data_ptr<at::BFloat16>();
  auto pf = fn.data_ptr<float>();
  auto po = out.data_ptr<float>();
  const float e = (float)eps, ik = 1.0f / (float)K;
  switch (JG) {
    case 6: norm_fn_kernel<6><<<grid, threads, shmem>>>(pr, pf, po, M, K, e, ik); break;
    case 4: norm_fn_kernel<4><<<grid, threads, shmem>>>(pr, pf, po, M, K, e, ik); break;
    case 3: norm_fn_kernel<3><<<grid, threads, shmem>>>(pr, pf, po, M, K, e, ik); break;
    case 2: norm_fn_kernel<2><<<grid, threads, shmem>>>(pr, pf, po, M, K, e, ik); break;
    default: norm_fn_kernel<1><<<grid, threads, shmem>>>(pr, pf, po, M, K, e, ik); break;
  }
  return out.view({n0, n1v, M});
}
"""


def _mod(_cache=[]):
    """Lazy: auto_bench drops module-level statements."""
    if not _cache:
        _cache.append(
            load_inline(
                name="norm_fn_maca",
                cpp_sources=(
                    "torch::Tensor norm_fn(torch::Tensor residual, torch::Tensor fn, "
                    "double eps, int64_t block);"
                ),
                cuda_sources=_cuda_source(),
                functions=["norm_fn"],
                # load_inline builds host code at -O0.
                extra_cflags=["-O3"],
                verbose=False,
            )
        )
    return _cache[0]


def _block_size():
    return int(os.environ.get("NORM_FN_BLOCK", DEFAULT_BLOCK))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        residual: torch.Tensor,
        mhc_fn: torch.Tensor,
        mhc_norm_weight: torch.Tensor | None,
        mhc_norm_eps: float,
    ) -> torch.Tensor:
        if mhc_norm_weight is not None:
            mhc_fn = mhc_fn * mhc_norm_weight
        residual = residual.flatten(2, 3).float()
        assert mhc_fn.dtype == residual.dtype == torch.float
        mhc_mult = mhc_fn.shape[0]
        rms_group_size = mhc_fn.shape[-1]
        mixes = torch.einsum(
            "mbk,nbk->mbn",
            residual.view(-1, 1, rms_group_size),
            mhc_fn.view(mhc_mult, 1, rms_group_size),
        )
        sqrsum = residual.view(-1, 1, rms_group_size).square().sum(-1)
        mixes = (
            mixes * (sqrsum.unsqueeze(-1) / rms_group_size + mhc_norm_eps).rsqrt()
        ).sum(-2)
        return mixes.view(*residual.shape[:2], -1)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._op = _mod().norm_fn
        self._block = _block_size()

    def forward(
        self,
        residual: torch.Tensor,
        mhc_fn: torch.Tensor,
        mhc_norm_weight: torch.Tensor | None,
        mhc_norm_eps: float,
    ) -> torch.Tensor:
        if mhc_norm_weight is not None:
            mhc_fn = mhc_fn * mhc_norm_weight
        return self._op(residual, mhc_fn, mhc_norm_eps, self._block)


def generate_norm_fn_test_data(n1, mhc_mult, hidden_size, generate_normw, device="cuda"):
    n0 = 1
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    if generate_normw:
        normw = torch.randn((mhc_hidden_size,), dtype=torch.float, device=device) * 0.1 + 1.0
    else:
        normw = None

    out_grad = torch.randn((n0, n1, mhc_mult3), dtype=torch.float, device=device)

    return [residual, fn, normw, out_grad, 1e-6]


def get_inputs():
    torch.manual_seed(233)
    residual, fn, normw, out_grad, mhc_norm_eps = generate_norm_fn_test_data(
        n1, mhc_mult, hidden_size, generate_normw
    )
    return [residual, fn, None, mhc_norm_eps]


def get_init_inputs():
    return []
