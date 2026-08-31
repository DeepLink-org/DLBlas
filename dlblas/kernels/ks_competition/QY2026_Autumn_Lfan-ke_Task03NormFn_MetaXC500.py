"""Task03 norm_fn — 优化实现 (v1)：一个融合 MXMACA kernel 顶掉整段 torch 参考。

参考实现把这件事摊成了一串独立算子：bf16→fp32 的 cast、einsum 走的那次小 GEMM、square、
sum、rsqrt、逐元素乘、再 sum、最后 view——每一步都是一次 kernel 启动加一趟中间张量往返显存，
而真正要算的只有 13 行 × 24 个长度 5120 的点积，外加每行一个平方和。1.6M 次乘加对 C500 来说
是零头，所以这个算子跟 engram_hash 一样：瓶颈在启动与访存，优化就是压成一次 launch。

kernel 布局：一个 block 负责一个 (row, j) 输出元素，共 13×24 = 312 个 block。每个 block 只做
一趟读、一次块内归约：沿 K 走的时候同时累 `acc += r*f`（点积）与 `sq += r*r`（RMS 分母），
读 residual 与 fn[j] 都是顺序合并访存，末了 0 号线程写 `dot * rsqrt(sq/K + eps)`。

★这个布局是量出来换的，不是一开始就这么写的。第一版是「一 block 一行、块内串 24 次归约」：
正确，但只有 13 个 block、每次归约都要 `__syncthreads`，实测 0.357 ms 对 v0 的 0.234 ms——
比参考还慢 (0.66×)。改成 312 个 block、每块一次归约后才跑赢。sqrsum 在 24 个 block 间重复算，
多读 23 遍 residual（10 KB/行）换掉全部同步，这笔账在这个尺寸上是划算的。

★harness 约束：auto_bench.py 的 `_filter_module_ast` 只保留 import、class/def 和字面量赋值，
模块级的其它语句会被静默丢弃，所以扩展必须在函数里懒加载。
"""

import os

import torch
from torch.utils.cpp_extension import load_inline

n1 = 13
mhc_mult = 4
hidden_size = 1280
generate_normw = False

# C500 上扫过 128/256/512/1024，见 README 的调参记录。
DEFAULT_BLOCK = 256


def _cuda_source():
    return r"""
#include <torch/extension.h>
#include <cstdint>

// MACA 的 bf16 叫 __maca_bfloat16 且没有到 float 的转换算子，所以干脆不碰那个类型：
// bf16 转 fp32 就是把这 16 位放到 fp32 的高 16 位，低位补零——位级精确，也不依赖任何内建函数。
__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
  union { uint32_t u; float f; } c;
  c.u = (uint32_t)b << 16;
  return c.f;
}

// 一 block 一个输出元素 (row, j)。K = rms_group_size (5120)，M = mhc_mult3 (24)。
__global__ void norm_fn_kernel(
    const uint16_t* __restrict__ residual,   // [rows, K] bf16 的原始位
    const float* __restrict__ fn,            // [M, K] fp32
    float* __restrict__ out,                 // [rows, M] fp32
    int M, int K, float eps, float inv_K) {
  extern __shared__ float red[];             // 2 * nthreads：点积与平方和各一半

  const int row = blockIdx.x / M;
  const int j = blockIdx.x - row * M;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const uint16_t* rrow = residual + (long)row * K;
  const float* frow = fn + (long)j * K;

  float acc = 0.f, sq = 0.f;
  // K 是 4 的倍数（本题 5120），一次取 4 个：fn 走 float4，residual 走 4 个 bf16 打包成的 uint2。
  // 这个 kernel 是延迟 bound（实测带宽只用到 ~195 GB/s），把访存指令数砍到 1/4 直接见效。
  if ((K & 3) == 0) {
    const int K4 = K >> 2;
    const float4* f4 = reinterpret_cast<const float4*>(frow);
    const uint2* r4 = reinterpret_cast<const uint2*>(rrow);
    for (int k = tid; k < K4; k += nthreads) {
      const uint2 rb = r4[k];
      const float4 fv = f4[k];
      const float r0 = bf16_to_f32((uint16_t)(rb.x & 0xffffu));
      const float r1 = bf16_to_f32((uint16_t)(rb.x >> 16));
      const float r2 = bf16_to_f32((uint16_t)(rb.y & 0xffffu));
      const float r3 = bf16_to_f32((uint16_t)(rb.y >> 16));
      acc = fmaf(r0, fv.x, acc); sq = fmaf(r0, r0, sq);
      acc = fmaf(r1, fv.y, acc); sq = fmaf(r1, r1, sq);
      acc = fmaf(r2, fv.z, acc); sq = fmaf(r2, r2, sq);
      acc = fmaf(r3, fv.w, acc); sq = fmaf(r3, r3, sq);
    }
  } else {
    for (int k = tid; k < K; k += nthreads) {
      const float r = bf16_to_f32(rrow[k]);
      acc = fmaf(r, frow[k], acc);
      sq = fmaf(r, r, sq);
    }
  }
  red[tid] = acc;
  red[nthreads + tid] = sq;
  __syncthreads();
  for (int s = nthreads >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      red[tid] += red[tid + s];
      red[nthreads + tid] += red[nthreads + tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[(long)row * M + j] = red[0] * rsqrtf(red[nthreads] * inv_K + eps);
  }
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

  auto out = torch::empty({rows, M}, fn.options());
  const int threads = (int)block;
  const size_t shmem = (size_t)(2 * threads) * sizeof(float);

  norm_fn_kernel<<<rows * M, threads, shmem>>>(
      (const uint16_t*)residual.data_ptr<at::BFloat16>(),
      fn.data_ptr<float>(), out.data_ptr<float>(), M, K,
      (float)eps, 1.0f / (float)K);
  return out.view({n0, n1v, M});
}
"""


def _mod(_cache=[]):
    """编译一次、缓存住的扩展。放在函数里是 harness 的硬要求（见模块头）。"""
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
