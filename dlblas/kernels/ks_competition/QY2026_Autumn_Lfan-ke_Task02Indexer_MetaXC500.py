"""KernelSwift Task02 Indexer for MetaX C500: fused scoring and topk, in-place rotary, bit-exact with torch."""

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.cpp_extension import load_inline

world_size = 1
rank = 0


def _default_dtype():
    return torch.bfloat16


@dataclass
class ModelArgs:
    """Model hyperparameters. Field names match the config JSON keys."""

    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


class Linear(nn.Module):
    """Linear layer supporting BF16, FP8, and FP4 weight formats with per-block scaling."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or _default_dtype()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert bias is None
    return F.linear(x, weight)


class ColumnParallelLinear(Linear):
    """Shards output dim across TP ranks. No all-reduce needed on output."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert out_features % world_size == 0
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings in-place. Uses conjugate for inverse (de-rotation)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def _cuda_source():
    return r"""
#include <torch/extension.h>
#include <cstdint>

__device__ __forceinline__ float bf16_f32(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}

// Round to nearest even like torch; inputs are finite.
__device__ __forceinline__ unsigned short f32_bf16(float f) {
  unsigned int u = __float_as_uint(f);
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// Same asymmetric FMAs as torch's complex multiply; a symmetric form is 1 ulp off.
__device__ __forceinline__ unsigned int rot(unsigned int raw, float fr, float fi) {
  const float re = bf16_f32((unsigned short)(raw & 0xFFFFu));
  const float im = bf16_f32((unsigned short)(raw >> 16));
  const float o0 = __fmaf_rn(re, fr, -__fmul_rn(im, fi));
  const float o1 = __fmaf_rn(im, fr, __fmul_rn(re, fi));
  return ((unsigned int)f32_bf16(o1) << 16) | (unsigned int)f32_bf16(o0);
}

// Do not take v's address: it spills v to scratch memory.
__global__ void rope_kern(unsigned short* __restrict__ q,   // [B,S,H,D] bf16
                          const float* __restrict__ fc,     // [S, rd/2, 2] fp32
                          int S, int H, int D, int rd, int qs) {
  extern __shared__ float fs[];
  const int s = blockIdx.x, b = blockIdx.y;
  for (int i = threadIdx.x; i < rd; i += blockDim.x) fs[i] = fc[(long long)s * rd + i];
  __syncthreads();

  const int quads = rd >> 3;
  const int h = threadIdx.x >> qs;
  const int u = threadIdx.x & (quads - 1);
  uint4* p = (uint4*)(q + ((((long long)b * S + s) * H + h) * D + (D - rd))) + u;
  uint4 v = *p;
  const float* f = fs + (u << 3);
  v.x = rot(v.x, f[0], f[1]);
  v.y = rot(v.y, f[2], f[3]);
  v.z = rot(v.z, f[4], f[5]);
  v.w = rot(v.w, f[6], f[7]);
  *p = v;
}

void rope_inplace(torch::Tensor q, torch::Tensor fc) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.is_contiguous(), "bad q");
  TORCH_CHECK(fc.is_cuda() && fc.scalar_type() == at::kFloat && fc.is_contiguous(), "bad fc");
  const int B = (int)q.size(0), S = (int)q.size(1);
  const int H = (int)q.size(2), D = (int)q.size(3);
  const int rd = (int)(fc.size(1) * fc.size(2));
  TORCH_CHECK(rd <= 64 && (rd & 7) == 0 && D >= rd, "unsupported rope width");
  const int quads = rd >> 3;
  int qs = 0; while ((1 << qs) < quads) ++qs;
  TORCH_CHECK((1 << qs) == quads, "rd/8 must be a power of two");
  rope_kern<<<dim3(S, B), H * quads, (size_t)rd * sizeof(float)>>>(
      (unsigned short*)q.data_ptr(), fc.data_ptr<float>(), S, H, D, rd, qs);
}

// One block per (b, s) row; t >= (s + 1) / ratio is never read.
__global__ void idx_reduce_topk_kernel(
    const unsigned short* __restrict__ sc4,   // [B,S,H,T] bf16
    const unsigned short* __restrict__ wt,    // [B,S,H]   bf16
    int64_t* __restrict__ out,                // [B,S,K]   int64
    int s0, int nseg, int S, int H, int T, int K, int ratio, int offset, int s_base) {
  extern __shared__ unsigned int keys[];
  __shared__ float wsh[64];

  const int b = blockIdx.x / nseg;
  const int s = s0 + blockIdx.x % nseg;
  const long long row = (long long)b * S + s;

  int valid = (s_base + s + 1) / ratio;   // global s when fed in chunks
  if (valid > T) valid = T;

  if (valid == 0) {
    for (int k = threadIdx.x; k < K; k += blockDim.x) out[row * K + k] = -1;
    return;
  }

  for (int h = threadIdx.x; h < H; h += blockDim.x) wsh[h] = bf16_f32(wt[row * H + h]);
  __syncthreads();

  const unsigned short* base = sc4 + row * H * T;
  for (int t = threadIdx.x; t < valid; t += blockDim.x) {
    float acc = 0.f;
    // In order with a bf16 round per step, as torch does.
    for (int h = 0; h < H; ++h) {
      float v = bf16_f32(base[(long long)h * T + t]);
      if (v < 0.f) v = 0.f;
      acc += bf16_f32(f32_bf16(v * wsh[h]));
    }
    unsigned short sb = f32_bf16(acc);
    if (sb == 0x8000u) sb = 0u;  // -0.0 ties with +0.0
    unsigned int ord = (sb & 0x8000u) ? (unsigned int)(unsigned short)(~sb)
                                      : (unsigned int)(unsigned short)(sb | 0x8000u);
    keys[t] = (ord << 16) | (unsigned int)(0xFFFFu - (unsigned int)t);
  }

  int P = 1;
  while (P < valid) P <<= 1;
  for (int t = valid + threadIdx.x; t < P; t += blockDim.x) keys[t] = 0u;
  __syncthreads();

  // Beyond K keys, sort K-key segments, then merge keeping only the top K.
  const int npair = P >> 1;
  int KS = 0; while ((1 << KS) < K) ++KS;          // segments need K to be a power of two
  if (P <= K || (K & (K - 1)) != 0) {
    for (int kk = 2; kk <= P; kk <<= 1)
      for (int jj = kk >> 1; jj > 0; jj >>= 1) {
        for (int p = threadIdx.x; p < npair; p += blockDim.x) {
          const int lo = p & (jj - 1);
          const int i = ((p - lo) << 1) + lo;
          const unsigned int a = keys[i], c = keys[i + jj];
          if (((i & kk) == 0) ? (a < c) : (a > c)) { keys[i] = c; keys[i + jj] = a; }
        }
        __syncthreads();
      }
  } else {
    // Shifts, not division: integer division is emulated and this runs ~2.5e8 times.
    const int HKS = KS - 1;
    const int HK = K >> 1;
    for (int kk = 2; kk <= K; kk <<= 1)
      for (int jj = kk >> 1; jj > 0; jj >>= 1) {
        for (int p = threadIdx.x; p < npair; p += blockDim.x) {
          const int g = p >> HKS, q = p & (HK - 1);
          const int lo = q & (jj - 1);
          const int ii = ((q - lo) << 1) + lo;
          const int i = ii + (g << KS);
          const unsigned int a = keys[i], c = keys[i + jj];
          if (((ii & kk) == 0) ? (a < c) : (a > c)) { keys[i] = c; keys[i + jj] = a; }
        }
        __syncthreads();
      }
    const int nsg = P >> KS;
    for (int step = 1; step < nsg; step <<= 1) {
      const int np = nsg / (step << 1);
      for (int p = threadIdx.x; p < np * K; p += blockDim.x) {
        const int pr = p >> KS, i = p & (K - 1);
        const int ga = pr * (step << 1);
        const unsigned int a = keys[(ga << KS) + i];
        const unsigned int c = keys[((ga + step) << KS) + (K - 1 - i)];
        if (c > a) keys[(ga << KS) + i] = c;
      }
      __syncthreads();
      for (int jj = HK; jj > 0; jj >>= 1) {
        for (int p = threadIdx.x; p < np * HK; p += blockDim.x) {
          const int pr = p >> HKS, q = p & (HK - 1);
          const int lo = q & (jj - 1);
          const int i = ((pr * (step << 1)) << KS) + ((q - lo) << 1) + lo;
          const unsigned int a = keys[i], c = keys[i + jj];
          if (a < c) { keys[i] = c; keys[i + jj] = a; }
        }
        __syncthreads();
      }
    }
  }

  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    out[row * K + k] = (k < valid)
        ? (int64_t)(int)(0xFFFFu - (keys[k] & 0xFFFFu)) + (int64_t)offset
        : (int64_t)(-1);
  }
}

static inline int pow2_ceil(int v) { int p = 1; while (p < v) p <<= 1; return p; }

torch::Tensor idx_reduce_topk(torch::Tensor sc4, torch::Tensor w, int64_t K,
                              int64_t ratio, int64_t offset, int64_t s_base) {
  TORCH_CHECK(sc4.is_cuda() && w.is_cuda(), "inputs must be on the accelerator");
  TORCH_CHECK(sc4.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
              "scores and weights must be bfloat16");
  TORCH_CHECK(sc4.is_contiguous() && w.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(sc4.dim() == 4 && w.dim() == 3, "bad shapes");

  const int B = (int)sc4.size(0), S = (int)sc4.size(1);
  const int H = (int)sc4.size(2), T = (int)sc4.size(3);
  TORCH_CHECK(H <= 64, "H must fit the weight cache");

  auto out = torch::empty({B, S, (int64_t)K}, sc4.options().dtype(torch::kLong));
  const unsigned short* p_sc = (const unsigned short*)sc4.data_ptr();
  const unsigned short* p_w = (const unsigned short*)w.data_ptr();
  int64_t* p_o = out.data_ptr<int64_t>();

  // One launch for all rows: results do not depend on shared size or thread count.
  int vmax = ((int)s_base + S) / (int)ratio;
  if (vmax > T) vmax = T;
  const int Pmax = pow2_ceil(vmax > 0 ? vmax : 1);
  idx_reduce_topk_kernel<<<B * S, 128, (size_t)Pmax * sizeof(unsigned int)>>>(
      p_sc, p_w, p_o, 0, S, S, H, T, (int)K, (int)ratio, (int)offset, (int)s_base);
  return out;
}
"""


def _mod(_cache=[]):
    """Lazy: auto_bench drops module-level statements."""
    if not _cache:
        _cache.append(
            load_inline(
                name="indexer_maca",
                cpp_sources=(
                    "torch::Tensor idx_reduce_topk(torch::Tensor sc4, torch::Tensor w, "
                    "int64_t K, int64_t ratio, int64_t offset, int64_t s_base);\n"
                    "void rope_inplace(torch::Tensor q, torch::Tensor fc);"
                ),
                cuda_sources=_cuda_source(),
                functions=["idx_reduce_topk", "rope_inplace"],
                verbose=False,
            )
        )
    return _cache[0]


class Model(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.
    Has its own Compressor (with Hadamard rotation) to build compressed KV for scoring."""

    def __init__(self, args: ModelArgs, freqs_cis: torch.Tensor, kv_cache: torch.Tensor, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self.kv_cache = kv_cache
        self.freqs_cis = freqs_cis

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, : end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1) >= torch.arange(
                1, seqlen + 1, device=x.device
            ).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


class ModelNew(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring."""

    def __init__(self, args: ModelArgs, freqs_cis: torch.Tensor, kv_cache: torch.Tensor, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim**-0.5
        self.compress_ratio = compress_ratio
        self.kv_cache = kv_cache
        self.freqs_cis = freqs_cis
        self._fc = torch.view_as_real(freqs_cis)
        _m = _mod()
        self._op = _m.idx_reduce_topk
        self._rope = _m.rope_inplace

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        self._rope(q, self._fc[start_pos:end_pos])
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, : end_pos // ratio])
        k = min(self.index_topk, end_pos // ratio)
        if start_pos == 0:
            return self._op(index_score.contiguous(), weights.contiguous(), k, ratio, offset, 0)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        return index_score.topk(k, dim=-1)[1] + offset


def _args():
    return ModelArgs(
        max_batch_size=8,
        max_seq_len=2600,
        dim=1024,
        index_n_heads=16,
        index_head_dim=64,
        index_topk=128,
        q_lora_rank=256,
        rope_head_dim=32,
    )


def get_inputs(device="cuda"):
    args = _args()
    batch_size = 8
    seq_len = 2600
    x = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16, device=device)
    qr = torch.randn(batch_size, seq_len, args.q_lora_rank, dtype=torch.bfloat16, device=device)
    start_pos = 0
    offset = 0
    return [x, qr, start_pos, offset]


def get_init_inputs(device="cuda"):
    args = _args()
    compress_ratio = 4
    max_seq_len = args.max_seq_len
    rope_theta = 10000.0
    freqs = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, args.rope_head_dim, 2, device=device)[: args.rope_head_dim // 2].float()
            / args.rope_head_dim
        )
    )
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).view(max_seq_len, -1)
    kv_cache = torch.randn(
        args.max_batch_size,
        args.max_seq_len // compress_ratio,
        args.index_head_dim,
        dtype=_default_dtype(),
        device=device,
    )
    return [args, freqs_cis, kv_cache, compress_ratio]
