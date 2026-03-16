# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code modified from here
https://github.com/albanie/collaborative-experts/blob/master/model/net_vlad.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _mm_rcr_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    for k in range(0, K, BLOCK_K):
        k_mask = k + offs_k < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Computes a @ b where a: [M, K], b: [K, N]
    if (not a.is_cuda) or (not b.is_cuda):
        return a @ b
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Triton path uses float32 for numerical parity."
    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Incompatible shapes for matmul"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _mm_rcr_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ],
    key=["BATCH", "M", "K", "N"],
)
@triton.jit
def _bmm_knd_kernel(
    a_ptr, b_ptr, c_ptr,
    BATCH, M, K, N,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # a: [B, M, K], b: [B, K, N] -> c: [B, M, N]
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_base = a_ptr + pid_b * stride_ab
    b_base = b_ptr + pid_b * stride_bb
    c_base = c_ptr + pid_b * stride_cb

    a_ptrs = a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    for k0 in range(0, K, BLOCK_K):
        k_mask = k0 + offs_k < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_bmm_knd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Computes batched matmul: [B, M, K] @ [B, K, N] -> [B, M, N]
    if (not a.is_cuda) or (not b.is_cuda):
        return torch.bmm(a, b)
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Triton path uses float32 for numerical parity."
    a = a.contiguous()
    b = b.contiguous()
    B, M, K = a.shape
    B2, K2, N = b.shape
    assert B == B2 and K == K2, "Incompatible shapes for batched matmul"
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (B, triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    _bmm_knd_kernel[grid](
        a, b, c,
        B, M, K, N,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
    )
    return c


# Fused VLAD accumulation (X^T @ A), centroid subtraction, and L2 normalization along D.
# Processes one (b, k) pair per program and iterates over N and D in tiles.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 128, "BLOCK_N": 32},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 128, "BLOCK_N": 64},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 256, "BLOCK_N": 32},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 256, "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 512, "BLOCK_N": 32},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 512, "BLOCK_N": 64},  num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 512, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["D", "N"],
)
@triton.jit
def _vlad_sub_norm_bk_kernel(
    x_ptr, a_ptr, c2_ptr, out_ptr,
    B, N, D, K,
    stride_xb, stride_xn, stride_xd,
    stride_ab, stride_an, stride_ak,
    stride_c0, stride_c1, stride_c2,
    stride_ob, stride_od, stride_ok,
    eps,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Base pointers
    x_base = x_ptr + pid_b * stride_xb
    a_base = a_ptr + pid_b * stride_ab
    o_base = out_ptr + pid_b * stride_ob

    # Accumulate along D: compute acc_d = sum_n A[b,n,k] * X[b,n,d]
    # Also accumulate sum_a = sum_n A[b,n,k] during the first D-tile to avoid an extra pass over A.
    sq_sum = 0.0
    sum_a = 0.0  # will be computed once during the first D tile

    d0 = 0
    while d0 < D:
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        n0 = 0
        while n0 < N:
            offs_n = n0 + tl.arange(0, BLOCK_N)
            n_mask = offs_n < N

            # Load X tile: [Dtile, Ntile]
            x_ptrs = x_base + offs_n[None, :] * stride_xn + offs_d[:, None] * stride_xd
            x_tile = tl.load(x_ptrs, mask=(d_mask[:, None] & n_mask[None, :]), other=0.0).to(tl.float32)

            # Load A vector: [Ntile]
            a_vec = tl.load(a_base + offs_n * stride_an + pid_k * stride_ak, mask=n_mask, other=0.0).to(tl.float32)

            # Accumulate sum_a only during the first D tile to reduce A traffic
            if d0 == 0:
                sum_a += tl.sum(a_vec, axis=0)

            # Weighted sum across N: acc += sum_j x[:, j] * a_vec[j]
            acc += tl.sum(x_tile * a_vec[None, :], axis=1)
            n0 += BLOCK_N

        # Subtract centroids: diff = acc - sum_a * C2[0, d, k]
        c2_ptrs = c2_ptr + offs_d * stride_c1 + pid_k * stride_c2  # c2[0, d, k]
        c2_tile = tl.load(c2_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        diff = acc - sum_a * c2_tile

        # Store diff to output buffer temporarily
        o_ptrs = o_base + offs_d * stride_od + pid_k * stride_ok
        tl.store(o_ptrs, diff, mask=d_mask)

        # Accumulate squared norm for normalization along D
        sq_sum += tl.sum(diff * diff, axis=0)
        d0 += BLOCK_D

    # 3) Normalize across D: out = diff / max(||diff||_2, eps)
    l2 = tl.sqrt(sq_sum)
    den = tl.where(l2 > eps, l2, eps)
    inv = 1.0 / den

    d0 = 0
    while d0 < D:
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        o_ptrs = o_base + offs_d * stride_od + pid_k * stride_ok
        vals = tl.load(o_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        vals = vals * inv
        tl.store(o_ptrs, vals, mask=d_mask)
        d0 += BLOCK_D


def triton_vlad_sub_norm_bk(x_bnd: torch.Tensor, a_bnk: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """
    Fused computation:
      v = X^T @ A  -> shape [B, D, K]
      v = v - (sum_n A) * C2
      v = F.normalize(v, dim=1)
    where:
      X: [B, N, D], A: [B, N, K], C2: [1, D, K].
    """
    if (not x_bnd.is_cuda) or (not a_bnk.is_cuda) or (not c2.is_cuda):
        # Reference fallback on CPU
        v = torch.bmm(x_bnd.transpose(1, 2), a_bnk)      # [B, D, K]
        a_sum = a_bnk.sum(dim=1, keepdim=True)           # [B, 1, K]
        v = v - a_sum * c2                                # broadcast [1, D, K]
        v = F.normalize(v)                                # normalize along D (dim=1)
        return v

    # Ensure contiguous memory layout for simple stride arithmetic
    x_bnd = x_bnd.contiguous()
    a_bnk = a_bnk.contiguous()
    c2 = c2.contiguous().to(x_bnd.dtype)

    B, N, D = x_bnd.shape
    B2, N2, K = a_bnk.shape
    assert B == B2 and N == N2 and c2.shape == (1, D, K)

    out = torch.empty((B, D, K), device=x_bnd.device, dtype=torch.float32)
    eps = 1e-12

    grid = (B, K)
    _vlad_sub_norm_bk_kernel[grid](
        x_bnd, a_bnk, c2, out,
        B, N, D, K,
        x_bnd.stride(0), x_bnd.stride(1), x_bnd.stride(2),
        a_bnk.stride(0), a_bnk.stride(1), a_bnk.stride(2),
        c2.stride(0), c2.stride(1), c2.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        eps,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        B = x.size(0)
        N = x.size(1)
        D = self.feature_size

        x_flat = x.view(-1, D)  # B x N x D -> (BN) x D

        if x_flat.device != self.clusters.device:
            msg = f"x.device {x_flat.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # Use highly-optimized cuBLAS for GEMM (assignment to clusters)
        assignment = x_flat @ self.clusters  # (BN x D) @ (D x (K+G)) -> BN x (K+G)

        # BatchNorm must remain in PyTorch to preserve running stats/affine params semantics
        assignment = self.batch_norm(assignment)

        # Softmax along clusters and drop ghost assignments
        assignment = F.softmax(assignment, dim=1)  # (BN x (K+G))
        assignment = assignment[:, :self.cluster_size]  # keep first K

        # Reshape to B x N x K
        assignment = assignment.view(B, N, self.cluster_size).contiguous()

        # Fused VLAD aggregation (X^T @ A), centroid subtraction and L2 normalization along D
        X_bnd = x_flat.view(B, N, D).contiguous()  # B x N x D
        vlad = triton_vlad_sub_norm_bk(X_bnd, assignment, self.clusters2)  # B x D x K (already normalized along D)

        # flattening + L2 norm across DK
        vlad = vlad.reshape(B, self.cluster_size * self.feature_size)  # B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK


batch_size = 32
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 0

def get_inputs():
  return [torch.randn(batch_size, num_features, feature_size)]

def get_init_inputs():
  return [num_clusters, feature_size, ghost_clusters]