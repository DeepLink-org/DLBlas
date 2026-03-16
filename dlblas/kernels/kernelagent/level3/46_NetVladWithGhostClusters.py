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


@triton.jit
def _vlad_fused_kernel(
    A_ptr,  # [B, N, K]
    X_ptr,  # [B, N, D]
    C_ptr,  # [1, D, K] -> we use [D, K]
    O_ptr,  # [B, D, K]
    B, N, K, D,
    stride_aB, stride_aN, stride_aK,
    stride_xB, stride_xN, stride_xD,
    stride_cD, stride_cK,
    stride_oB, stride_oD, stride_oK,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_b

    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    k_mask = k_offsets < K
    d_mask = d_offsets < D

    # Accumulators: acc for vlad (K x D), s for sum of assignments per cluster (K,)
    acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
    s = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Loop over N dimension
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    for n_tile in range(0, num_n_tiles):
        n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        # Load A tile: [BLOCK_N, BLOCK_K]
        a_ptrs = (
            A_ptr
            + b * stride_aB
            + n_offsets[:, None] * stride_aN
            + k_offsets[None, :] * stride_aK
        )
        a_tile = tl.load(a_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Load X tile: [BLOCK_N, BLOCK_D]
        x_ptrs = (
            X_ptr
            + b * stride_xB
            + n_offsets[:, None] * stride_xN
            + d_offsets[None, :] * stride_xD
        )
        x_tile = tl.load(x_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

        # GEMM accumulate: (K,N) x (N,D) -> (K,D)
        acc += tl.dot(tl.trans(a_tile), x_tile)
        # Sum reduce over N for each K
        s += tl.sum(a_tile, axis=0)

    # Load clusters2 tile: [BLOCK_D, BLOCK_K] from [D, K]
    c_ptrs = C_ptr + d_offsets[:, None] * stride_cD + k_offsets[None, :] * stride_cK
    c_tile = tl.load(c_ptrs, mask=d_mask[:, None] & k_mask[None, :], other=0.0)  # (D,K)
    c_tile_t = tl.trans(c_tile)  # (K,D)

    # Subtract a = a_sum * clusters2 -> acc - s[:,None] * c_tile_t
    acc = acc - s[:, None] * c_tile_t

    # Store to output O: [B, D, K] storing tile (D,K)
    o_ptrs = (
        O_ptr
        + b * stride_oB
        + d_offsets[:, None] * stride_oD
        + k_offsets[None, :] * stride_oK
    )
    # acc is (K,D) -> transpose to (D,K) for storage
    out_tile = tl.trans(acc)
    tl.store(o_ptrs, out_tile, mask=d_mask[:, None] & k_mask[None, :])


def _vlad_fused_triton(assignment_bnk: torch.Tensor, x_bnd: torch.Tensor, clusters2: torch.Tensor):
    # assignment_bnk: [B, N, K], x_bnd: [B, N, D], clusters2: [1, D, K]
    assert assignment_bnk.is_cuda and x_bnd.is_cuda and clusters2.is_cuda
    B, N, K = assignment_bnk.shape
    _, _, D = x_bnd.shape
    # Output B x D x K
    out = torch.empty((B, D, K), device=assignment_bnk.device, dtype=assignment_bnk.dtype)

    # Strides in elements
    saB, saN, saK = assignment_bnk.stride()
    sxB, sxN, sxD = x_bnd.stride()
    # clusters2 is [1, D, K] -> we pass strides for [D, K] dims
    scD, scK = clusters2.stride()[1], clusters2.stride()[2]
    soB, soD, soK = out.stride()

    # Tile sizes tuned for H100/H200 small-K, medium-D
    BLOCK_K = 32
    BLOCK_D = 128
    BLOCK_N = 32

    grid = (
        B,
        triton.cdiv(K, BLOCK_K),
        triton.cdiv(D, BLOCK_D),
    )
    _vlad_fused_kernel[grid](
        assignment_bnk, x_bnd, clusters2, out,
        B, N, K, D,
        saB, saN, saK,
        sxB, sxN, sxD,
        scD, scK,
        soB, soD, soK,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
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
        max_sample = x.size()[1]
        x_flat = x.view(-1, self.feature_size)  # B x N x D -> BN x D

        if x_flat.device != self.clusters.device:
            msg = f"x.device {x_flat.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        assignment = th.matmul(x_flat, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = self.batch_norm(assignment)
        assignment = F.softmax(assignment, dim=1)  # BN x (K+G) -> BN x (K+G)
        # remove ghost assigments
        assignment = assignment[:, :self.cluster_size]
        # -> B x N x K
        assignment_bnk = assignment.view(-1, max_sample, self.cluster_size).contiguous()

        # Prepare x in [B, N, D]
        x_bnd = x_flat.view(-1, max_sample, self.feature_size).contiguous()

        # Compute vlad = (assignment @ x) - a in one fused Triton kernel:
        # Output: B x D x K (already subtracted by a_sum * clusters2)
        if x_bnd.is_cuda and assignment_bnk.is_cuda and self.clusters2.is_cuda:
            vlad = _vlad_fused_triton(assignment_bnk, x_bnd, self.clusters2)
        else:
            # CPU / fallback path using reference ops
            a_sum = th.sum(assignment_bnk, dim=1, keepdim=True)  # B x 1 x K
            a = a_sum * self.clusters2  # B x D x K
            assignment_t = assignment_bnk.transpose(1, 2)  # B x K x N
            vlad_tmp = th.matmul(assignment_t, x_bnd)  # B x K x D
            vlad = vlad_tmp.transpose(1, 2)  # B x D x K
            vlad = vlad - a

        # L2 intra norm (along D)
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK


batch_size = 32
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 16

def get_inputs():
  return [torch.randn(batch_size, num_features, feature_size)]

def get_init_inputs():
  return [num_clusters, feature_size, ghost_clusters]