# ================== EVOLVE-BLOCK-START ==================
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
def vlad_compute_kernel(
    assignment_ptr, x_ptr, clusters2_ptr, output_ptr,
    B, N, D, K,
    assignment_stride_bn, assignment_stride_k,
    x_stride_b, x_stride_n, x_stride_d,
    clusters2_stride_d, clusters2_stride_k,
    output_stride_b, output_stride_d, output_stride_k,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = pid0 // D
    pid_d = pid0 % D

    if pid_b >= B or pid_d >= D or pid_k >= K:
        return

    vlad_part1 = 0.0
    a_sum_part = 0.0

    for n in range(0, N, BLOCK_SIZE_N):
        n_offsets = n + tl.arange(0, BLOCK_SIZE_N)
        mask = n_offsets < N

        a_ptrs = assignment_ptr + (pid_b * N + n_offsets) * assignment_stride_bn + pid_k * assignment_stride_k
        a_val = tl.load(a_ptrs, mask=mask, other=0.0)

        x_ptrs = x_ptr + pid_b * x_stride_b + n_offsets * x_stride_n + pid_d * x_stride_d
        x_val = tl.load(x_ptrs, mask=mask, other=0.0)

        vlad_part1 += tl.sum(a_val * x_val)
        a_sum_part += tl.sum(a_val)

    c_val = tl.load(clusters2_ptr + pid_d * clusters2_stride_d + pid_k * clusters2_stride_k)
    a = a_sum_part * c_val
    result = vlad_part1 - a

    output_ptrs = output_ptr + pid_b * output_stride_b + pid_d * output_stride_d + pid_k * output_stride_k
    tl.store(output_ptrs, result)

class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        max_sample = x.size()[1]
        x_flat = x.reshape(-1, self.feature_size)

        assignment = th.matmul(x_flat, self.clusters)
        assignment = self.batch_norm(assignment)
        assignment = F.softmax(assignment, dim=1)
        assignment = assignment[:, :self.cluster_size]

        assignment = assignment.contiguous()
        x = x.contiguous()

        B, N, D, K = x.size(0), max_sample, self.feature_size, self.cluster_size
        vlad = torch.empty((B, D, K), device=x.device, dtype=x.dtype)

        clusters2_2d = self.clusters2.squeeze(0)
        grid = (B * D, K)

        vlad_compute_kernel[grid](
            assignment, x, clusters2_2d, vlad,
            B, N, D, K,
            assignment.stride(0), assignment.stride(1),
            x.stride(0), x.stride(1), x.stride(2),
            clusters2_2d.stride(0), clusters2_2d.stride(1),
            vlad.stride(0), vlad.stride(1), vlad.stride(2),
            BLOCK_SIZE_N=32
        )

        vlad = F.normalize(vlad, dim=1)
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)
        return vlad

batch_size = 32
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 0

def get_inputs():
  return [torch.randn(batch_size, num_features, feature_size)]

def get_init_inputs():
  return [num_clusters, feature_size, ghost_clusters]

# =================== EVOLVE-BLOCK-END ===================