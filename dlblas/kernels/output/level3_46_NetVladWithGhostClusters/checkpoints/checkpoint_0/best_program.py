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
def fused_assignment_kernel(
    x_ptr, 
    clusters_ptr,
    running_mean_ptr,
    scale_ptr,
    bias_ptr,
    output_ptr,
    stride_x_batch,
    stride_x_d,
    stride_clusters_d,
    stride_running_mean_k,
    stride_scale_k,
    stride_bias_k,
    stride_output_b, 
    stride_output_n, 
    stride_output_k,
    D, 
    clusters, 
    cluster_size, 
    max_sample, 
    BLOCK_D: tl.constexpr, 
    BLOCK_CLUSTERS: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // max_sample
    n_idx = pid % max_sample
    
    # Initialize accumulator
    accum = tl.zeros((BLOCK_CLUSTERS,), dtype=tl.float32)
    
    # Compute matrix multiplication
    for d_offset in range(0, D, BLOCK_D):
        d_indices = tl.arange(0, BLOCK_D) + d_offset
        mask_d = d_indices < D
        
        # Load x values
        x_vals = tl.load(x_ptr + b_idx * stride_x_batch + n_idx * stride_x_d + d_indices, 
                         mask=mask_d, other=0.0)
        
        # Load clusters block
        clusters_block = tl.load(clusters_ptr + d_indices[:, None] * stride_clusters_d + 
                                tl.arange(0, BLOCK_CLUSTERS)[None, :],
                                mask=mask_d[:, None] & (tl.arange(0, BLOCK_CLUSTERS)[None, :] < clusters),
                                other=0.0)
        
        # Accumulate partial sums
        partial = tl.dot(x_vals, clusters_block)
        accum += partial

    # Load batch norm parameters
    k_indices = tl.arange(0, BLOCK_CLUSTERS)
    running_mean = tl.load(running_mean_ptr + k_indices, mask=k_indices < clusters, other=0.0)
    scale_vals = tl.load(scale_ptr + k_indices, mask=k_indices < clusters, other=0.0)
    bias_vals = tl.load(bias_ptr + k_indices, mask=k_indices < clusters, other=0.0)
    
    # Apply batch norm
    accum = (accum - running_mean) * scale_vals + bias_vals
    
    # Apply softmax
    max_val = tl.max(accum, axis=0)
    accum = accum - max_val
    exp_vals = tl.exp(accum)
    exp_sum = tl.sum(exp_vals, axis=0)
    softmax_out = exp_vals / exp_sum
    
    # Store only non-ghost clusters
    output_k_indices = tl.arange(0, cluster_size)
    output_vals = softmax_out[0:cluster_size]
    output_offset = b_idx * stride_output_b + n_idx * stride_output_n + output_k_indices
    tl.store(output_ptr + output_offset, output_vals, mask=output_k_indices < cluster_size)

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
        """Aggregates feature maps into a fixed size representation."""
        B, N, D = x.shape
        x_flat = x.reshape(B*N, D)
        
        # Precompute batch norm parameters for inference
        if not self.training:
            scale = self.batch_norm.weight / torch.sqrt(self.batch_norm.running_var + self.batch_norm.eps)
            bias = self.batch_norm.bias
            
            # Create output tensor
            assignment_out = torch.empty((B, N, self.cluster_size), 
                                        device=x.device, dtype=x.dtype)
            
            # Launch Triton kernel
            grid = (B*N,)
            fused_assignment_kernel[grid](
                x_flat, 
                self.clusters,
                self.batch_norm.running_mean,
                scale,
                bias,
                assignment_out,
                x_flat.stride(0),
                x_flat.stride(1),
                self.clusters.stride(0),
                self.batch_norm.running_mean.stride(0),
                scale.stride(0),
                bias.stride(0),
                assignment_out.stride(0),
                assignment_out.stride(1),
                assignment_out.stride(2),
                D,
                self.cluster_size + self.ghost_clusters,
                self.cluster_size,
                N,
                BLOCK_D=128,
                BLOCK_CLUSTERS=triton.next_power_of_2(self.cluster_size + self.ghost_clusters),
            )
        else:
            # Training path (original implementation)
            assignment = th.matmul(x_flat, self.clusters)
            assignment = self.batch_norm(assignment)
            assignment = F.softmax(assignment, dim=1)
            assignment = assignment[:, :self.cluster_size]
            assignment_out = assignment.view(B, N, self.cluster_size)

        a_sum = th.sum(assignment_out, dim=1, keepdim=True)
        a = a_sum * self.clusters2

        # Optimized VLAD computation
        vlad = th.matmul(x.transpose(1,2), assignment_out)
        vlad = vlad - a

        # Normalizations
        vlad = F.normalize(vlad, dim=1)
        vlad = vlad.reshape(B, -1)
        vlad = F.normalize(vlad)
        return vlad

batch_size = 32
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 16

def get_inputs():
  return [torch.randn(batch_size, num_features, feature_size)]

def get_init_inputs():
  return [num_clusters, feature_size, ghost_clusters]

# =================== EVOLVE-BLOCK-END ===================