"""
Feedforward module for E2Former.
"""

import torch
from torch import nn
import os
import sys
import math
from e3nn import o3

import triton
import triton.language as tl

from fairchem.core.models.equiformer_v2.activation import  GateActivation, S2Activation, SeparableS2Activation
from fairchem.core.models.equiformer_v2.so3 import (
    CoefficientMappingModule,
    FromS2Grid,
    SO3_LinearV2,
    ToS2Grid,
    SO3_Embedding
)

@triton.jit
def _linear_fw_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    ADD_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Helpful alignment hints for the compiler
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    # Masks
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    # Base pointers for first K-tile
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk       # [BM, BK]
    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn       # [BK, BN]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Preload first tiles
    k_row_mask = (offs_k[None, :] < K)
    k_col_mask = (offs_k[:, None] < K)
    a = tl.load(x_ptrs, mask=m_mask & k_row_mask, other=0.0, cache_modifier=".cg", eviction_policy="evict_first")
    b = tl.load(w_ptrs, mask=k_col_mask & n_mask, other=0.0, cache_modifier=".ca", eviction_policy="evict_last")

    # Iterate over K tiles with simple software pipelining
    k = 0
    while k + BLOCK_K < K:
        # Prefetch next tiles
        x_ptrs_next = x_ptrs + BLOCK_K * stride_xk
        w_ptrs_next = w_ptrs + BLOCK_K * stride_wk

        k_next = k + BLOCK_K
        next_row_mask = (k_next + offs_k[None, :]) < K
        next_col_mask = (k_next + offs_k[:, None]) < K

        a_next = tl.load(x_ptrs_next, mask=m_mask & next_row_mask, other=0.0, cache_modifier=".cg", eviction_policy="evict_first")
        b_next = tl.load(w_ptrs_next, mask=next_col_mask & n_mask, other=0.0, cache_modifier=".ca", eviction_policy="evict_last")

        # Compute on current tiles
        acc += tl.dot(a, b)

        # Advance
        a = a_next
        b = b_next
        x_ptrs = x_ptrs_next
        w_ptrs = w_ptrs_next
        k = k_next

    # Final tile
    acc += tl.dot(a, b)

    # Add bias
    if ADD_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0)
        acc += bias[None, :]

    # Store
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=m_mask & n_mask)


def _triton_linear_batched_lastdim(x, weight, bias):
    """
    Compute y = x @ weight.T + bias for x shape [B, 1, K] (or [B, K]).
    weight: [N, K], bias: [N]
    Returns y shape [B, 1, N] (or [B, N]).
    Heuristic: use cuBLAS-accelerated torch.nn.functional.linear for large problems;
    use Triton for small/medium problems to reduce launch overhead.
    """
    # If x is 3D with middle dim 1, prefer using high-performance F.linear directly for large GEMMs
    is_3d = (x.dim() == 3)
    if is_3d:
        B, one, K = x.shape
        assert one == 1, "Expected the middle dimension to be 1."
        M = B
    else:
        M, K = x.shape

    N = weight.shape[0]

    # Conditions for Triton execution
    can_triton = (x.is_cuda and weight.is_cuda and (bias is None or bias.is_cuda))
    # Simple heuristic threshold: for large GEMMs, cuBLAS (F.linear/addmm) is typically faster
    problem_size = M * N * K
    # Keep Triton only for relatively small problems to minimize overhead
    use_triton = can_triton and (problem_size <= (64 * 1024 * 1024))

    if not use_triton:
        # Fast path via cuBLAS using F.linear; preserves broadcasting and avoids manual reshapes
        y = torch.nn.functional.linear(x, weight, bias)
        return y

    # Triton path: collapse to 2D for GEMM
    if is_3d:
        x_mat = x.reshape(M, K)
    else:
        x_mat = x

    # Allocate output
    y = torch.empty((M, N), device=x_mat.device, dtype=x_mat.dtype)

    # Strides
    stride_xm, stride_xk = x_mat.stride()
    # Treat weight as logical [K, N]
    stride_wk = weight.stride(1)  # along K
    stride_wn = weight.stride(0)  # along N
    stride_ym, stride_yn = y.stride()

    # Tile sizes tuned for K=N=256 on NVIDIA H100/H200
    # Larger BK reduces the number of K-tiles; choose based on K
    if K >= 256:
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 128
        num_warps = 8
        num_stages = 5
    else:
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64
        num_warps = 4
        num_stages = 4

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _linear_fw_kernel[grid](
        x_mat, weight, bias if bias is not None else y, y,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        ADD_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    return y.reshape(B, 1, N) if is_3d else y


class SO3_Grid(torch.nn.Module):

    def __init__(
        self,
        lmax,
        mmax,
        normalization="integral",
        resolution=None,
    ):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = 2 * (self.lmax + 1)
        if lmax == mmax:
            self.long_resolution = 2 * (self.mmax + 1) + 1
        else:
            self.long_resolution = 2 * (self.mmax) + 1
        if resolution is not None:
            self.lat_resolution = resolution
            self.long_resolution = resolution

        self.mapping = CoefficientMappingModule([self.lmax], [self.lmax])

        device = 'cpu'

        to_grid = ToS2Grid(
            self.lmax,
            (self.lat_resolution, self.long_resolution),
            normalization=normalization,  # normalization="integral",
            device=device,
        )
        to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l**2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    to_grid_mat[:, :, start_idx : (start_idx + length)] * rescale_factor
                )
        to_grid_mat = to_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        from_grid = FromS2Grid(
            (self.lat_resolution, self.long_resolution),
            self.lmax,
            normalization=normalization,  # normalization="integral",
            device=device,
        )
        from_grid_mat = torch.einsum(
            "am, mbi -> bai", from_grid.sha, from_grid.shb
        ).detach()
        # rescale based on mmax
        if lmax != mmax:
            for l in range(lmax + 1):
                if l <= mmax:
                    continue
                start_idx = l**2
                length = 2 * l + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                from_grid_mat[:, :, start_idx : (start_idx + length)] = (
                    from_grid_mat[:, :, start_idx : (start_idx + length)]
                    * rescale_factor
                )
        from_grid_mat = from_grid_mat[
            :, :, self.mapping.coefficient_idx(self.lmax, self.mmax)
        ]

        # save tensors and they will be moved to GPU
        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)

    # Compute matrices to transform irreps to grid
    def get_to_grid_mat(self, device):
        return self.to_grid_mat

    # Compute matrices to transform grid to irreps
    def get_from_grid_mat(self, device):
        return self.from_grid_mat

    # Compute grid from irreps representation
    def to_grid(self, embedding, lmax, mmax):
        to_grid_mat = self.to_grid_mat[:, :, self.mapping.coefficient_idx(lmax, mmax)]
        grid = torch.einsum("bai, zic -> zbac", to_grid_mat, embedding)
        return grid

    def from_grid(self, grid, lmax, mmax):
        from_grid_mat = self.from_grid_mat[
            :, :, self.mapping.coefficient_idx(lmax, mmax)
        ]
        embedding = torch.einsum("bai, zbac -> zic", from_grid_mat, grid)
        return embedding

class SO3_Linear_e2former(torch.nn.Module):
    def __init__(self, in_features, out_features, lmax, bias=True):
        """
        1. Use `torch.einsum` to prevent slicing and concatenation
        2. Need to specify some behaviors in `no_weight_decay` and weight initialization.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = torch.nn.Parameter(
            torch.randn((self.lmax + 1), out_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer("expand_index", expand_index)
    
    @torch.compile()
    def forward(self, input_embedding):
        # with torch.profiler.record_function("SO3_Linear_e2former"):
        output_shape = input_embedding.shape[:-2]
        l_sum, hidden = input_embedding.shape[-2:]
        input_embedding = input_embedding.reshape(
            [output_shape.numel()] + [l_sum, hidden]
        )
        # triton_poi_fused_index_select_0
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )  # [(L_max + 1) ** 2, C_out, C_in]
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight
        )  # [N, (L_max + 1) ** 2, C_out]
    
        bias = self.bias.view(1, 1, self.out_features)
        # triton_poi_fused_add_1
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias

        out = out.reshape(output_shape + (l_sum, self.out_features))

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax})"

class ModelNew(torch.nn.Module):
    

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax,
        mmax=2,
        grid_resolution=18,
        use_gate_act=False,  # [True, False] Switch between gate activation and S2 activation
        use_grid_mlp=True,  # [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act=True,  # Separable S2 activation. Used for ablation study.
       
    ):
        super(ModelNew, self).__init__()

            
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.sphere_channels_all = self.sphere_channels
        self.so3_grid = torch.nn.ModuleList()
        self.lmax = lmax
        self.max_lmax = self.lmax
        self.lmax_list = [lmax]
        for l in range(lmax + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(lmax + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, m, resolution=grid_resolution  # , normalization="component"
                    )
                )
            self.so3_grid.append(SO3_m_grid)

        self.use_gate_act = use_gate_act  # [True, False] Switch between gate activation and S2 activation
        self.use_grid_mlp = use_grid_mlp  # [False, True] If `True`, use projecting to grids and performing MLPs for FFNs.
        self.use_sep_s2_act = (
            use_sep_s2_act  # Separable S2 activation. Used for ablation study.
        )

        self.so3_linear_1 = SO3_LinearV2(
            self.sphere_channels_all, self.hidden_channels, lmax=self.lmax
        )
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(
                        self.sphere_channels_all,
                        self.hidden_channels,
                        bias=True,
                    ),
                    nn.SiLU(),
                )
            else:
                self.scalar_mlp = None
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            )
        else:
            if self.use_gate_act:
                self.gating_linear = torch.nn.Linear(
                    self.sphere_channels_all,
                    self.lmax * self.hidden_channels,
                )
                self.gate_act = GateActivation(
                    self.lmax, self.lmax, self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    self.gating_linear = torch.nn.Linear(
                        self.sphere_channels_all, self.hidden_channels
                    )
                    self.s2_act = SeparableS2Activation(self.lmax, self.lmax)
                else:
                    self.gating_linear = None
                    self.s2_act = S2Activation(self.lmax, self.lmax)
        self.so3_linear_2 = SO3_LinearV2(
            self.hidden_channels, self.output_channels, lmax=self.lmax
        )

    def forward(self, input_embedding):
        
        with torch.profiler.record_function("FeedForwardNetwork_s2"):
            
            out_shape = input_embedding.shape[:-2]

            input_embedding = input_embedding.reshape(
                out_shape.numel(), (self.lmax + 1) ** 2, self.sphere_channels
            )
            x = SO3_Embedding(
                input_embedding.shape[0],
                self.lmax_list,
                self.sphere_channels,
                input_embedding.device,
                input_embedding.dtype,
            )
            x.embedding = input_embedding
            x = self._forward(x)

            return x.embedding.reshape(out_shape + (-1, self.output_channels))

    def _forward(self, input_embedding):
        gating_scalars = None
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                gating_scalars = self.scalar_mlp(
                    input_embedding.embedding.narrow(1, 0, 1)
                )
        else:
            if self.gating_linear is not None:
                gi = input_embedding.embedding.narrow(1, 0, 1)
                # Prefer cuBLAS-backed F.linear for large GEMMs; Triton is used for small problems
                gating_scalars = _triton_linear_batched_lastdim(
                    gi, self.gating_linear.weight, self.gating_linear.bias
                )

        input_embedding = self.so3_linear_1(input_embedding)

        if self.use_grid_mlp:
            # Project to grid
            input_embedding_grid = input_embedding.to_grid(
                self.so3_grid, lmax=self.max_lmax
            )
            # Perform point-wise operations
            input_embedding_grid = self.grid_mlp(input_embedding_grid)
            # Project back to spherical harmonic coefficients
            input_embedding._from_grid(
                input_embedding_grid, self.so3_grid, lmax=self.max_lmax
            )

            if self.use_sep_s2_act:
                input_embedding.embedding = torch.cat(
                    (
                        gating_scalars,
                        input_embedding.embedding.narrow(
                            1, 1, input_embedding.embedding.shape[1] - 1
                        ),
                    ),
                    dim=1,
                )
        else:
            if self.use_gate_act:
                input_embedding.embedding = self.gate_act(
                    gating_scalars, input_embedding.embedding
                )
            else:
                if self.use_sep_s2_act:
                    input_embedding.embedding = self.s2_act(
                        gating_scalars,
                        input_embedding.embedding,
                        self.so3_grid,
                    )
                else:
                    input_embedding.embedding = self.s2_act(
                        input_embedding.embedding, self.so3_grid
                    )

        return self.so3_linear_2(input_embedding)

# ==========================================
# Hyperparameters & Data Generation
# ==========================================

# Configuration from prompt
sphere_channels = 256
hidden_channels = 256
output_channels = 256
lmax = 3
mmax = 2 # Not used in simplified linear/act, but kept for context
grid_resolution = 18 # Not used in mock, but kept for context
use_gate_act = False
use_grid_mlp = False
use_sep_s2_act = True

# Data Shapes
N_nodes = 2240
dim_irreps_total = (lmax + 1) ** 2 # 16

def get_inputs():
    # torch.manual_seed(42)
    # input_embedding shape: [2240, 16, 256]
    input_embedding = torch.randn(N_nodes, dim_irreps_total, sphere_channels, dtype=torch.float32,device='cpu')
    return [input_embedding]
def get_init_inputs():
    return [
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax,
        mmax,
        grid_resolution,
        use_gate_act,
        use_grid_mlp,
        use_sep_s2_act
    ]