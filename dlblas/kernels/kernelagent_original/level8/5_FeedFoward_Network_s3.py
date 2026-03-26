"""
Feedforward module for E2Former.
"""

import torch
from torch import nn
import os
import sys
import math
from e3nn import o3

from fairchem.core.models.equiformer_v2.activation import  GateActivation, S2Activation, SeparableS2Activation
from fairchem.core.models.equiformer_v2.so3 import (
    CoefficientMappingModule,
    FromS2Grid,
    SO3_LinearV2,
    ToS2Grid,
    SO3_Embedding
)
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

class Model(torch.nn.Module):
    

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
        super(Model, self).__init__()

            
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
                gating_scalars = self.gating_linear(
                    input_embedding.embedding.narrow(1, 0, 1)
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