import math
import time
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# Try to import Triton for custom kernels
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

# ==========================================
# Triton kernel for weighted neighbor aggregation:
# aggregated[b, m, h] = sum_k alpha[b, k, h] * value[idx[b, k], m, h]
# Shapes:
# - alpha: [N, K, H]
# - value: [N, M, H]
# - idx  : [N, K] (int64)
# - out  : [N, M, H]
# ==========================================
if _TRITON_AVAILABLE:
    @triton.jit
    def weighted_gather_sum_kernel(
        alpha_ptr, value_ptr, idx_ptr, out_ptr,
        N, K, M, H,
        stride_alpha_b, stride_alpha_k, stride_alpha_h,
        stride_value_b, stride_value_m, stride_value_h,
        stride_idx_b, stride_idx_k,
        stride_out_b, stride_out_m, stride_out_h,
        BLOCK_H: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_hblk = tl.program_id(1)

        # Offsets within the H (channel) and M (irreps) tiles
        h_offsets = pid_hblk * BLOCK_H + tl.arange(0, BLOCK_H)
        m_offsets = tl.arange(0, BLOCK_M)

        # Boundary masks
        h_mask = h_offsets < H
        m_mask = m_offsets < M

        # Provide compiler hints for better vectorization
        tl.multiple_of(h_offsets, BLOCK_H)
        tl.multiple_of(m_offsets, BLOCK_M)
        tl.max_contiguous(h_offsets, BLOCK_H)

        # Initialize accumulator for this CTA's tile: [BLOCK_M, BLOCK_H]
        acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

        # Precompute base pointers for this batch n = pid_b
        alpha_b_ptr = alpha_ptr + pid_b * stride_alpha_b
        idx_b_ptr = idx_ptr + pid_b * stride_idx_b
        out_b_ptr = out_ptr + pid_b * stride_out_b

        # Loop over neighbors (dynamic K)
        for k in range(0, K):
            # Load neighbor index for (b, k) as int32 to reduce arithmetic width
            j = tl.load(idx_b_ptr + k * stride_idx_k, mask=True, other=0).to(tl.int32)

            # Load alpha vector tile [BLOCK_H] for this (b, k)
            alpha_vec_ptrs = alpha_b_ptr + k * stride_alpha_k + h_offsets * stride_alpha_h
            alpha_vec = tl.load(alpha_vec_ptrs, mask=h_mask, other=0.0)

            # Base pointer for value at neighbor node j
            val_base_j = value_ptr + j * stride_value_b

            # Load value tile [BLOCK_M, BLOCK_H] for neighbor node j
            val_ptrs = (
                val_base_j
                + m_offsets[:, None] * stride_value_m
                + h_offsets[None, :] * stride_value_h
            )
            val_tile = tl.load(val_ptrs, mask=(m_mask[:, None] & h_mask[None, :]), other=0.0)

            # FMA accumulate: acc[m, h] += val[m, h] * alpha[h]
            acc += val_tile * alpha_vec[None, :]

        # Store the result tile to out[b, :, :]
        out_ptrs = (
            out_b_ptr
            + m_offsets[:, None] * stride_out_m
            + h_offsets[None, :] * stride_out_h
        )
        tl.store(out_ptrs, acc, mask=(m_mask[:, None] & h_mask[None, :]))


# ==========================================
# Core Modules
# ==========================================

class SO3_Linear_e2former(nn.Module):
    """
    Standard SO3 Linear layer using einsum.
    """
    def __init__(self, in_features, out_features, lmax, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        self.weight = nn.Parameter(
            torch.randn((self.lmax + 1), out_features, in_features)
        )
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        self.bias = nn.Parameter(torch.zeros(out_features))

        expand_index = torch.zeros([(lmax + 1) ** 2]).long()
        for l in range(lmax + 1):
            start_idx = l**2
            length = 2 * l + 1
            expand_index[start_idx : (start_idx + length)] = l
        self.register_buffer("expand_index", expand_index)
    
    def forward(self, input_embedding):
        output_shape = input_embedding.shape[:-2]
        l_sum, hidden = input_embedding.shape[-2:]
        
        input_embedding = input_embedding.reshape(
            [output_shape.numel()] + [l_sum, hidden]
        )
        
        weight = torch.index_select(
            self.weight, dim=0, index=self.expand_index
        )
        
        out = torch.einsum(
            "bmi, moi -> bmo", input_embedding, weight
        )
    
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias

        out = out.reshape(output_shape + (l_sum, self.out_features))
        return out

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, lmax={self.lmax}"

class RadialFunction(nn.Module):
    """
    Construct a radial function (linear layers + layer normalization + SiLU) given a list of channels
    """
    def __init__(self, channels_list, use_layer_norm=True):
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue

            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1:
                break
            if use_layer_norm:
                modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(nn.SiLU())

        self.net = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.net(inputs)

class BaseAttentionOrder(nn.Module, ABC):
    def __init__(self, scalar_dim, num_attn_heads, edge_channel_list, lmax):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.num_attn_heads = num_attn_heads
        self.lmax = lmax
        
    @abstractmethod
    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data, **kwargs):
        pass

class ZeroOrderAttention(BaseAttentionOrder):
    def __init__(self, scalar_dim, num_attn_heads, edge_channel_list, lmax):
        super().__init__(scalar_dim, num_attn_heads, edge_channel_list, lmax)
    
        self.rad_func_intputhead = RadialFunction(
            edge_channel_list + [self.scalar_dim]
        )
        
        self.proj_zero = SO3_Linear_e2former(
            self.scalar_dim,
            self.scalar_dim,
            lmax=lmax,
        )

    def _aggregate_neighbors_triton(self, alpha_h, value, idx):
        """
        Triton-accelerated aggregation:
        alpha_h: [N, K, H], value: [N, M, H], idx: [N, K]
        returns aggregated: [N, M, H]
        """
        assert _TRITON_AVAILABLE, "Triton is not available"
        N, K, H = alpha_h.shape
        M = value.shape[1]
        # Ensure contiguity for pointer arithmetic
        alpha_h = alpha_h.contiguous()
        value = value.contiguous()
        idx = idx.contiguous()

        out = torch.empty((N, M, H), device=alpha_h.device, dtype=alpha_h.dtype)

        # Extract strides
        sa_b, sa_k, sa_h = alpha_h.stride()
        sv_b, sv_m, sv_h = value.stride()
        si_b, si_k = idx.stride()
        so_b, so_m, so_h = out.stride()

        # Tile sizes
        BLOCK_H = 64
        BLOCK_M = min(16, M)

        grid = (N, triton.cdiv(H, BLOCK_H))
        weighted_gather_sum_kernel[grid](
            alpha_h, value, idx, out,
            N, K, M, H,
            sa_b, sa_k, sa_h,
            sv_b, sv_m, sv_h,
            si_b, si_k,
            so_b, so_m, so_h,
            BLOCK_H=BLOCK_H, BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=1
        )
        return out
    
    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data, **kwargs):

        f_N1 = value.shape[0]
        f_sparse_idx_node = batched_data["f_sparse_idx_node"]

        # [N, Neighbors, scalar_dim]
        inputhead = self.rad_func_intputhead(x_edge)
        
        # Reshape alpha: [N, Neighbors, Heads, 1]
        # Reshape inputhead: [N, Neighbors, Heads, scalar_dim/Heads]
        # Element-wise multiplication broadcasts alpha across the hidden dimension of each head
        alpha = alpha.reshape(f_N1, -1, self.num_attn_heads, 1) * inputhead.reshape(
            alpha.shape[:2] + (self.num_attn_heads, -1)
        )
        
        # Flatten back to [N, Neighbors, scalar_dim]
        alpha = alpha.reshape(alpha.shape[:2] + (-1,))
        
        # Triton-accelerated neighbor aggregation
        use_triton = (
            _TRITON_AVAILABLE
            and alpha.is_cuda
            and value.is_cuda
            and f_sparse_idx_node.is_cuda
            and alpha.dtype == value.dtype == torch.float32
        )
        if use_triton:
            aggregated = self._aggregate_neighbors_triton(alpha, value, f_sparse_idx_node)
        else:
            # Fallback to original PyTorch implementation
            neighbor_values = value[f_sparse_idx_node]
            attended_values = alpha.unsqueeze(dim=2) * neighbor_values
            aggregated = torch.sum(attended_values, dim=1) 
        
        # Final Projection
        node_output = self.proj_zero(aggregated)
        
        return node_output

# ==========================================
# Main Model Wrapper
# ==========================================

class ModelNew(nn.Module):
    def __init__(self, scalar_dim, num_attn_heads, edge_channel_list, lmax):
        super().__init__()
        self.attn = ZeroOrderAttention(scalar_dim, num_attn_heads, edge_channel_list, lmax)

    def forward(self, alpha, value, x_edge, node_pos, edge_dis, batched_data):
        return self.attn(alpha, value, x_edge, node_pos, edge_dis, batched_data)

# ==========================================
# Hyperparameters & Data Generation
# ==========================================

SCALAR_DIM = 256    
NUM_ATTN_HEADS = 32
LMAX = 3
CHANNEL_LIST = [512, 128, 128]
N_NODES = 2233
N_NEIGHBORS = 20
DIM_IRREPS_TOTAL = (LMAX + 1) ** 2  # 16
EDGE_DIM = 512

def get_inputs():
    # Determine device based on availability
    device = 'cuda'
    torch.manual_seed(42)
    
    # alpha shape: [2233, 20, 32]
    alpha = torch.randn(N_NODES, N_NEIGHBORS, NUM_ATTN_HEADS, device=device)
    
    # value shape: [2233, 16, 256]
    value = torch.randn(N_NODES, DIM_IRREPS_TOTAL, SCALAR_DIM, device=device)
    
    # x_edge shape: [2233, 20, 512]
    x_edge = torch.randn(N_NODES, N_NEIGHBORS, EDGE_DIM, device=device)
    
    # f_sparse_idx_node shape: [2233, 20]
    f_sparse_idx_node = torch.randint(0, N_NODES, (N_NODES, N_NEIGHBORS), device=device)
    
    batched_data = {
        "f_sparse_idx_node": f_sparse_idx_node
    }
    
    node_pos = None
    edge_dis = None
    
    return [alpha, value, x_edge, node_pos, edge_dis, batched_data]

def get_init_inputs():
    return [SCALAR_DIM, NUM_ATTN_HEADS, CHANNEL_LIST, LMAX]