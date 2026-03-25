import math
import time
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

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
        
        # Gather neighbor values: [N, Neighbors, (L+1)^2, scalar_dim]
        neighbor_values = value[f_sparse_idx_node]
        
        # Apply attention weights
        # alpha unsqueeze -> [N, Neighbors, 1, scalar_dim]
        # Broadcasting over the Irreps dimension (dim 2)
        attended_values = alpha.unsqueeze(dim=2) * neighbor_values
        
        # Sum over neighbors (dim 1) -> [N, (L+1)^2, scalar_dim]
        aggregated = torch.sum(attended_values, dim=1) 
        
        # Final Projection
        node_output = self.proj_zero(aggregated)
        
        return node_output

# ==========================================
# Main Model Wrapper
# ==========================================

class Model(nn.Module):
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