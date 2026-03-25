import math
import torch
import torch.nn as nn
from torch import Tensor
import e3nn.o3 as o3
import triton
import triton.language as tl

# ==========================================
# Helper Modules (Dependencies)
# ==========================================
class SmoothLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope
    def forward(self, x):
        return (1 - self.alpha) * x * torch.sigmoid(x) + self.alpha * x
    def extra_repr(self):
        return "negative_slope={}".format(self.alpha)

class RadialFunction(nn.Module):
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

class SO3_Linear_e2former(nn.Module):
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
        out = torch.einsum("bmi, moi -> bmo", input_embedding, weight)
        bias = self.bias.view(1, 1, self.out_features)
        out[:, 0:1, :] = out.narrow(1, 0, 1) + bias
        out = out.reshape(output_shape + (l_sum, self.out_features))
        return out

# ==========================================
# Triton Kernel: compute concatenated per-l projections for i/j ends
# Produces features of shape [Q, E, 2*(L+1)*D] in order: [i_l0, j_l0, i_l1, j_l1, ...]
# ==========================================
@triton.jit
def compute_x0_features_kernel(
    Y_ptr,           # float32 [Q, E, M]
    NODE_ptr,        # float32 [Q, M, D]
    IDX_ptr,         # int32   [Q, E]
    OUT_ptr,         # float32 [Q, E, 2*(L+1)*D]
    Q: tl.constexpr, # number of nodes
    E: tl.constexpr, # number of neighbors
    M,               # total M = (L+1)^2
    D,               # attn_dim
    y_s0, y_s1, y_s2,
    n_s0, n_s1, n_s2,
    idx_s0, idx_s1,
    out_s0, out_s1, out_s2,
    BLOCK_D: tl.constexpr,
    LMAX: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_e = tl.program_id(1)
    pid_dt = tl.program_id(2)
    if (pid_q >= Q) or (pid_e >= E):
        return

    d_offsets = pid_dt * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D
    tl.multiple_of(d_offsets, 8)

    # Base pointers
    y_base = Y_ptr + pid_q * y_s0 + pid_e * y_s1
    idx_j = tl.load(IDX_ptr + pid_q * idx_s0 + pid_e * idx_s1)
    node_i_base = NODE_ptr + pid_q * n_s0
    node_j_base = NODE_ptr + idx_j * n_s0

    # Loop over l and accumulate dot products per-l for i-end and j-end
    for l in tl.static_range(0, LMAX + 1):
        start_m = l * l
        acc_i = tl.zeros([BLOCK_D], dtype=tl.float32)
        acc_j = tl.zeros([BLOCK_D], dtype=tl.float32)
        # Unrolled loop over m in current l-band
        for mm in tl.static_range(0, 2 * LMAX + 1):
            if mm < (2 * l + 1):
                m_index = start_m + mm
                y_val = tl.load(y_base + m_index * y_s2)
                # i-end
                v_i = tl.load(node_i_base + m_index * n_s1 + d_offsets * n_s2, mask=d_mask, other=0.0)
                # j-end
                v_j = tl.load(node_j_base + m_index * n_s1 + d_offsets * n_s2, mask=d_mask, other=0.0)
                acc_i += y_val * v_i
                acc_j += y_val * v_j

        # Store in interleaved order
        out_i = OUT_ptr + pid_q * out_s0 + pid_e * out_s1 + (2 * l) * D * out_s2
        out_j = OUT_ptr + pid_q * out_s0 + pid_e * out_s1 + (2 * l + 1) * D * out_s2
        tl.store(out_i + d_offsets * out_s2, acc_i, mask=d_mask)
        tl.store(out_j + d_offsets * out_s2, acc_j, mask=d_mask)

# ==========================================
# Main Model Structure
# ==========================================
class ModelNew(nn.Module):
    """
    Dot product based alpha computation with spherical harmonics.
    Re-implementation of DotAlphaModule structure.
    """
    def __init__(
        self,
        irreps_node_input,
        num_attn_heads,
        attn_scalar_head,
        attn_weight_input_dim,
        edge_channel_list,
        lmax,
        small_version=False,
    ):
        super().__init__()
        if isinstance(irreps_node_input, str):
            irreps_node_input = o3.Irreps(irreps_node_input)
        self.num_attn_heads = num_attn_heads
        self.attn_scalar_head = attn_scalar_head
        self.lmax = lmax
        self.scalar_dim = irreps_node_input[0][0]
        dim_factor = 8 if small_version else 1
        self.attn_dim = attn_weight_input_dim // dim_factor
        self.dot_linear = SO3_Linear_e2former(
            self.scalar_dim,
            self.attn_dim,
            lmax=lmax,
        )
        self.alpha_norm = nn.LayerNorm(attn_scalar_head)
        self.alpha_dot = nn.Parameter(
            torch.randn(num_attn_heads, attn_scalar_head)
        )
        std = 1.0 / math.sqrt(attn_scalar_head)
        nn.init.uniform_(self.alpha_dot, -std, std)
        self.fc_m0 = nn.Linear(
            2 * self.attn_dim * (lmax + 1),
            num_attn_heads * attn_scalar_head,
        )
        self.rad_func_m0 = RadialFunction(
            edge_channel_list + [2 * self.attn_dim * (lmax + 1)]
        )
        self.alpha_act = SmoothLeakyReLU(0.2)

    def forward(
        self, 
        x_edge: Tensor, 
        node_irreps_input: Tensor, 
        edge_vec: Tensor, 
        f_sparse_idx_node: Tensor
    ) -> Tensor:
        f_N1 = node_irreps_input.shape[0]
        # Linear projection
        node_irreps_input_dot = self.dot_linear(node_irreps_input)  # [Q, M, D], M=(lmax+1)^2

        # Compute spherical harmonics for all l at once (shape: [Q, E, M])
        Y_all = o3.spherical_harmonics(
            list(range(self.lmax + 1)), edge_vec, normalize=True
        )

        # Prepare inputs for Triton kernel
        Q = f_N1
        E = edge_vec.shape[1]
        D = self.attn_dim
        LMAX = self.lmax
        M = (LMAX + 1) ** 2
        assert Y_all.shape[-1] == M and node_irreps_input_dot.shape[1] == M

        Y_all = Y_all.contiguous()
        node_irreps_input_dot = node_irreps_input_dot.contiguous()
        idx_ji = f_sparse_idx_node.to(torch.int32).contiguous()

        # Output buffer: concatenated per-l [i, j] features -> [Q, E, 2*(L+1)*D]
        feat_dim = 2 * (LMAX + 1) * D
        x0_features = torch.empty(
            (Q, E, feat_dim),
            dtype=node_irreps_input_dot.dtype,
            device=node_irreps_input_dot.device,
        )

        # Strides in elements
        y_s0, y_s1, y_s2 = Y_all.stride()
        n_s0, n_s1, n_s2 = node_irreps_input_dot.stride()
        idx_s0, idx_s1 = idx_ji.stride()
        out_s0, out_s1, out_s2 = x0_features.stride()

        # Launch Triton kernel
        BLOCK_D = 128 if D >= 128 else 64
        grid = (Q, E, triton.cdiv(D, BLOCK_D))
        compute_x0_features_kernel[grid](
            Y_all, node_irreps_input_dot, idx_ji, x0_features,
            Q, E, M, D,
            y_s0, y_s1, y_s2,
            n_s0, n_s1, n_s2,
            idx_s0, idx_s1,
            out_s0, out_s1, out_s2,
            BLOCK_D=BLOCK_D, LMAX=LMAX,
            num_warps=4 if BLOCK_D == 128 else 2,
            num_stages=3,
        )

        # Compute alpha
        edge_m0 = self.rad_func_m0(x_edge)  # [Q, E, 2*D*(L+1)]
        x_0_alpha = self.fc_m0(x0_features * edge_m0)
        x_0_alpha = x_0_alpha.reshape(
            f_N1, -1, self.num_attn_heads, self.attn_scalar_head
        )
        x_0_alpha = self.alpha_norm(x_0_alpha)
        x_0_alpha = self.alpha_act(x_0_alpha)
        alpha = torch.einsum("qeik, ik -> qei", x_0_alpha, self.alpha_dot)
        return alpha

# ==========================================
# Hyperparameters & Data Generation
# ==========================================
# Hyperparameters based on the provided text
dtype = torch.float32
irreps_str = "256x0e+256x1e+256x2e+256x3e"
num_attn_heads = 32
attn_scalar_head = 16
attn_weight_input_dim = 256
edge_channel_list = [512, 128, 128]
lmax = 3
small_version = False
# Shapes
N_nodes = 2233
N_neighbors = 20
dim_edge = 512
dim_node_hidden = 256
dim_irreps_total = 16 # (lmax+1)^2 = 16

def get_inputs():
    torch.manual_seed(123)
    # x_edge: [2233, 20, 512]
    x_edge = torch.randn(N_nodes, N_neighbors, dim_edge, dtype=dtype, device="cuda")
    # node_irreps_input: [2233, 16, 256]
    node_irreps_input = torch.randn(N_nodes, dim_irreps_total, dim_node_hidden, dtype=dtype, device="cuda")
    # edge_vec: [2233, 20, 3]
    edge_vec = torch.randn(N_nodes, N_neighbors, 3, dtype=dtype, device="cuda")
    # f_sparse_idx_node: [2233, 20] (indices pointing to nodes)
    f_sparse_idx_node = torch.randint(0, N_nodes, (N_nodes, N_neighbors), dtype=torch.int64, device="cuda")
    return [
        x_edge,
        node_irreps_input,
        edge_vec,
        f_sparse_idx_node
    ]

def get_init_inputs():
    return [
        irreps_str, 
        num_attn_heads,
        attn_scalar_head,
        attn_weight_input_dim,
        edge_channel_list,
        lmax,
        small_version
    ]