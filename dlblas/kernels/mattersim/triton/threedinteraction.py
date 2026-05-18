import torch
import torch.nn as nn
import triton
import triton.language as tl

from torch.cuda import nvtx
from typing import Union, Optional

device = "cuda"


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    # Fast path with Triton for the common case used in ModelNew:
    # - reduce == "sum", dim == 0
    # - src is 2D [T, D]
    # - index is 1D [T]
    # - out is None (we allocate)
    # Otherwise, fallback to PyTorch reference implementation.
    if (
        reduce == "sum"
        and dim == 0
        and out is None
        and src.dim() == 2
        and index.dim() == 1
        and src.is_cuda
        and index.is_cuda
        and src.dtype == torch.float32
    ):
        T, D = src.shape
        # Heuristic: for very small problems, avoid Triton launch overhead
        if T == 0:
            out_rows = 0 if dim_size is None else int(dim_size)
            return torch.zeros((out_rows, D), dtype=src.dtype, device=src.device)
        if dim_size is not None:
            out_rows = int(dim_size)
        elif index.numel() == 0:
            out_rows = 0
        else:
            out_rows = int(index.max().item()) + 1

        # For tiny sizes, PyTorch scatter_add_ is typically faster
        if T * D <= 2048:
            # Avoid general _broadcast overhead for dim=0 by using a cheap expand
            bcast_index = index.view(T, 1).expand(T, D)
            out_t = torch.zeros((out_rows, D), dtype=src.dtype, device=src.device)
            return out_t.scatter_add_(0, bcast_index, src)

        return _scatter_add_2d_dim0_triton(src, index, out_rows)

    # Reference fallback to ensure full numerical correctness and generality
    assert reduce == "sum"  # for now, TODO
    bcast_index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif bcast_index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(bcast_index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, bcast_index, src)
    else:
        return out.scatter_add_(dim, bcast_index, src)


@triton.jit
def _scatter_add_2d_dim0_kernel(
    src_ptr,  # *f32
    idx_ptr,  # *i32 (destination row indices)
    out_ptr,  # *f32
    T,
    D,  # int32
    stride_src0,
    stride_src1,  # int32
    stride_out0,
    stride_out1,  # int32
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)

    row = pid_t
    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    row_mask = row < T
    mask = row_mask & (cols < D)

    if not row_mask:
        return

    # Load index for this row (expect int32)
    idx_val = tl.load(idx_ptr + row, mask=row_mask, other=0)

    # Load source tile
    src_offsets = row * stride_src0 + cols * stride_src1
    vals = tl.load(
        src_ptr + src_offsets,
        mask=mask,
        other=0.0,
        cache_modifier=".ca",
        eviction_policy="evict_first",
    )

    # Atomic add into out at [idx_val, cols]
    out_offsets = idx_val * stride_out0 + cols * stride_out1
    tl.atomic_add(out_ptr + out_offsets, vals, mask=mask)


def _scatter_add_2d_dim0_triton(
    src: torch.Tensor, index: torch.Tensor, out_rows: int
) -> torch.Tensor:
    # Expect src: [T, D], index: [T]
    assert src.dim() == 2 and index.dim() == 1
    T, D = src.shape
    # Allocate output
    out = torch.zeros((out_rows, D), dtype=src.dtype, device=src.device)
    if T == 0 or D == 0 or out_rows == 0:
        return out
    # Ensure contiguity for predictable strides
    src_c = src.contiguous()
    # Convert to int32 to avoid per-element cast in kernel
    idx_c = index.to(torch.int32).contiguous()
    # Choose tile and warps based on width
    if D <= 32:
        BLOCK_D = 32
        NUM_WARPS = 1
    elif D <= 64:
        BLOCK_D = 64
        NUM_WARPS = 2
    elif D <= 128:
        BLOCK_D = 128
        NUM_WARPS = 4
    else:
        BLOCK_D = 256
        NUM_WARPS = 8
    grid = (triton.cdiv(D, BLOCK_D), T)
    stride_src0, stride_src1 = src_c.stride()
    stride_out0, stride_out1 = out.stride()
    _scatter_add_2d_dim0_kernel[grid](
        src_c,
        idx_c,
        out,
        T,
        D,
        stride_src0,
        stride_src1,
        stride_out0,
        stride_out1,
        BLOCK_D=BLOCK_D,
        num_warps=NUM_WARPS,
        num_stages=1,
    )
    return out


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias, device=device)

    def forward(self, x):
        return self.linear(x)


class SigmoidLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x,
    ):
        return self.sigmoid(self.linear(x))


class SwishLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias, device=device)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x,
    ):
        x = self.linear(x)
        return x * self.sigmoid(x)


class ReLULayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias, device=device)
        self.relu = nn.ReLU()


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dims: list,
        activation: Union[list[Union[str, None]], str] = "swish",
        use_bias: bool = True,
    ):
        super().__init__()
        input_dim = in_dim
        if isinstance(activation, str) or activation is None:
            activation = [activation] * len(out_dims)
        else:
            assert len(activation) == len(
                out_dims
            ), "activation and out_dims must have the same length"
        module_list_g = []
        for i in range(len(out_dims)):
            if activation[i] == "swish":
                module_list_g.append(  # noqa: E501
                    SwishLayer(input_dim, out_dims[i], bias=use_bias)
                )
            elif activation[i] == "sigmoid":
                module_list_g.append(
                    SigmoidLayer(input_dim, out_dims[i], bias=use_bias)
                )
            elif activation[i] is None:
                module_list_g.append(  # noqa: E501
                    LinearLayer(input_dim, out_dims[i], bias=use_bias)
                )
            input_dim = out_dims[i]
        module_list_sigma = []
        activation[-1] = "sigmoid"
        input_dim = in_dim
        for i in range(len(out_dims)):
            if activation[i] == "swish":
                module_list_sigma.append(
                    SwishLayer(input_dim, out_dims[i], bias=use_bias)
                )
            elif activation[i] == "sigmoid":
                module_list_sigma.append(
                    SigmoidLayer(input_dim, out_dims[i], bias=use_bias)
                )
            elif activation[i] is None:
                module_list_sigma.append(
                    LinearLayer(input_dim, out_dims[i], bias=use_bias)
                )
            else:
                raise NotImplementedError
            input_dim = out_dims[i]
        self.g = nn.Sequential(*module_list_g)
        self.sigma = nn.Sequential(*module_list_sigma)

    def forward(
        self,
        x,
    ):
        return self.g(x) * self.sigma(x)


def polynomial(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Polynomial cutoff function
    Args:
        r (tf.Tensor): radius distance tensor
        cutoff (float): cutoff distance
    Returns: polynomial cutoff functions
    """
    ratio = torch.div(r, cutoff)
    result = (
        1
        - 6 * torch.pow(ratio, 5)
        + 15 * torch.pow(ratio, 4)
        - 10 * torch.pow(ratio, 3)
    )
    return torch.clamp(result, min=0.0)


class ModelNew(nn.Module):
    def __init__(
        self,
        max_n,
        max_l,
        cutoff,
        units,
        spherecal_dim,
        threebody_cutoff,
    ):
        super().__init__()
        # self.sbf = SphericalBesselFunction(
        #            max_l=max_l, max_n=max_n, cutoff=cutoff, smooth=smooth)
        # self.shf = SphericalHarmonicsFunction(max_l=max_l, use_phi=use_phi)
        self.atom_mlp = SigmoidLayer(in_dim=units, out_dim=spherecal_dim)
        # Linyu have modified the self.edge_gate_mlp
        # by adding swish activation and use_bias=False
        self.edge_gate_mlp = GatedMLP(
            in_dim=spherecal_dim,
            out_dims=[units],
            activation="swish",
            use_bias=False,  # noqa: E501
        )
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff

    def forward(
        self,
        edge_attr,
        three_basis,
        atom_attr,
        edge_index,
        three_body_index,
        edge_length,
        num_edges,
        num_triple_ij,
    ):
        atom_mask = (
            self.atom_mlp(atom_attr)[edge_index[0][three_body_index[:, 1]]]
            * polynomial(
                edge_length[three_body_index[:, 0]], self.threebody_cutoff  # noqa: E501
            )
            * polynomial(
                edge_length[three_body_index[:, 1]], self.threebody_cutoff  # noqa: E501
            )
        )
        three_basis = three_basis * atom_mask
        index_map = torch.arange(torch.sum(num_edges).item(), device=edge_length.device)
        index_map = torch.repeat_interleave(index_map, num_triple_ij).to(
            edge_length.device
        )
        # Triton-accelerated scatter-add along dim=0 (with small-size fallback)
        e_ij_tuda = scatter(
            three_basis,
            index_map,
            dim=0,
            reduce="sum",
            dim_size=torch.sum(num_edges).item(),
        )
        edge_attr_prime = edge_attr + self.edge_gate_mlp(e_ij_tuda)
        return edge_attr_prime


max_n = 4
max_l = 5
units = 64
spherical_dim = max_n * max_l
cutoff = threebody_cutoff = 1.0


def get_init_inputs():
    return [max_n, max_l, cutoff, units, spherical_dim, threebody_cutoff]


def get_inputs():
    edge_attr = torch.randn(8, units, device=device)
    num_triple_ij = torch.tensor([3, 2, 4, 1, 3, 2, 3, 2], device=device)
    total_triples = num_triple_ij.sum().item()
    three_basis = torch.randn(total_triples, spherical_dim, device=device)
    atom_attr = torch.randn(5, units, device=device)
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2, 2, 3],  # ~P~J~B~B
            [1, 2, 3, 2, 4, 3, 4, 4],  # ~[| ~G~J~B~B
        ],
        device=device,
    )
    three_body_index = torch.zeros(total_triples, 2, dtype=torch.long, device=device)

    idx = 0
    for edge_idx, count in enumerate(num_triple_ij):
        for _ in range(count):
            # 为边 edge_idx ~^~D建~I~S~^~K~L~Z~O~\~@~I~K~O~@~]边~\为第~L~]边
            j = torch.randint(0, 8, (1,), device=device).item()
            while j == edge_idx:
                j = torch.randint(0, 8, (1,), device=device).item()
            three_body_index[idx] = torch.tensor([edge_idx, j], device=device)
            idx += 1

    edge_length = torch.rand(8, device=device) * 3.0  # 边~U度~L~C~[ 0-3
    num_edges = torch.tensor([8], device=device)
    num_triple_ij = torch.tensor([3, 2, 4, 1, 3, 2, 3, 2], device=device)
    return [
        edge_attr,
        three_basis,
        atom_attr,
        edge_index,
        three_body_index,
        edge_length,
        num_edges,
        num_triple_ij,
    ]


out = ModelNew(*get_init_inputs()).forward(*get_inputs())
