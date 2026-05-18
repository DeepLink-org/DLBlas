import torch
import torch.nn as nn

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
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


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


class Model(nn.Module):
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
        index_map = torch.arange(torch.sum(num_edges).item()).to(
            edge_length.device
        )  # noqa: E501
        index_map = torch.repeat_interleave(index_map, num_triple_ij).to(
            edge_length.device
        )
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


out = Model(*get_init_inputs()).forward(*get_inputs())
