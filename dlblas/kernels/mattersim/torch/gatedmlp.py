import torch
import torch.nn as nn

from typing import Union

device = "cuda"


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

    def forward(
        self,
        x,
    ):
        return self.relu(self.linear(x))


class Model(nn.Module):
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


def get_init_inputs():
    return [10, [20, 30]]


def get_inputs():
    return [torch.randn(32, 10, device=device)]
