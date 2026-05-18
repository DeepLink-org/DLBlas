import torch
import torch.nn as nn
from typing import Union

import triton
import triton.language as tl

device = "cuda"


@triton.jit
def linear_bias_act_kernel(
    X_ptr,  # [M, K]
    W_ptr,  # [N, K]
    BIAS_ptr,  # [N] or unused if USE_BIAS=0
    Y_ptr,  # [M, N]
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    ACT: tl.constexpr,  # 0: none, 1: sigmoid, 2: swish
    USE_BIAS: tl.constexpr,  # 0 or 1
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

        x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)

        acc += tl.dot(x, tl.trans(w))

    if USE_BIAS:
        b = tl.load(BIAS_ptr + offs_n, mask=mask_n, other=0.0)
        y = acc + b[None, :]
    else:
        y = acc

    if ACT == 1:
        y = tl.sigmoid(y)
    elif ACT == 2:
        y = y * tl.sigmoid(y)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def two_path_two_layer_kernel(
    X_ptr,
    # g-path
    W1g_ptr,
    B1g_ptr,
    W2g_ptr,
    B2g_ptr,
    # sigma-path
    W1s_ptr,
    B1s_ptr,
    W2s_ptr,
    B2s_ptr,
    # output
    Y_ptr,
    # sizes
    M,
    K0,
    N1,
    N2,
    # strides
    stride_xm,
    stride_xk,
    stride_w1g_n,
    stride_w1g_k,
    stride_w2g_n,
    stride_w2g_k,
    stride_w1s_n,
    stride_w1s_k,
    stride_w2s_n,
    stride_w2s_k,
    stride_ym,
    stride_yn,
    # activations
    ACT1_G: tl.constexpr,
    ACT2_G: tl.constexpr,
    ACT1_S: tl.constexpr,
    ACT2_S: tl.constexpr,
    # bias flags
    USE_B1G: tl.constexpr,
    USE_B2G: tl.constexpr,
    USE_B1S: tl.constexpr,
    USE_B2S: tl.constexpr,
    # tiling
    BLOCK_M: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLOCK_K0: tl.constexpr,
    BLOCK_N1: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n2 = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n2 * BLOCK_N2 + tl.arange(0, BLOCK_N2)
    mask_m = offs_m < M
    mask_n2 = offs_n2 < N2

    acc_g = tl.zeros((BLOCK_M, BLOCK_N2), dtype=tl.float32)
    acc_s = tl.zeros((BLOCK_M, BLOCK_N2), dtype=tl.float32)

    for n1 in range(0, N1, BLOCK_N1):
        offs_n1 = n1 + tl.arange(0, BLOCK_N1)
        n1_mask = offs_n1 < N1

        h_g = tl.zeros((BLOCK_M, BLOCK_N1), dtype=tl.float32)
        h_s = tl.zeros((BLOCK_M, BLOCK_N1), dtype=tl.float32)

        for k0 in range(0, K0, BLOCK_K0):
            offs_k0 = k0 + tl.arange(0, BLOCK_K0)
            k_mask = offs_k0 < K0

            x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k0[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

            w1g_ptrs = (
                W1g_ptr
                + offs_n1[:, None] * stride_w1g_n
                + offs_k0[None, :] * stride_w1g_k
            )
            w1s_ptrs = (
                W1s_ptr
                + offs_n1[:, None] * stride_w1s_n
                + offs_k0[None, :] * stride_w1s_k
            )

            w1g = tl.load(w1g_ptrs, mask=n1_mask[:, None] & k_mask[None, :], other=0.0)
            w1s = tl.load(w1s_ptrs, mask=n1_mask[:, None] & k_mask[None, :], other=0.0)

            h_g += tl.dot(x, tl.trans(w1g))
            h_s += tl.dot(x, tl.trans(w1s))

        if USE_B1G:
            b1g = tl.load(B1g_ptr + offs_n1, mask=n1_mask, other=0.0)
            h_g += b1g[None, :]
        if USE_B1S:
            b1s = tl.load(B1s_ptr + offs_n1, mask=n1_mask, other=0.0)
            h_s += b1s[None, :]

        if ACT1_G == 1:
            h_g = tl.sigmoid(h_g)
        elif ACT1_G == 2:
            h_g = h_g * tl.sigmoid(h_g)

        if ACT1_S == 1:
            h_s = tl.sigmoid(h_s)
        elif ACT1_S == 2:
            h_s = h_s * tl.sigmoid(h_s)

        w2g_ptrs = (
            W2g_ptr + offs_n2[:, None] * stride_w2g_n + offs_n1[None, :] * stride_w2g_k
        )
        w2s_ptrs = (
            W2s_ptr + offs_n2[:, None] * stride_w2s_n + offs_n1[None, :] * stride_w2s_k
        )

        w2g = tl.load(w2g_ptrs, mask=mask_n2[:, None] & n1_mask[None, :], other=0.0)
        w2s = tl.load(w2s_ptrs, mask=mask_n2[:, None] & n1_mask[None, :], other=0.0)

        acc_g += tl.dot(h_g, tl.trans(w2g))
        acc_s += tl.dot(h_s, tl.trans(w2s))

    if USE_B2G:
        b2g = tl.load(B2g_ptr + offs_n2, mask=mask_n2, other=0.0)
        acc_g += b2g[None, :]
    if USE_B2S:
        b2s = tl.load(B2s_ptr + offs_n2, mask=mask_n2, other=0.0)
        acc_s += b2s[None, :]

    if ACT2_G == 1:
        acc_g = tl.sigmoid(acc_g)
    elif ACT2_G == 2:
        acc_g = acc_g * tl.sigmoid(acc_g)

    if ACT2_S == 1:
        acc_s = tl.sigmoid(acc_s)
    elif ACT2_S == 2:
        acc_s = acc_s * tl.sigmoid(acc_s)

    y = acc_g * acc_s

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n2[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n2[None, :])


def _linear_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    activation: str,
):
    # Ensure contiguous tensors for predictable strides
    x = x.contiguous()
    w = weight.contiguous()
    BIAS = (
        bias.contiguous()
        if bias is not None
        else torch.empty(0, device=x.device, dtype=x.dtype)
    )

    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Strides in elements
    stride_xm, stride_xk = x.stride()
    stride_wn, stride_wk = w.stride()
    stride_ym, stride_yn = y.stride()

    act_map = {"none": 0, "sigmoid": 1, "swish": 2}
    ACT = act_map.get(activation, 0)
    USE_BIAS = 1 if bias is not None else 0

    # Tuned for small MLP layers on H100/H200
    BLOCK_M = 32 if M <= 32 else 64
    BLOCK_N = 32 if N <= 32 else (64 if N <= 64 else 128)
    BLOCK_K = 16 if K <= 16 else (32 if K <= 32 else 64)

    # Favor low-overhead scheduling for small tiles
    tile_elems = BLOCK_M * BLOCK_N
    if tile_elems <= 1024:
        num_warps = 1
        num_stages = 1 if K <= BLOCK_K else 2
    elif tile_elems <= 4096:
        num_warps = 4
        num_stages = 2
    else:
        num_warps = 8
        num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    linear_bias_act_kernel[grid](
        x,
        w,
        BIAS if USE_BIAS else w,  # pass some valid ptr even if not used
        y,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_ym,
        stride_yn,
        ACT=ACT,
        USE_BIAS=USE_BIAS,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


def _two_path_mlp_triton(x: torch.Tensor, g_seq: nn.Sequential, s_seq: nn.Sequential):
    # Expect exactly two layers in each path
    g0, g1 = g_seq[0], g_seq[1]
    s0, s1 = s_seq[0], s_seq[1]

    # Extract weights and biases
    W1g, B1g = g0.linear.weight.contiguous(), (
        g0.linear.bias.contiguous()
        if g0.linear.bias is not None
        else torch.empty(0, device=x.device, dtype=x.dtype)
    )
    W2g, B2g = g1.linear.weight.contiguous(), (
        g1.linear.bias.contiguous()
        if g1.linear.bias is not None
        else torch.empty(0, device=x.device, dtype=x.dtype)
    )
    W1s, B1s = s0.linear.weight.contiguous(), (
        s0.linear.bias.contiguous()
        if s0.linear.bias is not None
        else torch.empty(0, device=x.device, dtype=x.dtype)
    )
    W2s, B2s = s1.linear.weight.contiguous(), (
        s1.linear.bias.contiguous()
        if s1.linear.bias is not None
        else torch.empty(0, device=x.device, dtype=x.dtype)
    )

    # Sizes
    x = x.contiguous()
    M, K0 = x.shape
    N1 = W1g.shape[0]
    N2 = W2g.shape[0]

    # Output
    y = torch.empty((M, N2), device=x.device, dtype=x.dtype)

    # Strides
    stride_xm, stride_xk = x.stride()
    stride_ym, stride_yn = y.stride()

    s_w1g_n, s_w1g_k = W1g.stride()
    s_w2g_n, s_w2g_k = W2g.stride()
    s_w1s_n, s_w1s_k = W1s.stride()
    s_w2s_n, s_w2s_k = W2s.stride()

    # Activation codes
    def act_code(layer):
        if isinstance(layer, SigmoidLayer):
            return 1
        if isinstance(layer, SwishLayer):
            return 2
        return 0

    ACT1_G = act_code(g0)
    ACT2_G = act_code(g1)
    ACT1_S = act_code(s0)
    ACT2_S = act_code(s1)

    USE_B1G = 1 if g0.linear.bias is not None else 0
    USE_B2G = 1 if g1.linear.bias is not None else 0
    USE_B1S = 1 if s0.linear.bias is not None else 0
    USE_B2S = 1 if s1.linear.bias is not None else 0

    # Tiling tuned for tiny MLPs
    BLOCK_M = 32 if M >= 32 else 16
    BLOCK_N2 = 32 if N2 <= 32 else 64
    BLOCK_N1 = 16 if N1 <= 16 else 32
    BLOCK_K0 = 16 if K0 <= 16 else 32

    # Lightweight scheduling to reduce overhead
    num_warps = 2 if (BLOCK_M * BLOCK_N2) >= 1024 else 1
    num_stages = 1 if K0 <= BLOCK_K0 else 2

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N2, BLOCK_N2))
    two_path_two_layer_kernel[grid](
        x,
        # g-path
        W1g,
        B1g if USE_B1G else W1g,
        W2g,
        B2g if USE_B2G else W2g,
        # sigma-path
        W1s,
        B1s if USE_B1S else W1s,
        W2s,
        B2s if USE_B2S else W2s,
        # out
        y,
        # sizes
        M,
        K0,
        N1,
        N2,
        # strides
        stride_xm,
        stride_xk,
        s_w1g_n,
        s_w1g_k,
        s_w2g_n,
        s_w2g_k,
        s_w1s_n,
        s_w1s_k,
        s_w2s_n,
        s_w2s_k,
        stride_ym,
        stride_yn,
        # activations
        ACT1_G=ACT1_G,
        ACT2_G=ACT2_G,
        ACT1_S=ACT1_S,
        ACT2_S=ACT2_S,
        # biases
        USE_B1G=USE_B1G,
        USE_B2G=USE_B2G,
        USE_B1S=USE_B1S,
        USE_B2S=USE_B2S,
        # tiling
        BLOCK_M=BLOCK_M,
        BLOCK_N2=BLOCK_N2,
        BLOCK_K0=BLOCK_K0,
        BLOCK_N1=BLOCK_N1,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


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
        # Fused Triton linear
        return _linear_triton(
            x, self.linear.weight, self.linear.bias, activation="none"
        )


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
        # Fused Triton linear + sigmoid
        return _linear_triton(
            x, self.linear.weight, self.linear.bias, activation="sigmoid"
        )


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
        # Fused Triton linear + swish
        return _linear_triton(
            x, self.linear.weight, self.linear.bias, activation="swish"
        )


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
        # Keep as-is (not used by default); could be Triton-fused if needed
        return self.relu(self.linear(x))


class ModelNew(nn.Module):
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
        # Fast path: fuse the two 2-layer MLP branches into a single Triton kernel
        try:
            if len(self.g) == 2 and len(self.sigma) == 2:
                return _two_path_mlp_triton(x, self.g, self.sigma)
        except Exception:
            pass
        # Fallback: separate evaluation
        return self.g(x) * self.sigma(x)


def get_init_inputs():
    return [10, [20, 30]]


def get_inputs():
    return [torch.randn(32, 10, device=device)]
