import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Favor wide-N tiles for small-batch M to improve occupancy on H200
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        # Balanced tiles
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        # Larger-N option
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=8),
        # Fallbacks
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=3, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _matmul_bias_activation_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ADD_BIAS: tl.constexpr,  # 1 to add bias, else 0
    ACT: tl.constexpr,       # 1 to apply ReLU, else 0
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Tile indices mapping (persistent 1D grid across (M,N))
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M
    num_pid_in_group = group_size * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group // num_pid_n)
    pid_n = pid_in_group % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        k_remaining = K - k_iter
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # Optional bias add
    if ADD_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    # Optional activation: ReLU
    if ACT:
        acc = tl.maximum(acc, 0)

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def _triton_linear_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, relu: bool) -> torch.Tensor:
    # Fallback to PyTorch when Triton can't be used or gradients are tracked
    if (not x.is_cuda) or (not weight.is_cuda) or torch.is_grad_enabled() or x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        out = F.linear(x, weight, bias)
        if relu:
            out = F.relu(out)
        return out

    # Shapes
    M, K = x.shape
    out_features, in_features = weight.shape
    assert in_features == K, "Incompatible shapes for linear layer"
    N = out_features

    # Contiguous inputs for predictable strides
    a = x.contiguous()
    # B = W^T for A(M,K) @ B(K,N); make B contiguous for coalesced loads over N
    b = weight.t().contiguous()
    # Output always computed in float32 in-kernel, cast back to input dtype after
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Strides in elements
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = out.stride()

    # Bias pointer - safe placeholder when not used
    bias_ptr = bias if (bias is not None) else out

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _matmul_bias_activation_kernel[grid](
        a, b, bias_ptr, out,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        int(bias is not None),
        int(relu),
    )

    # Cast result back to original dtype
    if x.dtype != torch.float32:
        out = out.to(x.dtype)
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()

        layers = []
        current_input_size = input_size

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size

        layers.append(nn.Linear(current_input_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Fast path using Triton when possible; otherwise fallback to PyTorch
        modules = list(self.network)
        i = 0
        out = x
        while i < len(modules):
            m = modules[i]
            if isinstance(m, nn.Linear):
                relu_next = (i + 1 < len(modules)) and isinstance(modules[i + 1], nn.ReLU)
                out = _triton_linear_forward(out, m.weight, m.bias, relu=relu_next)
                i += 2 if relu_next else 1
            elif isinstance(m, nn.ReLU):
                # Should be skipped if fused; keep for robustness
                out = F.relu(out)
                i += 1
            else:
                out = m(out)
                i += 1
        return out


# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [2000, 2000]  # Example of deep and narrow layers
output_size = 10

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]