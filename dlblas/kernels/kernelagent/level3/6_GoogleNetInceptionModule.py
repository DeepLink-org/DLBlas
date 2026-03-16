import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _matmul_bias_kernel(
    a_ptr,  # (M, K)
    b_ptr,  # (K, N)
    bias_ptr,  # (N,)
    c_ptr,  # (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    if ADD_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _matmul_bias(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # a: (M, K), b: (K, N), bias: (N,)
    assert a.is_cuda and b.is_cuda and bias.is_cuda
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb and bias.numel() == N
    # Compute in fp32 for parity with nn.Conv2d default
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Launch kernel
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_bias_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        ADD_BIAS=True,
        num_warps=8, num_stages=3,
    )
    return c


def _conv1x1_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # Fallback when gradients are needed or unsupported device
    if (not x.is_cuda) or torch.is_grad_enabled():
        return F.conv2d(x, weight, bias, stride=1, padding=0)

    # Shapes: x: (N, C_in, H, W), weight: (C_out, C_in, 1, 1), bias: (C_out,)
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    # Prepare matrices for GEMM: (M, K) @ (K, N) -> (M, N)
    x_mk = x.permute(0, 2, 3, 1).contiguous().view(-1, C_in)  # (M, K)
    w_kn = weight.view(C_out, C_in).t().contiguous()          # (K, N)
    b_n = (bias if bias is not None else torch.zeros(C_out, device=x.device, dtype=weight.dtype)).contiguous()

    # Triton matmul + bias
    out_mn = _matmul_bias(x_mk, w_kn, b_n)  # (M, N), fp32

    # Reshape back to NCHW, cast to original dtype
    y = out_mn.view(N, H, W, C_out).permute(0, 3, 1, 2)
    if y.dtype != x.dtype:
        y = y.to(x.dtype)
    return y


def _fused_three_1x1_triton(
    x: torch.Tensor,
    w1: torch.Tensor, b1: torch.Tensor,
    w2: torch.Tensor, b2: torch.Tensor,
    w3: torch.Tensor, b3: torch.Tensor,
):
    # Fallback when gradients are needed or unsupported device
    if (not x.is_cuda) or torch.is_grad_enabled():
        y1 = F.conv2d(x, w1, b1, stride=1, padding=0)
        y2 = F.conv2d(x, w2, b2, stride=1, padding=0)
        y3 = F.conv2d(x, w3, b3, stride=1, padding=0)
        return y1, y2, y3

    N, C_in, H, W = x.shape
    # Prepare input matrix
    x_mk = x.permute(0, 2, 3, 1).contiguous().view(-1, C_in)  # (M, K)
    # Prepare concatenated weights (K, N_total) and bias (N_total,)
    def prep(w, b):
        Cout = w.shape[0]
        w_kn = w.view(Cout, C_in).t().contiguous()
        b_n = (b if b is not None else torch.zeros(Cout, device=x.device, dtype=w.dtype)).contiguous()
        return w_kn, b_n, Cout

    w1_kn, b1_n, C1 = prep(w1, b1)
    w2_kn, b2_n, C2 = prep(w2, b2)
    w3_kn, b3_n, C3 = prep(w3, b3)

    w_cat = torch.cat([w1_kn, w2_kn, w3_kn], dim=1).contiguous()  # (K, C1+C2+C3)
    b_cat = torch.cat([b1_n.to(w_cat.dtype), b2_n.to(w_cat.dtype), b3_n.to(w_cat.dtype)], dim=0).contiguous()

    out_cat = _matmul_bias(x_mk, w_cat, b_cat)  # (M, C1+C2+C3), fp32

    y1_mn, y2_mn, y3_mn = torch.split(out_cat, [C1, C2, C3], dim=1)
    y1 = y1_mn.view(N, H, W, C1).permute(0, 3, 1, 2).to(x.dtype)
    y2 = y2_mn.view(N, H, W, C2).permute(0, 3, 1, 2).to(x.dtype)
    y3 = y3_mn.view(N, H, W, C3).permute(0, 3, 1, 2).to(x.dtype)
    return y1, y2, y3


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(ModelNew, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        # Autograd or non-CUDA: preserve exact PyTorch path for numerical & gradient parity
        if (not x.is_cuda) or torch.is_grad_enabled():
            branch1x1 = self.branch1x1(x)
            branch3x3 = self.branch3x3(x)
            branch5x5 = self.branch5x5(x)
            branch_pool = self.branch_pool(x)
            outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
            return torch.cat(outputs, 1)

        # Fused Triton path for the three 1x1 convolutions on the same input x
        b1, red3, red5 = _fused_three_1x1_triton(
            x,
            self.branch1x1.weight, self.branch1x1.bias,
            self.branch3x3[0].weight, self.branch3x3[0].bias,
            self.branch5x5[0].weight, self.branch5x5[0].bias,
        )
        # 3x3 and 5x5 convolutions using PyTorch (keeps semantics)
        branch3x3 = self.branch3x3[1](red3)
        branch5x5 = self.branch5x5[1](red5)

        # Pool branch: maxpool then fast 1x1 projection via Triton
        pooled = self.branch_pool[0](x)
        branch_pool = _conv1x1_triton(pooled, self.branch_pool[1].weight, self.branch_pool[1].bias)

        outputs = [b1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

# Test code
in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]