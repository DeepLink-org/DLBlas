# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _gemm_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptr += offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptr += offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_mask_x = offs_k[None, :] < k_remaining
        k_mask_w = offs_k[:, None] < k_remaining
        x_val = tl.load(x_ptr, mask=k_mask_x, other=0.0)
        w_val = tl.load(w_ptr, mask=k_mask_w, other=0.0)
        acc += tl.dot(x_val, w_val, allow_tf32=True)

        x_ptr += BLOCK_K * stride_xk
        w_ptr += BLOCK_K * stride_wk

    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_n
        bias_val = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
        acc += bias_val[None, :]

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, acc, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_dim = max_dim
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        M, K = x.shape
        N = self.out_features
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
        _gemm_kernel[grid](
            x, self.weight, self.bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        x = out
        x = torch.max(x, dim=self.max_dim, keepdim=True).values
        x = x - x.mean(dim=1, keepdim=True)
        x = torch.nn.functional.gelu(x)
        return x

batch_size = 128
in_features = 512
out_features = 1024
max_dim = 1

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, max_dim]
# =================== EVOLVE-BLOCK-END ===================