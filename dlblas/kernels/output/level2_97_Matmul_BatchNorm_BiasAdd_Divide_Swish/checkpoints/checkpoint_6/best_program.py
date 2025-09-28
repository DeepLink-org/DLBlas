# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _forward_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_bn,
    stride_om, stride_on,
    divide_value: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    input_ptrs = input_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    weight_ptrs = weight_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining
        a = tl.load(input_ptrs, mask=k_mask[None, :] & (offs_m[:, None] < M), other=0.0)
        b = tl.load(weight_ptrs, mask=k_mask[None, :] & (offs_n[:, None] < N), other=0.0)
        acc += tl.dot(a, tl.trans(b))
        input_ptrs += BLOCK_K * stride_ik
        weight_ptrs += BLOCK_K * stride_wk

    bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias_vals[None, :]
    acc = acc / divide_value
    sig = tl.sigmoid(acc)
    acc = acc * sig

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :]
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, acc, mask=output_mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = divide_value
        self.fused_weight = None
        self.fused_bias = None

    def _precompute_fused_params(self):
        weight = self.matmul.weight.data
        linear_bias = self.matmul.bias.data if self.matmul.bias is not None else 0
        bn_weight = self.bn.weight.data
        bn_bias = self.bn.bias.data
        running_mean = self.bn.running_mean
        running_var = self.bn.running_var
        eps = self.bn.eps

        denom = 1.0 / torch.sqrt(running_var + eps)
        fused_w = weight * (bn_weight * denom).view(-1, 1)
        fused_b = (linear_bias - running_mean) * (bn_weight * denom) + bn_bias + self.bias
        return fused_w, fused_b

    def forward(self, x):
        if self.training:
            x = self.matmul(x)
            x = self.bn(x)
            x = x + self.bias
            x = x / self.divide_value
            return x * torch.sigmoid(x)
        else:
            if self.fused_weight is None or self.fused_bias is None:
                self.fused_weight, self.fused_bias = self._precompute_fused_params()
            
            M, K = x.shape
            N = self.fused_weight.shape[0]
            output = torch.empty((M, N), device=x.device, dtype=x.dtype)
            
            grid = lambda opt: (triton.cdiv(M, opt.BLOCK_M) * triton.cdiv(N, opt.BLOCK_N),)
            _forward_kernel[grid](
                x, self.fused_weight, self.fused_bias, output,
                M, N, K,
                x.stride(0), x.stride(1),
                self.fused_weight.stride(0), self.fused_weight.stride(1),
                self.fused_bias.stride(0),
                output.stride(0), output.stride(1),
                self.divide_value,
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
            )
            return output

batch_size = 128
in_features = 1024
out_features = 512
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]
# =================== EVOLVE-BLOCK-END ===================