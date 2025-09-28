# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_gemm_bn_scale_softmax(
    x_ptr, w_ptr, b_ptr, gamma_ptr, beta_ptr, scale_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_bn,
    stride_gamman, stride_betan,
    stride_outputm, stride_outputn,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    
    # Load bias and add
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]
    
    # Online mean/variance computation
    mean = tl.sum(acc, axis=1) / N
    mean_bc = mean[:, None]
    var = tl.sum((acc - mean_bc) * (acc - mean_bc), axis=1) / N
    var_bc = var[:, None]
    
    # Batch normalization
    gamma = tl.load(gamma_ptr + offs_n, mask=offs_n < N, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=offs_n < N, other=0.0)
    normalized = (acc - mean_bc) / tl.sqrt(var_bc + eps)
    scaled = normalized * gamma[None, :] + beta[None, :]
    
    # Apply scale parameter
    scale_val = tl.load(scale_ptr)
    scaled = scaled * scale_val
    
    # Softmax
    max_val = tl.max(scaled, axis=1)
    exp_vals = tl.exp(scaled - max_val[:, None])
    sum_exp = tl.sum(exp_vals, axis=1)
    softmax_out = exp_vals / sum_exp[:, None]
    
    # Store results
    output_ptrs = output_ptr + offs_m[:, None] * stride_outputm + offs_n[None, :] * stride_outputn
    tl.store(output_ptrs, softmax_out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn_eps = bn_eps
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.scale = nn.Parameter(torch.ones(scale_shape))
        
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        M, K = x.shape
        N = self.out_features
        
        output = torch.empty(M, N, device=x.device, dtype=torch.float32)
        
        grid = lambda opt: (triton.cdiv(M, opt.BLOCK_M) * triton.cdiv(N, opt.BLOCK_N),)
        
        fused_gemm_bn_scale_softmax[grid](
            x, self.weight, self.bias, self.gamma, self.beta, self.scale, output,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            self.bias.stride(0),
            self.gamma.stride(0), self.beta.stride(0),
            output.stride(0), output.stride(1),
            self.bn_eps,
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=32
        )
        
        return output

batch_size = 128
in_features = 1024
out_features = 512
bn_eps = 1e-5
bn_momentum = 0.1
scale_shape = (1,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, scale_shape]
# =================== EVOLVE-BLOCK-END ===================