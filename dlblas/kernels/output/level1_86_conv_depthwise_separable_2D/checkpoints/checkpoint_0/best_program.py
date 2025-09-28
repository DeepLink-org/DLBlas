# ================== EVOLVE-BLOCK-START ==================
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, bias_ptr, output_ptr,
    stride, padding, dilation, kernel_size,
    in_channels, height, width, height_out, width_out,
    x_bs, x_cs, x_hs, x_ws,
    w_cs, w_ks, w_hs, w_ws,
    out_bs, out_cs, out_hs, out_ws,
    BIAS_EXISTS: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid = tl.program_id(0)
    n_elements = in_channels * height_out * width_out
    batch_idx = pid // n_elements
    pid = pid % n_elements
    
    ch_idx = pid // (height_out * width_out)
    hw_idx = pid % (height_out * width_out)
    h_idx = hw_idx // width_out
    w_idx = hw_idx % width_out
    
    # Vectorized computation for kernel elements
    kh = tl.arange(0, kernel_size)
    kw = tl.arange(0, kernel_size)
    
    h_in = h_idx * stride + kh[:, None] * dilation - padding
    w_in = w_idx * stride + kw[None, :] * dilation - padding
    
    # Create masks for valid positions
    h_mask = (h_in >= 0) & (h_in < height)
    w_mask = (w_in >= 0) & (w_in < width)
    valid_mask = h_mask & w_mask
    
    # Compute memory offsets
    x_offset = batch_idx * x_bs + ch_idx * x_cs
    w_offset = ch_idx * w_cs
    
    # Vectorized loads and computation
    acc = 0.0
    for ki in range(0, kernel_size, BLOCK_H):
        for kj in range(0, kernel_size, BLOCK_W):
            ki_mask = (ki + tl.arange(0, BLOCK_H)) < kernel_size
            kj_mask = (kj + tl.arange(0, BLOCK_W)) < kernel_size
            block_mask = ki_mask[:, None] & kj_mask[None, :]
            
            h_block = h_in[ki:ki+BLOCK_H, kj:kj+BLOCK_W]
            w_block = w_in[ki:ki+BLOCK_H, kj:kj+BLOCK_W]
            valid_block = valid_mask[ki:ki+BLOCK_H, kj:kj+BLOCK_W]
            
            x_vals = tl.load(
                x_ptr + x_offset + h_block * x_hs + w_block * x_ws,
                mask=valid_block & block_mask,
                other=0.0
            )
            w_vals = tl.load(
                w_ptr + w_offset + (ki + tl.arange(0, BLOCK_H))[:, None] * w_hs + 
                         (kj + tl.arange(0, BLOCK_W))[None, :] * w_ws,
                mask=block_mask,
                other=0.0
            )
            acc += tl.sum(x_vals * w_vals)
    
    if BIAS_EXISTS:
        bias_val = tl.load(bias_ptr + ch_idx)
        acc += bias_val
        
    out_offset = batch_idx * out_bs + ch_idx * out_cs + h_idx * out_hs + w_idx * out_ws
    tl.store(output_ptr + out_offset, acc)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    USE_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = offs_k < (K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=mask_k[None, :] & (offs_m[:, None] < M), other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if USE_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride = self.depthwise.stride[0]
        padding = self.depthwise.padding[0]
        dilation = self.depthwise.dilation[0]
        kernel_size = self.depthwise.kernel_size[0]
        
        B, C, H, W = x.shape
        H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        
        x = x.contiguous()
        depthwise_out = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)
        
        grid_depth = (B * C * H_out * W_out,)
        depthwise_conv2d_kernel[grid_depth](
            x, 
            self.depthwise.weight.contiguous(), 
            self.depthwise.bias.contiguous() if self.depthwise.bias is not None else None,
            depthwise_out,
            stride, padding, dilation, kernel_size,
            C, H, W, H_out, W_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.depthwise.weight.stride(0), self.depthwise.weight.stride(1), 
            self.depthwise.weight.stride(2), self.depthwise.weight.stride(3),
            depthwise_out.stride(0), depthwise_out.stride(1), 
            depthwise_out.stride(2), depthwise_out.stride(3),
            BIAS_EXISTS=self.depthwise.bias is not None,
            BLOCK_H=4, BLOCK_W=4
        )
        
        # Reshape for efficient matmul
        depthwise_out_reshaped = depthwise_out.permute(0, 2, 3, 1).reshape(-1, C)
        weight_t = self.pointwise.weight.view(self.pointwise.weight.shape[0], C).t().contiguous()
        
        M, K = depthwise_out_reshaped.shape
        N = weight_t.shape[1]
        pointwise_out_reshaped = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
        matmul_kernel[grid](
            depthwise_out_reshaped, 
            weight_t, 
            pointwise_out_reshaped,
            self.pointwise.bias if self.pointwise.bias is not None else torch.empty(0, device=x.device),
            M, N, K,
            depthwise_out_reshaped.stride(0), depthwise_out_reshaped.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            pointwise_out_reshaped.stride(0), pointwise_out_reshaped.stride(1),
            USE_BIAS=self.pointwise.bias is not None,
            BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32
        )
        
        pointwise_out = pointwise_out_reshaped.view(B, H_out, W_out, N).permute(0, 3, 1, 2)
        return pointwise_out

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]
# =================== EVOLVE-BLOCK-END ===================