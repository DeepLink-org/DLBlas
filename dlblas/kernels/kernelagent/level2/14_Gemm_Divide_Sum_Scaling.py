import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rowwise_dot_kernel(
    x_ptr,           # *f32/*f16, shape [M, K]
    s_ptr,           # *f32/*f16, shape [K]
    out_ptr,         # *f32/*f16, shape [M, 1] (we write column 0)
    M: tl.constexpr, # int
    K: tl.constexpr, # int
    stride_xm,       # int: stride for dim-0 of x in elements
    stride_xk,       # int: stride for dim-1 of x in elements
    stride_outm,     # int: stride for dim-0 of out in elements
    scale,           # f32 scalar (apply once at the end)
    BLOCK_M: tl.constexpr,  # tile size along M
    BLOCK_K: tl.constexpr,  # tile size along K
):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Accumulator for each row in the block
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    k0 = 0
    while k0 < K:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        x = tl.load(
            x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0, cache_modifier=".cg"
        ).to(tl.float32)

        # Load S tile [BLOCK_K]
        s = tl.load(s_ptr + offs_k, mask=mask_k, other=0.0, cache_modifier=".cg").to(tl.float32)

        # Accumulate row-wise dot products
        acc += tl.sum(x * s[None, :], axis=1)
        k0 += BLOCK_K

    # Apply the final scale once
    acc = acc * scale

    # Store result directly
    tl.store(out_ptr + offs_m * stride_outm, acc, mask=mask_m)


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    Equivalent fused form:
      y = scaling_factor * sum((x @ W^T) / 2, dim=1, keepdim=True)
        = (scaling_factor / 2) * (x @ sum(W, dim=0))
      Output shape: (batch_size, 1)
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Fallback to original computation when not on CUDA or when autograd is needed
        if (not x.is_cuda) or x.requires_grad or self.weight.requires_grad:
            y = torch.matmul(x, self.weight.T)          # Gemm
            y = y / 2                                    # Divide
            y = torch.sum(y, dim=1, keepdim=True)        # Sum over hidden dim
            y = y * self.scaling_factor                  # Scaling
            return y

        # Triton-optimized path (CUDA, inference)
        x = x.contiguous()
        M, K = x.shape

        # Compute s = sum(weight, dim=0) and fuse host-side scaling to reduce per-tile work
        s_eff = (self.weight.sum(dim=0) * (float(self.scaling_factor) * 0.5)).contiguous()

        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

        # Fixed tiling reduces Python overhead and performs well for small K on H200
        BLOCK_M = 128
        BLOCK_K = 128
        grid = ((M + BLOCK_M - 1) // BLOCK_M,)

        # scale is already fused into s_eff; pass 1.0 here
        _rowwise_dot_kernel[grid](
            x, s_eff, out,
            M, K,
            x.stride(0), x.stride(1),
            out.stride(0),
            1.0,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )
        return out


batch_size = 128
input_size = 10
hidden_size = 20
scaling_factor = 1.5

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]