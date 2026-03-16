import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rowwise_lse_leaky_gelu2(
    x_ptr,           # pointer to [B, N] input
    y_ptr,           # pointer to [B, 1] output
    stride_xm,       # stride between rows for x
    stride_xn,       # stride between cols for x
    stride_ym,       # stride between rows for y
    NEG_SLOPE: tl.constexpr,  # leaky ReLU negative slope
    N: tl.constexpr,          # number of columns (out_features)
    BLOCK_N: tl.constexpr,    # tile size along N
):
    pid = tl.program_id(0)  # row id

    # Precompute arange once and base row pointer for better ILP
    r = tl.arange(0, BLOCK_N)
    row_ptr = x_ptr + pid * stride_xm

    # Streaming LogSumExp in a single pass for numeric stability and fewer global loads
    m = tl.full((1,), -float("inf"), dtype=tl.float32)  # running max
    s = tl.zeros((1,), dtype=tl.float32)                # running sum of exp shifted by m
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + r
        mask = offs < N
        # Load masked lanes as -inf so we can drop further masking/where ops
        vals = tl.load(row_ptr + offs * stride_xn, mask=mask, other=-float("inf")).to(tl.float32)
        # tile max
        tile_max = tl.max(vals, axis=0)
        m_new = tl.maximum(m, tile_max)
        # accumulate sum in the new max domain
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(vals - m_new), axis=0)
        m = m_new

    # Final LogSumExp
    lse = m + tl.log(s)

    # Two LeakyReLU applications fused into one: for x<0 multiply by slope^2
    slope_sq = NEG_SLOPE * NEG_SLOPE
    x = tl.where(lse >= 0.0, lse, lse * slope_sq)

    # Two GELU (exact, erf-based) applications
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    cdf = 0.5 * (1.0 + libdevice.erf(x * inv_sqrt2))
    x = x * cdf
    cdf = 0.5 * (1.0 + libdevice.erf(x * inv_sqrt2))
    x = x * cdf

    # Store result
    y_offs = pid * stride_ym + tl.arange(0, 1)
    tl.store(y_ptr + y_offs, x)


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), followed by LogSumExp, LeakyReLU, 
    LeakyReLU, GELU, and GELU activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.neg_slope = 0.01

    def forward(self, x):
        # Gemm using cuBLAS via PyTorch
        x = self.linear(x)

        # Fused Row-wise LogSumExp + 2x LeakyReLU + 2x GELU using Triton
        if x.is_cuda:
            B, N = x.shape
            x_c = x.contiguous()
            y = torch.empty((B, 1), device=x.device, dtype=x.dtype)

            # Kernel launch: one program per row
            grid = (B,)
            # Choose a reasonable tile size along N (fits common shapes well)
            BLOCK_N = 128
            _rowwise_lse_leaky_gelu2[grid](
                x_c, y,
                x_c.stride(0), x_c.stride(1),
                y.stride(0),
                NEG_SLOPE=self.neg_slope,
                N=N,
                BLOCK_N=BLOCK_N,
                num_warps=4,
                num_stages=2,
            )
            return y
        else:
            # CPU fallback: preserve exact semantics
            x = torch.logsumexp(x, dim=1, keepdim=True)
            x = torch.nn.functional.leaky_relu(x, negative_slope=self.neg_slope)
            x = torch.nn.functional.leaky_relu(x, negative_slope=self.neg_slope)
            x = torch.nn.functional.gelu(x)
            x = torch.nn.functional.gelu(x)
            return x


batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]