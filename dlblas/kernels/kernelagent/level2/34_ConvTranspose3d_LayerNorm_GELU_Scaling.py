import torch
import torch.nn as nn
import triton
import triton.language as tl

# Help cuDNN pick the fastest algo for fixed shapes
torch.backends.cudnn.benchmark = True


@triton.autotune(
    configs=[
        # Add lighter/wider mixes to improve occupancy on small N
        triton.Config({'BLOCK_SIZE_N': 64, 'ROWS_PER_CTA': 8}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'ROWS_PER_CTA': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'ROWS_PER_CTA': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 128, 'ROWS_PER_CTA': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 128, 'ROWS_PER_CTA': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'ROWS_PER_CTA': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_N': 256, 'ROWS_PER_CTA': 16}, num_warps=8, num_stages=3),
    ],
    key=['n_cols'],
)
@triton.jit
def _layernorm_gelu_scale_kernel(
    x_ptr,           # *[n_rows, n_cols]
    y_ptr,           # *[n_rows, n_cols] (stores to dtype(y_ptr))
    w_ptr,           # *[n_cols]
    b_ptr,           # *[n_cols]
    n_rows,          # total number of rows = prod(shape[:-1])
    n_cols,          # size of last dim
    inv_n_cols,      # 1.0 / n_cols
    eps,             # eps for layernorm
    scale,           # scaling factor after GELU
    ROWS_PER_CTA: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    col_mask = cols < n_cols

    # Load affine params once per-CTA into registers (fp32 for stability)
    gamma = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    beta = tl.load(b_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    inv_sqrt2 = 0.70710678118654752440084436210485  # 1/sqrt(2)

    # Persistent-CTA double-buffered prefetch over rows
    row0 = pid * ROWS_PER_CTA
    offs0 = row0 * n_cols + cols
    mask0 = (row0 < n_rows) & col_mask
    x_buf = tl.load(x_ptr + offs0, mask=mask0, other=0.0).to(tl.float32)

    for r in tl.static_range(ROWS_PER_CTA):
        row = row0 + r
        row_valid = row < n_rows

        # Use prefetched row
        x = x_buf

        # Prefetch next row to overlap memory latency with compute
        if r + 1 < ROWS_PER_CTA:
            next_row = row + 1
            offs_n = next_row * n_cols + cols
            mask_n = (next_row < n_rows) & col_mask
            x_buf = tl.load(x_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

        # Compute statistics
        mu = tl.sum(x, axis=0) * inv_n_cols
        xc = x - mu
        var = tl.sum(xc * xc, axis=0) * inv_n_cols
        rstd = tl.math.rsqrt(var + eps)

        # Normalize + affine
        y = xc * rstd
        y = y * gamma + beta
        # Exact GELU
        y = 0.5 * y * (1.0 + tl.math.erf(y * inv_sqrt2))
        # Scale
        y = y * scale

        offs_store = row * n_cols + cols
        mask_store = row_valid & col_mask
        # Store in the dtype of y_ptr (implicit cast from fp32)
        tl.store(y_ptr + offs_store, y, mask=mask_store)


def layernorm_gelu_scale_triton(x: torch.Tensor,
                                weight: torch.Tensor,
                                bias: torch.Tensor,
                                eps: float,
                                scale: float) -> torch.Tensor:
    # Fused LayerNorm (over last dim) + GELU (exact) + scaling.
    # Falls back to PyTorch for unsupported cases.
    if (
        x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight is not None
        and bias is not None
        and x.shape[-1] == weight.numel() == bias.numel()
    ):
        xc = x.contiguous()
        n_cols = xc.shape[-1]
        n_rows = xc.numel() // n_cols

        if n_rows == 0 or n_cols == 0:
            y = torch.nn.functional.layer_norm(xc, (n_cols,), weight, bias, eps=eps)
            y = torch.nn.functional.gelu(y)
            y = y * scale
            return y

        # Allocate output in input dtype to reduce bandwidth
        y_out = torch.empty_like(xc, dtype=xc.dtype)

        # Ensure affine params are fp32 and contiguous on the same device
        w = weight.contiguous().to(dtype=torch.float32, device=xc.device)
        b = bias.contiguous().to(dtype=torch.float32, device=xc.device)

        grid = lambda meta: (triton.cdiv(n_rows, meta['ROWS_PER_CTA']),)
        _layernorm_gelu_scale_kernel[grid](
            xc, y_out, w, b,
            n_rows, n_cols, 1.0 / float(n_cols), float(eps), float(scale),
        )
        return y_out
    else:
        # Reference path: exact PyTorch semantics
        y = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps=eps)
        y = torch.nn.functional.gelu(y)
        y = y * scale
        return y


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, layer normalization, GELU activation, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv_transpose(x)
        # Fused LayerNorm (over last dim) + GELU + scaling
        x = layernorm_gelu_scale_triton(x, self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps, self.scaling_factor)
        return x


batch_size = 128
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]