import torch
import torch.nn as nn

# Triton imports with safe fallback
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _lse_hswish_bias_clamp_kernel(
    x_ptr,             # *const float
    out_ptr,           # *float
    min_bias_ptr,      # *const float (1 element)
    M,                 # int32: total number of elements per (N*D*H*W)
    STRIDE_C,          # int32: stride between channels (D*H*W)
    C: tl.constexpr,   # number of channels to reduce over (compile-time constant)
    BLOCK: tl.constexpr,  # block size along M
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    # Load the minimum bias across channels
    min_bias = tl.load(min_bias_ptr).to(tl.float32)

    # Use a numerically-stable two-pass logsumexp across channels to cut exp calls in half
    neg_inf = -float("inf")
    base_ptr = x_ptr + offs

    # Pass 1: compute maximum over channels
    m = tl.full((BLOCK,), neg_inf, dtype=tl.float32)
    for c in tl.static_range(0, C):
        v = tl.load(base_ptr + c * STRIDE_C, mask=mask, other=neg_inf).to(tl.float32)
        m = tl.maximum(m, v)
    # Avoid NaNs for masked lanes in the next pass
    m = tl.where(mask, m, 0.0)

    # Pass 2: compute sum(exp(v - m))
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in tl.static_range(0, C):
        v = tl.load(base_ptr + c * STRIDE_C, mask=mask, other=neg_inf).to(tl.float32)
        s += tl.exp(v - m)

    # Combine to get logsumexp; guard masked lanes
    lse = tl.where(mask, m + tl.log(s), 0.0)

    # HardSwish: x * sigmoid(x + 3) / 6
    t = lse + 3.0
    sig = 1.0 / (1.0 + tl.exp(-t))
    h = lse * sig * (1.0 / 6.0)

    # Subtract min_bias and clamp to [-1, 1]
    z = h - min_bias
    z = tl.maximum(tl.minimum(z, 1.0), -1.0)

    # Store result
    tl.store(out_ptr + offs, z, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, LogSumExp, HardSwish, subtraction, clamp, and maximum operations.
    Fused with a Triton kernel:
      After conv_transpose:
        - compute logsumexp over channels,
        - apply HardSwish,
        - subtract min(bias) and clamp,
        - result equals max over channels of clamp(h - bias_c) due to clamp monotonicity.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # ConvTranspose3d as in the original implementation
        y = self.conv_transpose(x)

        # Fallback to PyTorch if no CUDA/Triton is available
        if (not TRITON_AVAILABLE) or (not y.is_cuda):
            x1 = torch.logsumexp(y, dim=1, keepdim=True)
            x1 = x1 * torch.sigmoid(x1 + 3) / 6
            x1 = x1 - self.bias
            x1 = torch.clamp(x1, min=-1, max=1)
            x1 = torch.max(x1, dim=1, keepdim=True)[0]
            return x1

        # Triton fused path:
        # Using equivalence: max_c clamp(h - bias_c, -1, 1) = clamp(h - min_c(bias_c), -1, 1)
        # where h = HardSwish(logsumexp(y, dim=1)).
        B, C, D, H, W = y.shape
        M = B * D * H * W
        # Ensure contiguous for predictable strides
        y = y.contiguous()
        # Stride along channel dimension in elements
        stride_c = y.stride(1)

        # Prepare output tensor [B, 1, D, H, W]
        out = torch.empty((B, 1, D, H, W), device=y.device, dtype=y.dtype).contiguous()

        # Compute min bias across channels (bias shape: [C,1,1,1])
        min_bias = self.bias.min()
        # Ensure dtype/device match and pointer-compatible
        min_bias_t = min_bias.to(device=y.device, dtype=y.dtype)

        # Launch Triton kernel
        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK"]),)

        _lse_hswish_bias_clamp_kernel[grid](
            y,                         # x_ptr
            out.view(-1),              # out_ptr flattened
            min_bias_t,                # min_bias_ptr
            M,                         # total elements over (N*D*H*W)
            stride_c,                  # stride between channels
            C=C,                       # number of channels (constexpr)
            BLOCK=1024,                # tuned block size
            num_warps=8,               # more warps for H200 throughput
            num_stages=3,
        )

        return out

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]