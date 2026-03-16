import torch
import torch.nn as nn
import triton
import triton.language as tl

# Try to import libdevice for math functions like erf
try:
    from triton.language.extra import libdevice
except Exception:
    from triton.language import libdevice


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _fused_scale_lrelu_gelu(
    x_ptr,          # *float32, input tensor (NCHW) contiguous
    m_ptr,          # *float32, multiplier tensor flattened with shape (C,)
    y_ptr,          # *float32, output tensor (same shape as x)
    n_elements,     # int32, total elements B*C*H*W
    C,              # int32, number of channels
    HW,             # int32, product H*W
    negative_slope: tl.constexpr,  # float constant
    BLOCK_SIZE: tl.constexpr,      # tile size
):
    pid = tl.program_id(axis=0)
    arange = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + arange
    mask = offsets < n_elements
    tl.multiple_of(offsets, 16)

    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")

    # Compute channel index for each element: ((idx // (H*W)) % C)
    plane_idx = offsets // HW
    c_idx = plane_idx % C
    scale = tl.load(m_ptr + c_idx, mask=mask, other=1.0)

    # Compute in fp32 for numerical stability and correctness
    x32 = x.to(tl.float32)
    s32 = scale.to(tl.float32)
    v = x32 * s32

    # Branchless LeakyReLU: v = v + (neg - 1) * min(v, 0)
    v = v + (negative_slope - 1.0) * tl.minimum(v, 0.0)

    # GELU (exact): 0.5 * v * (1 + erf(v / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    e = libdevice.erf(v * inv_sqrt2)
    y32 = 0.5 * v * (1.0 + e)

    y = y32.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, multiplies by a learnable scalar, applies LeakyReLU, and then GELU.
    Fused Triton kernel is used to apply: y = GELU(LeakyReLU(conv(x) * multiplier))
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        # Fast path: fused Triton kernel on CUDA
        if x.is_cuda:
            x = x.contiguous()
            B, C, H, W = x.shape
            out = torch.empty_like(x)
            # multiplier has shape (C,1,1); flatten to (C,)
            m = self.multiplier.contiguous().view(-1).to(device=x.device, dtype=x.dtype)
            n_elements = x.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            _fused_scale_lrelu_gelu[grid](
                x, m, out,
                n_elements,
                C, H * W,
                self.leaky_relu.negative_slope,
            )
            return out
        # Fallback (CPU / non-CUDA)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        x = torch.nn.functional.gelu(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape]