import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_POS': 32}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_POS': 64}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_POS': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_POS': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_POS': 512}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_POS': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_POS': 2048}, num_warps=8, num_stages=5),
    ],
    key=['C', 'DHW'],
)
@triton.jit
def _clamp_softmax_mul2_tiled_ncdhw(
    x_ptr, y_ptr,
    N, C, DHW,
    stride_n, stride_c,
    clamp_min, clamp_max, scale,
    OUT_DTYPE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_POS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_tile = tl.program_id(1)

    # tile over positions within DHW
    pos_start = pid_tile * BLOCK_POS
    offs_p = pos_start + tl.arange(0, BLOCK_POS)
    mask_p = offs_p < DHW

    # channel lanes
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base_n = pid_n * stride_n

    # 2D tile pointer: [C, POS]
    ptrs = x_ptr + base_n + offs_c[:, None] * stride_c + offs_p[None, :]
    mask = mask_c[:, None] & mask_p[None, :]

    # Load and compute in fp32
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

    # clamp first
    x = tl.minimum(tl.maximum(x, clamp_min), clamp_max)

    # set masked lanes to -inf so they don't affect reductions across C
    neg_inf = -float('inf')
    x = tl.where(mask, x, neg_inf)

    # stable softmax over C (axis=0)
    x_max = tl.max(x, axis=0)
    x = tl.exp(x - x_max[None, :])
    denom = tl.sum(x, axis=0)
    inv = scale / denom[None, :]
    out = x * inv

    # write back
    tl.store(y_ptr + base_n + offs_c[:, None] * stride_c + offs_p[None, :], out.to(OUT_DTYPE), mask=mask)


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, average pooling, clamping, softmax, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    @staticmethod
    def _fused_clamp_softmax_mul2_tiled(x: torch.Tensor, clamp_min: float, clamp_max: float, scale: float = 2.0):
        # Fused: clamp -> softmax(dim=1) -> *scale for NCDHW, tiled over spatial positions for coalesced access
        if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            y = torch.clamp(x, clamp_min, clamp_max)
            return torch.softmax(y, dim=1) * scale

        x = x.contiguous()
        N, C, D, H, W = x.shape
        DHW = D * H * W
        y = torch.empty_like(x)

        sN, sC, sD, sH, sW = x.stride()  # only sN and sC are used since DHW is contiguous

        # choose BLOCK_C as next power-of-two >= C but at least 32 to map well to a warp
        BLOCK_C = max(32, _next_power_of_2(C))

        OUT_DTYPE = tl.float32
        if x.dtype == torch.float16:
            OUT_DTYPE = tl.float16
        elif x.dtype == torch.bfloat16:
            OUT_DTYPE = tl.bfloat16

        grid = lambda META: (N, triton.cdiv(DHW, META['BLOCK_POS']))
        _clamp_softmax_mul2_tiled_ncdhw[grid](
            x, y,
            N, C, DHW,
            sN, sC,
            float(clamp_min), float(clamp_max), float(scale),
            OUT_DTYPE=OUT_DTYPE,
            BLOCK_C=BLOCK_C,
        )
        return y

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        x = self.conv_transpose(x)
        x = self.avg_pool(x)
        # Fused: clamp -> softmax(dim=1) -> *2 using tiled Triton kernel
        x = self._fused_clamp_softmax_mul2_tiled(x, self.clamp_min, self.clamp_max, 2.0)
        return x


batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]