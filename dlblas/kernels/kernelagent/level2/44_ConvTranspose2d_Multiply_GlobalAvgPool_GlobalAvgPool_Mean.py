import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _global_avg_mul_kernel(
    x_ptr,          # *[NC*HW] flattened input
    out_ptr,        # *[NC] flattened output
    NC,             # number of (n,c) planes
    HW,             # spatial size H*W
    multiplier,     # scalar multiplier
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Guard against OOB
    if pid >= NC:
        return

    base = x_ptr + pid * HW
    idx = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros((), dtype=tl.float32)

    off = tl.zeros((), dtype=tl.int32)
    # Fast path: no masking needed when HW is divisible by BLOCK_SIZE
    if (HW % BLOCK_SIZE) == 0:
        while off < HW:
            offsets = off + idx
            vals = tl.load(base + offsets)
            acc += tl.sum(vals.to(tl.float32), axis=0)
            off += BLOCK_SIZE
    else:
        while off < HW:
            offsets = off + idx
            mask = offsets < HW
            vals = tl.load(base + offsets, mask=mask, other=0.0)
            acc += tl.sum(vals.to(tl.float32), axis=0)
            off += BLOCK_SIZE

    scale = (multiplier) / HW.to(tl.float32)
    mean_val = acc * scale
    tl.store(out_ptr + pid, mean_val)


def _fused_mul_global_avg(x: torch.Tensor, multiplier: float) -> torch.Tensor:
    # Ensure contiguous layout [N, C, H, W] -> flattened [NC, HW]
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    # Heuristic: choose BLOCK_SIZE as a power-of-two up to 2048 for good occupancy
    def next_pow2(v: int) -> int:
        return 1 << (v - 1).bit_length()
    bs = min(next_pow2(HW), 2048)

    # Tune warps/stages to better utilize Hopper SMs
    num_warps = 4 if bs <= 1024 else 8
    num_stages = 4

    grid = (NC,)
    _global_avg_mul_kernel[grid](
        x.view(-1),
        out.view(-1),
        NC,
        HW,
        float(multiplier),
        BLOCK_SIZE=bs,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, multiplies by a scalar, applies global average pooling, 
    another global average pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fused multiply and global average pooling over H and W using Triton
        x = _fused_mul_global_avg(x, self.multiplier)
        # Second mean over 1x1 is a no-op; keeping output identical without extra kernel launch
        return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier]