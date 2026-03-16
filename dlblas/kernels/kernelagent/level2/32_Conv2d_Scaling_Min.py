import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


@triton.jit
def _scale_min_channel_kernel(
    x_ptr,  # pointer to input tensor [B, C, H, W]
    y_ptr,  # pointer to output tensor [B, 1, H, W]
    scale,  # python scalar
    B, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Map one program instance to one (n, h, w) location
    pid = tl.program_id(axis=0)
    n_hw = H * W
    n = pid // n_hw
    hw = pid % n_hw
    h = hw // W
    w = hw % W

    # Base offset for this (n, h, w)
    x_base = n * stride_xn + h * stride_xh + w * stride_xw

    # Determine reduction type based on the sign of scale
    is_pos = scale >= 0.0

    # Initialize accumulator to +inf for min or -inf for max in the same dtype as x
    x0 = tl.load(x_ptr + x_base)
    acc_min_init = x0 * 0 + float("inf")
    acc_max_init = x0 * 0 - float("inf")
    acc = tl.where(is_pos, acc_min_init, acc_max_init)

    offs_c = tl.arange(0, BLOCK_C)
    # Loop over channels in chunks of BLOCK_C and accumulate min/max
    for c0 in range(0, C, BLOCK_C):
        c_idx = c0 + offs_c
        mask_c = c_idx < C
        x_ptrs = x_ptr + x_base + c_idx * stride_xc
        x_vals = tl.load(x_ptrs, mask=mask_c, other=0.0, cache_modifier=".cg")
        # Ensure masked lanes don't affect reduction
        neutral = tl.where(is_pos, float("inf"), -float("inf"))
        x_vals = tl.where(mask_c, x_vals, neutral)
        block_red = tl.min(x_vals, axis=0) if is_pos else tl.max(x_vals, axis=0)
        acc = tl.minimum(acc, block_red) if is_pos else tl.maximum(acc, block_red)

    # Apply scale after reduction (fewer multiplies)
    acc = acc * scale

    # Store result to y at channel 0
    y_offs = n * stride_yn + h * stride_yh + w * stride_yw  # channel dim = 0
    tl.store(y_ptr + y_offs, acc)


class ModelNew(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, out_height, out_width).
        """
        x = self.conv(x)

        # Use Triton kernel to fuse scaling and channel-wise min on CUDA when autograd is not required.
        use_triton = (
            HAS_TRITON
            and x.is_cuda
            and not x.requires_grad
            and x.dim() == 4
            and x.shape[1] > 0
        )
        if use_triton:
            B, C, H, W = x.shape
            y = torch.empty((B, 1, H, W), device=x.device, dtype=x.dtype)

            # Choose BLOCK_C with at least warp width for better vectorization
            def _next_pow2(v: int) -> int:
                return 1 if v <= 1 else 1 << (int(math.ceil(math.log2(v))))
            BLOCK_C = min(128, max(32, _next_pow2(C)))

            grid = (B * H * W,)
            _scale_min_channel_kernel[grid](
                x, y, self.scale_factor,
                B, H, W,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                C=C, BLOCK_C=BLOCK_C,
                num_warps=8, num_stages=2,
            )
            return y
        else:
            # Fallback to PyTorch when on CPU or when autograd is needed
            x = x * self.scale_factor
            x = torch.min(x, dim=1, keepdim=True)[0]
            return x


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]