import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fused_min_sum_gelu_add_bias(
    x_ptr,            # *f32, [N, C, H, W]
    bias_ptr,         # *f32, [C, 1, 1]
    y_ptr,            # *f32, [N, C, 1, W]
    N: tl.constexpr,  # int
    C: tl.constexpr,  # int
    H: tl.constexpr,  # int
    W: tl.constexpr,  # int
    sxn, sxc, sxh, sxw,    # strides for x
    sbc,                   # stride for bias along C
    syn, syc, syh, syw,    # strides for y
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_w_blk = tl.program_id(1)
    pid_cblk = tl.program_id(2)

    # Offsets and masks along width for this program
    w_offsets = pid_w_blk * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Channel tile handled by this program id
    c_offsets = pid_cblk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < C

    # Base offsets for input/output batch n
    x_base_n = pid_n * sxn
    out_base_n = pid_n * syn

    # Precompute w pointer increments to save MADs in inner loops
    w_ptrs = w_offsets * sxw
    tl.multiple_of(w_ptrs, values=1)

    # Accumulator over H of per-(min over C)
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    INF = 3.4028235e38  # fp32 max

    # Unroll height by 4 for better ILP
    h = 0
    while (h + 3) < H:
        # Initialize per-height minima
        min0 = tl.full((BLOCK_W,), INF, dtype=tl.float32)
        min1 = tl.full((BLOCK_W,), INF, dtype=tl.float32)
        min2 = tl.full((BLOCK_W,), INF, dtype=tl.float32)
        min3 = tl.full((BLOCK_W,), INF, dtype=tl.float32)

        c_start = 0
        while c_start < C:
            c_tile = c_start + tl.arange(0, BLOCK_C)
            mask_ct = c_tile < C
            # Base [C_tile, W_tile] pointer (independent of h)
            base_cw = (
                x_ptr
                + x_base_n
                + c_tile[:, None] * sxc
                + w_ptrs[None, :]
            )
            cmask_w = mask_ct[:, None] & mask_w[None, :]

            # Load 4 heights from the same C/W tile
            v0 = tl.load(base_cw + (h + 0) * sxh, mask=cmask_w, other=INF, cache_modifier=".cg").to(tl.float32)
            v1 = tl.load(base_cw + (h + 1) * sxh, mask=cmask_w, other=INF, cache_modifier=".cg").to(tl.float32)
            v2 = tl.load(base_cw + (h + 2) * sxh, mask=cmask_w, other=INF, cache_modifier=".cg").to(tl.float32)
            v3 = tl.load(base_cw + (h + 3) * sxh, mask=cmask_w, other=INF, cache_modifier=".cg").to(tl.float32)

            # Reduce across C-tile
            min0 = tl.minimum(min0, tl.min(v0, axis=0))
            min1 = tl.minimum(min1, tl.min(v1, axis=0))
            min2 = tl.minimum(min2, tl.min(v2, axis=0))
            min3 = tl.minimum(min3, tl.min(v3, axis=0))

            c_start += BLOCK_C

        # Accumulate valid results into acc with a single mask application
        sum_mins = (min0 + min1) + (min2 + min3)
        acc += tl.where(mask_w, sum_mins, 0.0)
        h += 4

    # Handle remaining rows if H % 4 != 0
    while h < H:
        cur_min = tl.full((BLOCK_W,), INF, dtype=tl.float32)
        c_start = 0
        while c_start < C:
            c_tile = c_start + tl.arange(0, BLOCK_C)
            mask_ct = c_tile < C
            base_cw = (
                x_ptr
                + x_base_n
                + c_tile[:, None] * sxc
                + w_ptrs[None, :]
            )
            x_vals = tl.load(base_cw + h * sxh, mask=mask_ct[:, None] & mask_w[None, :], other=INF, cache_modifier=".cg").to(tl.float32)
            cur_min = tl.minimum(cur_min, tl.min(x_vals, axis=0))
            c_start += BLOCK_C
        acc += tl.where(mask_w, cur_min, 0.0)
        h += 1

    # Apply GELU: 0.5*x*(1+erf(x/sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    gelu_vals = 0.5 * acc * (1.0 + libdevice.erf(acc * inv_sqrt2))

    # Load bias for the channel tile
    bias_vals = tl.load(bias_ptr + c_offsets * sbc, mask=mask_c, other=0.0).to(tl.float32)

    # Write out for this channel tile with bias broadcasting
    out_ptrs = (
        y_ptr
        + out_base_n
        + c_offsets[:, None] * syc
        + w_offsets[None, :] * syw
    )
    out_tile = gelu_vals[None, :] + bias_vals[:, None]
    tl.store(out_ptrs, out_tile, mask=mask_c[:, None] & mask_w[None, :])


class ModelNew(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    Fused with a Triton kernel for post-convolution operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Conv transpose via PyTorch/cuDNN
        x = self.conv_transpose(x)  # [N, C, H, W]

        # Allocate output: after ops -> [N, C, 1, W]
        N, C, H, W = x.shape
        y = torch.empty((N, C, 1, W), device=x.device, dtype=x.dtype)

        # Ensure contiguous for correct striding
        x_c = x.contiguous()
        b_c = self.bias.contiguous()

        # Launch Triton kernel: fuse min over C, sum over H, GELU, and bias add
        BLOCK_W = 64
        BLOCK_C = 32
        grid = (N, triton.cdiv(W, BLOCK_W), triton.cdiv(C, BLOCK_C))

        _fused_min_sum_gelu_add_bias[grid](
            x_c, b_c, y,
            N, C, H, W,
            x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
            b_c.stride(0),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            BLOCK_W=BLOCK_W,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )
        return y


batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]