import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 128, "BLOCK_R": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 128, "BLOCK_R": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 256, "BLOCK_R": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 128}, num_warps=4, num_stages=2),
    ],
    key=["C"],
)
@triton.jit
def _softmax_sigmoid_fused_5d(
    x_ptr, y_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_R
    rows = row_start + tl.arange(0, BLOCK_R)
    total_rows = N * D * H * W
    row_mask = rows < total_rows

    # Decompose rows -> (n, d, h, w)
    w_idx = rows % W
    tmp = rows // W
    h_idx = tmp % H
    tmp = tmp // H
    d_idx = tmp % D
    n_idx = tmp // D

    # Base offsets for each row
    base = (n_idx * stride_n + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w).to(tl.int64)

    # Fast path: if the whole channel dimension fits in one tile, do a single load/compute/store
    if C <= BLOCK_C:
        ch = tl.arange(0, BLOCK_C)
        ch_mask = ch < C
        ptrs = x_ptr + base[:, None] + (ch[None, :] * stride_c)
        x = tl.load(ptrs, mask=row_mask[:, None] & ch_mask[None, :], other=-float("inf"))
        x_f32 = x.to(tl.float32)
        m = tl.max(x_f32, axis=1)
        e = tl.exp(x_f32 - m[:, None])
        l = tl.sum(e, axis=1)
        soft = e * (1.0 / l[:, None])
        sig = 1.0 / (1.0 + tl.exp(-soft))
        out_ptrs = y_ptr + base[:, None] + (ch[None, :] * stride_c)
        tl.store(out_ptrs, sig, mask=row_mask[:, None] & ch_mask[None, :])
        return

    # General path: loop over channels to compute max, sumexp, then write
    m = tl.full((BLOCK_R,), -float("inf"), dtype=tl.float32)
    c0 = 0
    # Pass 1: max
    while c0 < C:
        ch = c0 + tl.arange(0, BLOCK_C)
        ch_mask = ch < C
        ptrs = x_ptr + base[:, None] + (ch[None, :] * stride_c)
        x = tl.load(ptrs, mask=row_mask[:, None] & ch_mask[None, :], other=-float("inf"))
        x = x.to(tl.float32)
        m = tl.maximum(m, tl.max(x, axis=1))
        c0 += BLOCK_C

    # Pass 2: sum of exp(x - max)
    l = tl.zeros((BLOCK_R,), dtype=tl.float32)
    c0 = 0
    while c0 < C:
        ch = c0 + tl.arange(0, BLOCK_C)
        ch_mask = ch < C
        ptrs = x_ptr + base[:, None] + (ch[None, :] * stride_c)
        x = tl.load(ptrs, mask=row_mask[:, None] & ch_mask[None, :], other=-float("inf"))
        x = x.to(tl.float32)
        e = tl.exp(x - m[:, None])
        l += tl.sum(e, axis=1)
        c0 += BLOCK_C

    inv_l = 1.0 / l

    # Pass 3: write sigmoid(softmax)
    c0 = 0
    while c0 < C:
        ch = c0 + tl.arange(0, BLOCK_C)
        ch_mask = ch < C
        ptrs = x_ptr + base[:, None] + (ch[None, :] * stride_c)
        x = tl.load(ptrs, mask=row_mask[:, None] & ch_mask[None, :], other=-float("inf"))
        x = x.to(tl.float32)
        soft = tl.exp(x - m[:, None]) * inv_l[:, None]
        sig = 1.0 / (1.0 + tl.exp(-soft))
        out_ptrs = y_ptr + base[:, None] + (ch[None, :] * stride_c)
        tl.store(out_ptrs, sig, mask=row_mask[:, None] & ch_mask[None, :])
        c0 += BLOCK_C


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, applies Softmax and Sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        # Keep modules for fallback / parity
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D, H, W).
        """
        x = self.conv_transpose(x)

        # Use Triton fused kernel if on CUDA and float32, else fallback to PyTorch ops.
        if x.is_cuda and x.dtype == torch.float32:
            N, C, D, H, W = x.shape
            y = torch.empty_like(x)
            sN, sC, sD, sH, sW = x.stride()
            total_rows = N * D * H * W

            def grid(meta):
                return (triton.cdiv(total_rows, meta["BLOCK_R"]),)

            _softmax_sigmoid_fused_5d[grid](
                x, y,
                N, C, D, H, W,
                sN, sC, sD, sH, sW,
            )
            return y
        else:
            # Fallback path (numerically equivalent)
            x = self.softmax(x)
            x = self.sigmoid(x)
            return x


batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]