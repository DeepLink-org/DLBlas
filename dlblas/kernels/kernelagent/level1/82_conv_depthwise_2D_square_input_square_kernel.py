import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _dwconv2d_kernel(
    x_ptr,        # *f32 [N, C, H, W]
    w_ptr,        # *f32 [C, 1, K, K]
    b_ptr,        # *f32 [C] or dummy
    y_ptr,        # *f32 [N, C, H_OUT, W_OUT]
    N, C, H, W,
    H_OUT, W_OUT,
    S: tl.constexpr,     # stride (square)
    P: tl.constexpr,     # padding (square)
    K: tl.constexpr,     # kernel size (square)
    stride_xN, stride_xC, stride_xH, stride_xW,
    stride_wC, stride_wH, stride_wW,
    stride_yN, stride_yC, stride_yH, stride_yW,
    HAS_BIAS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program ids
    pid_nc = tl.program_id(0)      # over N*C
    pid_h = tl.program_id(1)       # over H_OUT
    pid_w = tl.program_id(2)       # over W_OUT tiles

    # derive n, c from pid_nc
    n = pid_nc // C
    c = pid_nc % C

    oh = pid_h  # one row per program on axis-1

    w_start = pid_w * BLOCK_W
    ow = w_start + tl.arange(0, BLOCK_W)
    out_mask = ow < W_OUT

    # base pointers/offsets
    y_base = n * stride_yN + c * stride_yC + oh * stride_yH
    x_plane_base = n * stride_xN + c * stride_xC

    # initialize accumulator
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # compute input top-left coordinate for this output row
    ih0 = oh * S - P
    iw0 = ow * S - P
    w_base_c = c * stride_wC

    # Fast path for common case: stride=1, padding=0 -> no interior bound checks needed
    if (S == 1) & (P == 0):
        # For this case, for all lanes with ow < W_OUT, all KxK taps are guaranteed in-bounds.
        for kh in tl.static_range(0, K):
            ih = oh + kh  # valid by construction for all lanes with out_mask
            x_row_ptr = x_ptr + x_plane_base + ih * stride_xH
            w_row_base = w_base_c + kh * stride_wH

            # start pointer for kw=0
            x_ptrs = x_row_ptr + iw0 * stride_xW
            # unrolled over kernel width with pointer bumping
            for kw in tl.static_range(0, K):
                x_val = tl.load(x_ptrs, mask=out_mask, other=0.0)
                w_val = tl.load(w_ptr + w_row_base + kw * stride_wW)
                acc += x_val * w_val
                x_ptrs += stride_xW
    else:
        # General path with full boundary checks
        for kh in tl.static_range(0, K):
            ih = ih0 + kh
            valid_h = (ih >= 0) & (ih < H)

            x_row_ptr = x_ptr + x_plane_base + ih * stride_xH
            w_row_base = w_base_c + kh * stride_wH

            # start pointer for kw=0 and advance each step
            x_ptrs = x_row_ptr + iw0 * stride_xW
            for kw in tl.static_range(0, K):
                # width validity for this kw
                valid_w = (iw0 + kw >= 0) & (iw0 + kw < W)
                mask = out_mask & valid_h & valid_w

                x_val = tl.load(x_ptrs, mask=mask, other=0.0)
                w_val = tl.load(w_ptr + w_row_base + kw * stride_wW)
                acc += x_val * w_val

                x_ptrs += stride_xW

    # add bias if present
    if HAS_BIAS:
        b_val = tl.load(b_ptr + c)
        acc += b_val

    # store result
    y_ptrs = y_ptr + y_base + ow * stride_yW
    tl.store(y_ptrs, acc, mask=out_mask)


class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # keep parameters & initialization identical to reference
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        # Fallback to PyTorch if not CUDA
        if not x.is_cuda:
            return self.conv2d(x)

        # Ensure contiguous tensors for predictable strides
        x = x.contiguous()
        w = self.conv2d.weight.contiguous()
        b = self.conv2d.bias
        if b is not None:
            b = b.contiguous()

        N, C, H, W = x.shape
        K = w.shape[-1]  # square kernel
        # use module stride/padding (square)
        S = self.conv2d.stride[0]
        P = self.conv2d.padding[0]

        # output dimensions (match PyTorch conv2d)
        H_OUT = (H + 2 * P - K) // S + 1
        W_OUT = (W + 2 * P - K) // S + 1

        y = torch.empty((N, C, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

        # strides in elements
        stride_xN, stride_xC, stride_xH, stride_xW = x.stride()
        stride_wC, _, stride_wH, stride_wW = w.stride()
        stride_yN, stride_yC, stride_yH, stride_yW = y.stride()

        # Grid: (N*C, H_OUT, ceil_div(W_OUT, BLOCK_W))
        # Keep mapping but optimize inner kernel
        BLOCK_W = 256
        grid = (
            N * C,
            H_OUT,
            triton.cdiv(W_OUT, BLOCK_W),
        )

        # Choose num_warps based on tile width
        num_warps = 8 if BLOCK_W >= 256 else 4

        _dwconv2d_kernel[grid](
            x, w, (b if b is not None else y), y,
            N, C, H, W,
            H_OUT, W_OUT,
            S=S, P=P, K=K,
            stride_xN=stride_xN, stride_xC=stride_xC, stride_xH=stride_xH, stride_xW=stride_xW,
            stride_wC=stride_wC, stride_wH=stride_wH, stride_wW=stride_wW,
            stride_yN=stride_yN, stride_yC=stride_yC, stride_yH=stride_yH, stride_yW=stride_yW,
            HAS_BIAS=(1 if b is not None else 0),
            BLOCK_W=BLOCK_W,
            num_warps=num_warps,
            num_stages=2,  # leaner pipeline for better occupancy on memory-bound kernel
        )
        return y


# Test code
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]