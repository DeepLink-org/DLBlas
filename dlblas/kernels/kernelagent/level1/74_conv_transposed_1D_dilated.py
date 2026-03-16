import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _conv_transpose1d_kernel(
    x_ptr,            # *f32, [B, Cin, Lin]
    w_ptr,            # *f32, [Cin, Cout, K]
    bias_ptr,         # *f32, [Cout] (optional)
    y_ptr,            # *f32, [B, Cout, Lout]
    B: tl.constexpr,  # batch size
    Cin: tl.constexpr,              # in channels
    Cout,             # out channels
    Lin,              # input length
    Lout,             # output length
    K: tl.constexpr,  # kernel size
    STRIDE: tl.constexpr,           # stride
    PADDING: tl.constexpr,          # padding
    DILATION: tl.constexpr,         # dilation
    HAS_BIAS: tl.constexpr,         # whether to add bias
    BLOCK_COUT: tl.constexpr,       # tile size along out-channels
    BLOCK_T: tl.constexpr,          # tile size along time dimension
):
    # program ids
    pid0 = tl.program_id(axis=0)  # over (B, T-blocks)
    pid1 = tl.program_id(axis=1)  # over Cout blocks

    # decompose pid0 into batch and time-block id
    T_BLOCKS = tl.cdiv(Lout, BLOCK_T)
    b = pid0 // T_BLOCKS
    tb = pid0 % T_BLOCKS

    # tile offsets
    oc_offsets = pid1 * BLOCK_COUT + tl.arange(0, BLOCK_COUT)
    t_offsets = tb * BLOCK_T + tl.arange(0, BLOCK_T)

    oc_mask = oc_offsets < Cout
    t_mask = t_offsets < Lout

    # accumulators
    acc = tl.zeros((BLOCK_COUT, BLOCK_T), dtype=tl.float32)

    # base pointers per batch
    x_batch_base = (b * Cin) * Lin
    y_batch_base = (b * Cout) * Lout

    # Precompute base for stride==1 path
    base_t = t_offsets + PADDING

    # Loop over in-channels and kernel elements (compile-time unrolled)
    for ic in tl.static_range(0, Cin):
        x_ic_base = x_batch_base + ic * Lin
        w_ic_base = (ic * Cout) * K
        w_ptr_ic = w_ptr + w_ic_base + oc_offsets * K

        if STRIDE == 1:
            # Fast path: no div/mod
            for k in tl.static_range(0, K):
                t_in = base_t - k * DILATION
                vmask = (t_in >= 0) & (t_in < Lin) & t_mask
                t_in_safe = tl.where(vmask, t_in, 0)
                x_vals = tl.load(x_ptr + x_ic_base + t_in_safe, mask=vmask, other=0.0).to(tl.float32)
                w_vec = tl.load(w_ptr_ic + k, mask=oc_mask, other=0.0).to(tl.float32)
                acc += w_vec[:, None] * x_vals[None, :]
        else:
            # General path: require divisibility by stride
            for k in tl.static_range(0, K):
                n_vec = base_t - k * DILATION
                div_ok = (n_vec % STRIDE) == 0
                t_in = n_vec // STRIDE
                vmask = div_ok & (t_in >= 0) & (t_in < Lin) & t_mask
                t_in_safe = tl.where(vmask, t_in, 0)
                x_vals = tl.load(x_ptr + x_ic_base + t_in_safe, mask=vmask, other=0.0).to(tl.float32)
                w_vec = tl.load(w_ptr_ic + k, mask=oc_mask, other=0.0).to(tl.float32)
                acc += w_vec[:, None] * x_vals[None, :]

    # Add bias if present
    if HAS_BIAS:
        b_vec = tl.load(bias_ptr + oc_offsets, mask=oc_mask, other=0.0).to(tl.float32)
        acc += b_vec[:, None]

    # Store results
    y_idx = y_batch_base + oc_offsets[:, None] * Lout + t_offsets[None, :]
    mask_2d = oc_mask[:, None] & t_mask[None, :]
    tl.store(y_ptr + y_idx, acc, mask=mask_2d)


class ModelNew(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Keep PyTorch module to own parameters and ensure drop-in API compatibility
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        # Fallback to PyTorch for non-CUDA or unsupported dtypes
        if (not x.is_cuda) or (x.dtype not in (torch.float32,)):
            return self.conv1d_transpose(x)

        w = self.conv1d_transpose.weight.contiguous()  # [Cin, Cout, K]
        b = self.conv1d_transpose.bias
        stride = self.conv1d_transpose.stride[0]
        padding = self.conv1d_transpose.padding[0]
        dilation = self.conv1d_transpose.dilation[0]
        out_pad = self.conv1d_transpose.output_padding[0] if hasattr(self.conv1d_transpose, "output_padding") else 0

        B, Cin, Lin = x.shape
        Cout = w.shape[1]
        K = w.shape[2]

        # Output length formula from PyTorch docs
        Lout = (Lin - 1) * stride - 2 * padding + dilation * (K - 1) + out_pad + 1

        # Ensure contiguous inputs
        x_c = x.contiguous()
        # Output buffer
        y = torch.empty((B, Cout, Lout), device=x.device, dtype=torch.float32)

        # Tuned tile sizes for better occupancy on H200
        BLOCK_T = 128
        BLOCK_COUT = 64
        grid = (triton.cdiv(Lout, BLOCK_T) * B, triton.cdiv(Cout, BLOCK_COUT))

        _conv_transpose1d_kernel[grid](
            x_c, w, (b if b is not None else y), y,
            B, Cin, Cout, Lin, Lout, K,
            stride, padding, dilation,
            HAS_BIAS=(b is not None),
            BLOCK_COUT=BLOCK_COUT,
            BLOCK_T=BLOCK_T,
            num_warps=8,
            num_stages=3,
        )

        # Cast back to original dtype if needed
        if y.dtype != x.dtype:
            y = y.to(dtype=x.dtype)
        return y

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 5
length = 256
stride = 1
padding = 0
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]