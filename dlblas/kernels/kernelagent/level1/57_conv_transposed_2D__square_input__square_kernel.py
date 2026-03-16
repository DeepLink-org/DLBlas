import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _flip_transpose_weight_kernel(
    w_in_ptr,   # *T  [IC, OC_per_g, K, K]
    w_out_ptr,  # *T  [OC, IC_per_g, K, K]
    IC,         # in_channels (total)
    OC,         # out_channels (total)
    K,          # kernel size (square)
    n_elements, # OC * (IC // G) * K * K
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    # Infer groups
    total = IC * OC * K * K
    G = total // n_elements
    oc_per_g = OC // G
    ic_per_g = IC // G

    # Decode linear index for OUT layout: (OC, IC_per_g, K, K)
    # Use fewer div/mod: first isolate spatial kk then split channel dims
    K2 = K * K
    p = offs // K2         # flatten (OC, IC_per_g)
    kk = offs - p * K2     # faster than offs % K2
    icg = p % ic_per_g
    oc = p // ic_per_g

    # group index and local indices
    g = oc // oc_per_g
    ocg = oc - g * oc_per_g
    ic = g * ic_per_g + icg

    # spatial flip: (kh, kw) -> (K-1-kh, K-1-kw) encoded via kk
    src_kk = (K2 - 1) - kk

    # Strides for input layout (IC, OC_per_g, K, K)
    s_ic_in = oc_per_g * K2
    s_ocg_in = K2

    src_index = ic * s_ic_in + ocg * s_ocg_in + src_kk

    vals = tl.load(w_in_ptr + src_index, mask=mask, other=0)
    tl.store(w_out_ptr + offs, vals, mask=mask)


class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        # Cache for converted conv2d weights to avoid redundant kernel launches
        self._w_cache = None
        self._w_cache_version = None
        self._w_cache_key = None  # (device, dtype, shape, groups)

    def _weight_to_conv2d(self, weight: torch.Tensor, groups: int) -> torch.Tensor:
        """
        Convert ConvTranspose2d weight (IC, OC_per_g, K, K) into Conv2d weight (OC, IC_per_g, K, K)
        with spatial flip. Supports groups.
        """
        IC, OC_per_g, K, K2 = weight.shape
        assert K == K2, "Kernel must be square"
        OC = self.conv_transpose2d.out_channels
        IC_per_g = IC // groups
        assert OC_per_g * groups == OC, "Mismatch in grouped channels"

        n_elements = OC * IC_per_g * K * K

        w_in = weight.contiguous()
        w_out = torch.empty((OC, IC_per_g, K, K), device=w_in.device, dtype=w_in.dtype)

        if TRITON_AVAILABLE and w_in.is_cuda and w_out.is_cuda:
            BLOCK = 4096
            grid = lambda META: ((n_elements + BLOCK - 1) // BLOCK,)
            _flip_transpose_weight_kernel[grid](
                w_in, w_out, IC, OC, K, n_elements, BLOCK=BLOCK, num_warps=4, num_stages=2
            )
        else:
            # CPU / non-cuda fallback
            g = groups
            w = w_in.view(g, IC // g, OC // g, K, K)
            w = w.flip(-1, -2).permute(0, 2, 1, 3, 4).contiguous()
            w_out.copy_(w.view(OC, IC // g, K, K))

        return w_out

    def _get_cached_w_conv(self) -> torch.Tensor:
        Wt = self.conv_transpose2d.weight
        ct = self.conv_transpose2d
        key = (Wt.device, Wt.dtype, tuple(Wt.shape), ct.groups)
        if (
            self._w_cache is None
            or self._w_cache_version != Wt._version
            or self._w_cache_key != key
        ):
            self._w_cache = self._weight_to_conv2d(Wt, ct.groups)
            self._w_cache_version = Wt._version
            self._w_cache_key = key
        return self._w_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        convt = self.conv_transpose2d

        # Normalize params to tuples
        stride = convt.stride if isinstance(convt.stride, tuple) else (convt.stride, convt.stride)
        padding = convt.padding if isinstance(convt.padding, tuple) else (convt.padding, convt.padding)
        out_pad = convt.output_padding if isinstance(convt.output_padding, tuple) else (convt.output_padding, convt.output_padding)
        dilation = convt.dilation if isinstance(convt.dilation, tuple) else (convt.dilation, convt.dilation)

        # Fast path: stride=1, dilation=1, output_padding=0
        # Identity: conv_transpose2d(x, W, P) == conv2d(x, flip(W).permute(1,0,2,3), padding=K-1-P), with groups
        if (
            TRITON_AVAILABLE
            and x.is_cuda
            and stride == (1, 1)
            and dilation == (1, 1)
            and out_pad == (0, 0)
        ):
            K = convt.kernel_size if isinstance(convt.kernel_size, tuple) else (convt.kernel_size, convt.kernel_size)
            assert K[0] == K[1], "Kernel must be square"
            k = K[0]
            qh = k - 1 - padding[0]
            qw = k - 1 - padding[1]
            if qh >= 0 and qw >= 0:
                W_conv = self._get_cached_w_conv()

                # Use NHWC fast path when available to improve cuDNN performance on Hopper
                x_fast = x
                if x_fast.is_cuda:
                    x_fast = x_fast.contiguous(memory_format=torch.channels_last)

                y = F.conv2d(
                    x_fast,
                    W_conv,
                    bias=convt.bias,
                    stride=1,
                    padding=(qh, qw),
                    dilation=1,
                    groups=convt.groups,
                )
                return y  # same values/shape as reference

        # General-case fallback to preserve exact PyTorch semantics
        return convt(x)


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
width = 128
height = 128

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization