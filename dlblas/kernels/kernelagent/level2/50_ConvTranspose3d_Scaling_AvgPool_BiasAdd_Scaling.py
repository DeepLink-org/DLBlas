import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _avgpool3d_k2s2_bias_scale_fused(
    x_ptr,                # *float32 [N, C, D, H, W] contiguous NCDHW
    bias_ptr,             # *float32 [C]
    y_ptr,                # *float32 [N, C, D2, H2, W2] contiguous NCDHW
    N, C, D, H, W,        # input dims
    D2, H2, W2,           # output dims = floor(D/2), floor(H/2), floor(W/2)
    scale1,               # float
    scale2,               # float
    n_elements,           # total output elements = N*C*D2*H2*W2
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Fewer integer divisions for index de-linearization
    out_spatial = D2 * H2 * W2
    nc = offs // out_spatial                          # combined n*C + c
    rem = offs - nc * out_spatial
    t0 = rem // W2
    w2 = rem - t0 * W2
    d2 = t0 // H2
    h2 = t0 - d2 * H2
    c = nc % C                                        # for bias indexing

    # Map to input coordinates (stride=2, kernel=2, padding=0)
    w = w2 * 2
    h = h2 * 2
    d = d2 * 2

    # Strides for contiguous NCDHW
    stride_w = 1
    stride_h = W
    stride_d = H * W
    stride_nc = D * H * W

    # Linear base index for the (n, c, d, h, w) in input
    base = nc * stride_nc + d * stride_d + h * stride_h + w * stride_w

    # Offsets for the 2x2x2 kernel window
    o0 = 0
    o1 = 1
    o2 = stride_h
    o3 = stride_h + 1
    o4 = stride_d
    o5 = stride_d + 1
    o6 = stride_d + stride_h
    o7 = stride_d + stride_h + 1

    # Load 8 values
    x0 = tl.load(x_ptr + base + o0, mask=mask, other=0.0, cache_modifier=".ca")
    x1 = tl.load(x_ptr + base + o1, mask=mask, other=0.0, cache_modifier=".ca")
    x2 = tl.load(x_ptr + base + o2, mask=mask, other=0.0, cache_modifier=".ca")
    x3 = tl.load(x_ptr + base + o3, mask=mask, other=0.0, cache_modifier=".ca")
    x4 = tl.load(x_ptr + base + o4, mask=mask, other=0.0, cache_modifier=".ca")
    x5 = tl.load(x_ptr + base + o5, mask=mask, other=0.0, cache_modifier=".ca")
    x6 = tl.load(x_ptr + base + o6, mask=mask, other=0.0, cache_modifier=".ca")
    x7 = tl.load(x_ptr + base + o7, mask=mask, other=0.0, cache_modifier=".ca")

    # Pairwise reduction to shorten dependency chain
    s0 = x0 + x1
    s1 = x2 + x3
    s2 = x4 + x5
    s3 = x6 + x7
    s = (s0 + s1) + (s2 + s3)

    # Fuse scales to reduce arithmetic
    alpha = scale1 * scale2 * 0.125
    beta = scale2

    # Load channel bias and apply final affine transform
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = s * alpha + b * beta

    # Store to output
    tl.store(y_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, scaling, average pooling, bias addition, and scaling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

    def forward(self, x):
        x = self.conv_transpose(x)

        # Fast fused AvgPool3d(kernel=2, stride=2) + bias + final scaling using Triton.
        # Matches: x = x * self.scale1; x = avg_pool(x); x = x + self.bias; x = x * self.scale2
        # Fused as: out = (avg_pool(x * scale1) + bias) * scale2
        if x.is_cuda:
            # Ensure contiguous memory (NCDHW)
            x = x.contiguous()
            N, C, D, H, W = x.shape
            D2, H2, W2 = D // 2, H // 2, W // 2
            out = torch.empty((N, C, D2, H2, W2), device=x.device, dtype=x.dtype)

            # Bias expected shape: (C, 1, 1, 1); view to (C,)
            bias_1d = self.bias.view(C).contiguous()

            n_elements = N * C * D2 * H2 * W2
            BLOCK = 256
            grid = lambda META: ((n_elements + BLOCK - 1) // BLOCK,)

            _avgpool3d_k2s2_bias_scale_fused[grid](
                x, bias_1d, out,
                N, C, D, H, W,
                D2, H2, W2,
                float(self.scale1.item()),
                float(self.scale2.item()),
                n_elements,
                BLOCK_SIZE=BLOCK,
                num_warps=4,
                num_stages=2,
            )
            return out
        else:
            # CPU fallback: exact original semantics
            x = x * self.scale1
            x = self.avg_pool(x)
            x = x + self.bias
            x = x * self.scale2
            return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
scale1 = 0.5
scale2 = 1.0
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape]