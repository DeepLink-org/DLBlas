import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_post_ops_bias_kernel(
    x_ptr,             # *f32
    bias_ptr,          # *f32
    y_ptr,             # *f32
    n_elements,        # i32
    C,                 # i32
    stride_c,          # i32 (elements)
    bias_stride_c,     # i32 (elements)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Hints to compiler for better vectorization/coalescing
    tl.multiple_of(block_start, BLOCK_SIZE)
    tl.max_contiguous(offs, BLOCK_SIZE)

    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # 1) ReLU
    x = tl.maximum(x, 0.0)

    # 2) LeakyReLU after ReLU is a no-op; omit to save work while preserving semantics

    # 3) GELU (exact): 0.5 * u * (1 + erf(u / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    u = x
    e = tl.math.erf(u * inv_sqrt2)
    x = (u * (1.0 + e)) * 0.5

    # 4) Sigmoid: since x >= 0 after ReLU->GELU, use simplified stable form
    # Use exp2 for slightly faster evaluation: exp(-x) = exp2(-x * log2(e))
    LOG2E = 1.4426950408889634
    x = 1.0 / (1.0 + tl.exp2(-x * LOG2E))

    # Bias add with fast-path if the whole block lies within a single channel chunk
    rem_in_chan = stride_c - (block_start % stride_c)
    one_channel_block = rem_in_chan >= BLOCK_SIZE
    c0 = ((block_start // stride_c) % C).to(tl.int32)
    b0 = tl.load(bias_ptr + c0 * bias_stride_c)

    if one_channel_block:
        out = x + b0
    else:
        c_idx = (offs // stride_c) % C
        b = tl.load(bias_ptr + c_idx * bias_stride_c, mask=mask, other=0.0)
        out = x + b

    tl.store(y_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    Fused the post-conv elementwise ops + bias addition into a single Triton kernel for performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        # If on CUDA, fuse the activation chain and bias add via Triton kernel
        if x.is_cuda:
            y = x.contiguous()
            n, c, d, h, w = y.shape
            n_elements = y.numel()
            b = self.bias
            stride_c = y.stride(1)
            bias_stride_c = b.stride(0)
            BLOCK = 4096
            grid = lambda meta: (triton.cdiv(n_elements, BLOCK),)
            _fused_post_ops_bias_kernel[grid](
                y, b, y,
                n_elements,
                c,
                stride_c,
                bias_stride_c,
                BLOCK_SIZE=BLOCK,
                num_warps=8,
                num_stages=4,
            )
            return y
        else:
            # CPU fallback to exact PyTorch sequence for correctness
            x = torch.relu(x)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
            x = torch.nn.functional.gelu(x)
            x = torch.sigmoid(x)
            x = x + self.bias
            return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]