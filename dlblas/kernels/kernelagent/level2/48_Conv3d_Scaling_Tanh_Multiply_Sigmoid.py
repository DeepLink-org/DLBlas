import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_pointwise_ncdhw_kernel(
    x_ptr,           # *f32
    sf_ptr,          # *f32, shape [C]
    bias_ptr,        # *f32, shape [C]
    out_ptr,         # *f32
    n_elements,      # int
    C,               # int
    DHW,             # int = D*H*W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    m = offs < n_elements

    # Hints for better vectorization/coalescing
    tl.multiple_of(offs, 16)
    tl.max_contiguous(offs, 16)

    # Load input
    x = tl.load(x_ptr + offs, mask=m, other=0.0)

    # Compute per-block channel info. For most practical cases when BLOCK_SIZE <= DHW,
    # each block touches at most two channels. We exploit this to broadcast per-channel
    # parameters instead of gathering them for every element.
    # Compute first/last indices for this block (clamp last to n_elements-1).
    last_pred = block_start + BLOCK_SIZE - 1
    last_idx = tl.where(last_pred < n_elements, last_pred, n_elements - 1)

    c0 = (block_start // DHW) % C
    c1 = (last_idx // DHW) % C

    # Number of elements in this block that still belong to channel c0
    within_tile = block_start % DHW
    cut = DHW - within_tile  # distance to next channel boundary
    cut = tl.where(cut < BLOCK_SIZE, cut, BLOCK_SIZE)

    # Decide path: if DHW >= BLOCK_SIZE, at most one boundary inside the block
    use_two_scalar_path = DHW >= BLOCK_SIZE

    if use_two_scalar_path:
        # Two-scalar broadcast path: load at most two scalars per parameter.
        sf0 = tl.load(sf_ptr + c0)
        b0 = tl.load(bias_ptr + c0)
        sf1 = tl.load(sf_ptr + c1)
        b1 = tl.load(bias_ptr + c1)

        idx = tl.arange(0, BLOCK_SIZE)
        use_second = idx >= cut
        sf = tl.where(use_second, sf1, sf0)
        b = tl.where(use_second, b1, b0)
    else:
        # General gather path (handles cases when BLOCK_SIZE > DHW).
        c_idx = (offs // DHW) % C
        sf = tl.load(sf_ptr + c_idx, mask=m, other=0.0)
        b = tl.load(bias_ptr + c_idx, mask=m, other=0.0)

    # Fused pointwise:
    # 1) scale
    x = x * sf
    # 2) tanh(x) via 2*sigmoid(2x) - 1 (one exp)
    sig2x = 1.0 / (1.0 + tl.exp(-2.0 * x))
    x = 2.0 * sig2x - 1.0
    # 3) multiply by bias
    x = x * b
    # 4) sigmoid
    x = 1.0 / (1.0 + tl.exp(-x))

    # Store
    tl.store(out_ptr + offs, x, mask=m)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)

        # Fast fused pointwise with Triton on CUDA; fallback to PyTorch otherwise.
        if x.is_cuda:
            x = x.contiguous()
            N, C, D, H, W = x.shape
            n_elements = x.numel()
            DHW = D * H * W

            # Flatten channel-wise params to [C]
            sf = self.scaling_factor.reshape(C).contiguous()
            bs = self.bias.reshape(C).contiguous()

            # In-place compute to avoid extra allocation
            BLOCK = 4096
            grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
            _fused_pointwise_ncdhw_kernel[grid](
                x, sf, bs, x,
                n_elements, C, DHW,
                BLOCK_SIZE=BLOCK,
                num_warps=8,
            )
            return x

        # CPU or non-CUDA fallback (exact original semantics)
        x = x * self.scaling_factor 
        x = torch.tanh(x)
        x = x * self.bias
        x = torch.sigmoid(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]