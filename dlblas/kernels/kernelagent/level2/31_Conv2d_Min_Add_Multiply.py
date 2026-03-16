import torch
import torch.nn as nn

# Try to import Triton; provide a safe fallback if unavailable
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# Fused Triton kernel: y = (min(x, const_value) + bias[c]) * scaling_factor
@triton.jit
def _fused_min_bias_scale_kernel(
    x_ptr,          # *x dtype tensor (N*C*H*W)
    bias_ptr,       # *x dtype tensor (C,)
    out_ptr,        # *x dtype tensor (N*C*H*W)
    n_elements,     # total number of elements (N*C*H*W)
    hw,             # H*W
    C,              # channels
    const_value,    # scalar float
    scaling,        # scalar float
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0)

    # Compute channel index for bias broadcasting: c = (index // (H*W)) % C
    hw_i = tl.full([1], hw, dtype=tl.int32)
    C_i = tl.full([1], C, dtype=tl.int32)
    offs_i = offs.to(tl.int32)
    c_idx = (offs_i // hw_i) % C_i

    # Load bias per-channel
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0)

    # Ensure scalar constants match input dtype
    const_val = tl.full([1], const_value, x.dtype)
    scale_val = tl.full([1], scaling, x.dtype)

    # Fused compute: y = (min(x, const) + b) * scale
    y = tl.where(x < const_val, x, const_val)
    y = (y + b) * scale_val

    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    """
    Simple model that performs a convolution, takes the minimum with a constant, adds a bias term, and multiplies by a scaling factor.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = float(constant_value)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.conv(x)

        # If Triton is available and tensor is on CUDA, run the fused kernel for post-conv ops
        if _TRITON_AVAILABLE and x.is_cuda:
            # Ensure contiguous memory for linear indexing
            x = x.contiguous()
            N, C, H, W = x.shape
            n_elements = x.numel()
            hw = H * W

            # Bias is (C,1,1) -> flatten to (C,) for easy indexing
            bias_1d = self.bias.reshape(-1).to(dtype=x.dtype, device=x.device)

            # In-place compute: write results back to x
            BLOCK_SIZE = 4096
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            _fused_min_bias_scale_kernel[grid](
                x,                      # x_ptr
                bias_1d,                # bias_ptr
                x,                      # out_ptr (in-place)
                n_elements,
                hw,
                C,
                self.constant_value,
                self.scaling_factor,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
            )
            return x
        else:
            # Fallback PyTorch implementation (exact original semantics)
            x = torch.min(x, torch.tensor(self.constant_value))
            x = x + self.bias
            x = x * self.scaling_factor
            return x

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]