import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _fused_bias_div_swish_flat_kernel(
    x_ptr,        # *float32, flattened [M*N]
    y_ptr,        # *float32, flattened [M*N] (can alias x_ptr for in-place)
    bias_ptr,     # *float32, shape (1,) scalar bias
    inv_div,      # float32 scalar = 1.0 / divide_value
    N_ELEMENTS,   # total number of elements = M * N
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load scalar bias once per program
    b = tl.load(bias_ptr)

    # z = (x + b) * inv_div
    z = (x + b) * inv_div

    # Swish: z * sigmoid(z)
    s = 1.0 / (1.0 + tl.exp(-z))
    y = z * s

    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(nn.Module):
    """
    Model that performs a matrix multiplication, batch normalization, bias addition, division, and Swish activation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, bias_shape=(1,), divide_value=1.0):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = float(divide_value)

    def forward(self, x):
        x = self.matmul(x)
        x = self.bn(x)

        # Fuse: +bias, /divide_value, and Swish activation using Triton
        if x.is_cuda:
            # Ensure bias resides on the same device without host syncs
            bias_dev = self.bias if self.bias.device == x.device else self.bias.to(device=x.device)

            # Use a flat 1D kernel over all elements for maximal coalescing
            N_elems = x.numel()
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(N_elems, BLOCK_SIZE),)

            inv_div = 1.0 / self.divide_value

            # In-place to reduce memory traffic
            _fused_bias_div_swish_flat_kernel[grid](
                x, x, bias_dev, inv_div, N_elems,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=8,
                num_stages=3,
            )
        else:
            # CPU path preserves exact semantics
            x = x + self.bias
            x = x / self.divide_value
            x = x * torch.sigmoid(x)
        return x


batch_size = 128
in_features = 1024
out_features = 512
bn_eps = 1e-5
bn_momentum = 0.1
bias_shape = (1,)
divide_value = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bn_eps, bn_momentum, bias_shape, divide_value]