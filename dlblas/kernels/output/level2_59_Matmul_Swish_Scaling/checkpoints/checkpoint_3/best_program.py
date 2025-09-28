# ================== EVOLVE-BLOCK-START ==================
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernel for fused linear + swish + scale
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=8),
    ],
    key=['out_features', 'in_features'],
)
@triton.jit
def _fused_linear_swish_kernel(
    # Pointers to matrices
    x_ptr, w_ptr, b_ptr, output_ptr,
    # Matrix dimensions
    batch_size, in_features, out_features,
    # Strides
    stride_x0, stride_x1,
    stride_w0, stride_w1,
    stride_out0, stride_out1,
    scaling_factor: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Map program ids to the block of the output they should compute.
    pid = tl.program_id(0)
    num_out_blocks = tl.cdiv(out_features, BLOCK_SIZE_N)
    batch_idx = pid // num_out_blocks
    out_block_idx = pid % num_out_blocks

    # This program will compute a block of the output for the given batch index and a block of the out_features.
    offs_out = out_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Create pointers for the weight block: [BLOCK_SIZE_N, BLOCK_SIZE_K]
    w_ptrs = w_ptr + offs_out[:, None] * stride_w0 + offs_k[None, :] * stride_w1

    # Accumulator for the output block
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Loop over the in_features dimension in blocks of BLOCK_SIZE_K
    num_k_blocks = tl.cdiv(in_features, BLOCK_SIZE_K)
    for k in range(0, num_k_blocks):
        k_offs = k * BLOCK_SIZE_K
        # Load a block of x for the current batch and the current K block.
        x_ptrs = x_ptr + batch_idx * stride_x0 + (k_offs + offs_k) * stride_x1
        mask_x = (k_offs + offs_k) < in_features
        x_block = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Load a block of weights for the current K block and the current out block.
        mask_w = (offs_out[:, None] < out_features) & ((k_offs + offs_k)[None, :] < in_features)
        w_block = tl.load(w_ptrs + k_offs * stride_w1, mask=mask_w, other=0.0)

        # Compute the dot product for this block
        acc += tl.sum(w_block * x_block[None, :], axis=1)

    # Load bias for the current out block
    b_ptrs = b_ptr + offs_out
    mask_bias = offs_out < out_features
    bias = tl.load(b_ptrs, mask=mask_bias, other=0.0)
    acc += bias

    # Apply Swish: acc * sigmoid(acc)
    swish_output = acc * tl.sigmoid(acc)
    # Scale
    swish_output = swish_output * scaling_factor

    # Write back the result
    out_ptrs = output_ptr + batch_idx * stride_out0 + offs_out * stride_out1
    tl.store(out_ptrs, swish_output, mask=mask_bias)

def fused_linear_swish(x, weight, bias, scaling_factor):
    # Check sizes
    batch, in_features = x.shape
    out_features, _ = weight.shape
    # Allocate output
    output = torch.empty((batch, out_features), device=x.device, dtype=x.dtype)
    # The grid is batch * ceil(out_features / 64) to ensure sufficient coverage
    grid = (batch * triton.cdiv(out_features, 64),)
    # Launch kernel without passing BLOCK_SIZE parameters
    _fused_linear_swish_kernel[grid](
        x, weight, bias, output,
        batch, in_features, out_features,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        scaling_factor,
    )
    return output

class ModelNew(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Swish activation, and scales the result.
    Uses a fused Triton kernel for the operation.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        # initialize the parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # We use the fused kernel
        return fused_linear_swish(x, self.weight, self.bias, self.scaling_factor)

batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
# =================== EVOLVE-BLOCK-END ===================